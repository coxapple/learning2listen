# base_models.py (수정 예시)

import matplotlib.pyplot as plt
import os
import seaborn as sns

import torch
import torch.nn as nn
from torch.nn import functional as F

from einops.layers.torch import Rearrange
from utils.base_model_util import get_activation, get_shape_list

EPSILON = 1e-10


class Norm(nn.Module):
    """ Norm Layer """
    def __init__(self, fn, size):
        super().__init__()
        self.norm = nn.LayerNorm(size, eps=1e-5)
        self.fn = fn

    def forward(self, x_data):
        if isinstance(x_data, dict):
            x_norm = self.fn({'x_a': x_data['x_a'],
                              'x_b': self.norm(x_data['x_b'])})
            return x_norm
        else:
            x, mask_info = x_data
            x_norm, _ = self.fn((self.norm(x), mask_info))
            return (x_norm, mask_info)

class Residual(nn.Module):
    """ Residual Layer """
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x_data):
        if isinstance(x_data, dict):
            x_resid = self.fn(x_data)['x_b']
            return {'x_a': x_data['x_a'], 'x_b': x_resid + x_data['x_b']}
        else:
            x, mask_info = x_data
            x_resid, _ = self.fn(x_data)
            return (x_resid + x, mask_info)

class MLP(nn.Module):
    """ MLP Layer """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.activation = get_activation("gelu")
        self.l2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x_data):
        if isinstance(x_data, dict):
            out = self.l2(self.activation(self.l1(x_data['x_b'])))
            return {'x_a': x_data['x_a'], 'x_b': out}
        else:
            x, mask_info = x_data
            out = self.l2(self.activation(self.l1(x)))
            return (out, mask_info)

class CrossModalAttention(nn.Module):
    """ Cross Modal Attention Layer
        Given 2 modalities (a, b), computes the K,V from modality b
        and Q from modality a.
    """
    def __init__(self, in_dim, dim, heads=8, in_dim2=None):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        if in_dim2 is not None:
            self.to_kv = nn.Linear(in_dim2, in_dim2 * 2, bias=False)
        else:
            self.to_kv = nn.Linear(in_dim, dim * 2, bias=False)
        self.to_q = nn.Linear(in_dim, dim, bias=False)

        if in_dim2 is not None:
            dim2 = int((in_dim + in_dim2 * 2) / 3)  # 임시 계산 로직
        else:
            dim2 = dim

        self.to_out = nn.Linear(dim2, dim)

        self.rearrange_qkv = Rearrange(
            "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads
        )
        self.rearrange_out = Rearrange("b h n d -> b n (h d)")

    def forward(self, x_data):
        x_a = x_data['x_a']
        x_b = x_data['x_b']

        kv = self.to_kv(x_b)
        q = self.to_q(x_a)
        qkv = torch.cat((q, kv), dim=-1)
        qkv = self.rearrange_qkv(qkv)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = F.softmax(dots, dim=-1)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = self.rearrange_out(out)
        out = self.to_out(out)

        return {'x_a': x_a, 'x_b': out}

class Attention(nn.Module):
    """ Vanilla Self-Attention Layer """
    def __init__(self, in_dim, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(in_dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.rearrange_qkv = Rearrange("b n (qkv h d) -> qkv b h n d",
                                       qkv=3, h=self.heads)
        self.rearrange_out = Rearrange("b h n d -> b n (h d)")

    def forward(self, x_data):
        x, mask_info = x_data
        max_mask = mask_info['max_mask']
        mask = mask_info['mask']

        qkv = self.to_qkv(x)
        qkv = self.rearrange_qkv(qkv)
        q, k, v = qkv[0], qkv[1], qkv[2]

        dots = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        if max_mask is not None:
            dots[:, :, :max_mask, :max_mask] = dots[:, :, :max_mask, :max_mask]\
                .masked_fill(mask == 0., float('-inf'))
        attn = F.softmax(dots, dim=-1)

        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = self.rearrange_out(out)
        out = self.to_out(out)
        return (out, mask_info)

class Transformer(nn.Module):
    """ Transformer class
        cross_modal=False -> normal Transformer
        cross_modal=True  -> CrossModalAttention
    """
    def __init__(self,
                 in_size=50,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 cross_modal=False,
                 in_dim2=None):
        super().__init__()
        blocks = []
        self.cross_modal = cross_modal

        if cross_modal:
            # CrossModal
            for i in range(num_hidden_layers):
                blocks.extend([
                    Residual(Norm(CrossModalAttention(
                        in_size, hidden_size, heads=num_attention_heads,
                        in_dim2=in_dim2),
                        hidden_size)),
                    Residual(Norm(MLP(hidden_size, hidden_size, intermediate_size),
                                   hidden_size))
                ])
        else:
            # Vanilla Self-attn
            for i in range(num_hidden_layers):
                blocks.extend([
                    Residual(Norm(Attention(in_size, hidden_size,
                                            heads=num_attention_heads),
                                   hidden_size)),
                    Residual(Norm(MLP(hidden_size, hidden_size, intermediate_size),
                                   hidden_size))
                ])
        self.net = nn.Sequential(*blocks)

    def forward(self, x_data):
        if self.cross_modal:
            assert isinstance(x_data, dict)
            x_data = self.net(x_data)
            x = x_data['x_b']
        else:
            x, mask_info = x_data
            x, _ = self.net((x, mask_info))
        return x

class LinearEmbedding(nn.Module):
    """ Linear Layer for embedding
        size: input feature dimension (e.g. 209)
        dim : output dimension (e.g. 200)
    """
    def __init__(self, size, dim):
        super().__init__()
        self.net = nn.Linear(size, dim)

    def forward(self, x):
        return self.net(x)

class AudioEmbedding(nn.Module):
    """ Audio embedding layer
        - version='v6': simple MaxPool1d(4) etc.
        - size=128 (mel)
        - quant_factor: how many times we pool
    """
    def __init__(self, size=128, dim=200, quant_factor=1, version='v6'):
        super().__init__()
        self.version = version
        self.net = nn.MaxPool1d(4)       # e.g. 256->64
        layers = []
        for _ in range(quant_factor-1):
            layers.append(nn.MaxPool1d(2))
        self.squasher = nn.Sequential(*layers)
        self.proj = nn.Linear(size, dim)

    def forward(self, x):
        # x shape: (B, size=128, T=256) if channel-first
        # or (B, T=256, size=128) if channel-last => then permute
        # ex) if we do x=x.permute(0,2,1) => (B,128,256)
        x = self.net(x)
        x = self.squasher(x)
        # now shape e.g. (B,128,64)
        # project feature dimension(128->dim)
        x = x.permute(0,2,1)
        x = self.proj(x)
        x = x.permute(0,2,1)
        return x

class PositionEmbedding(nn.Module):
    """Position Embedding Layer"""
    def __init__(self, seq_length, dim):
        super().__init__()
        # seq_length e.g. 64
        self.pos_embedding = nn.Parameter(torch.zeros(seq_length, dim))

    def forward(self, x):
        # x shape: (B, T, dim)
        # if T <= seq_length, we slice or x + pos[:T]
        B, T, D = x.shape
        if T > self.pos_embedding.shape[0]:
            # 경고 or 처리 (ex. pos_embedding 확장 필요)
            pass
        pos = self.pos_embedding[:T, :]
        return x + pos.unsqueeze(0)

class CrossModalLayer(nn.Module):
    """Cross Modal Layer inspired by FACT [Li 2021]"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        model_config = self.config['transformer']
        self.transformer_layer = Transformer(
            in_size=model_config['hidden_size'],
            hidden_size=model_config['hidden_size'],
            num_hidden_layers=model_config['num_hidden_layers'],
            num_attention_heads=model_config['num_attention_heads'],
            intermediate_size=model_config['intermediate_size']
        )

        output_layer_config = self.config['output_layer']
        self.cross_norm_layer = nn.LayerNorm(self.config['in_dim'])
        self.cross_output_layer = nn.Linear(self.config['in_dim'],
                                            output_layer_config['out_dim'],
                                            bias=False)
        self.cross_pos_embedding = PositionEmbedding(
            self.config["sequence_length"], self.config['in_dim']
        )

    def forward(self, modal_a_sequences, modal_b_sequences, mask_info):
        """
        modal_a_sequences : (B, T_a, D)
        modal_b_sequences : (B, T_b, D)
        merges them -> transformer -> linear -> logits
        """
        _, _, modal_a_width = get_shape_list(modal_a_sequences)
        merged_sequences = modal_a_sequences
        if modal_b_sequences is not None:
            _, _, modal_b_width = get_shape_list(modal_b_sequences)
            if modal_a_width != modal_b_width:
                raise ValueError("Modal A hidden size != Modal B hidden size")
            merged_sequences = torch.cat([merged_sequences, modal_b_sequences],
                                         axis=1)  # (B, T_a+T_b, D)

        # position emb
        merged_sequences = self.cross_pos_embedding(merged_sequences)
        merged_sequences = self.transformer_layer((merged_sequences, mask_info))
        merged_sequences = self.cross_norm_layer(merged_sequences)
        logits = self.cross_output_layer(merged_sequences)
        return logits
