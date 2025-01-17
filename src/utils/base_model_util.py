# base_model_util.py (수정 버전)

import numpy as np
import six
import torch
import torch.nn as nn
import torch.nn.functional as F

def dropout(input_tensor, dropout_prob):
    """
    Perform dropout in PyTorch style.
    Args:
      input_tensor: float Tensor (B, T, F) or similar.
      dropout_prob: Python float, probability of dropping out.
    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    # PyTorch usage: nn.Dropout(p=...) is a module that we call on input_tensor
    output = nn.Dropout(p=dropout_prob)(input_tensor)
    return output


def create_look_ahead_mask(seq_length, batch_size=0):
    """
    Create a look ahead mask given a certain seq length.
    Args:
      seq_length: int, the length of the sequence.
      batch_size: if > 0, the mask will be repeated for that batch size.
    Returns:
      mask shape: (batch_size, seq_length, seq_length) or (seq_length, seq_length)
                  with 1 = masked, 0 = unmasked
    """
    # Fix: replaced 'troch' -> 'torch'
    mask = 1 - torch.tril(torch.ones((seq_length, seq_length)))
    # shape = (seq_length, seq_length), upper triangular is 1
    if batch_size > 0:
        mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
    return mask


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """
    Create 3D attention mask from a 2D tensor mask.
    Args:
      from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
      to_mask: int32 Tensor of shape [batch_size, to_seq_length].
    Returns:
      float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor)
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask)
    to_seq_length = to_shape[1]

    to_mask = torch.reshape(to_mask, (batch_size, 1, to_seq_length)).float()

    # We create a tensor of all ones for "from_tensor", then multiply by to_mask
    broadcast_ones = torch.ones((batch_size, from_seq_length, 1), dtype=torch.float32)
    mask = broadcast_ones * to_mask
    return mask


def gelu(x):
    """
    Gaussian Error Linear Unit (GELU).
    https://arxiv.org/abs/1606.08415
    """
    cdf = 0.5 * (1.0 + torch.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3)))))
    return x * cdf


def get_activation(activation_string):
    """
    Maps a string to an activation function, e.g. "relu" => nn.ReLU, "gelu" => gelu
    """
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return nn.ReLU
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return torch.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_shape_list(tensor):
    """
    Returns a list/tuple of the shape of tensor, preferring static dimensions.
    If any dimension is None or dynamic, we stop with an assertion (original code).
    """
    shape = list(tensor.size())
    # If there's a dynamic axis, original code prints error + asserts False
    # We'll keep it as is:
    if any(s is None for s in shape):
        print('something wrong with static shaping')
        assert False
    return shape


def gather_indexes(sequence_tensor, positions):
    """
    Gathers the vectors at the specific positions over a minibatch.
    sequence_tensor: (batch_size, seq_length, width)
    positions: (batch_size, n_positions)
    Returns:
      output_tensor: (batch_size, n_positions, width)
    """
    sequence_shape = get_shape_list(sequence_tensor)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    # Replace torch.range -> torch.arange
    flat_offsets = torch.arange(0, batch_size, dtype=torch.int32) * seq_length
    flat_offsets = flat_offsets.unsqueeze(1)  # (batch_size, 1)

    flat_positions = positions + flat_offsets  # shape (batch_size, n_positions)
    flat_positions = flat_positions.reshape(-1)  # 1D

    # Flatten sequence_tensor to (batch_size*seq_length, width)
    flat_sequence_tensor = sequence_tensor.reshape(batch_size * seq_length, width)

    # Use torch.gather on dim=0
    # we need index to be shape (batch_size*n_positions)
    # or we do index_select
    # For demonstration, let's do index_select:
    output_tensor = torch.index_select(flat_sequence_tensor, 0, flat_positions)
    # Now shape = (batch_size*n_positions, width)
    # Reshape back
    n_positions = positions.size(1)
    output_tensor = output_tensor.reshape(batch_size, n_positions, width)

    return output_tensor


def split_heads(x, batch_size, seq_length, num_joints, num_attention_heads,
                model_depth):
    """
    Split the embedding vector for different heads for the spatial attention.
    x: (batch_size, seq_len, num_joints, model_depth) or
       (batch_size, seq_len, model_depth)
    Returns:
      the split vector (batch_size, seq_len, num_heads, num_joints, depth) or
      (batch_size, num_heads, seq_len, depth)
    """
    depth = model_depth // num_attention_heads
    x_shape = x.shape
    if len(x_shape) == 4:
        # x_shape = (B, seq_len, num_joints, model_depth)
        x = x.reshape(batch_size, seq_length, num_joints, num_attention_heads, depth)
        return x.permute(0, 1, 3, 2, 4)
    elif len(x_shape) == 3:
        # x_shape = (B, seq_len, model_depth)
        x = x.reshape(batch_size, seq_length, num_attention_heads, depth)
        return x.permute(0, 2, 1, 3)
    else:
        raise ValueError("Unsupported input tensor dimension.")


def scaled_dot_product_attention(q, k, v, mask):
    """
    The scaled dot product attention mechanism.
    Attn(Q, K, V) = softmax((QK^T + mask)/sqrt(d_k)) V
    """
    # In PyTorch: matmul_qk = q @ k.transpose(-1, -2) typically.
    # This code is incomplete in the snippet, so we keep it as is.
    matmul_qk = q @ k.transpose(-1, -2)

    dk = k.shape[-1]
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(dk, dtype=torch.float32))

    if mask is not None:
        scaled_attention_logits = scaled_attention_logits + (mask * -1e9)

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = attention_weights @ v
    return output, attention_weights
