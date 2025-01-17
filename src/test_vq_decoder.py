import argparse
import json
import logging
import numpy as np
import os
import pickle
import scipy.io as sio

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.autograd import Variable

from modules.fact_model import setup_model, calc_logit_loss
from vqgan.vqmodules.gan_models import setup_vq_transformer
from utils.load_utils import *


def run_model(args, config, l_vq_model, generator, test_X, test_Y, test_audio,
              seq_len, patch_size, rng=None):
    """
    Modified for seq_len=64, patch_size=8.
    Generates future listener motion autoregressively.
    """

    batch_size = config['batch_size']

    #디버그
    total_samples = test_X.shape[0]
    expected_batches = total_samples // batch_size
    if total_samples % batch_size != 0:
        expected_batches += 1
    print(f"DEBUG: Expected total batches: {expected_batches}")



    batchinds = np.arange(test_X.shape[0] // min(test_X.shape[0], batch_size))

    max_mask_len = config['fact_model']['cross_modal_model']['max_mask_len']
    cut_point = config['fact_model']['listener_past_transformer_config']['sequence_length']

    cut_point = 4

    # listener 과거 구간 64, patch_size=8 => 512
    past_cut_point = cut_point * patch_size
    start_t = step_t = patch_size

    output_pred = output_gt = output_probs = None

    for bii, bi in enumerate(batchinds):
        #디버그
        print(f"DEBUG: Processing batch {bii} with start index {bi * batch_size}")
        idxStart = bi * batch_size
        speakerData_np = test_X[idxStart:(idxStart + batch_size), :, :]
        listenerData_np = test_Y[idxStart:(idxStart + batch_size), :, :]
        audioData_np = test_audio[idxStart:(idxStart + batch_size), :, :]

        # 첫 segment: listenerData_np[:,:seq_len,:] = 0 => test scenario
        listenerData_np[:, :seq_len, :] *= 0.

        # 첫 patch 예측
        prediction, probs, inputs, quant_size = generate_prediction(
            config, args, l_vq_model, generator,
            speakerData_np[:, : (seq_len + patch_size), :],
            listenerData_np[:, : seq_len, :],
            audioData_np[:, : (seq_len + patch_size) * 4, :],
            seq_len, patch_size, 0, cut_point
        )

        # [DEBUG] 첫 patch 결과 확인
        print(f"[DEBUG] (bii={bii}) first patch prediction.shape:", prediction.shape)
        print(f"[DEBUG] (bii={bii}) first patch probs.shape:", probs.shape)
        print(f"[DEBUG] (bii={bii}) inputs['listener_past'].shape:", inputs['listener_past'].shape)
        print(f"[DEBUG] (bii={bii}) quant_size:", quant_size)

        # listener past + 첫 prediction 합치기
        prediction = torch.cat((inputs['listener_past'], prediction[:, 0]), dim=-1)
        probs = torch.cat((
            torch.zeros((probs.shape[0], inputs['listener_past'].shape[1], probs.shape[2])).cuda(),
            probs[:, [0], :]
        ), dim=1)

        # [DEBUG] concat 후 shape
        print(f"[DEBUG] (bii={bii}) after cat => prediction.shape:", prediction.shape)

        # autoregressive로 계속 8프레임씩 예측
        for t in range(start_t, test_X.shape[1] - past_cut_point, step_t):
            # listener_in: (B, some_step, n_embed)
            slice_start = int(t / step_t)
            slice_end   = int((t + seq_len) / step_t)
            listener_in = prediction.data[:, slice_start : slice_end].cpu().numpy()

            curr_prediction, curr_probs, _, _ = generate_prediction(
                config, args, l_vq_model, generator,
                speakerData_np[:, t : (t + seq_len + patch_size), :],
                listener_in,
                audioData_np[:, t : (t + (seq_len + patch_size) * 4), :],
                seq_len, patch_size, int(t / step_t), cut_point,
                btc=quant_size
            )
            # [DEBUG] curr_prediction
            print(f"[DEBUG] (bii={bii}, t={t}) curr_prediction.shape:", curr_prediction.shape)
            prediction = torch.cat((prediction, curr_prediction[:, 0]), dim=1)
            probs = torch.cat((probs, curr_probs[:, [0], :]), dim=1)

        # [DEBUG] 최종 prediction shape
        print(f"[DEBUG] (bii={bii}) final prediction before decode shape:", prediction.shape)

        # 이제 full sequence로부터 VQ decode
        # remove initial GT portion from 'prediction'
        # [DEBUG] quant_size[-1]
        print(f"[DEBUG] quant_size[-1]: {quant_size[-1]}")
        prediction = prediction[:, quant_size[-1] :]

        # [DEBUG] prediction.shape after removing initial GT portion
        print(f"[DEBUG] (bii={bii}) prediction.shape after strip:", prediction.shape)

        decoded_pred = None
        for t in range(0, prediction.shape[-1], quant_size[-1]):
            chunk = prediction[:, t : t + quant_size[-1]]
            print(f"[DEBUG] decode chunk shape: {chunk.shape}, numel={chunk.numel()}")
            curr_dec = l_vq_model.module.decode_to_img(chunk, quant_size)
            # [DEBUG] curr_dec
            print(f"[DEBUG] curr_dec.shape after decode_to_img: {curr_dec.shape}")

            decoded_pred = curr_dec if decoded_pred is None else torch.cat((decoded_pred, curr_dec), axis=1)

        # re-attach the initial GT
        print(f"[DEBUG] decoded_pred.shape:", decoded_pred.shape)
        prediction = torch.cat((
            torch.from_numpy(listenerData_np[:, :seq_len, :]).cuda(),
            decoded_pred
        ), dim=1)

        print(f"[DEBUG] after re-attach => prediction.shape:", prediction.shape)

        # upper bound decode for GT
        decoded_gt = None
        for t in range(0, listenerData_np.shape[1], seq_len):
            tmp = torch.from_numpy(listenerData_np[:, t : t + seq_len, :]).float().cuda()
            _, gt_logit = l_vq_model.module.get_quant(tmp)
            tmp_decoded = l_vq_model.module.decode_to_img(gt_logit, quant_size)
            decoded_gt = tmp_decoded if decoded_gt is None else torch.cat((decoded_gt, tmp_decoded), axis=1)

        if output_pred is None:
            output_pred = prediction.data.cpu().numpy()
            output_probs = probs.data.cpu().numpy()
            output_gt = decoded_gt.data.cpu().numpy()
        else:
            output_pred = np.concatenate((output_pred, prediction.data.cpu().numpy()), axis=0)
            #디버그
            print(f"DEBUG: Collected predictions shape so far: {output_pred.shape}")
            output_probs = np.concatenate((output_probs, probs.data.cpu().numpy()), axis=0)
            output_gt = np.concatenate((output_gt, decoded_gt.data.cpu().numpy()), axis=0)

    print('out', output_pred.shape)
    return output_pred, output_probs, output_gt


def generate_prediction(config, args, l_vq_model, generator, test_X,
                        test_Y, test_audio, seq_len, patch_size,
                        mask_point, cut_point, btc=None):
    inputs, _, raw_listener, quant_size = create_data_vq(
        l_vq_model, test_X, test_Y, test_audio,
        seq_len, data_type=config['loss_config']['loss_type'],
        patch_size=patch_size, btc=btc
    )

    # [DEBUG]
    print("[DEBUG] generate_prediction => test_X.shape:", test_X.shape,
          "test_Y.shape:", (test_Y.shape if isinstance(test_Y, np.ndarray) else np.array(test_Y).shape),
          "test_audio.shape:", test_audio.shape)
    print("[DEBUG] inputs['speaker_full'].shape:", inputs['speaker_full'].shape)
    print("[DEBUG] inputs['listener_past'].shape:", inputs['listener_past'].shape)
    print("[DEBUG] inputs['audio_full'].shape:", inputs['audio_full'].shape)
    print("[DEBUG] quant_size:", quant_size)

    with torch.no_grad():
        quant_prediction = generator(
            inputs, config['fact_model']['cross_modal_model']['max_mask_len'],
            mask_point
        )
    # sample output
    prediction, probs = l_vq_model.module.get_logit(
        quant_prediction[:, :cut_point, :],
        sample_idx=args.sample_idx
    )
    print("[DEBUG] prediction.shape after get_logit:", prediction.shape,
          "probs.shape:", probs.shape)

    return prediction, probs, inputs, quant_size


def save_pred(args, config, tag, pipeline, test_files, unstd_pred, probs=None):
    """
    기존 DECA 56차 전제(pickle)에 맞는 부분이므로
    209차를 저장하려면 customize 필요.
    """


    B, T, _ = unstd_pred.shape
    preprocess = np.load(os.path.join('vqgan/', config['model_path'],
        '{}{}_preprocess_core.npz'.format(config['tag'], config['pipeline'])))
    body_mean_Y = preprocess['body_mean_Y']
    body_std_Y = preprocess['body_std_Y']
    test_Y = unstd_pred * body_std_Y + body_mean_Y

    # 원본은 exp:[:50], rot:[50:53], jaw:[53:]
    # => 209차라면 아래 part customize 필요
    for b in range(B):
        for t in range(T):
            #현재 우리 데이터는 테스트 파일 없음
            # vid, _, frame_num = test_files[b, t, :]
            vid = f"sample_{b}"
            frame_num = t  # 프레임 번호로 사용



            save_base = os.path.join('outputs/', vid,
                'results/{}predicted/'.format(args.etag+tag))
            if not os.path.exists(save_base):
                os.makedirs(save_base)
            save_path = os.path.join(save_base, '{:08d}.pkl'.format(int(frame_num)))

            # DECA 용 code -> 209차 gaze+pose라면 아래를 수정해야 함
            data = {
                'gaze_pose': torch.from_numpy(test_Y[b, t, :209]).cuda()[None, ...]
            }
            if probs is not None:
                data['prob'] = probs[b, int(t/8), :]  # patch_size=8
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
    print('done save', test_Y.shape)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)

    # seq_len=32  patch_size=8
    seq_len = 32
    patch_size = 8
    num_out = 1024

    with open(args.config) as f:
        config = json.load(f)
    pipeline = config['pipeline']
    tag = config['tag']

    # setup VQ-VAE
    with open(config['l_vqconfig']) as f:
        l_vqconfig = json.load(f)

    l_model_path = 'vqgan/' + l_vqconfig['model_path'] + \
        '{}{}_best.pth'.format(l_vqconfig['tag'], l_vqconfig['pipeline'])
    l_vq_model, _, _ = setup_vq_transformer(args, l_vqconfig,
                                            load_path=l_model_path,
                                            test=True)
    l_vq_model.eval()
    vq_configs = {'l_vqconfig': l_vqconfig, 's_vqconfig': None}

    # setup Predictor
    load_path = args.checkpoint
    print('> checkpoint', load_path)
    generator, _, _ = setup_model(config, l_vqconfig,
                                  mask_index=0, test=True, s_vqconfig=None,
                                  load_path=load_path)
    generator.eval()

    # load data
    out_num = 1 if config['data']['speaker'] == 'fallon' else 0
    test_X, test_Y, test_audio, test_files, _ = \
        load_test_data(config, pipeline, tag, out_num=out_num,
                       vqconfigs=vq_configs, smooth=True,
                       speaker=args.speaker, num_out=num_out)
    
    # 디버그 문구 추가: 로드된 데이터 shape 확인
    print("DEBUG: test_X.shape:", test_X.shape)
    print("DEBUG: test_Y.shape:", test_Y.shape)
    print("DEBUG: test_audio.shape:", test_audio.shape)
    print("DEBUG: Number of test segments:", test_X.shape[0])




    # run model
    unstd_pred, probs, unstd_ub = run_model(args, config,
        l_vq_model, generator, test_X, test_Y, test_audio,
        seq_len, patch_size, rng=rng
    )

    # compute L2
    # note: test_Y[:, seq_len:, :] vs. unstd_pred[:, seq_len:, :]
    # shape check
    #데이터가 batch size로 안 나누어 떨어질때 버림 처리
    # 원래 코드는 주석 처리함
    # minT = min(test_Y.shape[1], unstd_pred.shape[1])
    # overall_l2 = np.mean(
    #     np.linalg.norm(test_Y[:, seq_len: minT, :] - unstd_pred[:, seq_len: minT, :], axis=-1)
    # )
    # print('overall l2:', overall_l2)

    minB = min(test_Y.shape[0], unstd_pred.shape[0])
    minT = min(test_Y.shape[1], unstd_pred.shape[1])
    overall_l2 = np.mean(
        np.linalg.norm(
            test_Y[:minB, seq_len:minT, :] - unstd_pred[:minB, seq_len:minT, :],
            axis=-1
        )
    )
    print('overall l2:', overall_l2)



    # optional save
    if args.save:
        save_pred(args, l_vqconfig, tag, pipeline, test_files, unstd_pred, probs=probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--speaker', type=str, required=True)
    parser.add_argument('--etag', type=str, default='')
    parser.add_argument('--sample_idx', type=int, default=None)
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    print(args)
    main(args)