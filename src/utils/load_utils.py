import cv2
import numpy as np
import os
import pickle

import torch
from torchvision import transforms
from torch.autograd import Variable

EPSILON = 1e-10

#################################################
# 1) Bilateral Filtering
#################################################
def bilateral_filter(outputs):
    """
    smoothing function
    Applies bilateral filtering along temporal dim of sequence.
    outputs: (B, T, F) assumed float32
    """
    outputs = outputs.astype(np.float32)  # 혹시 float64라면 float32 변환
    outputs_smooth = np.zeros_like(outputs, dtype=np.float32)

    B, T, F = outputs.shape
    for b in range(B):
        for f in range(F):
            # opencv bilateralFilter는 (H,W) 2D 형태를 기대 -> (T,1) 사용
            tmp = outputs[b, :, f].reshape(-1, 1)
            smoothed = cv2.bilateralFilter(tmp, 5, 20, 20).reshape(-1)
            outputs_smooth[b, :, f] = smoothed
    return outputs_smooth

#################################################
# 2) create_data_vq() - 기존 구조 그대로 두되 사용 여부에 맞춰 수정
#################################################
def create_data_vq(l_vq_model, speakerData_np, listenerData_np, audioData_np,
                   seq_len, startpoint=0, midpoint=None, data_type='on_logit',
                   btc=None, patch_size=8):
    """
    data preparation function
    processes the data by truncating full input sequences to remove future info,
    and converts listener raw motion to listener codebook indices
    """

    speakerData = Variable(torch.from_numpy(speakerData_np),
                           requires_grad=False).cuda()
    listenerData = Variable(torch.from_numpy(listenerData_np),
                            requires_grad=False).cuda()
    audioData = Variable(torch.from_numpy(audioData_np),
                         requires_grad=False).cuda()

    ## future timesteps for speaker inputs (keep past and current context)
    speaker_full = speakerData[:, :(seq_len + patch_size), :]
    # 오디오는 예전 코드에서 *4 배수로 indexing 했음
    audio_full = audioData[:, :(seq_len + patch_size)*4, :]

    with torch.no_grad():
        if listenerData.dim() == 3:
            # if listener input is raw
            listener_past, listener_past_index = \
                l_vq_model.module.get_quant(listenerData[:, :seq_len, :])
            btc = (listener_past.shape[0],
                   listener_past.shape[2],
                   listener_past.shape[1])
            listener_past_index = torch.reshape(
                listener_past_index, (listener_past.shape[0], -1)
            )
        else:
            # if listener input is already in index format
            tmp_past_index = listenerData[:, :btc[1]]
            tmp_decoded = l_vq_model.module.decode_to_img(tmp_past_index, btc)
            new_past, new_past_index = l_vq_model.module.get_quant(
                tmp_decoded[:, :seq_len, :]
            )
            listener_past_index = torch.reshape(
                new_past_index, (new_past.shape[0], -1)
            )

        listener_future = None
        listener_future_index = None
        if listenerData.shape[1] > seq_len:
            listener_future, listener_future_index = \
                l_vq_model.module.get_quant(listenerData[:, seq_len:, :])
            listener_future_index = torch.reshape(
                listener_future_index,
                (listener_future.shape[0], -1)
            )

    raw_listener = listenerData[:, seq_len:, :] if listenerData.dim() == 3 else None
    inputs = {
        "speaker_full": speaker_full,
        "listener_past": listener_past_index,
        "audio_full": audio_full
    }
    return inputs, listener_future_index, raw_listener, btc

#################################################
# 3) mean/std 계산 유틸
#################################################
def mean_std_swap(data):
    """
    helper function to calc std and mean
    data: (B, T, F)
    returns: mean, std with shape (1,1,F)
    """
    B, T, F = data.shape
    mean = data.mean(axis=(0,1), keepdims=True)  # (1,1,F)
    std  = data.std(axis=(0,1), keepdims=True)   # (1,1,F)
    std += EPSILON
    return mean, std

def calc_stats(config, vqconfigs, tag, pipeline, train_X, train_Y, train_audio):
    """
    helper function to calculate std/mean for different cases
    - train_X, train_Y: (N, T, 209) 가정
    - train_audio: (N, T', 128) 가정
    """
    # vqconfigs 있는 경우 listener는 vqconfig 기반으로 사용할 수도 있지만
    # 일단 아래는 통일해서 새로 계산 (필요시 분기 처리)
    body_mean_X, body_std_X = mean_std_swap(train_X)
    body_mean_Y, body_std_Y = mean_std_swap(train_Y)
    body_mean_audio, body_std_audio = mean_std_swap(train_audio)

    # npz 저장
    np.savez_compressed(
        os.path.join(config['model_path'],
                     f"{tag}{pipeline}_preprocess_core.npz"),
        body_mean_X=body_mean_X,
        body_std_X=body_std_X,
        body_mean_Y=body_mean_Y,
        body_std_Y=body_std_Y,
        body_mean_audio=body_mean_audio,
        body_std_audio=body_std_audio
    )
    return body_mean_X, body_std_X, body_mean_Y, body_std_Y, body_mean_audio, body_std_audio

#################################################
# 4) load_test_data() - 테스트 시에만 호출
#################################################
def load_test_data(config, pipeline, tag, out_num=0, vqconfigs=None,
                   smooth=False, speaker=None, segment_tag='', num_out=None):
    """
    function to load test data from files
    (새로운 209차원 Gaze+Pose + 128차원 audio)
    """
    # base_dir = config['data']['basedir'] 오류있어서 하드코딩함
    base_dir = "./data"
    # base_dir = "../data"
    # 새로운 파일 경로
    A_pose_path = os.path.join(base_dir, "VCL/test", "A_gaze_pose_merged.npy")
    B_pose_path = os.path.join(base_dir, "VCL/test", "B_gaze_pose_merged.npy")
    A_audio_path = os.path.join(base_dir, "VCL/test", "A_audio.npy")

    # 실제 로드
    speaker_data = np.load(A_pose_path, allow_pickle=True)   # (N, T, 209)
    listener_data = np.load(B_pose_path, allow_pickle=True)  # (N, T, 209)
    audio_data = np.load(A_audio_path, allow_pickle=True)    # (N, T', 128)

    # 일부만 쓰기 (옵션)
    if num_out is not None:
        speaker_data = speaker_data[:num_out]
        listener_data = listener_data[:num_out]
        audio_data = audio_data[:num_out]

    # NaN 처리 + float32
    speaker_data = np.nan_to_num(speaker_data, nan=0.0).astype(np.float32)
    listener_data = np.nan_to_num(listener_data, nan=0.0).astype(np.float32)
    audio_data = np.nan_to_num(audio_data, nan=0.0).astype(np.float32)

    # (선택) 스무딩
    if smooth:
        speaker_data = bilateral_filter(speaker_data)
        listener_data = bilateral_filter(listener_data)
        # audio_data는 별도 처리 or pass

    # 학습 때 저장된 mean/std 로드
    preprocess = np.load(
        os.path.join(config['model_path'], f"{tag}{pipeline}_preprocess_core.npz"),
        allow_pickle=True
    )
    body_mean_X = preprocess['body_mean_X']        # shape (1,1,209)
    body_std_X  = preprocess['body_std_X']
    body_mean_Y = preprocess['body_mean_Y']        # shape (1,1,209)
    body_std_Y  = preprocess['body_std_Y']
    body_mean_audio = preprocess['body_mean_audio']  # shape (1,1,128)
    body_std_audio  = preprocess['body_std_audio']

    # 표준화
    speaker_data  = (speaker_data  - body_mean_X) / body_std_X
    listener_data = (listener_data - body_mean_Y) / body_std_Y
    audio_data    = (audio_data    - body_mean_audio) / body_std_audio

    # filepaths 대신 None 리턴 (기존코드 호환)
    filepaths = None
    # std_info dict
    std_info = {
        'body_mean_X': body_mean_X,
        'body_std_X': body_std_X,
        'body_mean_Y': body_mean_Y,
        'body_std_Y': body_std_Y,
        'body_mean_audio': body_mean_audio,
        'body_std_audio': body_std_audio
    }

    return speaker_data, listener_data, audio_data, filepaths, std_info

#################################################
# 5) load_data() - 학습/훈련 시 사용
#################################################
def load_data(config, pipeline, tag, rng, vqconfigs=None, segment_tag='',
              smooth=False):
    """
    function to load train data from files
    """
    # base_dir = config['data']['basedir'] 오류있어서 하드코딩함
    base_dir = "./data"
    # base_dir = "../data" 

    # (1) 새로운 경로
    A_pose_path = os.path.join(base_dir, "VCL/train", "A_gaze_pose_merged.npy")
    B_pose_path = os.path.join(base_dir, "VCL/train", "B_gaze_pose_merged.npy")
    A_audio_path = os.path.join(base_dir, "VCL/train", "A_audio.npy")

    # (2) 데이터 로드
    speaker_data  = np.load(A_pose_path, allow_pickle=True)   # (N, T, 209)
    listener_data = np.load(B_pose_path, allow_pickle=True)   # (N, T, 209)
    audio_data    = np.load(A_audio_path, allow_pickle=True)  # (N, T', 128)

    # (3) NaN 처리 -> 0
    speaker_data  = np.nan_to_num(speaker_data, nan=0.0).astype(np.float32)
    listener_data = np.nan_to_num(listener_data, nan=0.0).astype(np.float32)
    audio_data    = np.nan_to_num(audio_data,   nan=0.0).astype(np.float32)

    # (4) 스무딩
    if smooth:
        speaker_data  = bilateral_filter(speaker_data)
        listener_data = bilateral_filter(listener_data)
        # audio_data는 pass or 별도 처리

    # (5) Train/Test split (70:30)
    N = speaker_data.shape[0]
    train_N = int(N * 0.7)
    idx = np.random.permutation(N) if rng is None else rng.permutation(N)
    train_idx, test_idx = idx[:train_N], idx[train_N:]

    train_X = speaker_data[train_idx]  # (train_N, T, 209)
    test_X  = speaker_data[test_idx]
    train_Y = listener_data[train_idx] # (train_N, T, 209)
    test_Y  = listener_data[test_idx]
    train_audio = audio_data[train_idx] # (train_N, T', 128)
    test_audio  = audio_data[test_idx]

    print('===> in/out')
    print(train_X.shape, train_Y.shape, train_audio.shape)
    print("====> train/test", train_X.shape, test_X.shape)

    # (6) mean/std 계산 & 저장
    body_mean_X, body_std_X, body_mean_Y, body_std_Y, \
        body_mean_audio, body_std_audio = \
        calc_stats(config, vqconfigs, tag, pipeline, train_X, train_Y, train_audio)

    # (7) 표준화
    train_X     = (train_X     - body_mean_X) / body_std_X
    test_X      = (test_X      - body_mean_X) / body_std_X
    train_Y     = (train_Y     - body_mean_Y) / body_std_Y
    test_Y      = (test_Y      - body_mean_Y) / body_std_Y
    train_audio = (train_audio - body_mean_audio) / body_std_audio
    test_audio  = (test_audio  - body_mean_audio) / body_std_audio

    print("=====> standardization done")
    return train_X, test_X, train_Y, test_Y, train_audio, test_audio
