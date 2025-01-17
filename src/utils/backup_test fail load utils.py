import cv2
import numpy as np
import os
import pickle
import torch
from torchvision import transforms
from torch.autograd import Variable

EPSILON = 1e-10

speakerName = "./data/VCL/" # 데이터 폴더명 VCL로 고정 ; 경로 변경시 반드시 수정

def bilateral_filter(outputs):
    """ smoothing function
        Applies bilateral filtering along temporal dim of sequence.
        outputs: (B, T, F)
    """
    outputs = outputs.astype(np.float32)  # 미리 전체를 float32로 변환
    outputs_smooth = np.zeros(outputs.shape, dtype=np.float32)
    for b in range(outputs.shape[0]):
        for f in range(outputs.shape[2]):
            smoothed = cv2.bilateralFilter(
                outputs[b, :, f].reshape(-1, 1), 
                5, 20, 20
            ).reshape(-1)
            outputs_smooth[b, :, f] = smoothed
    return outputs_smooth

def create_data_vq(l_vq_model, speakerData_np, listenerData_np, audioData_np,
                   seq_len, startpoint=0, midpoint=None, data_type='on_logit',
                   btc=None, patch_size=8):
    speakerData = Variable(torch.from_numpy(speakerData_np),
                           requires_grad=False).cuda()
    listenerData = Variable(torch.from_numpy(listenerData_np),
                            requires_grad=False).cuda()
    audioData = Variable(torch.from_numpy(audioData_np),
                         requires_grad=False).cuda()

    speaker_full = speakerData[:, :(seq_len + patch_size), :]
    audio_full = audioData[:, :(seq_len + patch_size) * 4, :]

    with torch.no_grad():
        if listenerData.dim() == 3:
            listener_past, listener_past_index = \
                l_vq_model.module.get_quant(listenerData[:, :seq_len, :])
            btc = (listener_past.shape[0],
                   listener_past.shape[2],
                   listener_past.shape[1])
            listener_past_index = torch.reshape(listener_past_index,
                                                (listener_past.shape[0], -1))
        else:
            tmp_past_index = listenerData[:, :btc[1]]
            tmp_decoded = l_vq_model.module.decode_to_img(tmp_past_index, btc)
            new_past, new_past_index = l_vq_model.module.get_quant(
                tmp_decoded[:, :seq_len, :]
            )
            listener_past_index = torch.reshape(new_past_index,
                                                (new_past.shape[0], -1))

        listener_future = None
        listener_future_index = None
        if listenerData.shape[1] > seq_len:
            listener_future, listener_future_index = \
                l_vq_model.module.get_quant(listenerData[:, seq_len:, :])
            listener_future_index = torch.reshape(listener_future_index,
                                                  (listener_future.shape[0], -1))

    raw_listener = listenerData[:, seq_len:, :] if listenerData.dim() == 3 else None
    inputs = {
        "speaker_full": speaker_full,
        "listener_past": listener_past_index,
        "audio_full": audio_full
    }
    return inputs, listener_future_index, raw_listener, btc

def load_test_data(config, pipeline, tag, out_num=0, vqconfigs=None,
                   smooth=False, speaker=None, segment_tag='', num_out=None):
    base_dir = config['data']['basedir']
    A_pose_path = os.path.join(base_dir, speakerName + "test/" + "A_gaze_pose_merged.npy")
    B_pose_path = os.path.join(base_dir, speakerName + "test/" + "B_gaze_pose_merged.npy")
    A_audio_path = os.path.join(base_dir, speakerName + "test/" +  "A_audio.npy")

    speaker_data = np.load(A_pose_path, allow_pickle=True)[:, :, :209]
    listener_data = np.load(B_pose_path, allow_pickle=True)[:, :, :209]
    audio_data = np.load(A_audio_path, allow_pickle=True)

    if num_out is not None:
        speaker_data = speaker_data[:num_out]
        listener_data = listener_data[:num_out]
        audio_data = audio_data[:num_out]

    if smooth:
        speaker_data = bilateral_filter(speaker_data)
        listener_data = bilateral_filter(listener_data)

    preprocess = np.load(
        os.path.join(config['model_path'], f"{tag}{pipeline}_preprocess_core.npz"),
        allow_pickle=True
    )
    body_mean_X = preprocess['body_mean_X']
    body_std_X = preprocess['body_std_X']
    body_mean_audio = preprocess['body_mean_audio']
    body_std_audio = preprocess['body_std_audio']

    y_preprocess = np.load(os.path.join('vqgan/',
        vqconfigs['l_vqconfig']['model_path'],
        f"{vqconfigs['l_vqconfig']['tag']}{pipeline}_preprocess_core.npz"),
        allow_pickle=True
    )
    body_mean_Y = y_preprocess['body_mean_Y']
    body_std_Y = y_preprocess['body_std_Y']

    speaker_data = (speaker_data - body_mean_X) / body_std_X
    listener_data = (listener_data - body_mean_Y) / body_std_Y
    audio_data = (audio_data - body_mean_audio) / body_std_audio

    return speaker_data, listener_data, audio_data, None, {
        'body_mean_X': body_mean_X,
        'body_std_X': body_std_X,
        'body_mean_Y': body_mean_Y,
        'body_std_Y': body_std_Y
    }

def load_data(config, pipeline, tag, rng, vqconfigs=None, segment_tag='',
              smooth=False):
    base_dir = config['data']['basedir']
    A_pose_path = os.path.join(base_dir, speakerName + "train/" + "A_gaze_pose_merged.npy")
    B_pose_path = os.path.join(base_dir, speakerName + "train/" + "B_gaze_pose_merged.npy")
    A_audio_path = os.path.join(base_dir, speakerName + "train/" + "A_audio.npy")

    # 데이터 로드
    speaker_data = np.load(A_pose_path, allow_pickle=True)[:, :, :209]
    listener_data = np.load(B_pose_path, allow_pickle=True)[:, :, :209]
    audio_data = np.load(A_audio_path, allow_pickle=True)



    # 스무딩 (옵션)
    if smooth:
        speaker_data = bilateral_filter(speaker_data)
        listener_data = bilateral_filter(listener_data)

    # Train/Test Split (70:30 비율)
    N = speaker_data.shape[0]
    train_N = int(N * 0.7)
    idx = np.random.permutation(N)
    train_idx, test_idx = idx[:train_N], idx[train_N:]

    # 학습 및 테스트 데이터로 분리
    train_X = speaker_data[train_idx, :, :].astype(np.float32)
    test_X = speaker_data[test_idx, :, :].astype(np.float32)
    train_Y = listener_data[train_idx, :, :].astype(np.float32)
    test_Y = listener_data[test_idx, :, :].astype(np.float32)
    train_audio = audio_data[train_idx, :, :].astype(np.float32)
    test_audio = audio_data[test_idx, :, :].astype(np.float32)

    print("Train/Test split 완료")
    print("Train X:", train_X.shape, "Test X:", test_X.shape)

    train_X = np.nan_to_num(train_X)  # NaN을 0으로 변환
    train_Y = np.nan_to_num(train_Y)
    train_audio = np.nan_to_num(train_audio)



    return train_X, test_X, train_Y, test_Y, train_audio, test_audio