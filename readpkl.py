import pickle
import csv
import torch

def process_value(val):
    """
    주어진 값이 PyTorch 텐서인 경우 CPU로 이동시키고,
    NumPy 배열로 변환한 후 적절한 파이썬 객체로 반환.
    딕셔너리나 리스트 안에 있는 텐서도 재귀적으로 변환.
    """
    if isinstance(val, torch.Tensor):
        # CUDA 텐서를 CPU로 이동 및 NumPy로 변환
        val = val.detach().cpu()
        arr = val.numpy()
        # 스칼라 값이면 스칼라로 반환, 아니면 리스트로 반환
        return arr.item() if arr.shape == () else arr.tolist()
    elif isinstance(val, dict):
        return {k: process_value(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [process_value(item) for item in val]
    else:
        return val

# pickle 파일에서 데이터 불러오기
with open('./src/outputs/sample_0/results/delta_v6_predicted/00001645.pkl', 'rb') as f:
    data = pickle.load(f)

# 딕셔너리 내의 모든 PyTorch 텐서를 CPU로 이동시키고 변환
data = process_value(data)

# data가 dict인지 확인 후 key-value 형식으로 CSV에 저장
if isinstance(data, dict):
    with open('xxx.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Key', 'Value'])  # 헤더 작성
        for key, value in data.items():
            # 리스트나 배열 형태의 값도 문자열로 변환하여 저장
            writer.writerow([key, str(value)])
    print("딕셔너리를 key-value 형태로 CSV에 저장했습니다.")
else:
    print("데이터가 dict 형태가 아닙니다.")
