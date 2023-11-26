import torch
import numpy as np

a = np.load('ext/experiment/images/projected_w.npz')

# NumPy 배열을 PyTorch 텐서로 변환
tensor_data = torch.from_numpy(a['w'])  # 'your_array'는 npz 파일 내에서 사용하는 배열의 이름입니다.

# PyTorch 텐서를 파일로 저장 (pt 파일)
torch.save(tensor_data, 'projected_w.pt')
