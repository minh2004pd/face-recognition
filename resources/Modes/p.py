import torch

# Tạo một tensor 2 chiều với các giá trị ngẫu nhiên
x = torch.randn(2, 3)

# In tensor
print(x)

from tqdm import tqdm

for i in tqdm(range(1000000)):
    pass