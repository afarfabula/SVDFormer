import torch
import torch_directml

# 检查DirectML是否可用并设置为设备
if torch_directml.is_available():
    device = torch_directml.device()
    print(f"Using DirectML device: {device}")
else:
    device = torch.device("cpu")
    print("DirectML not available, using CPU.")

# 之后就可以像往常一样把你的tensors和models移动到这个device上
# tensor = tensor.to(device)
# model = model.to(device)