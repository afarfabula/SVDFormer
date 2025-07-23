import torch.nn as nn
import torch
 
class SVDNet(nn.Module):
    def __init__(self, dim=64, rank=32):
        super(SVDNet, self).__init__()
        self.dim = dim
        self.rank = rank
        self.input_dim = dim * dim * 2
 
        self.fc_U = nn.Linear(self.input_dim, dim * rank * 2)
        self.fc_V = nn.Linear(self.input_dim, dim * rank * 2)
        self.fc_S = nn.Linear(self.input_dim, rank)
 
    def forward(self, x):  # x: [B, 64, 64, 2] or [64, 64, 2]
        # 检查输入维度，如果为3维则增加一个批次维度
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(0) # [64, 64, 2] -> [1, 64, 64, 2]

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)  # Flatten to [B, 64*64*2]
 
        U = self.fc_U(x).view(batch_size, self.dim, self.rank, 2)
        V = self.fc_V(x).view(batch_size, self.dim, self.rank, 2)
        S = self.fc_S(x).view(batch_size, self.rank)
 
        U = self.normalize_columns(U)
        V = self.normalize_columns(V)
 
        # 如果输入没有批次维度，则在输出时移除
        if not is_batched:
            U = U.squeeze(0)
            V = V.squeeze(0)
            S = S.squeeze(0)

        return U, S, V
 
    def normalize_columns(self, mat): # mat: [B, dim, rank, 2]
        real = mat[..., 0]
        imag = mat[..., 1]
        norm = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        # norm的维度是[B, dim, rank]，需要扩展以匹配mat
        norm = norm.unsqueeze(-1).expand_as(mat)
        mat = mat / norm
        return mat

if __name__ == '__main__':
    # 1. 设置模型和设备
    device = 'cpu' # 如果配置了DirectML，可以换成 torch_directml.device()
    model = SVDNet().to(device)
    model.eval()

    # 测试1：无批次维度输入
    print("--- Testing with single sample ---")
    dummy_input_single = torch.randn(64, 64, 2).to(device)
    with torch.no_grad():
        U_out, S_out, V_out = model(dummy_input_single)
    print(f"Input shape:  {dummy_input_single.shape}")
    print(f"Output U shape: {U_out.shape}")
    print(f"Output S shape: {S_out.shape}")
    print(f"Output V shape: {V_out.shape}")
    assert U_out.shape == (64, 32, 2)
    assert S_out.shape == (32,)
    assert V_out.shape == (64, 32, 2)
    print("Single sample test passed!")

    # 测试2：有批次维度输入
    print("\n--- Testing with batched samples ---")
    batch_size = 4
    dummy_input_batch = torch.randn(batch_size, 64, 64, 2).to(device)
    with torch.no_grad():
        U_out_b, S_out_b, V_out_b = model(dummy_input_batch)
    print(f"Input shape:  {dummy_input_batch.shape}")
    print(f"Output U shape: {U_out_b.shape}")
    print(f"Output S shape: {S_out_b.shape}")
    print(f"Output V shape: {V_out_b.shape}")
    assert U_out_b.shape == (batch_size, 64, 32, 2)
    assert S_out_b.shape == (batch_size, 32)
    assert V_out_b.shape == (batch_size, 64, 32, 2)
    print("Batch sample test passed!")