import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch_directml

from transformer_svd import SVDTransformer

# 检查DirectML是否可用
if torch_directml.is_available():
    device = torch_directml.device()
    print(f"Using DirectML device: {device}")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# 数据加载函数 (与之前相同)
def load_training_data(data_dir, prefix):
    all_train_data = []
    files = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix + 'TrainData') and f.endswith('.npy')])
    for f in files:
        data_path = os.path.join(data_dir, f)
        data = np.load(data_path)
        all_train_data.append(data)
    return np.concatenate(all_train_data, axis=0)

# 复数矩阵乘法 (用于正交性损失)
def complex_bmm_real(a, b):
    real_part = a[..., 0] @ b[..., 0] - a[..., 1] @ b[..., 1]
    imag_part = a[..., 0] @ b[..., 1] + a[..., 1] @ b[..., 0]
    return torch.stack((real_part, imag_part), dim=-1)

# 正交性损失函数
def orthogonality_loss(U, V):
    B, R, M, _ = U.shape
    
    # U_H * U
    U_H = torch.stack((U[..., 0].transpose(-2, -1), -U[..., 1].transpose(-2, -1)), dim=-1)
    U_H_U = complex_bmm_real(U_H, U)
    
    # V_H * V
    V_H = torch.stack((V[..., 0].transpose(-2, -1), -V[..., 1].transpose(-2, -1)), dim=-1)
    V_H_V = complex_bmm_real(V_H, V)
    
    I = torch.eye(R, device=U.device).unsqueeze(0).expand(B, -1, -1)
    I_complex = torch.stack([I, torch.zeros_like(I)], dim=-1)
    
    loss_U = torch.mean(torch.linalg.norm((U_H_U - I_complex).view(B, -1), dim=1))
    loss_V = torch.mean(torch.linalg.norm((V_H_V - I_complex).view(B, -1), dim=1))
    
    return loss_U + loss_V


def main():
    # --- 配置 ---
    data_dir = './CompetitionData1'
    prefix = 'Round1'
    batch_size = 32
    learning_rate = 1e-4
    epochs = 50
    ortho_weight = 0.1 # 正交性损失的权重

    # --- 数据加载 ---
    print("Loading data...")
    train_data = load_training_data(data_dir, prefix)
    train_tensor = torch.from_numpy(train_data).float()
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Data loaded.")

    # --- 模型、损失和优化器 ---
    model = SVDTransformer(input_dim=64, embed_dim=128, rank=32, nhead=8, num_encoder_layers=4, dim_feedforward=512).to(device)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 训练循环 ---
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (H_batch,) in enumerate(train_loader):
            H_batch = H_batch.to(device)

            # --- 解析SVD作为监督信号 (在CPU上计算) ---
            H_complex_cpu = torch.complex(H_batch[..., 0], H_batch[..., 1]).cpu()
            U_true_cpu, S_true_cpu, V_true_cpu = torch.linalg.svd(H_complex_cpu, full_matrices=False)
            U_true = torch.stack([U_true_cpu.real, U_true_cpu.imag], dim=-1).to(device)
            V_true = torch.stack([V_true_cpu.real, V_true_cpu.imag], dim=-1).to(device)
            S_true = S_true_cpu.to(device)
            
            # --- 前向传播 ---
            pred_U, pred_S, pred_V = model(H_batch)

            # --- 计算损失 ---
            # 调整维度以匹配: pred_U/V is (B, R, M, 2), true_U/V is (B, M, R, 2)
            pred_U_permuted = pred_U.permute(0, 2, 1, 3)
            pred_V_permuted = pred_V.permute(0, 2, 1, 3)

            loss_mse_U = mse_loss(pred_U_permuted, U_true)
            loss_mse_V = mse_loss(pred_V_permuted, V_true)
            loss_mse_S = mse_loss(pred_S, S_true)
            
            loss_ortho = orthogonality_loss(pred_U, pred_V)
            
            loss = loss_mse_U + loss_mse_V + loss_mse_S + ortho_weight * loss_ortho

            # --- 反向传播和优化 ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}")

    # --- 保存模型 ---
    torch.save(model.state_dict(), 'transformer_svd_model.pth')
    print("Training complete. Model saved to transformer_svd_model.pth")

if __name__ == '__main__':
    main()