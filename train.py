import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch_directml
from debug_svd_loss import*

from solution import SVDNet

# 1. 数据加载函数 (与之前相同)
def load_training_data(data_dir, prefix):
    all_train_data = []
    all_train_labels = []

    files = sorted(os.listdir(data_dir))
    data_files = [f for f in files if f.startswith(prefix + 'TrainData1') and f.endswith('.npy')]
    label_files = [f for f in files if f.startswith(prefix + 'TrainLabel1') and f.endswith('.npy')]

    print(f"Found {len(data_files)} data files and {len(label_files)} label files.")

    for data_file in data_files:
        data_path = os.path.join(data_dir, data_file)
        all_train_data.append(np.load(data_path))

    for label_file in label_files:
        label_path = os.path.join(data_dir, label_file)
        all_train_labels.append(np.load(label_path))

    X_train = np.concatenate(all_train_data, axis=0)
    Y_train = np.concatenate(all_train_labels, axis=0)

    return X_train, Y_train

# 辅助函数：从 U, S, V 重构矩阵 (实数版本)
def reconstruct_from_svd_real(U, S, V):
    """
    Reconstructs a complex matrix from its SVD components represented as real tensors.
    U: (B, M, R, 2)
    S: (B, R)
    V: (B, N, R, 2)
    Returns: (B, M, N, 2)
    """
    # In our case, M=N=R=64

    # Scale columns of U by S. S needs to be broadcast to (B, 1, R, 1)
    # to multiply with U (B, M, R, 2)
    US = U * S.unsqueeze(1).unsqueeze(-1)

    # Compute V_H (conjugate transpose of V)
    # V is (B, N, R, 2). Transpose should be on N and R dims -> (B, R, N, 2)
    V_t = V.transpose(-3, -2)
    # Conjugate by negating the imaginary part
    V_H = V_t.clone()
    V_H[..., 1] *= -1

    # Perform complex matmul: H = US @ V_H
    US_real = US[..., 0]
    US_imag = US[..., 1]
    V_H_real = V_H[..., 0]
    V_H_imag = V_H[..., 1]

    # H_real = US_real @ V_H_real - US_imag @ V_H_imag
    H_real = torch.matmul(US_real, V_H_real) - torch.matmul(US_imag, V_H_imag)

    # H_imag = US_real @ V_H_imag + US_imag @ V_H_real
    H_imag = torch.matmul(US_real, V_H_imag) + torch.matmul(US_imag, V_H_real)

    pred_H = torch.stack((H_real, H_imag), dim=-1)
    return pred_H

# --- Custom Loss Function ---




# --- Main Training Function --- #
def main():
    # U, V 的形状是 (batch, M/N, R, 2), S 的形状是 (batch, R)
    # 我们需要在实数域完成 H = U * S * V^H 的计算
    
    # 将S扩展为对角矩阵
    S_diag = torch.diag_embed(S)
    
    # U * S
    # U: (batch, M, R, 2), S_diag: (batch, R, R)
    # 需要将S_diag扩展维度以匹配U
    S_diag_expanded = S_diag.unsqueeze(-1).expand(-1, -1, -1, 2)
    # (batch, M, R, 2) * (batch, R, R, 2) -> 错误，维度不匹配
    # 我们需要执行复数乘法 U_complex * S_diag_complex
    # U_complex = U[..., 0] + 1j * U[..., 1]
    # S_diag_complex = S_diag (因为S是实数)
    # (a+bi)*c = ac + bci
    US = torch.einsum('bmri,brj->bmji', U, S_diag)

    # (U*S) * V^H
    # V^H: V的共轭转置。形状 (batch, N, R, 2) -> (batch, R, N, 2) 且虚部取反
    V_T = V.transpose(-2, -3)
    V_H = V_T.clone()
    V_H[..., 1] *= -1

    # (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    # US: (batch, M, R, 2), V_H: (batch, R, N, 2)
    # result_real = US_real * VH_real - US_imag * VH_imag
    # result_imag = US_real * VH_imag + US_imag * VH_real
    H_real = torch.einsum('bmri,brni->bmn', US[..., 0], V_H[..., 0]) - torch.einsum('bmri,brni->bmn', US[..., 1], V_H[..., 1])
    H_imag = torch.einsum('bmri,brni->bmn', US[..., 0], V_H[..., 1]) + torch.einsum('bmri,brni->bmn', US[..., 1], V_H[..., 0])
    
    return torch.stack([H_real, H_imag], dim=-1)


# 2. 主训练逻辑 (严格模仿demo_code.py)
if __name__ == '__main__':
    PathSet = {0: "./DebugData", 1: "./CompetitionData1", 2: "./CompetitionData2", 3: "./CompetitionData3"}
    PrefixSet = {0: "Round0", 1: "Round1", 2: "Round2", 3: "Round3"}
    
    Ridx = 1
    DATA_DIR = PathSet[Ridx]
    PREFIX = PrefixSet[Ridx]

    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-2
    MODEL_SAVE_PATH = f'svdnet_model_{PREFIX}.pth'

    if torch_directml.is_available():
        device = torch_directml.device()
        print(f"Using DirectML device for training.")
    else:
        device = 'cpu'
        print("DirectML not available, falling back to CPU.")

    print(f"Loading data from {DATA_DIR} with prefix {PREFIX}...")
    X, Y_true = load_training_data(DATA_DIR, PREFIX)
    
    # 直接将numpy数组转为tensor，不进行任何复数转换
    X_tensor = torch.from_numpy(X).float()
    Y_true_tensor = torch.from_numpy(Y_true).float()

    dataset = TensorDataset(X_tensor, Y_true_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Data loaded successfully.")

    model = SVDNet().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, weights_only=True))
    # criterion = nn.MSELoss() # Using custom loss function now
    # 在main函数中，创建优化器后添加
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 在每个epoch结束后添加
    scheduler.step()

    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(inputs.shape)

            optimizer.zero_grad()

            # 直接将(..., 2)格式的实数张量送入模型
            pred_U, pred_S, pred_V = model(inputs)
            #print("pred_U shape:", pred_U.shape)
            #print("pred_S shape:", pred_S.shape)
            #print("pred_V shape:", pred_V.shape)

            # 在实数域重构并计算损失
            #pred_H = reconstruct_from_svd_real(pred_U, pred_S, pred_V)
            loss = svd_approximation_error_loss(pred_U, pred_S, pred_V, labels)
            

            # 在loss.backward()后，optimizer.step()前添加
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(dataloader)}], Loss: {loss.item():.6f}')

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.6f}")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Training finished. Model saved to {MODEL_SAVE_PATH}")