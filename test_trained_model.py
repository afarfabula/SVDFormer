import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch_directml

from solution import SVDNet

# 从debug_svd_loss.py中复制的损失函数
def complex_bmm_real(a, b):
    real_part = a[..., 0] @ b[..., 0] - a[..., 1] @ b[..., 1]
    imag_part = a[..., 0] @ b[..., 1] + a[..., 1] @ b[..., 0]
    return torch.stack((real_part, imag_part), dim=-1)

def reconstruct_from_svd_real(U, S, V):
    S_diag = torch.diag_embed(S)
    S_diag_complex = torch.stack((S_diag, torch.zeros_like(S_diag)), dim=-1)
    V_H = torch.stack((V[..., 0].transpose(-2, -1), -V[..., 1].transpose(-2, -1)), dim=-1)
    US = complex_bmm_real(U, S_diag_complex)
    reconstructed_H = complex_bmm_real(US, V_H)
    return reconstructed_H

def svd_approximation_error_loss(pred_U, pred_S, pred_V, labels):
    device = pred_U.device
    B, M, R = pred_U.shape[0], pred_U.shape[1], pred_U.shape[2]
    N = pred_V.shape[1]

    reconstructed_H = reconstruct_from_svd_real(pred_U, pred_S, pred_V)
    
    # Reshape to vectors for robust norm calculation
    reconstruction_error = torch.linalg.norm((reconstructed_H - labels).view(B, -1), dim=1) / torch.linalg.norm(labels.view(B, -1), dim=1)
    loss1 = torch.mean(reconstruction_error)

    U_H = torch.stack((pred_U[..., 0].transpose(-2, -1), -pred_U[..., 1].transpose(-2, -1)), dim=-1)
    U_H_U = complex_bmm_real(U_H, pred_U)
    
    # Create Identity matrix on CPU and then move to device to avoid DirectML issues
    I = torch.eye(R, device='cpu').unsqueeze(0).expand(B, -1, -1).to(device)
    I_complex = torch.stack((I, torch.zeros_like(I)), dim=-1)

    diff_U = U_H_U - I_complex
    loss2 = torch.mean(torch.linalg.norm(diff_U.view(B, -1), dim=1))

    V_H = torch.stack((pred_V[..., 0].transpose(-2, -1), -pred_V[..., 1].transpose(-2, -1)), dim=-1)
    V_H_V = complex_bmm_real(V_H, pred_V)
    diff_V = V_H_V - I_complex
    loss3 = torch.mean(torch.linalg.norm(diff_V.view(B, -1), dim=1))

    total_loss = loss1 + (loss2 + loss3)
    return total_loss, loss1, loss2, loss3

# 数据加载函数（从train_direct_supervision.py复制）
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

if __name__ == '__main__':
    # 配置参数
    PathSet = {0: "./DebugData", 1: "./CompetitionData1", 2: "./CompetitionData2", 3: "./CompetitionData3"}
    PrefixSet = {0: "Round0", 1: "Round1", 2: "Round2", 3: "Round3"}
    
    Ridx = 1
    DATA_DIR = PathSet[Ridx]
    PREFIX = PrefixSet[Ridx]
    MODEL_PATH = f'svdnet_model_{PREFIX}_direct.pth'
    BATCH_SIZE = 32

    # 设置设备
    try:
        if torch_directml.is_available():
            device = torch_directml.device()
            print(f"Using DirectML device for testing.")
        else:
            device = 'cpu'
            print("DirectML not available, falling back to CPU.")
    except ImportError:
        device = 'cpu'
        print("torch_directml not found, falling back to CPU.")

    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file {MODEL_PATH} not found!")
        print("Please run train_direct_supervision.py first to train the model.")
        exit(1)

    # 加载数据
    print(f"Loading test data from {DATA_DIR} with prefix {PREFIX}...")
    X, Y_true = load_training_data(DATA_DIR, PREFIX)
    
    X_tensor = torch.from_numpy(X).float()
    Y_true_tensor = torch.from_numpy(Y_true).float()

    dataset = TensorDataset(X_tensor, Y_true_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("Data loaded successfully.")

    # 加载训练好的模型
    print(f"Loading trained model from {MODEL_PATH}...")
    model = SVDNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 测试模型
    print("\nStarting model testing with SVD approximation error loss...")
    total_loss = 0.0
    total_reconstruction_loss = 0.0
    total_orthogonal_U_loss = 0.0
    total_orthogonal_V_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 模型预测
            pred_U, pred_S, pred_V = model(inputs)

            # 计算SVD近似误差损失
            batch_loss, recon_loss, ortho_U_loss, ortho_V_loss = svd_approximation_error_loss(
                pred_U, pred_S, pred_V, labels
            )

            total_loss += batch_loss.item()
            total_reconstruction_loss += recon_loss.item()
            total_orthogonal_U_loss += ortho_U_loss.item()
            total_orthogonal_V_loss += ortho_V_loss.item()
            num_batches += 1

            if (i + 1) % 10 == 0:
                print(f'Batch [{i+1}/{len(dataloader)}], '
                      f'Total Loss: {batch_loss.item():.6f}, '
                      f'Reconstruction: {recon_loss.item():.6f}, '
                      f'Orthogonal U: {ortho_U_loss.item():.6f}, '
                      f'Orthogonal V: {ortho_V_loss.item():.6f}')

    # 计算平均损失
    avg_total_loss = total_loss / num_batches
    avg_reconstruction_loss = total_reconstruction_loss / num_batches
    avg_orthogonal_U_loss = total_orthogonal_U_loss / num_batches
    avg_orthogonal_V_loss = total_orthogonal_V_loss / num_batches

    print("\n" + "="*80)
    print("TESTING RESULTS SUMMARY")
    print("="*80)
    print(f"Average Total Loss: {avg_total_loss:.6f}")
    print(f"Average Reconstruction Loss: {avg_reconstruction_loss:.6f}")
    print(f"Average Orthogonal U Loss: {avg_orthogonal_U_loss:.6f}")
    print(f"Average Orthogonal V Loss: {avg_orthogonal_V_loss:.6f}")
    print(f"Total batches processed: {num_batches}")
    print(f"Total samples tested: {num_batches * BATCH_SIZE}")
    print("="*80)

    # 额外的模型性能分析
    print("\nPerformance Analysis:")
    if avg_reconstruction_loss < 0.1:
        print("✓ Excellent reconstruction quality (< 0.1)")
    elif avg_reconstruction_loss < 0.5:
        print("✓ Good reconstruction quality (< 0.5)")
    else:
        print("⚠ Poor reconstruction quality (>= 0.5)")
    
    if avg_orthogonal_U_loss < 0.1 and avg_orthogonal_V_loss < 0.1:
        print("✓ Good orthogonality constraints satisfied")
    else:
        print("⚠ Orthogonality constraints not well satisfied")

    print(f"\nTesting completed. Model: {MODEL_PATH}")