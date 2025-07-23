import torch
import torch_directml

if torch_directml.is_available():
    device = torch_directml.device()
    print(f"Using DirectML device for training.")
else:
    device = 'cpu'
    print("DirectML not available, falling back to CPU.")

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
    return total_loss

if __name__ == '__main__':
    B, M, N = 32, 64, 64
    R = min(M, N)

    # 1. 在CPU上创建一个随机的复数矩阵 (B, M, N)
    labels_complex_cpu = torch.randn(B, M, N, dtype=torch.complex64, device='cpu')
    
    # 将其转换为实数表示法 (B, M, N, 2) 并移动到目标设备
    labels = torch.stack((labels_complex_cpu.real, labels_complex_cpu.imag), dim=-1).to(device)

    # 2. 在CPU上使用 torch.linalg.svd 计算解析解
    U_true_complex_cpu, S_true_cpu, Vh_true_complex_cpu = torch.linalg.svd(labels_complex_cpu, full_matrices=False)

    # Vh 是 V 的共轭转置, 我们需要 V
    V_true_complex_cpu = Vh_true_complex_cpu.mH

    # 3. 将复数形式的 U 和 V 转换为实数表示法 (..., 2) 并移动到目标设备
    pred_U = torch.stack((U_true_complex_cpu.real, U_true_complex_cpu.imag), dim=-1).to(device)
    pred_V = torch.stack((V_true_complex_cpu.real, V_true_complex_cpu.imag), dim=-1).to(device)
    pred_S = S_true_cpu.to(device)

    # 4. 使用解析解验证损失函数
    # 理想情况下，这个损失应该非常接近于0
    loss = svd_approximation_error_loss(pred_U, pred_S, pred_V, labels)
    print(f'Loss with analytical SVD solution: {loss.item()}')

    # 也可以测试随机输入
    print("\nTesting with random (non-analytical) inputs:")
    pred_U_rand = torch.randn(B, M, R, 2, device=device)
    pred_S_rand = torch.randn(B, R, device=device).abs() # Singular values should be non-negative
    pred_V_rand = torch.randn(B, N, R, 2, device=device)
    loss_rand = svd_approximation_error_loss(pred_U_rand, pred_S_rand, pred_V_rand, labels)
    print(f'Loss with random SVD components: {loss_rand.item()}')
