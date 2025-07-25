
import torch
import torch.nn as nn
import torch.nn.functional as F

class SVDDNN(nn.Module):
    def __init__(self, input_size=64, rank=16):  # 减少rank从32到16
        super(SVDDNN, self).__init__()
        self.input_size = input_size
        self.rank = rank
        
        # 大幅减少卷积层通道数
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, padding=1)    # 64->32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 128->64
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 256->128
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # 512->256
        
        # 对应的批归一化层
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # 添加dropout层防止过拟合
        self.dropout = nn.Dropout(0.2)
        
        # 计算卷积后的尺寸
        conv_output_size = input_size // (2**4)  # 64 // 16 = 4
        flattened_size = 256 * conv_output_size * conv_output_size  # 256 * 4 * 4 = 4096
        
        # 添加中间层减少参数量
        hidden_dim = 512  # 中间隐藏层维度
        
        # 共享的特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(flattened_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 分别的输出层（参数量大幅减少）
        self.fc_U = nn.Linear(hidden_dim, input_size * rank * 2)  # 512 -> 2048
        self.fc_S = nn.Linear(hidden_dim, rank)                   # 512 -> 16
        self.fc_V = nn.Linear(hidden_dim, rank * input_size * 2)  # 512 -> 2048
        
        # Activation for singular values (ensure non-negative)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch_size, 2, 64, 64)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # 共享特征提取
        features = self.feature_extractor(x)
        
        # Compute outputs
        U = self.fc_U(features).view(-1, self.input_size, self.rank, 2)
        S = self.sigmoid(self.fc_S(features)) * 100
        V = self.fc_V(features).view(-1, self.rank, self.input_size, 2)
        
        return U, S, V

    def compute_reconstructed_matrix(self, U, S, V):
        # 使用实数表示的复数矩阵乘法
        def complex_matmul_real(a, b):
            real_part = torch.matmul(a[..., 0], b[..., 0]) - torch.matmul(a[..., 1], b[..., 1])
            imag_part = torch.matmul(a[..., 0], b[..., 1]) + torch.matmul(a[..., 1], b[..., 0])
            return torch.stack([real_part, imag_part], dim=-1)
        
        # 创建对角矩阵S
        S_diag = torch.diag_embed(S)
        S_diag_complex = torch.stack([S_diag, torch.zeros_like(S_diag)], dim=-1)
        
        # U @ S
        US = complex_matmul_real(U, S_diag_complex)
        
        # (U @ S) @ V
        reconstructed = complex_matmul_real(US, V)
        
        return reconstructed

class SVDLoss(nn.Module):
    def __init__(self, rank=16):  # 对应减少的rank
        super(SVDLoss, self).__init__()
        self.rank = rank

    def forward(self, input_matrix, U, S, V):
        # Input shape: (batch_size, 2, 64, 64) -> (batch_size, 64, 64, 2)
        input_real = input_matrix.permute(0, 2, 3, 1).contiguous()
        
        # 重构矩阵
        model = SVDDNN(rank=self.rank)
        reconstructed = model.compute_reconstructed_matrix(U, S, V)
        
        # Frobenius norm loss
        diff = (input_real - reconstructed).contiguous()
        frobenius_loss = torch.norm(diff.reshape(diff.shape[0], -1), p=2, dim=1).mean()
        
        # 正交性约束 (使用实数矩阵乘法)
        def complex_matmul_real(a, b):
            real_part = torch.matmul(a[..., 0], b[..., 0]) - torch.matmul(a[..., 1], b[..., 1])
            imag_part = torch.matmul(a[..., 0], b[..., 1]) + torch.matmul(a[..., 1], b[..., 0])
            return torch.stack([real_part, imag_part], dim=-1)
        
        # U^H @ U 应该是单位矩阵
        U_H = torch.stack([U[..., 0].transpose(-1, -2), -U[..., 1].transpose(-1, -2)], dim=-1)
        U_H_U = complex_matmul_real(U_H, U)
        
        identity = torch.eye(self.rank, device=U.device)
        identity_complex = torch.stack([identity, torch.zeros_like(identity)], dim=-1)
        identity_complex = identity_complex.unsqueeze(0).repeat(U.shape[0], 1, 1, 1)
        
        diff_U = (U_H_U - identity_complex).contiguous()
        ortho_loss_U = torch.norm(diff_U.reshape(diff_U.shape[0], -1), p=2, dim=1).mean()
        
        # V @ V^H 应该是单位矩阵
        V_H = torch.stack([V[..., 0].transpose(-1, -2), -V[..., 1].transpose(-1, -2)], dim=-1)
        V_V_H = complex_matmul_real(V, V_H)
        
        diff_V = (V_V_H - identity_complex).contiguous()
        ortho_loss_V = torch.norm(diff_V.reshape(diff_V.shape[0], -1), p=2, dim=1).mean()
        
        # 总损失
        total_loss = frobenius_loss + 0.1 * (ortho_loss_U + ortho_loss_V)
        return total_loss

# 修复测试代码部分（从第150行开始替换）
if __name__ == "__main__":
    # Initialize model and loss
    model = SVDDNN(input_size=64, rank=16)
    criterion = SVDLoss(rank=16)
    
    # 检查模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 修复：正确计算各部分参数量
    conv_params = sum(p.numel() for p in model.conv1.parameters()) + \
                  sum(p.numel() for p in model.conv2.parameters()) + \
                  sum(p.numel() for p in model.conv3.parameters()) + \
                  sum(p.numel() for p in model.conv4.parameters())
    
    bn_params = sum(p.numel() for p in model.bn1.parameters()) + \
                sum(p.numel() for p in model.bn2.parameters()) + \
                sum(p.numel() for p in model.bn3.parameters()) + \
                sum(p.numel() for p in model.bn4.parameters())
    
    fc_params = sum(p.numel() for p in model.feature_extractor.parameters()) + \
                sum(p.numel() for p in model.fc_U.parameters()) + \
                sum(p.numel() for p in model.fc_S.parameters()) + \
                sum(p.numel() for p in model.fc_V.parameters())
    
    print(f"Convolutional layers: {conv_params:,} parameters")
    print(f"Batch normalization: {bn_params:,} parameters")
    print(f"Fully connected layers: {fc_params:,} parameters")
    
    # Dummy input (batch_size=2, 2, 64, 64)
    input_matrix = torch.randn(2, 2, 64, 64)
    
    try:
        print("\nForward pass...")
        U, S, V = model(input_matrix)
        
        print("Computing loss...")
        loss = criterion(input_matrix, U, S, V)
        print(f"Loss: {loss.item():.6f}")
        
        # Verify output shapes
        print(f"\nOutput shapes:")
        print(f"U shape: {U.shape}")  # Expected: (2, 64, 16, 2)
        print(f"S shape: {S.shape}")  # Expected: (2, 16)
        print(f"V shape: {V.shape}")  # Expected: (2, 16, 64, 2)
        
        # 测试反向传播
        print("\nTesting backward pass...")
        loss.backward()
        print("✅ All tests passed!")
        
        # 内存使用估算
        model_size_mb = total_params * 4 / (1024 * 1024)  # 假设float32
        print(f"\nEstimated model size: {model_size_mb:.2f} MB")
        
        # 与原始模型的对比
        original_params = 68_931_552  # 原始模型参数量
        reduction_ratio = (original_params - total_params) / original_params * 100
        print(f"Parameter reduction: {reduction_ratio:.1f}%")
        print(f"Model size reduction: {original_params//total_params:.1f}x smaller")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

