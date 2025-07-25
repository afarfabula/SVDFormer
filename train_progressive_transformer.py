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

# 数据加载函数
def load_training_data(data_dir, prefix):
    all_train_data = []
    files = sorted([f for f in os.listdir(data_dir) if f.startswith(prefix + 'TrainData') and f.endswith('.npy')])
    for f in files:
        data_path = os.path.join(data_dir, f)
        data = np.load(data_path)
        all_train_data.append(data)
    return np.concatenate(all_train_data, axis=0)

# 复数矩阵乘法（DirectML兼容）
def complex_bmm_real(a, b):
    """复数批量矩阵乘法，避免使用torch.complex"""
    real_part = a[..., 0] @ b[..., 0] - a[..., 1] @ b[..., 1]
    imag_part = a[..., 0] @ b[..., 1] + a[..., 1] @ b[..., 0]
    return torch.stack((real_part, imag_part), dim=-1)

# 正交性损失函数（支持部分分量）
def orthogonality_loss(U, V, num_components=None):
    """计算正交性损失
    Args:
        U: [B, R, M, 2] - 预测的U矩阵（复数表示为实部虚部）
        V: [B, R, N, 2] - 预测的V矩阵（复数表示为实部虚部）
        num_components: 使用的分量数量
    """
    if num_components is not None:
        U = U[:, :num_components, :, :]
        V = V[:, :num_components, :, :]
    
    B, R, M, _ = U.shape
    _, _, N, _ = V.shape
    
    def complex_matmul_real(a, b):
        real_part = torch.matmul(a[..., 0], b[..., 0]) - torch.matmul(a[..., 1], b[..., 1])
        imag_part = torch.matmul(a[..., 0], b[..., 1]) + torch.matmul(a[..., 1], b[..., 0])
        return torch.stack([real_part, imag_part], dim=-1)
    
    # U^H @ U = I_R
    U_reshaped = U.transpose(1, 2)  # [B, M, R, 2]
    U_H_reshaped = torch.stack([U_reshaped[..., 0].transpose(-2, -1), -U_reshaped[..., 1].transpose(-2, -1)], dim=-1)
    U_H_U = complex_matmul_real(U_H_reshaped, U_reshaped)
    
    # V^H @ V = I_R
    V_reshaped = V.transpose(1, 2)  # [B, N, R, 2]
    V_H_reshaped = torch.stack([V_reshaped[..., 0].transpose(-2, -1), -V_reshaped[..., 1].transpose(-2, -1)], dim=-1)
    V_H_V = complex_matmul_real(V_H_reshaped, V_reshaped)
    
    # 创建单位矩阵
    identity = torch.zeros(R, R, device=U.device, dtype=U.dtype)
    for i in range(R):
        identity[i, i] = 1.0
    
    identity_complex = torch.stack([identity, torch.zeros_like(identity)], dim=-1)
    identity_complex = identity_complex.unsqueeze(0).expand(B, -1, -1, -1)
    
    # 计算损失
    diff_U = U_H_U - identity_complex
    ortho_loss_U = torch.mean(torch.norm(diff_U.view(B, -1), p=2, dim=1))
    
    diff_V = V_H_V - identity_complex
    ortho_loss_V = torch.mean(torch.norm(diff_V.view(B, -1), p=2, dim=1))
    
    return ortho_loss_U + ortho_loss_V

# 重构损失函数（DirectML兼容）
# 重构损失函数（简化版本，修复维度问题）
def reconstruction_loss(pred_U, pred_S, pred_V, H_true, num_components=None):
    """计算重构损失（向量化版本，大幅提升性能）
    Args:
        pred_U, pred_S, pred_V: 预测的SVD分量
        H_true: 真实的输入矩阵 (B, 64, 64, 2)
        num_components: 使用的分量数量
    """
    if num_components is not None:
        pred_U = pred_U[:, :num_components, :, :]
        pred_S = pred_S[:, :num_components]
        pred_V = pred_V[:, :num_components, :, :]
    
    B, R, M, _ = pred_U.shape
    
    # 向量化重构：U @ diag(S) @ V^H
    # pred_U: (B, R, M, 2), pred_S: (B, R), pred_V: (B, R, M, 2)
    
    # 将 S 扩展为对角矩阵形式
    pred_S_expanded = pred_S.unsqueeze(-1).unsqueeze(-1)  # (B, R, 1, 1)
    
    # U * S (广播)
    US_real = pred_U[:, :, :, 0] * pred_S_expanded.squeeze(-1)  # (B, R, M)
    US_imag = pred_U[:, :, :, 1] * pred_S_expanded.squeeze(-1)  # (B, R, M)
    
    # V^H (共轭转置)
    VH_real = pred_V[:, :, :, 0]  # (B, R, M)
    VH_imag = -pred_V[:, :, :, 1]  # (B, R, M)
    
    # 计算 US @ VH (批量矩阵乘法)
    # (B, R, M, 1) @ (B, R, 1, M) -> (B, R, M, M)
    US_real_expanded = US_real.unsqueeze(-1)  # (B, R, M, 1)
    US_imag_expanded = US_imag.unsqueeze(-1)  # (B, R, M, 1)
    VH_real_expanded = VH_real.unsqueeze(-2)  # (B, R, 1, M)
    VH_imag_expanded = VH_imag.unsqueeze(-2)  # (B, R, 1, M)
    
    # 复数矩阵乘法: (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    component_real = (US_real_expanded * VH_real_expanded - 
                     US_imag_expanded * VH_imag_expanded)  # (B, R, M, M)
    component_imag = (US_real_expanded * VH_imag_expanded + 
                     US_imag_expanded * VH_real_expanded)  # (B, R, M, M)
    
    # 对所有分量求和
    reconstructed_real = torch.sum(component_real, dim=1)  # (B, M, M)
    reconstructed_imag = torch.sum(component_imag, dim=1)  # (B, M, M)
    
    # 计算重构误差
    H_real = H_true[:, :, :, 0]
    H_imag = H_true[:, :, :, 1]
    
    # 计算 |H - H_reconstructed|^2
    diff_real = H_real - reconstructed_real
    diff_imag = H_imag - reconstructed_imag
    
    # |a + bi|^2 = a^2 + b^2
    #loss = torch.mean (torch.sqrt(diff_real ** 2 + diff_imag ** 2)/torch.sqrt(H_real ** 2 + H_imag ** 2))
    loss = torch.mean(diff_real ** 2 + diff_imag ** 2)
    return loss

# 渐进式训练函数
def progressive_training(resume_from=None):
    # --- 配置 ---
    data_dir = './CompetitionData1'
    prefix = 'Round1'
    batch_size = 32 # 减小批量大小以适应多阶段训练
    learning_rate = 1e-4
    epochs = 30
    ortho_weight = 1.0
    recon_weight = 1.0
    
    # 每个阶段监督的分量数量
    #stage_components = [1,4, 8, 16, 32,32]
    stage_components = [1,1, 1, 1, 1,1]
    iterations = 6
    
    print(f"=== 渐进式训练配置 ===")
    print(f"迭代次数: {iterations}")
    print(f"各阶段监督分量: {stage_components}")
    print(f"批量大小: {batch_size}")
    print(f"学习率: {learning_rate}")
    print(f"训练轮数: {epochs}")
    if resume_from:
        print(f"从权重文件恢复训练: {resume_from}")
    
    # --- 数据加载 ---
    print("\n加载训练数据...")
    train_data = load_training_data(data_dir, prefix)
    train_tensor = torch.from_numpy(train_data).float()
    train_dataset = TensorDataset(train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"数据加载完成，共 {len(train_data)} 个样本")
    
    # --- 模型、损失和优化器 ---
    model = SVDTransformer(
        input_dim=64, 
        embed_dim=128, 
        rank=32, 
        nhead=8, 
        num_encoder_layers=4, 
        dim_feedforward=512,
        max_iters=10,
        #use_conv_encoder=True
        use_vit_encoder=True
    ).to(device)
    
    # 加载预训练权重
    start_epoch = 0
    if resume_from:
        try:
            checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
            
            # 如果保存的是完整的checkpoint（包含epoch信息）
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                start_epoch = checkpoint.get('epoch', 0)
                print(f"✅ 成功加载checkpoint，从第 {start_epoch + 1} 轮开始训练")
            else:
                # 如果只是模型权重
                model.load_state_dict(checkpoint)
                print(f"✅ 成功加载模型权重")
                
        except FileNotFoundError:
            print(f"❌ 权重文件 {resume_from} 不存在，从头开始训练")
        except Exception as e:
            print(f"❌ 加载权重失败: {e}，从头开始训练")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    # 如果从checkpoint恢复，调整scheduler
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # --- 训练循环 ---
    print("\n开始渐进式训练...")
    
    for epoch in range(start_epoch, epochs):  # 从start_epoch开始
        model.train()
        total_loss = 0
        stage_losses = [0.0] * iterations
        
        for i, (H_batch,) in enumerate(train_loader):
            H_batch = H_batch.to(device)
            
            # --- 前向传播获取所有阶段输出 ---
            all_stages = model(H_batch, iterations=iterations, return_all_stages=True)
            
            # --- 计算每个阶段的损失 ---
            total_batch_loss = 0
            
            for stage_idx, (pred_U, pred_S, pred_V) in enumerate(all_stages):
                num_components = stage_components[stage_idx]
                
                # 重构损失（使用当前阶段的分量数）
                recon_loss = reconstruction_loss(pred_U, pred_S, pred_V, H_batch, num_components)
                
                # 正交性损失（使用当前阶段的分量数）
                ortho_loss = orthogonality_loss(pred_U, pred_V, num_components)
                
                # 阶段总损失（仅使用重构损失和正交性损失）
                stage_loss = (recon_weight * recon_loss + ortho_weight * ortho_loss)
                
                total_batch_loss += stage_loss
                stage_losses[stage_idx] += stage_loss.item()
            
            # --- 反向传播和优化 ---
            optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            # 每个batch打印信息
            if (i + 1) % 300 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(train_loader)}]")
                print(f"  总损失: {total_batch_loss.item():.4f}")
                print(f"  各阶段损失: {[f'{loss:.4f}' for loss in [stage_losses[j]/(i+1) for j in range(iterations)]]}")

        torch.save(model.state_dict(), 'progressive_transformer_final.pth')
        # 更新学习率
        scheduler.step()
        
        # 打印训练信息
        avg_loss = total_loss / len(train_loader)
        avg_stage_losses = [loss / len(train_loader) for loss in stage_losses]
        
        print(f"\nEpoch [{epoch+1}/{epochs}] 完成")
        print(f"  平均总损失: {avg_loss:.4f}")
        print(f"  各阶段平均损失: {[f'{loss:.4f}' for loss in avg_stage_losses]}")
        print(f"  学习率: {scheduler.get_last_lr()[0]:.6f}")
        
        # 每10个epoch显示详细损失信息
        if (epoch + 1) % 10 == 0:
            print(f"\n=== Epoch {epoch+1} 详细损失信息 ===")
            # 重新计算一个batch的详细损失用于显示
            model.eval()
            with torch.no_grad():
                sample_batch = next(iter(train_loader))
                H_batch = sample_batch[0].to(device)
                
                # 前向传播
                all_stages = model(H_batch, iterations=iterations, return_all_stages=True)
                
                # 计算各阶段的详细损失
                for stage_idx, (pred_U, pred_S, pred_V) in enumerate(all_stages):
                    num_components = stage_components[stage_idx]
                    stage_name = f"阶段{stage_idx+1}"
                    
                    # 重构损失
                    recon_loss = reconstruction_loss(pred_U, pred_S, pred_V, H_batch, num_components)
                    
                    # 正交性损失
                    ortho_loss = orthogonality_loss(pred_U, pred_V, num_components)
                    
                    print(f"  {stage_name} (分量数: {num_components}):")
                    print(f"    重构损失: {recon_loss.item():.6f}")
                    print(f"    正交损失: {ortho_loss.item():.6f}")
            
            model.train()
        
        # 每10个epoch保存一次模型
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'progressive_transformer_epoch_{epoch+1}.pth')
            print(f"  模型已保存: progressive_transformer_epoch_{epoch+1}.pth")
    
    print("\n训练完成！")
    torch.save(model.state_dict(), 'progressive_transformer_final.pth')
    print("最终模型已保存: progressive_transformer_final.pth")
    
    # --- 训练总结 ---
    print("\n=== 训练总结 ===")
    print(f"总训练轮数: {epochs}")
    print(f"最终学习率: {scheduler.get_last_lr()[0]:.6f}")
    print(f"各阶段监督分量数: {stage_components}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

def test_progressive_model():
    """测试训练好的渐进式模型"""
    print("\n=== 测试渐进式模型 ===")
    
    # 确保模型配置与保存时一致
    model = SVDTransformer(
        input_dim=64, 
        embed_dim=128, 
        rank=32, 
        nhead=8, 
        num_encoder_layers=4, 
        dim_feedforward=512,
        max_iters=10,
        use_vit_encoder=True  # 添加这一行，匹配保存的模型
    ).to(device)
    
    try:
        model.load_state_dict(torch.load('progressive_transformer_final.pth', map_location=device, weights_only=False))
        print("模型加载成功")
    except FileNotFoundError:
        print("未找到训练好的模型，请先运行训练")
        return
    
    model.eval()
    
    # 加载训练数据 - 修改这里的路径和前缀
    print("加载训练数据...")
    try:
        train_data = load_training_data('./CompetitionData1', 'Round1')  # 修改路径和前缀
        print(f"训练数据形状: {train_data.shape}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("生成随机测试数据...")
        # 如果没有训练数据，生成随机数据进行测试
        train_data = np.random.randn(200, 64, 64, 2)
        print(f"使用随机数据，形状: {train_data.shape}")
    
    # 随机采样200个样本
    num_samples = min(200, len(train_data))
    indices = torch.randperm(len(train_data))[:num_samples]
    sampled_data = train_data[indices]
    
    # 转换为tensor并移到设备
    test_input = torch.from_numpy(sampled_data).float().to(device)
    print(f"采样了 {num_samples} 个训练样本进行测试")
    
    # 定义阶段分量数
    stage_components = [1,4, 8, 16, 32,32]
    
    with torch.no_grad():
        # 测试所有阶段输出
        all_stages = model(test_input, iterations=6, return_all_stages=True)
        
        print(f"\n测试输入维度: {test_input.shape}")
        print(f"输出阶段数: {len(all_stages)}")
        
        # 计算每个阶段的平均损失
        for i, (U, S, V) in enumerate(all_stages):
            print(f"\n阶段 {i+1} (监督 {stage_components[i]} 个分量):")
            print(f"  U: {U.shape}")
            print(f"  S: {S.shape}")
            print(f"  V: {V.shape}")
            
            # 计算重构误差
            recon_loss = reconstruction_loss(U, S, V, test_input, stage_components[i])
            print(f"  平均重构误差: {recon_loss.item():.6f}")
            
            # 计算正交性损失
            ortho_loss = orthogonality_loss(U, V, stage_components[i])
            print(f"  平均正交损失: {ortho_loss.item():.6f}")

if __name__ == '__main__':
    print("=== SVD Transformer 渐进式训练 ===")
    print("1. 从头开始训练")
    print("2. 从权重文件恢复训练")
    print("3. 测试模型")
    
    choice = input("请选择操作 (1/2/3): ").strip()
    
    if choice == '1':
        progressive_training()
    elif choice == '2':
        weight_file = input("请输入权重文件路径 (例如: progressive_transformer_final.pth): ").strip()
        if not weight_file:
            weight_file = 'progressive_transformer_final.pth'  # 默认文件
        progressive_training(resume_from=weight_file)
    elif choice == '3':
        test_progressive_model()
    else:
        print("无效选择，默认从头开始训练")
        progressive_training()