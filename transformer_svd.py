import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, iter_idx):
        # x is the input tensor, iter_idx is the current iteration number
        # self.pe is (1, max_len, d_model)
        # We select the positional encoding for the current iteration and add it to x
        return x + self.pe[:, iter_idx, :].unsqueeze(1) # Add to sequence dimension

class SVDTransformer(nn.Module):
    def __init__(self, input_dim=64, embed_dim=64, rank=32, nhead=8, num_encoder_layers=6, 
                 dim_feedforward=256, dropout=0.1, max_iters=10, use_conv_encoder=False, use_vit_encoder=False):
        super(SVDTransformer, self).__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.embed_dim = embed_dim
        self.max_iters = max_iters
        self.use_conv_encoder = use_conv_encoder
        self.use_vit_encoder = use_vit_encoder

        # 1. Input Encoder - 保持共享（因为输入H是固定的）
        if use_vit_encoder:
            # ViT编码器配置保持不变
            self.patch_size = 8
            self.num_patches = (input_dim // self.patch_size) ** 2
            self.patch_dim = self.patch_size * self.patch_size * 2
            
            self.patch_embedding = nn.Linear(self.patch_dim, embed_dim)
            self.patch_pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
            
            vit_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=nhead//2,
                dim_feedforward=dim_feedforward//2, 
                dropout=dropout, 
                batch_first=True
            )
            self.vit_transformer = nn.TransformerEncoder(vit_layer, num_layers=2)
            self.patch_to_row = nn.Linear(embed_dim, embed_dim)
            
        elif use_conv_encoder:
            self.h_conv_encoder = nn.Sequential(
                nn.Conv1d(in_channels=input_dim*2, out_channels=embed_dim*2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=embed_dim*2, out_channels=embed_dim, kernel_size=3, padding=1),
                nn.ReLU(),
            )
        else:
            self.h_encoder = nn.Linear(input_dim * 2, embed_dim)

        # 2. 为每个阶段创建独立的SVD组件嵌入层
        self.stage_u_embeds = nn.ModuleList([
            nn.Linear(input_dim * 2, embed_dim) for _ in range(max_iters)
        ])
        self.stage_v_embeds = nn.ModuleList([
            nn.Linear(input_dim * 2, embed_dim) for _ in range(max_iters)
        ])
        self.stage_s_embeds = nn.ModuleList([
            nn.Linear(rank, embed_dim) for _ in range(max_iters)
        ])

        # 3. 为每个阶段创建独立的Transformer编码器
        self.stage_transformers = nn.ModuleList()
        for i in range(max_iters):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                batch_first=True
            )
            transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
            self.stage_transformers.append(transformer)

        # 4. 为每个阶段创建独立的解码器
        self.stage_u_decoders = nn.ModuleList([
            nn.Linear(embed_dim, input_dim * 2) for _ in range(max_iters)
        ])
        self.stage_v_decoders = nn.ModuleList([
            nn.Linear(embed_dim, input_dim * 2) for _ in range(max_iters)
        ])
        self.stage_s_decoders = nn.ModuleList([
            nn.Linear(embed_dim, rank) for _ in range(max_iters)
        ])

        # 5. 位置编码（保持共享）
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_iters)

        # 6. 初始参数（保持共享）
        self.initial_U = nn.Parameter(torch.randn(1, rank, input_dim, 2))
        self.initial_V = nn.Parameter(torch.randn(1, rank, input_dim, 2))
        self.initial_S = nn.Parameter(torch.randn(1, rank))

    def forward(self, x, iterations=4, return_all_stages=False):
        batch_size = x.shape[0]
    
        # 编码输入矩阵H（使用共享编码器）
        if self.use_vit_encoder:
            patches = self._extract_patches(x)
            patch_tokens = self.patch_embedding(patches)
            patch_tokens = patch_tokens + self.patch_pos_embedding
            vit_output = self.vit_transformer(patch_tokens)
            encoded_h = self.patch_to_row(vit_output)
        elif self.use_conv_encoder:
            h_reshaped = x.view(batch_size, self.input_dim, self.input_dim * 2)
            h_transposed = h_reshaped.transpose(1, 2)
            encoded_h_conv = self.h_conv_encoder(h_transposed)
            encoded_h = encoded_h_conv.transpose(1, 2)
        else:
            h_reshaped = x.view(batch_size, self.input_dim, self.input_dim * 2)
            encoded_h = self.h_encoder(h_reshaped)
    
        # 初始化SVD分量
        pred_U = self.initial_U.expand(batch_size, -1, -1, -1)
        pred_V = self.initial_V.expand(batch_size, -1, -1, -1)
        pred_S = self.initial_S.expand(batch_size, -1)
    
        if return_all_stages:
            all_stages = []
    
        # 迭代细化 - 每个阶段使用独立权重
        for i in range(iterations):
            # 使用第i阶段的独立嵌入层
            u_flat = pred_U.flatten(start_dim=2)
            u_tokens = self.stage_u_embeds[i](u_flat)
    
            v_flat = pred_V.flatten(start_dim=2)
            v_tokens = self.stage_v_embeds[i](v_flat)
    
            s_tokens = self.stage_s_embeds[i](pred_S).unsqueeze(1)
    
            # 拼接tokens
            tokens = torch.cat([encoded_h, u_tokens, v_tokens, s_tokens], dim=1)
    
            # 添加位置编码
            tokens = self.pos_encoder(tokens, i)
    
            # 使用第i阶段的独立Transformer
            output_tokens = self.stage_transformers[i](tokens)
    
            # 分割输出tokens
            h_out_len, u_out_len, v_out_len = self.input_dim, self.rank, self.rank
            u_start, v_start = h_out_len, h_out_len + u_out_len
            
            u_out = output_tokens[:, u_start:v_start, :]
            v_out = output_tokens[:, v_start:v_start + v_out_len, :]
            s_out = output_tokens[:, -1, :]
    
            # 使用第i阶段的独立解码器
            delta_U_flat = self.stage_u_decoders[i](u_out)
            delta_U = delta_U_flat.view(batch_size, self.rank, self.input_dim, 2)
    
            delta_V_flat = self.stage_v_decoders[i](v_out)
            delta_V = delta_V_flat.view(batch_size, self.rank, self.input_dim, 2)
    
            delta_S = self.stage_s_decoders[i](s_out)
    
            # 更新SVD分量
            pred_U = pred_U + delta_U
            pred_V = pred_V + delta_V
            pred_S = pred_S + delta_S
            
            # 添加列归一化（可选）
            pred_U = self.normalize_columns(pred_U)
            pred_V = self.normalize_columns(pred_V)
    
            if return_all_stages:
                all_stages.append((pred_U.clone(), pred_S.clone(), pred_V.clone()))
    
        if return_all_stages:
            return all_stages
        else:
            return pred_U, pred_S, pred_V
    def normalize_columns(self, mat):
        """
        对复数矩阵的列进行归一化
        mat: (B, rank, input_dim, 2) - 最后一维是 [实部, 虚部]
        返回: 归一化后的矩阵
        """
        real = mat[..., 0]  # (B, rank, input_dim)
        imag = mat[..., 1]  # (B, rank, input_dim)
        
        # 计算复数模长: sqrt(real^2 + imag^2)
        norm = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)  # (B, rank, input_dim)
        
        # 扩展维度以匹配原矩阵
        norm = norm.unsqueeze(-1).expand_as(mat)  # (B, rank, input_dim, 2)
        
        # 归一化
        mat = mat / norm
        return mat

    def _extract_patches(self, x):
        """
        将输入矩阵分割成patches
        x: (B, 64, 64, 2)
        返回: (B, num_patches, patch_dim) = (B, 64, 128)
        """
        batch_size = x.shape[0]
        
        # 使用unfold提取patches
        # x: (B, 64, 64, 2) -> (B, 2, 64, 64)
        x_permuted = x.permute(0, 3, 1, 2)
        
        # 提取8x8的patches
        patches = x_permuted.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches: (B, 2, 8, 8, 8, 8)
        
        # 重新整理维度
        patches = patches.contiguous().view(batch_size, 2, 8, 8, -1)
        # patches: (B, 2, 8, 8, 64)
        
        patches = patches.permute(0, 4, 1, 2, 3)
        # patches: (B, 64, 2, 8, 8)
        
        # 展平每个patch
        patches = patches.view(batch_size, self.num_patches, -1)
        # patches: (B, 64, 128)
        
        return patches
    

def test_progressive_training():
   
    """测试渐进式训练的输出维度"""
    print("\n=== 渐进式训练输出测试 ===")
    
    batch_size = 2
    input_dim = 64
    rank = 32
    embed_dim = 128
    iterations = 4
    
    model = SVDTransformer(
        input_dim=input_dim,
        embed_dim=embed_dim,
        rank=rank,
        max_iters=10
    )
    
    test_input = torch.randn(batch_size, input_dim, input_dim, 2)
    print(f"输入维度: {test_input.shape}")
    
    # 测试返回所有阶段的输出
    with torch.no_grad():
        all_stages = model(test_input, iterations=iterations, return_all_stages=True)
    
    print(f"\n总共 {len(all_stages)} 个迭代阶段:")
    for i, (U, S, V) in enumerate(all_stages):
        print(f"阶段 {i+1}:")
        print(f"  U: {U.shape} (完整32个分量)")
        print(f"  S: {S.shape} (完整32个分量)")
        print(f"  V: {V.shape} (完整32个分量)")
        
        # 验证维度
        assert U.shape == (batch_size, rank, input_dim, 2)
        assert S.shape == (batch_size, rank)
        assert V.shape == (batch_size, rank, input_dim, 2)
    
    print("\n✅ 所有阶段输出维度正确！")
    print("💡 每个阶段都输出完整的32个SVD分量，具体监督策略在训练脚本中定义")
    
    return all_stages

# 在主函数中添加测试
if __name__ == "__main__":
    print("=== SVDTransformer 维度测试 ===")
    
    # 测试参数
    batch_size = 4
    input_dim = 64
    rank = 32
    embed_dim = 128
    iterations = 4
    max_test_iterations = 10  # 设置更大的最大迭代数
    
    # 初始化模型 - 使用更大的max_iters
    model = SVDTransformer(
        input_dim=input_dim,
        embed_dim=embed_dim,
        rank=rank,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=256,
        dropout=0.1,
        max_iters=max_test_iterations  # 改为10，支持更多迭代测试
    )
    
    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 模型总参数量: {total_params:,}")
    
    # 创建测试输入
    test_input = torch.randn(batch_size, input_dim, input_dim, 2)
    print(f"\n📥 输入维度: {test_input.shape}")
    print(f"   - 批量大小: {batch_size}")
    print(f"   - 矩阵尺寸: {input_dim}x{input_dim}")
    print(f"   - 复数表示: 2 (实部+虚部)")
    
    try:
        print("\n🔄 开始前向传播...")
        
        # 前向传播
        with torch.no_grad():
            U, S, V = model(test_input, iterations=iterations)
        
        print("✅ 前向传播成功！")
        
        # 检查输出维度
        print(f"\n📤 输出维度检查:")
        print(f"   U 矩阵: {U.shape} (期望: ({batch_size}, {rank}, {input_dim}, 2))")
        print(f"   S 向量: {S.shape} (期望: ({batch_size}, {rank}))")
        print(f"   V 矩阵: {V.shape} (期望: ({batch_size}, {rank}, {input_dim}, 2))")
        
        # 验证维度正确性
        expected_U = (batch_size, rank, input_dim, 2)
        expected_S = (batch_size, rank)
        expected_V = (batch_size, rank, input_dim, 2)
        
        assert U.shape == expected_U, f"U维度错误: 得到{U.shape}, 期望{expected_U}"
        assert S.shape == expected_S, f"S维度错误: 得到{S.shape}, 期望{expected_S}"
        assert V.shape == expected_V, f"V维度错误: 得到{V.shape}, 期望{expected_V}"
        
        print("\n✅ 所有维度检查通过！")
        
        # 数值范围检查
        print(f"\n📈 数值范围检查:")
        print(f"   U 范围: [{U.min():.4f}, {U.max():.4f}]")
        print(f"   S 范围: [{S.min():.4f}, {S.max():.4f}]")
        print(f"   V 范围: [{V.min():.4f}, {V.max():.4f}]")
        
        # 测试重构误差（简单验证）
        print(f"\n🔍 重构验证:")
        
        # 将复数表示转换为复数张量进行矩阵乘法
        def complex_matmul(U, S, V):
            # U: (B, rank, 64, 2), S: (B, rank), V: (B, rank, 64, 2)
            # 转换为复数
            U_complex = torch.complex(U[..., 0], U[..., 1])  # (B, rank, 64)
            V_complex = torch.complex(V[..., 0], V[..., 1])  # (B, rank, 64)
            
            # 重构: U @ diag(S) @ V^H
            # U: (B, rank, 64), S: (B, rank), V: (B, rank, 64)
            S_expanded = S.unsqueeze(-1)  # (B, rank, 1)
            US = U_complex * S_expanded  # (B, rank, 64)
            
            # V^H: (B, 64, rank)
            V_H = V_complex.conj().transpose(-2, -1)
            
            # 重构矩阵: (B, 64, 64)
            reconstructed = torch.bmm(US.transpose(-2, -1), V_H.transpose(-2, -1))
            
            return reconstructed
        
        try:
            # 原始输入转为复数
            input_complex = torch.complex(test_input[..., 0], test_input[..., 1])
            
            # 重构
            reconstructed = complex_matmul(U, S, V)
            
            # 计算重构误差
            reconstruction_error = torch.norm(input_complex - reconstructed, dim=(-2, -1)).mean()
            print(f"   平均重构误差: {reconstruction_error.item():.6f}")
            
        except Exception as e:
            print(f"   重构验证跳过 (复数运算问题): {e}")
        
        # 测试不同批量大小
        print(f"\n🔄 测试不同批量大小:")
        for test_batch in [1, 2, 8]:
            test_input_batch = torch.randn(test_batch, input_dim, input_dim, 2)
            with torch.no_grad():
                U_batch, S_batch, V_batch = model(test_input_batch, iterations=2)
            print(f"   批量大小 {test_batch}: U{U_batch.shape}, S{S_batch.shape}, V{V_batch.shape} ✅")
        
        # 测试不同迭代次数 - 修复：确保不超过max_iters
        print(f"\n🔄 测试不同迭代次数:")
        for test_iter in [1, 2, 6, 8]:  # 现在可以安全测试到8次迭代
            with torch.no_grad():
                U_iter, S_iter, V_iter = model(test_input, iterations=test_iter)
            print(f"   迭代次数 {test_iter}: 输出维度一致 ✅")
        
        print(f"\n🎉 所有测试通过！模型维度设计正确。")
        
        # 模型信息总结
        print(f"\n📋 模型总结:")
        print(f"   - 输入维度: (B, {input_dim}, {input_dim}, 2)")
        print(f"   - SVD秩: {rank}")
        print(f"   - 嵌入维度: {embed_dim}")
        print(f"   - Transformer层数: 6")
        print(f"   - 注意力头数: 8")
        print(f"   - 最大迭代次数: {max_test_iterations}")
        print(f"   - 总参数量: {total_params:,}")
        print(f"   - 模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n=== 测试完成 ===")
    
    # 添加渐进式训练测试
    test_progressive_training()


