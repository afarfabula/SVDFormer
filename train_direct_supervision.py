import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch_directml

from solution import SVDNet

# 1. 数据加载函数 (与 train.py 相同)
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

# 2. 主训练逻辑
if __name__ == '__main__':
    PathSet = {0: "./DebugData", 1: "./CompetitionData1", 2: "./CompetitionData2", 3: "./CompetitionData3"}
    PrefixSet = {0: "Round0", 1: "Round1", 2: "Round2", 3: "Round3"}
    
    Ridx = 1
    DATA_DIR = PathSet[Ridx]
    PREFIX = PrefixSet[Ridx]

    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3 # 调整学习率
    MODEL_SAVE_PATH = f'svdnet_model_{PREFIX}_direct.pth'

    try:
        if torch_directml.is_available():
            device = torch_directml.device()
            print(f"Using DirectML device for training.")
        else:
            device = 'cpu'
            print("DirectML not available, falling back to CPU.")
    except ImportError:
        device = 'cpu'
        print("torch_directml not found, falling back to CPU.")

    print(f"Loading data from {DATA_DIR} with prefix {PREFIX}...")
    X, Y_true = load_training_data(DATA_DIR, PREFIX)
    
    X_tensor = torch.from_numpy(X).float()
    Y_true_tensor = torch.from_numpy(Y_true).float()

    dataset = TensorDataset(X_tensor, Y_true_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Data loaded successfully.")

    model = SVDNet().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    criterion = nn.MSELoss() # 使用MSE作为基础损失函数
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print("Starting training with direct supervision...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # --- 解析解计算 ---
            # 1. 将标签从实数表示 (B, M, N, 2) 转换到复数张量
            #    这步需要在CPU上完成，因为DirectML不支持复数
            labels_cpu = labels.to('cpu')
            labels_complex = torch.complex(labels_cpu[..., 0], labels_cpu[..., 1])

            # 2. 在CPU上计算SVD
            U_true_c, S_true_cpu, Vh_true_c = torch.linalg.svd(labels_complex, full_matrices=False)
            V_true_c = Vh_true_c.mH # V = (V^H)^H

            # 3. 将解析解转换回实数表示并移动到目标设备
            U_true_full = torch.stack((U_true_c.real, U_true_c.imag), dim=-1).to(device)
            S_true_full = S_true_cpu.to(device)
            V_true_full = torch.stack((V_true_c.real, V_true_c.imag), dim=-1).to(device)
            optimizer.zero_grad()

            # 模型预测
            pred_U, pred_S, pred_V = model(inputs)


            # --- 裁剪解析解以匹配模型输出 --- 
            # 这是临时的解决方案，根本问题在于模型定义
            R_model = pred_U.shape[2]
            U_true = U_true_full[:, :, :R_model, :]
            S_true = S_true_full[:, :R_model]
            V_true = V_true_full[:, :, :R_model, :]
            # --- 解析解计算结束 ---


            # 计算各项损失
            loss_U = criterion(pred_U, U_true)
            loss_S = criterion(pred_S, S_true)
            loss_V = criterion(pred_V, V_true)

            # 总损失
            total_batch_loss = loss_U + loss_S + loss_V

            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if (i + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(dataloader)}], Loss: {total_batch_loss.item():.6f} (U: {loss_U.item():.6f}, S: {loss_S.item():.6f}, V: {loss_V.item():.6f})')

            total_loss += total_batch_loss.item()
        
        scheduler.step() # 在每个epoch结束后更新学习率
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Average Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]}")
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(f"Training finished. Model saved to {MODEL_SAVE_PATH}")