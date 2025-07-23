import os
import numpy as np
import torch
import torch_directml

from solution import SVDNet
from debug_svd_loss import svd_approximation_error_loss

def load_test_data(data_dir, prefix):
    test_data_path = os.path.join(data_dir, prefix + 'TestData1.npy')
    test_labels_path = os.path.join(data_dir, prefix + 'TrainLabel1.npy') # Using TrainLabel for evaluation as TestLabel is not available
    
    if not os.path.exists(test_data_path):
        print(f"Error: Test data file not found at {test_data_path}")
        return None, None
    if not os.path.exists(test_labels_path):
        print(f"Error: Label file not found at {test_labels_path}")
        return None, None
        
    X_test = np.load(test_data_path)
    Y_test = np.load(test_labels_path)
    return X_test, Y_test

def main():
    # --- 配置 ---
    data_dir = './CompetitionData1'
    prefix = 'Round1'
    model_path = 'svdnet_model_Round1_direct.pth'
    batch_size = 512 # 设置一个合理的批量大小

    # --- 设备设置 ---
    try:
        if torch_directml.is_available():
            device = torch_directml.device()
            print(f"Using DirectML device for evaluation.")
        else:
            device = 'cpu'
            print("DirectML not available, falling back to CPU.")
    except ImportError:
        device = 'cpu'
        print("torch_directml not found, falling back to CPU.")

    # --- 加载模型 ---
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    model = SVDNet().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    print(f"Model loaded from {model_path}")

    # --- 加载数据 ---
    X_test, Y_test = load_test_data(data_dir, prefix)
    if X_test is None:
        return
        
    X_test_tensor = torch.from_numpy(X_test).float()
    Y_test_tensor = torch.from_numpy(Y_test).float()

    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, Y_test_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    print(f"Loaded {len(X_test_tensor)} samples for evaluation.")

    # --- 评估 ---
    total_loss = 0
    total_reconstruction_error = 0
    total_ortho_error = 0
    num_batches = 0

    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            pred_U, pred_S, pred_V = model(X_batch)
            
            loss, reconstruction_error, ortho_error = svd_approximation_error_loss(pred_U, pred_S, pred_V, Y_batch)
            
            total_loss += loss.item()
            total_reconstruction_error += reconstruction_error.item()
            total_ortho_error += ortho_error.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_reconstruction_error = total_reconstruction_error / num_batches
    avg_ortho_error = total_ortho_error / num_batches

    print("--- Evaluation Results ---")
    print(f"Average Total Loss: {avg_loss:.6f}")
    print(f"  - Average Reconstruction Error: {avg_reconstruction_error:.6f}")
    print(f"  - Average Orthogonality Error: {avg_ortho_error:.6f}")

if __name__ == '__main__':
    main()