import os
import numpy as np
from solution import SVDNet
import torch
import torch_directml

# 读取配置文件函数
def read_cfg_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        line_fmt = [line.rstrip('\n').split(' ') for line in lines]
    info = line_fmt
    samp_num = int(info[0][0])
    M = int(info[1][0])
    N = int(info[2][0])
    IQ = int(info[3][0])
    R = int(info[4][0])
    return samp_num, M, N, IQ, R
 
if __name__ == "__main__":
    print("<<< Welcome to 2025 Wireless Algorithm Contest! >>>\n")
    ## 不同轮次的输入数据可放在不同文件夹中便于管理，这里用户可以自定义
    PathSet = {0: "./DebugData", 1: "./CompetitionData1", 2: "./CompetitionData2", 3: "./CompetitionData3"}
    PrefixSet = {0: "Round0", 1: "Round1", 2: "Round2", 3: "Round3"}
 
    Ridx = 0  # 设置比赛轮次索引，指明数据存放目录。0:Test; 1: 1st round; 2: 2nd round ...
    PathRaw = PathSet[Ridx]
    Prefix = PrefixSet[Ridx]
   
    # 查找文件夹中包含的所有比赛/测试数据文件，非本轮次数据请不要放在目标文件夹中
    files = os.listdir(PathRaw)
    Caseidxes = []
    for f in sorted(files):
        if f.find('CfgData') != -1 and f.endswith('.txt'):
            Caseidxes.append(f.split('CfgData')[-1].split('.txt')[0])
 
    ## 创建对象并处理
    for Caseidx in Caseidxes:
        print('Processing Round ' + str(Ridx) + ' Case ' + str(Caseidx))
       
        # 读取配置文件 RoundYCfgDataX.txt
        cfg_path = PathRaw + '/' + Prefix + 'CfgData' + Caseidx + '.txt'
        _, M, N, IQ, R = read_cfg_file(cfg_path)
 
        # 读取信道输入文件 RoundYTestDataX.npy
        H_data_file = PathRaw + '/' + Prefix + 'TestData' + Caseidx + '.npy'
        H_data_all = np.load(H_data_file)
        if (M, N, IQ) != H_data_all.shape[1:4]:
            raise ValueError("Channel data loading error!")
 
        # 输出模型结果
        samp_num = H_data_all.shape[0]
        U_out_all = np.zeros((samp_num, M, R, IQ), dtype=float)
        S_out_all = np.zeros((samp_num, R), dtype=float)
        V_out_all = np.zeros((samp_num, N, R, IQ), dtype=float)
        
        # 正确设置device，使用torch-directml
        if torch_directml.is_available():
            device = torch_directml.device()
            print(f"Using DirectML device for computation.")
        else:
            device = 'cpu'
            print("DirectML not available, falling back to CPU.")

        model = SVDNet().to(device)
        # 模型定义后选手需要读入训练好的模型参数----
        #-------------------------------------
        for samp_idx in range(samp_num):
            H_data = H_data_all[samp_idx, ...]  # 模型输入的非理想信道数据
            H_data = torch.tensor(H_data)
            #########模型代码在solution.py中的SVDNet类定义########
            #########参赛者需用自己模型代码替代SVDNet##############
            with torch.no_grad():
                H_data = H_data.to(device)
                print(H_data.shape)
                U_out, S_out, V_out = model(H_data)
                print(U_out.shape)
                print(S_out.shape)
                print(V_out.shape)
            ###################################################
            U_out_all[samp_idx, ...] = U_out.cpu().numpy()
            S_out_all[samp_idx, ...] = S_out.cpu().numpy()
            V_out_all[samp_idx, ...] = V_out.cpu().numpy()
        U_out_all = U_out_all.astype(np.float32)
        S_out_all = S_out_all.astype(np.float32)
        V_out_all = V_out_all.astype(np.float32)
 
        # 保存输出结果
        TestOutput_file = Prefix + 'TestOutput' + Caseidx + '.npz'
        np.savez(TestOutput_file, U_out=U_out_all, S_out=S_out_all, V_out=V_out_all)
        print("File saved.")