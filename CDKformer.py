import torch
import torch.nn as nn
import torchaudio

class conformer(nn.Module):
    def __init__(self):
        super(conformer, self).__init__()
        self.conformer = torchaudio.models.Conformer(
            input_dim=12*51, 
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
            depthwise_conv_kernel_size=15,
        )

    def forward(self, x):
        x = x.reshape(x.size(0), x.size(1), -1)
        # 确保lengths张量在与x相同的设备上
        lengths = torch.ones(x.size(0), device=x.device)
        shared_features = self.conformer(x, lengths)[0].squeeze(dim=1)
        return shared_features

class MultiTaskModel(nn.Module):
    def __init__(self, backbone, input_dim, ind, num_predict_list):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone
        self.num_tasks = len(num_predict_list)
        self.task_branches = nn.ModuleList()
        self.ind = ind
        self.task_branches.append(nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 36)
        ))
        self.task_branches.append(nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 36)
        ))


    def forward(self, x):
        features = self.backbone(x)
        if self.ind == 0:
            outputs = self.task_branches[0](features)
        if self.ind == 1:
            outputs = self.task_branches[1](features)

        return outputs


def conformermulti(num_classes, ind):
    backbone = conformer()
    num_predict_list = [num_classes, num_classes]  # the num of num_classes should be ajust when model num change
    input_dim = 12*51
    model = MultiTaskModel(backbone, input_dim, ind, num_predict_list)
    return model

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据并移到指定设备
    x = torch.randn(2, 1, 12, 51).to(device)
    
    # 创建模型并移到指定设备
    model = conformermulti(num_classes=36, ind=0).to(device)
    
    # 前向传播
    output = model(x)
    print(output.shape)
