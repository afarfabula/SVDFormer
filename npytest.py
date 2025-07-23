import numpy as np
import os

# 请确保这里的路径是正确的
label_path = os.path.join('CompetitionData1/Round1TrainLabel1.npy')
label = np.load(label_path, allow_pickle=True)
print(label.shape)
label_path = os.path.join('CompetitionData1/Round1TrainData1.npy')
label = np.load(label_path, allow_pickle=True)
print(label.shape)