import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_size = 0.85

# class MyDataset(Dataset):
#     def __init__(self):
#         super(MyDataset, self).__init__()
#         self.data = pd.read_csv('./data_9.csv')
#         self.data = self.data[['Alti', 'TEM', 'DPT', 'RHU', 'WIN_D_INST', 'WIN_S_INST', 'GST', 'PRE_1h']]
#         self.data = self.data.values
#         self.data = self.data.astype(np.float32)
#         self.data = torch.from_numpy(self.data)
#         self.data = self.data.reshape(-1, 8, 1)
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def __getitem__(self, index):
#         return self.data[index]
#


df = pd.read_csv('./data_9.csv')
print(df.shape)