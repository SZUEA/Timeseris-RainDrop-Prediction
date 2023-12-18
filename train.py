import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models import Informer

train_size = 1
x_stand = StandardScaler()
y_stand = StandardScaler()
s_len = 6
pre_len = 6
batch_size = 32
device = "cuda"
lr = 5e-5
epochs = 100


def create_data(datas):
    values = []
    labels = []

    lens = datas.shape[0]
    datas = datas.values
    for index in range(0, lens - pre_len - s_len):
        value = datas[index:index + s_len, [0, 2, 3, 4, 5]]
        label = datas[index + s_len - pre_len:index + s_len + pre_len, [0, 1]]

        values.append(value)
        labels.append(label)

    return values, labels


def read_data(each):
    datas = pd.DataFrame(each)

    xs = datas.values[:, [2, 3, 4, 5]]
    ys = datas.values[:, 1]

    x_stand.fit(xs)
    y_stand.fit(ys[:, None])

    values, labels = create_data(datas)

    train_x, test_x, train_y, test_y = train_test_split(values, labels, train_size=train_size)

    return train_x, test_x, train_y, test_y


# 自定义数据集
class RainDropData(Dataset):
    def __init__(self, values, labels):
        self.values, self.labels = values, labels

    def __len__(self):
        return len(self.values)

    def create_time(self, data):
        time = data[:, 0]
        time = pd.to_datetime(time)

        week = np.int32(time.dayofweek)[:, None]
        month = np.int32(time.month)[:, None]
        day = np.int32(time.day)[:, None]
        time_data = np.concatenate([month, week, day], axis=-1)

        return time_data

    def __getitem__(self, item):
        value = self.values[item]
        label = self.labels[item]

        value_t = self.create_time(value)
        label_t = self.create_time(label)

        value = x_stand.transform(value[:, 1:])
        label = y_stand.transform(label[:, 1][:, None])
        value = np.float32(value)
        label = np.float32(label)
        return value, label, value_t, label_t


def train():
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    model = Informer(
        enc_in=4,
        dec_in=1,
        c_out=1,
        out_len=pre_len,
    )
    model.train()
    model.to(device)
    loss_fc = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_datas = []
    test_datas = []

    ds = np.load('dataset/raindrop.npy', allow_pickle=True)

    for each in ds:
        train_x, test_x, train_y, test_y = read_data(each)

        train_data = RainDropData(train_x, train_y)
        train_data = DataLoader(train_data, shuffle=True, batch_size=batch_size)

        test_data = RainDropData(test_x, test_y)
        test_data = DataLoader(test_data, shuffle=True, batch_size=batch_size)

        train_datas.append(train_data)
        test_datas.append(test_data)

    min_loss = 99999999
    for epoch in range(epochs):
        loss_tot = 0
        for train_data, test_data in zip(train_datas, test_datas):
            for step, (x, y, xt, yt) in enumerate(train_data):
                mask = torch.zeros_like(y)[:, pre_len:].to(device)

                x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
                dec_y = torch.cat([y[:, :pre_len], mask], dim=1)
                logits = model(x, xt, dec_y, yt)
                loss = loss_fc(logits, y[:, pre_len:])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                os.system('cls')
                s = "train ==> epoch:{} - step:{} - loss:{}".format(epoch, step, loss)
                loss_tot += loss
                print(s)
            model.train()
        # save the model
        if loss_tot < min_loss:
            min_loss = loss_tot
            if epoch > 3:
                # delete the old model
                os.remove(f'checkpoint/epoch_{epoch - 3}.ckpt.pth')
            torch.save(model.state_dict(), f'checkpoint/informer.best.ckpt.pth')
        torch.save(model.state_dict(), f'checkpoint/informer.lastest.ckpt.pth')

            # model.eval()
            # with torch.no_grad():
            #     for step, (x, y, xt, yt) in enumerate(test_data):
            #         mask = torch.zeros_like(y)[:, pre_len:].to(device)
            #         x, y, xt, yt = x.to(device), y.to(device), xt.to(device), yt.to(device)
            #         dec_y = torch.cat([y[:, :pre_len], mask], dim=1)
            #         logits = model(x, xt, dec_y, yt)
            #         loss = loss_fc(logits, y[:, pre_len:])
            #         s = "test ==> epoch:{} - step:{} - loss:{}".format(epoch, step, loss)
            #         print(s)
            #         os.system('cls')


if __name__ == '__main__':
    train()
