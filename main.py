import pickle
import sys
import random

import numpy as np
import torch.utils.data
from torch.utils.data import DataLoader

from dataset import AudioDataset
from network import AudioNet
from trainer import AudioTrainer

import seaborn as sns
import matplotlib.pyplot as plt

mode = sys.argv[1]

sample_num = 4800
microphone_num = 14
batch_size = 10
output_num = 7

val_size = 1000

random.seed(24)


def collate_data(array):
    data = [[], []]
    for x, y in array:
        data[0] += [x]
        data[1] += [y]
    return data


if mode == "train":
    training_data = np.load("/home/soundr-share/train_set3.npy", allow_pickle=True)
    data_X = training_data[0]
    data_y = training_data[1]

    total_size = int(data_X.shape[0])
    new_order = list(range(total_size))
    random.shuffle(new_order)
    new_order = np.array(new_order)
    train_X = data_X[new_order[range(0, total_size - val_size)]]
    train_y = data_y[new_order[range(0, total_size - val_size)]]

    val_X = data_X[new_order[range(total_size - val_size, total_size)]]
    val_y = data_y[new_order[range(total_size - val_size, total_size)]]

    cat_train_y = np.concatenate(train_y)
    plt.clf()
    sns.scatterplot(cat_train_y[:, 0], cat_train_y[:, 2], linewidth=0)
    # plt.show()

    cat_val_y = np.concatenate(val_y)
    plt.clf()
    sns.scatterplot(cat_val_y[:, 0], cat_val_y[:, 2], linewidth=0)
    # plt.show()

    train_loader = DataLoader(
        AudioDataset(train_X, train_y), batch_size=batch_size, shuffle=True, collate_fn=collate_data)
    val_loader = DataLoader(
        AudioDataset(val_X, val_y), batch_size=batch_size, collate_fn=collate_data)

    model = AudioNet(sample_num, microphone_num, output_num)
    trainer = AudioTrainer(model, train_loader, val_loader)
    trainer.train()
    trainer.close("./train.json")

elif mode == "test":
    raise NotImplemented
