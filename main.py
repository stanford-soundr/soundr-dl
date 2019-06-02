import pickle
import sys
import random

import numpy as np
import torch.utils.data

from dataset import AudioDataset
from network import AudioNet
from trainer import AudioTrainer

mode = sys.argv[1]

sample_num = 1280
microphone_num = 7
batch_size = 100

val_size = 1000

random.seed(42)

if mode == "train":
    with open("/home/jackie/Downloads/train_set.pickle", "rb") as train_set_file:
        data_X, data_y = pickle.load(train_set_file)

    data_X = np.transpose(data_X, (0, 2, 1))
    data_X = np.delete(data_X, [7, 8, 9, 10, 11, 12, 13], 1)
    data_y = np.delete(data_y, [3, 4, 5, 6], 1)
    total_size = data_X.shape[0]
    new_order = list(range(total_size))
    random.shuffle(new_order)
    new_order = np.array(new_order)
    train_X = torch.Tensor(data_X[new_order[range(0, total_size - val_size)]])
    train_y = torch.Tensor(data_y[new_order[range(0, total_size - val_size)]])

    val_X = torch.Tensor(data_X[new_order[range(total_size - val_size, total_size)]])
    val_y = torch.Tensor(data_y[new_order[range(total_size - val_size, total_size)]])

    train_loader = torch.utils.data.DataLoader(AudioDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(AudioDataset(val_X, val_y), batch_size=batch_size)

    model = AudioNet(sample_num, microphone_num)
    trainer = AudioTrainer(model, train_loader, val_loader)
    trainer.train()
    trainer.close("./train.json")

elif mode == "test":
    raise NotImplemented