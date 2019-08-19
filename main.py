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

from params import *

random.seed(24)


def collate_data(array):
    data = [[], []]
    for x, y in array:
        data[0] += [x]
        data[1] += [y]
    return data


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "train":
        # training_data = np.load("/home/soundr-share/train_set3.1.npy", allow_pickle=True)
        # data_X = training_data[0]
        # data_y = training_data[1]

        data_X = np.load("/home/soundr-share/train_set13_input.npy", allow_pickle=True)
        data_y = np.load("/home/soundr-share/train_set13_output.npy", allow_pickle=True)

        total_size = int(data_X.shape[0])
        print(total_size)
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
        training_data = np.load("/home/soundr-share/train_set3.1.npy", allow_pickle=True)
        data_X = training_data[0]
        data_y = training_data[1]
        with open("/home/soundr_dl-share/checkpoints/20190626T164421/modelTrained_560000_0.8386740684509277.pickle", "rb") as model_file:
            checkpoint = torch.load(model_file)

        model = AudioNet(sample_num, microphone_num, output_num)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_data = torch.Tensor(data_X[0:1]).to(device)
        output = model(test_data)
        prediction = output
        print(prediction)
