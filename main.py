import pickle
import sys
import random

import numpy as np
import torch.utils.data

from dataset import AudioDataset
from network import AudioNet
from trainer import AudioTrainer

import seaborn as sns
import matplotlib.pyplot as plt

mode = sys.argv[1]

sample_num = 2560
microphone_num = 14
batch_size = 100
output_num = 7

val_size = 1000

random.seed(24)

if mode == "train":
    with open("/home/soundr-share/train_set2.pickle", "rb") as train_set_file:
        data_X, data_y = pickle.load(train_set_file)

    data_X = np.transpose(data_X, (0, 2, 1))
    # data_X = np.delete(data_X, [7, 8, 9, 10, 11, 12, 13], 1)
    # data_y = np.delete(data_y, [3, 4, 5, 6], 1)
    # data_y = np.concatenate((data_y[:, 0:3], data_y[:, 4:7], data_y[:, 3:4]), axis=1)
    total_size = int(data_X.shape[0] / 1.8)
    new_order = list(range(total_size))
    random.shuffle(new_order)
    new_order = np.array(new_order)
    train_X = torch.Tensor(data_X[new_order[range(0, total_size - val_size)]])
    train_y = torch.Tensor(data_y[new_order[range(0, total_size - val_size)]])

    val_X = torch.Tensor(data_X[new_order[range(total_size - val_size, total_size)]])
    val_y = torch.Tensor(data_y[new_order[range(total_size - val_size, total_size)]])

    plt.clf()
    sns.scatterplot(train_y[:, 0], train_y[:, 2], linewidth=0)
    plt.show()

    plt.clf()
    sns.scatterplot(val_y[:, 0], val_y[:, 2], linewidth=0)
    plt.show()

    train_loader = torch.utils.data.DataLoader(AudioDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(AudioDataset(val_X, val_y), batch_size=batch_size)

    model = AudioNet(sample_num, microphone_num, output_num)
    trainer = AudioTrainer(model, train_loader, val_loader)
    trainer.train()
    trainer.close("./train.json")

elif mode == "test":
    with open("/home/soundr-share/train_set2.pickle", "rb") as train_set_file:
        data_X, data_y = pickle.load(train_set_file)
    with open("/home/soundr-share/checkpoints/20190626T164421/modelTrained_560000_0.8386740684509277.pickle", "rb") as model_file:
        checkpoint = torch.load(model_file)

    data_X = np.transpose(data_X, (0, 2, 1))
    model = AudioNet(sample_num, microphone_num, output_num)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = torch.Tensor(data_X[0:1]).to(device)
    output = model(test_data)
    prediction = output
    print(prediction)





