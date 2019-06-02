import torch
import torch.utils.data.DataLoader as DataLoader
from network import AudioNet
from dataset import AudioDataset
from trainer import AudioTrainer
import sys
import tensorflow as tf

mode = sys.argv[1]

# TODO: modify these parameters
sample_num = 540
microphone_num = 10

if mode == "train":
    # TODO: load X and y
    train_X = None
    train_y = None

    train_loader = DataLoader(AudioDataset(train_X, train_y), batch_size=100, shuffle=True)
    model = AudioDataset(sample_num, microphone_num)
    trainer = AudioTrainer(model, train_loader)
    trainer.train()

elif mode == "test":
    raise NotImplemented