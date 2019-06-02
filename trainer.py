import torch
import torch.nn as nn
import tensorflow as tf

class AudioTrainer:
    def __init__(self, model, dataloader, max_step = 10000):
        self.model = model
        self.dataloader = dataloader
        self.max_step = max_step
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)

    def train(self):
        self.model.train()
        for input, reference in self.dataloader:
            output = self.model(input)
            loss = self.criterion(output, reference)
            tf.summary.scalar('train_loss', loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()