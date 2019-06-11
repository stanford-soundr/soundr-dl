import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import PIL.Image
import io
from torchvision.transforms import ToTensor

class AudioTrainer:
    def __init__(self, model, train_loader, val_loader, max_step=10000000):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_step = max_step
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
        self.writer = SummaryWriter()
        self.n_iter = 0

    def train(self):
        while self.n_iter < self.max_step or True:
            for input, reference in self.train_loader:
                if len(input) <= 1:
                    continue
                self.model.train()
                output = self.model(input)
                loss = self.criterion(output, reference)
                self.writer.add_scalar('train/loss', loss, self.n_iter)

                train_error = reference - output
                train_error_dist = torch.sqrt(torch.sum(train_error ** 2, dim=1))
                avg_train_error_dist = torch.mean(train_error_dist)
                self.writer.add_scalar('train/avg_dist', avg_train_error_dist, self.n_iter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.n_iter += 1
                print(self.n_iter)

                if self.n_iter % 100 == 0:
                    plt.clf()
                    output_np = output.cpu().detach().numpy()
                    sns.scatterplot(output_np[:, 0], output_np[:, 2], linewidth=0)
                    self.writer.add_figure('train/output', plt.gcf(), self.n_iter)
                    plt.close()
                    self.model.eval()
                    result = None
                    loss = None
                    result_y = None
                    for val_input, val_reference in self.train_loader:
                        val_output = self.model(val_input)
                        val_loss = self.criterion(val_output, val_reference)
                        if result_y is None:
                            result_y = val_output.cpu().detach().numpy()
                        else:
                            result_y = np.concatenate((result_y, val_output.cpu().detach().numpy()), axis=0)
                        val_loss_np = val_loss.cpu().detach().numpy()
                        if loss is None:
                            loss = val_loss_np
                        else:
                            loss += val_loss_np
                        error = (val_reference - val_output)
                        error_dist = torch.sqrt(torch.sum(error ** 2, dim=1))
                        error_dist_np = error_dist.cpu().detach().numpy()
                        if result is None:
                            result = error_dist_np
                        else:
                            result = np.concatenate((result, error_dist_np), axis=0)
                    avg_dist = np.average(result)
                    self.writer.add_scalar('val/avg_dist', avg_dist, self.n_iter)
                    avg_loss = np.average(loss)
                    self.writer.add_scalar('val/loss', avg_loss, self.n_iter)
                    plt.clf()
                    sns.scatterplot(result_y[:, 0], result_y[:, 2], linewidth=0)
                    self.writer.add_figure('val/output', plt.gcf(), self.n_iter)
                    plt.close()

    def close(self, path):
        self.writer.export_scalars_to_json(path)
        self.writer.close()
