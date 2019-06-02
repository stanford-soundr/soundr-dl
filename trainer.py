import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
import numpy as np

class AudioTrainer:
    def __init__(self, model, train_loader, val_loader, max_step = 10000000):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_step = max_step
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-6)
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

                train_error = reference - loss
                train_error_dist = torch.sqrt(torch.sum(train_error ** 2, dim=1))
                avg_train_error_dist = torch.mean(train_error_dist)
                self.writer.add_scalar('train/avg_dist', avg_train_error_dist, self.n_iter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.n_iter += 1
                print(self.n_iter)

                if self.n_iter % 100 == 0:
                    self.model.eval()
                    result = None
                    loss = None
                    for val_input, val_reference in self.train_loader:
                        val_output = self.model(val_input)
                        val_loss = self.criterion(val_output, val_reference)
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
                            result += error_dist_np
                    avg_dist = np.average(result)
                    self.writer.add_scalar('val/avg_dist', avg_dist, self.n_iter)
                    avg_loss = np.average(loss)
                    self.writer.add_scalar('val/loss', avg_loss, self.n_iter)



    def close(self, path):
        self.writer.export_scalars_to_json(path)
        self.writer.close()
