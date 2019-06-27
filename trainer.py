import datetime as dt
import os

import math
import matplotlib.pyplot as plt
import numpy as np
import quaternion
import seaborn as sns
import torch
import torch.nn as nn
import torchgeometry.core
from tensorboardX import SummaryWriter

eps = 1e-8


def quat_to_rot_mat(quat):
    return torchgeometry.angle_axis_to_rotation_matrix(torchgeometry.quaternion_to_angle_axis(quat))


def quat_to_angle(quat):
    angle = quat.normalized().angle()
    if angle > math.pi:
        return 2 * math.pi - angle
    else:
        return angle


class AudioTrainer:
    def __init__(self, model, train_loader, val_loader, max_step=10000000):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_step = max_step
        # self.criterion = nn.MSELoss()
        self.pos_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-2)
        self.writer = SummaryWriter()
        self.n_iter = 0
        self.min_avg_loss = math.inf

    #loss calculation function
    def criterion(self, output, reference):
        output_pos = output[:, 0:3]
        reference_pos = reference[:, 0:3]
        pos_loss = self.pos_criterion(output_pos, reference_pos)
        output_quat = output[:, 3:7]
        reference_quat = reference[:, 3:7]
        output_rot_mat = quat_to_rot_mat(output_quat)
        reference_rot_mat = quat_to_rot_mat(reference_quat)
        output_rot_mat_t = torch.transpose(output_rot_mat, 1, 2)
        rot_mat_prod = torch.matmul(output_rot_mat_t, reference_rot_mat)
        trace = torch.sum(torch.diagonal(rot_mat_prod, 0, 1, 2), dim=1)
        # quat_loss = torch.mean(torch.abs(torch.acos(torch.clamp((trace - 2) / 2, min=-(1-eps), max=1-eps))))
        quat_loss = torch.mean(torch.abs(torch.acos((trace - 2) / 2.00001)))
        # (quat_loss + pos_loss).backward()
        return pos_loss, quat_loss

    def train(self):
        # torch.autograd.set_detect_anomaly(True)
        output_parent_dir = "/home/soundr-share/checkpoints"
        current_datetime = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        output_dir = os.path.join(output_parent_dir, current_datetime)
        os.mkdir(output_dir)
        while self.n_iter < self.max_step or True:
            for input, reference in self.train_loader:
                if len(input) <= 1:
                    continue
                self.model.train()
                output = self.model(input)
                pos_loss, quat_loss = self.criterion(output, reference)
                loss = pos_loss + quat_loss
                self.writer.add_scalar('train/pos_loss', pos_loss, self.n_iter)
                self.writer.add_scalar('train/quat_loss', quat_loss, self.n_iter)

                train_error = reference[:, 0:3] - output[:, 0:3]
                train_error_dist = torch.sqrt(torch.sum(train_error ** 2, dim=1))
                avg_train_error_dist = torch.mean(train_error_dist)
                self.writer.add_scalar('train/avg_dist', avg_train_error_dist, self.n_iter)

                train_quat = output[:, 3:7].cpu().detach().numpy()
                reference_quat = quaternion.as_quat_array(reference[:, 3:7].cpu().detach().numpy())
                train_quat[:, 0] = -train_quat[:, 0]
                train_quat = quaternion.as_quat_array(train_quat)
                train_quat_error = train_quat * reference_quat
                # train_quat_error_roll = quaternion.as_quat_array(np.roll(quaternion.as_float_array(train_quat_error), 3, 1))
                # train_quat_vector = quaternion.as_rotation_vector(train_quat_error)
                # train_quat_error = np.sqrt(np.sum(train_quat_vector ** 2, axis=1))
                train_quat_diff = np.array(list(map(quat_to_angle, train_quat_error)))
                avg_train_quat_error = np.average(train_quat_diff)
                self.writer.add_scalar('train/avg_angle', avg_train_quat_error, self.n_iter)

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
                    result_quat = None
                    pos_loss = None
                    quat_loss = None
                    result_y = None
                    for val_input, val_reference in self.train_loader:
                        val_output = self.model(val_input)
                        val_loss_pos, val_loss_quat = self.criterion(val_output, val_reference)
                        if result_y is None:
                            result_y = val_output.cpu().detach().numpy()
                        else:
                            result_y = np.concatenate((result_y, val_output.cpu().detach().numpy()), axis=0)
                        pos_loss_np = val_loss_pos.cpu().detach().numpy()
                        quat_loss_np = val_loss_quat.cpu().detach().numpy()
                        if pos_loss is None:
                            pos_loss = np.array([pos_loss_np])
                        else:
                            pos_loss = np.concatenate((pos_loss, [pos_loss_np]), axis=0)
                        if quat_loss is None:
                            quat_loss = np.array([quat_loss_np])
                        else:
                            quat_loss = np.concatenate((quat_loss, [quat_loss_np]), axis=0)

                        error = (val_reference[:, 0:3] - val_output[:, 0:3])
                        error_dist = torch.sqrt(torch.sum(error ** 2, dim=1))
                        error_dist_np = error_dist.cpu().detach().numpy()
                        if result is None:
                            result = error_dist_np
                        else:
                            result = np.concatenate((result, error_dist_np), axis=0)

                        val_quat = val_output[:, 3:7].cpu().detach().numpy()
                        val_reference_quat = quaternion.as_quat_array(val_reference[:, 3:7].cpu().detach().numpy())
                        val_quat[:, 0] = - val_quat[:, 0]
                        val_quat = quaternion.as_quat_array(val_quat)
                        val_quat_error = val_quat * val_reference_quat
                        # val_quat_error_roll = quaternion.as_quat_array(np.roll(quaternion.as_float_array(val_quat_error), 3, 1))
                        # val_quat_vector = quaternion.as_rotation_vector(val_quat_error)
                        # val_quat_error = np.sqrt(np.sum(val_quat_vector ** 2, axis=1))
                        val_quat_diff = np.array(list(map(quat_to_angle, val_quat_error)))
                        if result_quat is None:
                            result_quat = val_quat_diff
                        else:
                            result_quat = np.concatenate((result_quat, val_quat_diff), axis=0)

                    avg_dist = np.average(result)
                    self.writer.add_scalar('val/avg_dist', avg_dist, self.n_iter)
                    avg_quat = np.average(result_quat)
                    self.writer.add_scalar('val/avg_angle', avg_quat, self.n_iter)
                    avg_loss_pos = np.average(pos_loss)
                    self.writer.add_scalar('val/pos_loss', avg_loss_pos, self.n_iter)
                    avg_loss_quat = np.average(quat_loss)
                    self.writer.add_scalar('val/quat_loss', avg_loss_quat, self.n_iter)
                    plt.clf()
                    sns.scatterplot(result_y[:, 0], result_y[:, 2], linewidth=0)
                    self.writer.add_figure('val/output', plt.gcf(), self.n_iter)
                    plt.close()

                    avg_loss = avg_loss_quat + avg_loss_pos
                    if self.n_iter % 10000 == 0 or avg_loss < self.min_avg_loss:
                        self.min_avg_loss = avg_loss
                        filename = f"modelTrained_{self.n_iter}_{loss.cpu().detach().numpy()}.pickle"
                        print(f"Creating checkpoint: {filename}")
                        filepath = os.path.join(output_dir, filename)
                        torch.save({'epoch': self.n_iter,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'loss': loss}, filepath)

    def close(self, path):
        self.writer.export_scalars_to_json(path)
        self.writer.close()
