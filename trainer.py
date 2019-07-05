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

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


class AudioTrainer:
    def __init__(self, model, train_loader, val_loader, max_step=10000000):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_step = max_step
        # self.criterion = nn.MSELoss()
        self.pos_criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-2)
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
            for test in self.train_loader:
                input, reference = test
                if len(input) <= 1:
                    continue
                self.model.train()
                input = torch.nn.utils.rnn.pack_sequence(input, enforce_sorted=False)
                reference = torch.nn.utils.rnn.pack_sequence(reference, enforce_sorted=False)
                output = self.model(input)
                output = output.data
                reference = reference.data
                pos_loss, quat_loss = self.criterion(output, reference)
                loss = quat_loss + pos_loss
                self.writer.add_scalar('train/pos_loss', pos_loss, self.n_iter)
                self.writer.add_scalar('train/quat_loss', quat_loss, self.n_iter)

                train_error = reference[:, 0:3] - output[:, 0:3]
                train_error_dist = torch.sqrt(torch.sum(train_error ** 2, dim=1))
                avg_train_error_dist = torch.mean(train_error_dist)
                self.writer.add_scalar('train/avg_dist', avg_train_error_dist, self.n_iter)

                train_quat = quaternion.as_quat_array(output[:, 3:7].cpu().detach().numpy())
                reference_quat = quaternion.as_quat_array(reference[:, 3:7].cpu().detach().numpy())
                train_quat = np.invert(train_quat)
                train_quat_error = train_quat * reference_quat
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
                    reference_np = reference.cpu().detach().numpy()
                    sns.kdeplot(reference_np[:, 0] - output_np[:, 0],
                                reference_np[:, 2] - output_np[:, 2],
                                shade=True)
                    plt.xlim(-1.5, 1.5)
                    plt.ylim(-1.5, 1.5)
                    self.writer.add_figure('train/pos_diff', plt.gcf(), self.n_iter)
                    plt.close()
                    train_quat_vec = quaternion.rotate_vectors(train_quat_error, [0, 0, 1])
                    plt.clf()
                    # sns.scatterplot(train_quat_vec[:, 0], train_quat_vec[:, 2], linewidth=0)
                    ax = plt.subplot(111, polar=True)
                    rho, phi = cart2pol(train_quat_vec[:, 2], train_quat_vec[:, 0])
                    ax.scatter(x=phi, y=rho)
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    plt.ylim(0, 1)
                    self.writer.add_figure('train/quat_diff', plt.gcf(), self.n_iter)
                    plt.close()
                    self.model.eval()
                    result = None
                    result_quat = None
                    result_quat_vec = None
                    pos_loss = None
                    quat_loss = None
                    result_y = None
                    reference_y = None
                    for val_input, val_reference in self.train_loader:
                        val_input = torch.nn.utils.rnn.pack_sequence(val_input, enforce_sorted=False)
                        val_reference = torch.nn.utils.rnn.pack_sequence(val_reference, enforce_sorted=False)
                        val_output = self.model(val_input)
                        val_output = val_output.data
                        val_reference = val_reference.data
                        val_loss_pos, val_loss_quat = self.criterion(val_output, val_reference)
                        if result_y is None:
                            result_y = val_output.cpu().detach().numpy()
                        else:
                            result_y = np.concatenate((result_y, val_output.cpu().detach().numpy()), axis=0)

                        if reference_y is None:
                            reference_y = val_reference.cpu().detach().numpy()
                        else:
                            reference_y = np.concatenate((reference_y, val_reference.cpu().detach().numpy()), axis=0)

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

                        val_quat = quaternion.as_quat_array(val_output[:, 3:7].cpu().detach().numpy())
                        val_reference_quat = quaternion.as_quat_array(val_reference[:, 3:7].cpu().detach().numpy())
                        val_quat = np.invert(val_quat)
                        val_quat_error = val_quat * val_reference_quat
                        val_quat_diff = np.array(list(map(quat_to_angle, val_quat_error)))
                        if result_quat is None:
                            result_quat = val_quat_diff
                        else:
                            result_quat = np.concatenate((result_quat, val_quat_diff), axis=0)

                        val_quat_vec = quaternion.rotate_vectors(val_quat_error, [0, 0, 1])
                        if result_quat_vec is None:
                            result_quat_vec = val_quat_vec
                        else:
                            result_quat_vec = np.concatenate((result_quat_vec, val_quat_vec), axis=0)

                    avg_dist = np.average(result)
                    self.writer.add_scalar('val/avg_dist', avg_dist, self.n_iter)
                    avg_quat = np.average(result_quat)
                    self.writer.add_scalar('val/avg_angle', avg_quat, self.n_iter)
                    avg_loss_pos = np.average(pos_loss)
                    self.writer.add_scalar('val/pos_loss', avg_loss_pos, self.n_iter)
                    avg_loss_quat = np.average(quat_loss)
                    self.writer.add_scalar('val/quat_loss', avg_loss_quat, self.n_iter)
                    plt.clf()
                    sns.kdeplot(reference_y[:, 0] - result_y[:, 0], reference_y[:, 2] - result_y[:, 2], shade=True)
                    plt.xlim(-1.5, 1.5)
                    plt.ylim(-1.5, 1.5)
                    self.writer.add_figure('val/pos_diff', plt.gcf(), self.n_iter)
                    plt.close()
                    plt.clf()
                    # sns.scatterplot(result_quat_vec[:, 0], result_quat_vec[:, 2], linewidth=0)
                    ax = plt.subplot(111, polar=True)
                    rho, phi = cart2pol(result_quat_vec[:, 2], result_quat_vec[:, 0])
                    ax.scatter(x=phi, y=rho)
                    ax.set_theta_zero_location('N')
                    ax.set_theta_direction(-1)
                    plt.ylim(0, 1)
                    self.writer.add_figure('val/quat_diff', plt.gcf(), self.n_iter)
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
