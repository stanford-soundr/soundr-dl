import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioNet(nn.Module):
    def __init__(self, sample_num, microphone_num):
        super(AudioNet, self).__init__()
        self.n = sample_num
        self.kernel1_size = 7
        self.kernel2_size = 5
        self.kernel3_size = 3

        self.conv1 = nn.Conv1d(in_channels=microphone_num, out_channels=96, kernel_size=self.kernel1_size).to(device)
        self.conv1_bn = nn.BatchNorm1d(96).to(device)
        self.maxpool1 = nn.MaxPool1d(kernel_size=self.kernel1_size).to(device)
        self.n1 = int((self.n - self.kernel1_size + 1) / self.kernel1_size)

        # self.conv2 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=self.kernel1_size).to(device)
        # self.n2 = self.n1 - self.kernel1_size + 1

        self.conv3 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=self.kernel2_size).to(device)
        self.conv3_bn = nn.BatchNorm1d(128).to(device)
        self.maxpool3 = nn.MaxPool1d(kernel_size=self.kernel2_size).to(device)
        self.n3 = int((self.n1 - self.kernel2_size + 1) / self.kernel2_size)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=self.kernel2_size).to(device)
        self.conv4_bn = nn.BatchNorm1d(128).to(device)
        self.maxpool4 = nn.MaxPool1d(kernel_size=self.kernel2_size).to(device)
        self.n4 = int((self.n3 - self.kernel2_size + 1) / self.kernel2_size)

        # self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=self.kernel3_size).to(device)
        # self.n5 = self.n4 - self.kernel3_size + 1

        self.mlp6 = nn.Linear(in_features=self.n4 * 128, out_features=500).to(device)
        self.mlp6_bn = nn.BatchNorm1d(500).to(device)
        self.mlp7 = nn.Linear(in_features=500, out_features=3).to(device)
        self.dropout = nn.Dropout(p=0.5).to(device) # the dropout module will be automatically turned off in evaluation mode

    def forward(self, input):
        conv1_output = F.relu(self.conv1(input))
        conv1_bn_output = self.conv1_bn(conv1_output)
        maxpool1_output = self.maxpool1(conv1_bn_output)
        # conv2_output = self.conv2(maxpool1_output)
        conv3_output = F.relu(self.conv3(maxpool1_output))
        conv3_bn_output = self.conv3_bn(conv3_output)
        maxpool3_output = self.maxpool3(conv3_bn_output)
        conv4_output = F.relu(self.conv4(maxpool3_output))
        conv4_bn_output = self.conv4_bn(conv4_output)
        maxpool4_output = self.maxpool4(conv4_bn_output)
        # conv5_output = self.conv5(maxpool4_output)
        flatten_conv5_output = maxpool4_output.view(maxpool4_output.size(0), -1)
        mlp6_output = self.dropout(F.relu(self.mlp6(flatten_conv5_output)))
        mlp6_bn_output = self.mlp6_bn(mlp6_output)
        mlp7_output = self.mlp7(mlp6_bn_output)
        return mlp7_output
