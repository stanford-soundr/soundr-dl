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

        self.conv2 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=self.kernel1_size).to(device)
        self.n2 = self.n1 - self.kernel1_size + 1

        self.conv3 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=self.kernel2_size).to(device)
        self.conv3_bn = nn.BatchNorm1d(128).to(device)
        self.maxpool3 = nn.MaxPool1d(kernel_size=self.kernel2_size).to(device)
        self.n3 = int((self.n1 - self.kernel2_size + 1) / self.kernel2_size)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=self.kernel2_size).to(device)
        self.conv4_bn = nn.BatchNorm1d(128).to(device)
        self.maxpool4 = nn.MaxPool1d(kernel_size=self.kernel2_size).to(device)
        self.n4 = int((self.n3 - self.kernel2_size + 1) / self.kernel2_size)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=self.kernel3_size).to(device)
        self.n5 = self.n4 - self.kernel3_size + 1

        self.mlp6 = nn.Linear(in_features=self.n5 * 128, out_features=500).to(device)
        self.mlp6_bn = nn.BatchNorm1d(500).to(device)
        self.mlp7 = nn.Linear(in_features=500, out_features=3).to(device)
        self.dropout = nn.Dropout(p=0.5).to(device) # the dropout module will be automatically turned off in evaluation mode

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.conv1_bn(x)
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv3_bn(x)
        x = self.maxpool3(x)
        x = F.relu(self.conv4(x))
        x = self.conv4_bn(x)
        x = self.maxpool4(x)
        x = F.relu(self.conv5(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.mlp6(x)))
        x = self.mlp6_bn(x)
        x = self.mlp7(x)
        return x
