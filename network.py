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

        self.conv1 = nn.Conv1d(in_channels=microphone_num, out_channels=96, kernel_size=self.kernel1_size)
        self.maxpool1 = nn.MaxPool1d(kernel_size=self.kernel1_size)
        self.n1 = int((self.n - 1) / self.kernel1_size)

        self.conv2 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=self.kernel1_size)
        self.n2 = self.n1 - self.kernel1_size + 1

        self.conv3 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=self.kernel2_size)
        self.maxpool3 = nn.MaxPool1d(kernel_size=self.kernel2_size)
        self.n3 = int((self.n2 - 1) / self.kernel2_size)

        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=self.kernel2_size)
        self.maxpool4 = nn.MaxPool1d(kernel_size=self.kernel2_size)
        self.n4 = int((self.n3 - 1) / self.kernel2_size)

        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=self.kernel3_size)
        self.n5 = self.n4 - self.kernel3_size + 1

        self.mlp6 = nn.Linear(in_features=self.n5, out_features=500)
        self.mlp7 = nn.Linear(in_features=500, out_features=3)
        self.dropout = nn.Dropout(p=0.5) # the dropout module will be automatically turned off in evaluation mode

    def forward(self, input):
        conv1_output = self.conv1(input)
        conv2_output = self.conv2(conv1_output)
        conv3_output = self.conv3(conv2_output)
        conv4_output = self.conv4(conv3_output)
        conv5_output = self.conv5(conv4_output)
        flatten_conv5_output = conv5_output.view(-1)
        mlp6_output = self.dropout(F.relu(self.mlp6(flatten_conv5_output)))
        mlp7_output = self.mlp7(mlp6_output)
        return mlp7_output
