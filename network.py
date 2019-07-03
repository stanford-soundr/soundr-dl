import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioNet(nn.Module):
    def __init__(self, sample_num, microphone_num, output_num):
        super(AudioNet, self).__init__()
        self.n = sample_num
        self.kernel1_size = 7
        self.kernel2_size = 5
        self.kernel3_size = 3

        #layer 1: mic_num inputs and 128 outputs
        #BatchNorm: Normalize the activations of the previous layer at each batch, i.e. applies a transformation
        # that maintains the mean activation close to 0 and the activation standard deviation close to 1.
        #MaxPool: summarizes convolutional layer, keeps num of params low
        self.conv1 = nn.Conv1d(in_channels=microphone_num, out_channels=128, kernel_size=self.kernel1_size).to(device)
        self.conv1_bn = nn.BatchNorm1d(128).to(device)
        self.maxpool1 = nn.MaxPool1d(kernel_size=self.kernel1_size).to(device)
        self.n1 = int((self.n - self.kernel1_size + 1) / self.kernel1_size)

        #layer 2 only has a convolution operation; input: 128 and output: 128
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=self.kernel1_size).to(device)
        self.n2 = self.n1 - self.kernel1_size + 1

        #layer 3 convolves, normalizes, and maximizes; input: 128 and output: 192
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=192, kernel_size=self.kernel2_size).to(device)
        self.conv3_bn = nn.BatchNorm1d(192).to(device)
        self.maxpool3 = nn.MaxPool1d(kernel_size=self.kernel2_size).to(device)
        self.n3 = int((self.n2 - self.kernel2_size + 1) / self.kernel2_size)

        #layer 4 is similar to layer 3; input: 192 and output: 192
        self.conv4 = nn.Conv1d(in_channels=192, out_channels=192, kernel_size=self.kernel2_size).to(device)
        self.conv4_bn = nn.BatchNorm1d(192).to(device)
        self.maxpool4 = nn.MaxPool1d(kernel_size=self.kernel2_size).to(device)
        self.n4 = int((self.n3 - self.kernel2_size + 1) / self.kernel2_size)

        #layer 5 has only a convolutional layer with input: 192 and output: 192
        self.conv5 = nn.Conv1d(in_channels=192, out_channels=192, kernel_size=self.kernel3_size).to(device)
        self.n5 = self.n4 - self.kernel3_size + 1

        self.conv5_1 = nn.Conv1d(in_channels=192, out_channels=192, kernel_size=self.kernel3_size).to(device)
        self.n5_1 = self.n5 - self.kernel3_size + 1

        # layer 7 produces final output; dropout rate set to 1/2
        self.mlp6 = nn.Linear(in_features=self.n5_1 * 192, out_features=700).to(device)
        self.mlp6_bn = nn.BatchNorm1d(700).to(device)

        # add LSTM here
        self.lstm = nn.LSTM(input_size=700,hidden_size=700,num_layers=2).to(device)

        self.mlp7 = nn.Linear(in_features=700, out_features=output_num).to(device)
        self.dropout = nn.Dropout(p=0.5).to(device) # the dropout module will be automatically turned off in evaluation mode

    #the forward pass through the network built above
    def forward(self, input, hidden):
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
        x = F.relu(self.conv5_1(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.mlp6(x)))
        x = self.mlp6_bn(x)

        x, hidden = self.lstm(x,hidden)

        x = self.mlp7(x)
        pos = x[:, 0:3]
        pre_quat = x[:, 3:7]
        quat = F.tanh(pre_quat)
        x = torch.cat((pos, quat), 1)
        return x
