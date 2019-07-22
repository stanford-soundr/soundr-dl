import torch.utils.data
import torch


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        # X: L x S x N x C
        # y: L x S x 3
        # (L: total sequence count, S: sequence length, N: sample number, C: microphone number)
        assert X.shape[0] == y.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = X
        self.y = y

    def __getitem__(self, index):
        return torch.Tensor(self.X[index][:, 0:7]).to(self.device), torch.Tensor(self.y[index]).to(self.device)

    def __len__(self):
        return self.y.shape[0]
