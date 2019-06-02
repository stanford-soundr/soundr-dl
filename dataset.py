import torch.utils.data.Dataset as Dataset
import torch

class AudioDataset(Dataset):
    def __init__(self, X, y):
        # X: L x N x C
        # y: L x 3
        # (L: total data point count, N: sample number, C: microphone number)
        assert X.size(0) == y.size(0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X = X.to(device)
        self.y = y.to(device)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

    def __len__(self):
        return self.y.size(0)
