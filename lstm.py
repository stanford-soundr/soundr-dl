import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
