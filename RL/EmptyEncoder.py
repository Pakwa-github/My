import torch.nn as nn
import torch

class EmptyEncoder(nn.Module):
    def __init__(self, output_dim=1024):
        super(EmptyEncoder, self).__init__()
        self.output_dim = output_dim

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        fc = nn.Linear(x.shape[1], self.output_dim).to(x.device)
        return fc(x)