import torch
from torch import nn


class NN(nn.Module):

    def __init__(self, d_in, d_out):
        super(NN, self).__init__()
        self.inp = nn.Linear(d_in, 64)
        self.int1 = nn.Linear(64, 128)
        self.int2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, d_out)
        self.softm = nn.LogSoftmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, ob):
        out = self.relu(self.inp(ob))
        out = self.relu(self.int1(out))
        out = self.out(out)
        return self.softm(out)





