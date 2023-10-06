import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(2, 1)
        # self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.b = nn.Parameter(torch.randn(1, requires_grad=True))
        # self.c = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x):
        output = self.linear(x)
        return output