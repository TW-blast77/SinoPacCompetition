import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(20, 916)
        self.leaky_relu1 = nn.LeakyReLU()
        self.linear2 = nn.Linear(916, 8192)
        self.leaky_relu2 = nn.LeakyReLU()
        self.linear3 = nn.Linear(8192, 916)
        self.leaky_relu3 = nn.LeakyReLU()
        self.linear4 = nn.Linear(916, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.leaky_relu1(x)
        x = self.linear2(x)
        x = self.leaky_relu2(x)
        x = self.linear3(x)
        x = self.leaky_relu3(x)
        output = self.linear4(x)
        return output