import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(46, 512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, 8192)
        self.relu4 = nn.ReLU()
        self.linear5 = nn.Linear(8192, 4096)
        self.relu6 = nn.ReLU()
        self.linear7 = nn.Linear(4096, 2048)
        self.relu8 = nn.ReLU()
        self.linear9 = nn.Linear(2048, 1024)
        self.leaky_relu10 = nn.LeakyReLU()
        self.linear11 = nn.Linear(1024, 512)
        self.sigmoid12 = nn.Sigmoid()
        self.linear13 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu4(x)
        x = self.linear5(x)
        x = self.relu6(x)
        x = self.linear7(x)
        x = self.relu8(x)
        x = self.linear9(x)
        x = self.leaky_relu10(x)
        x = self.linear11(x)
        x = self.sigmoid12(x)
        output = self.linear13(x)
        return output