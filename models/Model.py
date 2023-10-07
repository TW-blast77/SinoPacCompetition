import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(20, 916)
        self.linear2 = nn.Linear(916, 8192)
        self.dropout1 = nn.Dropout(0.2)
        self.linear3 = nn.Linear(8192, 916)
        self.dropout2 = nn.Dropout(0.3)
        self.linear4 = nn.Linear(916, 81)
        self.dropout3 = nn.Dropout(0.4)
        self.linear5 = nn.Linear(81, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.dropout1(x)
        x = self.linear3(x)
        x = self.dropout2(x)
        x = self.linear4(x)
        x = self.dropout3(x)
        output = self.linear5(x)
        return output