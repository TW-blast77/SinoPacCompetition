from __future__ import print_function
import torch, csv
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from models.Model import Net

def main():
    #
    # Setup
    #
    torch.manual_seed(1)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    #
    # Initial
    #
    model = Net()#.to(device)
    lr = 5e-5
    epochs = 1500
    MSELoss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    #
    # Load data
    #
    data = []
    with open('processed.csv', newline='') as csvfile:
        # 讀取 CSV 檔案內容
        rows = csv.reader(csvfile)
        for row in rows:
            data.append(
                list(map(float, row))
            )

    #
    # Train
    #
    for epoch in range(epochs):
        for row in data:
            model.train()

            x = torch.from_numpy(np.array(row[:2])).float()
            y = torch.from_numpy(np.array(row[2])).float()

            # 不用再手動做 output 了
            # yhat = a + b * x_tensor
            yhat = model(x)
            loss = MSELoss(
                torch.reshape(y, [1]),
                yhat
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"x={row[0]}, y={row[1]}, z={row[2]}, pred={yhat}")
            print(f"loss={loss.item()}")


if __name__ == '__main__':
    main()