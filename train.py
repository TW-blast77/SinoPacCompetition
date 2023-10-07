from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from models.Model import Net
from sinodataset import SinoDataset, SinoTestDataset
from torch.utils.data.dataloader import DataLoader

def main():
    # 
    # 初始化模型、優化器、損失函數 取得 cuda device, 設定種子以重現結果
    #
    torch.manual_seed(1)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Net().to(device)
    epochs = 10
    lossfunction = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-6, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    #
    # 初始化 DataLoaders, Datasets
    #
    train_data = SinoDataset('data/training_revised.csv')
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=False, num_workers=4)
    test_data = SinoTestDataset('data/public_revised.csv')
    test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=4)

    #
    # 開始訓練
    #
    for epoch in range(epochs):
        iter_count, epoch_loss = 0, 0
        for data, lable in train_loader:
            model.train()
            data = data.to(device)
            lable = lable.to(device)
            yhat = model(data)
            loss = lossfunction(lable, yhat)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            iter_count+=1
            epoch_loss+=loss.item()
            print(f"iter={iter_count}, iter_loss={loss.item()}")
        print(f"Epochs={epoch}, Epoch_loss={epoch_loss/iter_count}")
    
    #
    # 測試
    #
    # for data, lable in test_loader:
    #     model.train()
    #     data = data.to(device)
    #     yhat = model(data)
    #     print(yhat)

if __name__ == '__main__':
    main()