from __future__ import print_function
import os
import torch
import math
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from models.Model import Net
from sinodataset import SinoDataset
from dotenv import load_dotenv

def mape_loss(output, target):
    return torch.mean(torch.abs((target - output) / target))

def train(model, train_loader, loss_function, optimizer) -> float:
    iter_train_count, epoch_train_loss = 0, 0
    for data, lable in train_loader:
        model.train()
        data = data.to(device).unsqueeze(1)
        lable = lable.to(device).unsqueeze(1)
        yhat = model(data)
        loss = loss_function(lable, yhat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter_train_count+=1
        epoch_train_loss+=loss.item()
    return epoch_train_loss/iter_train_count

def valid(model, valid_loader, loss_function) -> float:
    iter_valid_count, epoch_valid_loss = 0, 0
    for data, lable in valid_loader:
        model.eval()
        data = data.to(device).unsqueeze(1)
        lable = lable.to(device).unsqueeze(1)
        yhat = model(data)
        loss = loss_function(lable, yhat)

        iter_valid_count+=1
        epoch_valid_loss+=loss.item()
    return epoch_valid_loss/iter_valid_count


if __name__ == '__main__':
    load_dotenv()
    TORCH_SEED              = int(os.getenv("TORCH_SEED", 1))
    EPOCHS                  = int(os.getenv("EPOCHS", 10))
    LEARNING_RATE           = float(os.getenv("LEARNING_RATE", 1e-5))
    TRAIN_DATASET_RATIO     = float(os.getenv("TRAIN_DATASET_RATIO", 0.9))
    VALID_DATASET_RATIO     = float(os.getenv("VALID_DATASET_RATIO", 0.1))
    TRAIN_BATCH_SIZE        = int(os.getenv("TRAIN_BATCH_SIZE", 64))
    VALID_BATCH_SIZE        = int(os.getenv("VALID_BATCH_SIZE", 64))
    TRAIN_CSV_PATH          = os.getenv("TRAIN_CSV_PATH", "data/training_revised.csv")
    TRAIN_MODEL_PT_PATH     = os.getenv("TRAIN_MODEL_PT_PATH", "models/")
    TRAIN_MODEL_NAME        = os.getenv("TRAIN_MODEL_NAME", "apan.pt")
    TRAIN_BEST_MODEL_NAME   = os.getenv("TRAIN_BEST_MODEL_NAME", "apan_best.pt")


    torch.manual_seed(TORCH_SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-7)


    train_data = SinoDataset(TRAIN_CSV_PATH)
    train_set, valid_set = random_split(dataset=train_data, lengths=(TRAIN_DATASET_RATIO, VALID_DATASET_RATIO))
    train_loader = DataLoader(dataset=train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(dataset=valid_set, batch_size=VALID_BATCH_SIZE, shuffle=True, num_workers=4)

    lowest_valid_loss = math.inf
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, mape_loss, optimizer)
        valid_loss = valid(model, valid_loader, mape_loss)
        if lowest_valid_loss > valid_loss:
            torch.save(model, TRAIN_MODEL_PT_PATH+TRAIN_BEST_MODEL_NAME)
            lowest_valid_loss = valid_loss
        print(f"\nEpochs={epoch}, avg_train_loss={train_loss}, avg_valid_loss={valid_loss}")


    torch.save(model, TRAIN_MODEL_PT_PATH+TRAIN_MODEL_NAME)