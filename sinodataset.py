import torch
import math
import pandas as pd

def cast_to_float(value) -> float:
    try:
        return 0.0 if math.isnan(float(value)) else float(value)
    except Exception:
        return 0.0

class SinoDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, csv_path: str):
        self.__df = pd.read_csv(csv_path)

    def __getitem__(self, index):
        origin_row = self.__df.iloc[index].to_list()[1:]
        float_row = list(map(
            cast_to_float,
            origin_row
        ))
        return torch.Tensor(float_row[:-1]), torch.Tensor([origin_row[-1]])
    
    def __len__(self):
        return self.__df.__len__()

class SinoTestDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, csv_path: str):
        self.__df = pd.read_csv(csv_path)

    def __getitem__(self, index):
        origin_row = self.__df.iloc[index].to_list()[1:]
        float_row = list(map(
            cast_to_float,
            origin_row
        ))
        return torch.Tensor(float_row), 0
    
    def __len__(self):
        return self.__df.__len__()