import pandas as pd
from math import radians
from src.latlng2twd97 import LatLonToTWD97

def convert_external_data_coordination(df) -> tuple:
    convertor = LatLonToTWD97()
    x_list, y_list = [], []
    for index in range(df.__len__()):
        lat = df["lat"].iloc[index]
        lng = df["lng"].iloc[index]
        x, y = convertor.convert(radians(lat), radians(lng))
        x_list.append(x)
        y_list.append(y)
    
    return x_list, y_list


if __name__ == "__main__":
    external_data_list = [
        "ATM資料", "便利商店", "公車站點資料", "國中基本資料", "國小基本資料",
        "大學基本資料", "捷運站點資料", "火車站點資料", "腳踏車站點資料",
        "郵局據點資料", "醫療機構基本資料", "金融機構基本資料", "高中基本資料"
    ]

    for filename in external_data_list:
        external_data = pd.read_csv(f'external_data/{filename}.csv')
        x_list, y_list = convert_external_data_coordination(external_data)
        external_data["橫坐標"] = x_list
        external_data["縱坐標"] = y_list
        external_data.to_csv(f'external_data/{filename}_revised.csv')
        print(f"已儲存轉換檔案{filename}_revised.csv...")

    