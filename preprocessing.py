import pandas as pd
import ramda as R
import numpy as np

def get_encoded_dict(df, encoded_col_list) -> dict:
    """
    Args:
        df:
            Dataframe object read from training csv
        encoded_col_list:
            List with different item
            ex: ["縣市", "鄉鎮市區", "路名", "使用分區"]
    Returns:
        Encoded dict
        ex:
            {
                "縣市": {"台北市": 0, "台中市": 1 ...},
                "鄉鎮市區": {"大安區": 0, "萬華區": 1 ...}
                ...
            }
    """
    res = {}
    for key in encoded_col_list:
        res[key] = R.pipe(
            R.map(str),
            sorted,
            R.group_with(R.equals),
            R.map(set),
            R.map(lambda city: str(city.pop())),
            enumerate,
            list,
            R.map(lambda x: (x[1], x[0])),
            R.from_pairs
        )(df[key])
    return res

def cal_external_data_nearest_distance(org_df, ex_df) -> list:
    """
    Args:
        org_df:
            Dataframe object read from training csv
        ex_df:
            Dataframe object read from external data
    Returns:
        List with each line have the nearest distance
        ex: [ 65.2, 63, 12, 84, 100, 632, 14, ... ]
    """
    nearest_distance_list = []
    for index in range(org_df.__len__()):
        building_x = org_df["橫坐標"].iloc[index]
        building_y = org_df["縱坐標"].iloc[index]
        nearest_distance = np.min(
            np.sqrt(
                (ex_df["橫坐標"] - building_x) ** 2 + (ex_df["縱坐標"] - building_y) ** 2
            )
        )
        nearest_distance_list.append(nearest_distance)
    return nearest_distance_list

def cal_external_data_count(org_df, ex_df, distance=1000) -> list:
    """
    Args:
        org_df:
            Dataframe object read from training csv
        ex_df:
            Dataframe object read from external data
    Returns:
        List with each line have the quantity of the facility
        ex: [ 10, 5, 3, 7, 5, 3 ]
    """
    near_count_list = []
    for index in range(org_df.__len__()):
        building_x = org_df["橫坐標"].iloc[index]
        building_y = org_df["縱坐標"].iloc[index]
        distance_array = np.sqrt((ex_df["橫坐標"] - building_x) ** 2 + (ex_df["縱坐標"] - building_y) ** 2)
        nearest_distance = np.count_nonzero(
            distance_array[distance_array > distance]
        )
        near_count_list.append(nearest_distance)
    return near_count_list

if __name__ == "__main__":
    train_data = pd.read_csv('data/training_data.csv')
    train_data["備註"] = train_data["備註"].fillna("0")

    #
    # 編碼給定的欄位
    #
    update_col_list = [
        "縣市", "鄉鎮市區", "路名", "使用分區", "主要用途",
        "路名", "建物型態", "主要建材", "備註"
    ]
    encoded_col_dict = get_encoded_dict(train_data, update_col_list)
    for index, row in train_data.iterrows():
        for key, value in encoded_col_dict.items():
            row_col_value = row.get(key)
            encoded_number = value.get(row_col_value)
            train_data.at[index, key] = encoded_number

    #
    # 利用額外資料擴充特徵欄位
    #
    external_data_list = [
        "ATM資料", "便利商店", "公車站點資料", "國中基本資料", "國小基本資料",
        "大學基本資料", "捷運站點資料", "火車站點資料", "腳踏車站點資料",
        "郵局據點資料", "醫療機構基本資料", "金融機構基本資料", "高中基本資料"
    ]
    nearest_distance_list = []
    near_facility_count_list = []
    for ex_data_path in external_data_list:
        ex_pd = pd.read_csv(f"external_data/{ex_data_path}_revised.csv")
        train_data[f"最近的{ex_data_path}"] = cal_external_data_nearest_distance(train_data, ex_pd)
        train_data[f"方圓1km{ex_data_path}"] = cal_external_data_count(train_data, ex_pd, 1000)
        nearest_distance_list.append(f"最近的{ex_data_path}")
        near_facility_count_list.append(f"方圓1km{ex_data_path}")
        print(f"資料{ex_data_path}特徵計算完成...")

    #
    # 對給定的欄位做正規化
    #
    normalized_cols_list = [
        "縣市", "鄉鎮市區", "路名", "土地面積", "使用分區", "移轉層次",
        "總樓層數", "主要用途", "主要建材", "建物型態", "屋齡", "建物面積", "車位面積", "車位個數",
        "橫坐標", "縱坐標", "備註", "主建物面積", "陽台面積", "附屬建物面積"
    ] + nearest_distance_list + near_facility_count_list
    for col in normalized_cols_list:
        train_data[col] = (train_data[col] - train_data[col].mean()) / train_data[col].std()
        print(f"欄位{col}標準化完成...")

    #
    # 保存
    #
    train_data.to_csv("data/training_revised.csv", index=False)
