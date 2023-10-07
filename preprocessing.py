import pandas as pd
import ramda as R

def get_encoded_list(origin_list: list) -> dict:
    """
        Args:
            origin_list:
                a list with different item ex: ["A", "B", "A", "C", "A", "B"]
        Returns:
            a encoded dict ex: { "A": 0, "B": 1, "C": 2 }
    """
    return R.pipe(
        sorted,
        R.group_with(R.equals),
        R.map(set),
        R.map(lambda city: str(city.pop())),
        enumerate,
        list,
        R.map(lambda x: (x[1], x[0])),
        R.from_pairs
    )(origin_list)

if __name__ == "__main__":
    #
    # 初始化資料以及將備註的空欄位填 0
    #
    train_data = pd.read_csv ('data/public_dataset.csv')
    train_data["備註"] = train_data["備註"].fillna("0")

    #
    # 取得編碼表
    #
    update_cols = { "縣市": {}, "鄉鎮市區": {}, "路名": {}, "使用分區": {}, "主要用途": {}, "主要建材": {}, "備註": {} }
    for key in update_cols.keys():
        encoded_number_list = get_encoded_list(train_data[key])
        update_cols.get(key).update(encoded_number_list)

    #
    # 將編碼表對應到欄位上更新
    #
    for index, row in train_data.iterrows():
        for key, value in update_cols.items():
            row_col_value = row.get(key)
            encoded_number = value.get(row_col_value)
            train_data.at[index, key] = encoded_number


    #
    # 對該欄位做歸一化
    #
    normalized_cols = [
        "縣市", "鄉鎮市區", "路名", "土地面積", "使用分區", "移轉層次",
        "總樓層數", "主要用途", "主要建材", "屋齡", "建物面積", "車位面積", "車位個數",
        "橫坐標", "縱坐標", "備註", "主建物面積", "陽台面積", "附屬建物面積"
    ]
    for col in normalized_cols:
        train_data[col] = (train_data[col] - train_data[col].mean()) / train_data[col].std()

    
    #
    # 保存更動
    #
    train_data.to_csv("data/public_revised.csv", index=False)
