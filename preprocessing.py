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
    # Init
    train_data = pd.read_csv ('data/training_data.csv')
    update_cols = { "縣市": {}, "鄉鎮市區": {}, "路名": {}, "使用分區": {}, "主要用途": {}, "主要建材": {} }

    # Get encode dictionary
    for key in update_cols.keys():
        encoded_number_list = get_encoded_list(train_data[key])
        update_cols.get(key).update(encoded_number_list)

    # Update every row col value
    for index, row in train_data.iterrows():
        for key, value in update_cols.items():
            row_col_value = row.get(key)
            encoded_number = value.get(row_col_value)
            train_data.at[index, key] = encoded_number

    # Save
    train_data.to_csv("data/training_revised.csv", index=False)
