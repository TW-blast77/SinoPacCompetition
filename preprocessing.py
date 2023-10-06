import pandas as pd
import ramda as R

def get_encoded_list(origin_list: list) -> dict:
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

    # Get id dictionary
    for col in update_cols: update_cols[col] = get_encoded_list(train_data[col])

    # Update every row
    for index, row in train_data.iterrows():
        for key, value in update_cols.items():
            train_data.at[index, key] = value[row[key]]

    # Save
    train_data.to_csv("Revised.csv", index=False)
