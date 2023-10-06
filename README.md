# SinoPac AI competition
The competition web URL is [here](https://tbrain.trendmicro.com.tw/Competitions/Details/30)

## Rule
1. All models save in directory models.
2. Don't upload competition datasets to github.
3. Please create a branch before you add or remove something.

## Preprocessing
1. Encoded every chinese value into integer.
2. Split the "建物型態" into 2 line which are "電梯樓層" and "有無電梯"
``` bash
# Run preprocessing
# You must have csv, data/train_data.csv
python preprocessing.py
```