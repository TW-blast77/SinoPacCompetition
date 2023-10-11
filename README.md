# SinoPac AI competition
The competition web URL is [here](https://tbrain.trendmicro.com.tw/Competitions/Details/30)

## Datasets
Please download the dataset from the SinoPac AI competition website.

### Training dataset
Put the training dataset in ./data directory.
``` bash
# Create directory
mkdir data
# Move your datasets
mv <Your dataset location> ./data
```

### External dataset
Move the external data
``` bash
# Create directory
mkdir external_data
# Move your datasets
mv <Your external dataset location> ./external_data
```


## Rule
1. All models save in directory models.
2. Don't upload competition datasets to github.
3. Please create a branch before you add or remove something.


## Usage

### Setup .env file
1. Copy your .env.
    ``` bash
    # copy your own .env file
    cp .env.example .env
    ```
2. Setup your own .env
    ``` bash
    vim .env
    # or
    gedit .env
    ```

### External data preprocessing
1. Run the external_data_convertor.py
    ``` bash
    # You must have csv, ex: external_data/大學基本資料.csv...
    python external_data_convertor.py
    ```

### Preprocessing
1. Run the preprocessing.py
    ``` bash
    # You must have csv, ex: data/train_data.csv
    python preprocessing.py
    ```
### Training
1. Start training
    ``` bash
    # After setup your .env file & preprocess
    python train.py
    ```