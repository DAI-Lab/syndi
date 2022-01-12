import os
import pandas as pd
DATA_PATH = os.path.join(_BASE_PATH,
    'data'
) 
TRAIN_PATH = os.path.join(DATA_PATH,
    'train.csv'
)
TEST_PATH = os.path.join(DATA_PATH,
    'test.csv'
)

def load_demo_data():
    train_path
    train = pd.load_csv(TRAIN_PATH)
    test = pd.load_csv(TEST_PATH)
    return train, test
