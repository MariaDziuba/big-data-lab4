from typing import List
from sklearn.model_selection import train_test_split
import pandas as pd

import pickle

def save_ckpt(object, path: str):
    with open(path, 'wb') as f:
        pickle.dump(object, f)

def load_ckpt(path: str):
    with open(path, 'rb') as f:
        object = pickle.load(f)
    return object