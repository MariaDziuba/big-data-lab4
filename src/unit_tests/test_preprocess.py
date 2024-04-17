import os
import configparser
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
from preprocess import Preprocessor
import pandas as pd

def test_preprocessor():
    config = configparser.ConfigParser()
    config.read('config.ini')
    path_to_tmp_test_data = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_test_data'])

    preprocessor = Preprocessor()
    is_test_false = preprocessor.load_and_preprocess_data(path_to_tmp_test_data, isTest=False)
    is_test_true = preprocessor.load_and_preprocess_data(path_to_tmp_test_data, isTest=True)
    assert type(is_test_false) is tuple
    assert type(is_test_true) is list
    X_train, y_train = is_test_false
    assert X_train[0] == 'business good' and X_train[1] == 'football popular sport'
