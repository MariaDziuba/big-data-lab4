import os
import configparser
import sys 
import path
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
from train import Trainer
import pandas as pd
from test_utils import get_tmp_test_data, tmp_dir_from_path

def test_trainer(
        path_to_train_data: str, 
        path_to_model_ckpt: str, 
        path_to_vectorizer_ckpt: str, 
):
    tmp_dir_from_path(path_to_vectorizer_ckpt)
    trainer = Trainer()
    trainer.train(path_to_train_data, path_to_model_ckpt, path_to_vectorizer_ckpt)
    assert os.path.exists(path_to_model_ckpt), "Model was not saved after training"
    assert os.path.exists(path_to_vectorizer_ckpt), "Vectorizer was not saved after training"

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    path_to_train_data = os.path.join(cur_dir.parent.parent.parent, config['data']['path_to_train_data'])
    path_to_vectorizer_ckpt = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_vectorizer_ckpt'])
    path_to_model_ckpt = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_model_ckpt'])
    test_trainer(path_to_train_data, path_to_model_ckpt, path_to_vectorizer_ckpt)