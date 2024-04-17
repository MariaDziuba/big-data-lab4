import os
import configparser
import sys 
import path
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
from train import Trainer
import pandas as pd

def test_trainer():
    config = configparser.ConfigParser()
    config.read('config.ini')
    path_to_train_data = os.path.join(cur_dir.parent.parent.parent, config['data']['path_to_train_data'])
    path_to_vectorizer_ckpt = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_vectorizer_ckpt'])
    path_to_model_ckpt = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_model_ckpt'])

    trainer = Trainer()
    trainer.train(path_to_train_data, path_to_model_ckpt, path_to_vectorizer_ckpt)
    assert os.path.exists(path_to_model_ckpt), "Model was not saved after training"
    assert os.path.exists(path_to_vectorizer_ckpt), "Vectorizer was not saved after training"

