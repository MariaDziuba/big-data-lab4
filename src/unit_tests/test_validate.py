import os
import configparser
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
from train import Trainer
import pandas as pd
from sklearn.dummy import DummyClassifier
from preprocess import Preprocessor
from utils import load_ckpt, save_ckpt
from validate import Validator

def test_validator():
    config = configparser.ConfigParser()
    config.read('config.ini')
    path_to_train_data = os.path.join(cur_dir.parent.parent.parent, config['data']['path_to_train_data'])
    path_to_tmp_val_data = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_val_data'])
    path_to_vectorizer_ckpt = os.path.join(cur_dir.parent.parent.parent, config['vectorizer']['path_to_vectorizer_ckpt'])
    path_to_dummy_model_ckpt = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_dummy_model_ckpt'])
    path_to_tmp_metrics = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_metrics'])

    dummy = DummyClassifier(strategy="most_frequent")
    preprocessor = Preprocessor()
    X_train, y_train = preprocessor.load_and_preprocess_data(path_to_train_data, isTest=False)
    vectorizer = load_ckpt(path_to_vectorizer_ckpt)
    train_features = vectorizer.fit_transform(X_train)
    dummy.fit(train_features, y_train)
    save_ckpt(dummy, path_to_dummy_model_ckpt)
    validator = Validator()
    validator.validate(path_to_tmp_val_data, path_to_dummy_model_ckpt, path_to_vectorizer_ckpt, path_to_tmp_metrics)
    assert os.path.exists(path_to_tmp_metrics), "Submission file was not generated"
    tmp_metrics = pd.read_csv(path_to_tmp_metrics)
    assert tmp_metrics.iloc[0]["metric"] == "f1_macro" and round(tmp_metrics.iloc[0]["value"], 2) == 0.04
    assert tmp_metrics.iloc[1]["metric"] == "accuracy" and round(tmp_metrics.iloc[1]["value"], 2) == 0.11
    assert tmp_metrics.iloc[2]["metric"] == "precision_macro" and round(tmp_metrics.iloc[2]["value"], 2) == 0.02
    assert tmp_metrics.iloc[3]["metric"] == "recall_macro" and round(tmp_metrics.iloc[3]["value"], 2) == 0.2
