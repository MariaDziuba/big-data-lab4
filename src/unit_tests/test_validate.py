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
from new_validate import Validator
from conftest import db

def test_validator():
    config = configparser.ConfigParser()
    config.read('config.ini')
    # path_to_train_data = os.path.join(cur_dir.parent.parent.parent, config['data']['path_to_train_data'])
    # path_to_tmp_val_data = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_val_data'])
    path_to_vectorizer_ckpt = os.path.join(cur_dir.parent.parent.parent, config['vectorizer']['path_to_vectorizer_ckpt'])
    path_to_dummy_model_ckpt = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_dummy_model_ckpt'])
    # path_to_tmp_metrics = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_metrics'])

    dummy = DummyClassifier(strategy="most_frequent")
    preprocessor = Preprocessor()
    X_train, y_train = preprocessor.load_and_preprocess_data(db, "train", isTest=False)
    vectorizer = load_ckpt(path_to_vectorizer_ckpt)
    train_features = vectorizer.fit_transform(X_train)
    dummy.fit(train_features, y_train)
    save_ckpt(dummy, path_to_dummy_model_ckpt)
    validator = Validator()
    attrs = vars(validator)
    # {'kids': 0, 'name': 'Dog', 'color': 'Spotted', 'age': 10, 'legs': 2, 'smell': 'Alot'}
    # now dump this in some way or another
    print(', '.join("%s: %s" % item for item in attrs.items()))
    validator.validate(db, "tmp_val", "tmp_metrics", path_to_dummy_model_ckpt, path_to_vectorizer_ckpt)
    # assert os.path.exists(path_to_tmp_metrics), "Submission file was not generated"
    assert db.table_exists("tmp_metrics")['result'].iloc[0]
    # tmp_metrics = pd.read_csv(path_to_tmp_metrics)
    tmp_metrics = db.read_table("tmp_metrics")
    assert tmp_metrics.iloc[0]["Metric"] == "f1_macro" and round(tmp_metrics.iloc[0]["Value"], 2) == 0.04
    assert tmp_metrics.iloc[1]["Metric"] == "accuracy" and round(tmp_metrics.iloc[1]["Value"], 2) == 0.11
    assert tmp_metrics.iloc[2]["Metric"] == "precision_macro" and round(tmp_metrics.iloc[2]["Value"], 2) == 0.02
    assert tmp_metrics.iloc[3]["Metric"] == "recall_macro" and round(tmp_metrics.iloc[3]["Value"], 2) == 0.2
