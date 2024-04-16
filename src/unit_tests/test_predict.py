import os
import configparser
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
from predict import Predictor
from test_utils import get_tmp_test_data, tmp_dir_from_path
import pandas as pd

def test_predictor(
        path_to_test_data: str, 
        path_to_model_ckpt: str, 
        path_to_vectorizer_ckpt: str, 
        path_to_submission: str
):
    tmp_dir_from_path(path_to_test_data)
    get_tmp_test_data(path_to_test_data)
    predictor = Predictor()
    predictor.predict(path_to_test_data, path_to_model_ckpt, path_to_vectorizer_ckpt, path_to_submission)
    assert os.path.exists(path_to_submission), "Submission file was not generated"
    submission_tmp = pd.read_csv(path_to_submission)
    assert submission_tmp.iloc[0]["Category"] == "business" and submission_tmp.iloc[0]["ArticleId"] == 1, "Wrong prediction in line 1"
    assert submission_tmp.iloc[1]["Category"] == "sport" and submission_tmp.iloc[1]["ArticleId"] == 2, "Wrong prediction in line 2"

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')
    path_to_test_data = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_test_data'])
    path_to_vectorizer_ckpt = os.path.join(cur_dir.parent.parent.parent, config['vectorizer']['path_to_vectorizer_ckpt'])
    path_to_model_ckpt = os.path.join(cur_dir.parent.parent.parent, config['model']['path_to_model_ckpt'])
    path_to_submission = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_submission'])
    test_predictor(path_to_test_data, path_to_model_ckpt, path_to_vectorizer_ckpt, path_to_submission)
