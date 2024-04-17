import os
import pandas as pd
import pytest
import shutil
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
tmp_dir = os.path.join(cur_dir.parent.parent.parent, "tmp")
import configparser


@pytest.fixture(scope="session", autouse=True)
def run_around_tests():
    # before tests:
    os.makedirs(tmp_dir, mode=0o777, exist_ok=True)

    config = configparser.ConfigParser()
    config.read('config.ini')

    path_to_tmp_val_data = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_val_data'])
    path_to_tmp_test_data = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_tmp_test_data'])

    test_df = pd.DataFrame({
        "ArticleId": [1, 2],
        "Text": ["Business is good!", "Football is a popular sport"],
        "Category": ["business", "sport"]
    })
    test_df.to_csv(path_to_tmp_test_data, index=False)

    val_df = pd.DataFrame({
        "ArticleId": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "Text": [
            "70s was a decade of legendary rock stars",
            "Taylor Swift is a famous American singer",
            "The goverment decided to increase taxes",
            "It is the largest oil company in Japan",
            "I think he is a great guitarist",
            "Tell search engines that your website exists",
            "LG releases a new flagship smartphone next month",
            "He's the starting goalie on the hockey team",
            "She wants to become an excellent long-distance swimmer"
        ],
        "Category": [
            "entertainment",
            "entertainment",
            "politics",
            "business",
            "entertainment",
            "tech",
            "tech",
            "sport",
            "sport",
        ]
    })
    val_df.to_csv(path_to_tmp_val_data, index=False)
    yield
    # after tests:
    shutil.rmtree(tmp_dir)
