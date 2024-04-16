import os
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def get_tmp_test_data(tmp_path: str) -> str:
    if not os.path.exists(tmp_path):
        train_df = pd.DataFrame({
            "ArticleId": [1, 2],
            "Text": ["Business is good!", "Football is a popular sport"],
            "Category": ["business", "sport"]
        })
        train_df.to_csv(tmp_path, index=False)
        return train_df
    else:
        return pd.read_csv(tmp_path)


@pytest.fixture(scope="module")
def get_tmp_val_data(tmp_path: str) -> str:
    if not os.path.exists(tmp_path):
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
        val_df.to_csv(tmp_path, index=False)
        return val_df
    else:
        return pd.read_csv(tmp_path)
    

@pytest.fixture(scope="module")    
def tmp_dir_from_path(tmp_path: str):
    tmp_dir = os.path.dirname(tmp_path)
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
