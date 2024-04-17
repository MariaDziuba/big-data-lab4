import os
import pandas as pd
import pytest
import shutil
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
tmp_dir = os.path.join(cur_dir.parent.parent, "/tmp")

@pytest.fixture(autouse=True)
def run_around_tests():
    # before tests:
    os.makedirs(tmp_dir, exist_ok=True)

    train_df = pd.DataFrame({
        "ArticleId": [1, 2],
        "Text": ["Business is good!", "Football is a popular sport"],
        "Category": ["business", "sport"]
    })
    train_df.to_csv(tmp_dir, index=False)

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
    val_df.to_csv(tmp_dir, index=False)
    yield
    # after tests:
    shutil.rmtree(tmp_dir)
