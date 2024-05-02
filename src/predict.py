import pickle
from utils import load_ckpt
from preprocess import Preprocessor
import pandas as pd
import configparser
from src.db import Database


class Predictor:

    def predict(
            self, 
            db: Database,
            submission_table: str,
            test_table: str,
            path_to_model_ckpt: str, 
            path_to_vectorizer_ckpt: str,
            add_cols = None
    ):
        clf = load_ckpt(path_to_model_ckpt)
        preprocessor = Preprocessor()
        X_test = preprocessor.load_and_preprocess_data(db, test_table, isTest=True)
        vectorizer = load_ckpt(path_to_vectorizer_ckpt)
        test_features = vectorizer.transform(X_test)
        predicted = clf.predict(test_features)
        dct = {'ArticleId': preprocessor.get_article_ids(), 'Category': predicted.tolist()}
        if add_cols != None:
            dct.update(add_cols)
        submission = pd.DataFrame(dct)
        db.insert_df(submission_table, submission)