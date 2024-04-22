from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from sklearn.base import BaseEstimator
from utils import save_ckpt
from preprocess import Preprocessor
import configparser
from db import Database

class Trainer:
    
    def train(
            self, 
            db: Database,
            train_table: str,
            # path_to_train_data: str, 
            path_to_model_ckpt: str, 
            path_to_vectorizer_ckpt: str, 
            model: BaseEstimator = MultinomialNB(),
            vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, stop_words='english')
    ):
        preprocessor = Preprocessor()
        X_train, y_train = preprocessor.load_and_preprocess_data(db, train_table, isTest=False)
        train_features = vectorizer.fit_transform(X_train)
        clf = model.fit(train_features, y_train)
        save_ckpt(clf, path_to_model_ckpt)
        save_ckpt(vectorizer, path_to_vectorizer_ckpt)
        
