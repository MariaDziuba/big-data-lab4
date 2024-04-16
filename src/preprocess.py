import nltk
import subprocess
from typing import List, Tuple
import numpy as np
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import string
import pandas as pd

class Preprocessor:

    article_ids: List

    def __init__(self):
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stop_words = stopwords.words('english')

    def clean_text(self, doc):
        text = "".join([char for char in doc if char not in string.punctuation and not char.isdigit()])
    
        tokens = nltk.word_tokenize(text.lower())
    
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
    
        return ' '.join(tokens)
    
    def preprocess_data(self, train_data: pd.DataFrame):
        train_data['Clean_text'] = train_data['Text'].apply(lambda x: self.clean_text(x))
        return train_data
    
    def load_data(self, path_to_train_data: str) -> pd.DataFrame:
        train_data = pd.read_csv(path_to_train_data)
        return train_data
        
    def load_and_preprocess_data(self, path_to_data: str, isTest: bool)-> Tuple[List, List]:
        data = self.load_data(path_to_data)
        data = self.preprocess_data(data)
        self.article_ids = data['ArticleId'].to_list()
        X = data['Clean_text'].tolist()
        if isTest:
            return X
        y = data['Category'].to_list()
        return X, y
    
    def get_article_ids(self):
        return self.article_ids

    