import pickle
from utils import load_ckpt
from preprocess import Preprocessor
import pandas as pd
import configparser

class Predictor:

    def predict(
            self, 
            path_to_test_data: str, 
            path_to_model_ckpt: str, 
            path_to_vectorizer_ckpt: str,
            path_to_submission: str
    ):
        clf = load_ckpt(path_to_model_ckpt)
        preprocessor = Preprocessor()
        X_test = preprocessor.load_and_preprocess_data(path_to_test_data, isTest=True)
        vectorizer = load_ckpt(path_to_vectorizer_ckpt)
        test_features = vectorizer.transform(X_test)
        predicted = clf.predict(test_features)
        submission = pd.DataFrame({'ArticleId': preprocessor.get_article_ids, 'Category': predicted.tolist()}) 
        submission.to_csv(path_to_submission, index=False)

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('config.ini')

    path_to_test_data = config['data']['path_to_test_data']
    path_to_vectorizer_ckpt = config['vectorizer']['path_to_vectorizer_ckpt']
    path_to_model_ckpt = config['model']['path_to_model_ckpt']
    path_to_submission = config['results']['path_to_submission']
    
    predictor = Predictor()
    predictor.predict(path_to_test_data, path_to_model_ckpt, path_to_vectorizer_ckpt, path_to_submission)