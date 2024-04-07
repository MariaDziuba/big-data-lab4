from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import pickle
from utils import load_ckpt
from preprocess import Preprocessor
import pandas as pd

metrics = {
    "f1_macro" : (f1_score, "macro"),
    "accuracy" : (accuracy_score, ""),
    "precision_macro" : (precision_score, "macro"),
    "recall_macro" : (recall_score, "macro"),
}

class Validator:
    def validate(
            self,
            path_to_val_data: str, 
            path_to_model_ckpt: str, 
            path_to_vectorizer_ckpt: str, 
            path_to_metrics: str
    ):
        clf = load_ckpt(path_to_model_ckpt)
        preprocessor = Preprocessor()
        X_val, y_val = preprocessor.load_and_preprocess_data(path_to_val_data, isTest=False)
        vectorizer = load_ckpt(path_to_vectorizer_ckpt)
        val_features = vectorizer.transform(X_val)
        predicted = clf.predict(val_features)

        metrics_list = []
        values_list = []
        
        for k, v in metrics.items():
            metrics_list.append(k)
            if v[1] == "":
                values_list.append(v[0](y_val, predicted))
            else:
                values_list.append(v[0](y_val, predicted, average=v[1]))

        metrics_df = pd.DataFrame({"metric": metrics_list, "value": values_list})
        metrics_df.to_csv(path_to_metrics)

if __name__ == "__main__":
    path_to_val_data = '/Users/modzyuba1/ITMO/big-data-lab1/data/bbc_news_val.csv'
    path_to_vectorizer_ckpt = '/Users/modzyuba1/ITMO/big-data-lab1/ckpts/tfidf_vectorizer.pkl'
    path_to_model_ckpt = '/Users/modzyuba1/ITMO/big-data-lab1/ckpts/tfidf_svc_model.pkl'
    path_to_metrics = '/Users/modzyuba1/ITMO/big-data-lab1/metrics/metrics.csv'
    validator = Validator()
    validator.validate(path_to_val_data, path_to_model_ckpt, path_to_vectorizer_ckpt, path_to_metrics)