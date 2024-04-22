from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
import pickle
from utils import load_ckpt
from preprocess import Preprocessor
import pandas as pd
import configparser
from db import Database

metrics = {
    "f1_macro" : (f1_score, "macro"),
    "accuracy" : (accuracy_score, ""),
    "precision_macro" : (precision_score, "macro"),
    "recall_macro" : (recall_score, "macro"),
}

class Validator:
    def validate(
            self,
            db: Database,
            val_table: str,
            metrics_table: str,
            # path_to_val_data: str, 
            path_to_model_ckpt: str, 
            path_to_vectorizer_ckpt: str, 
            # path_to_metrics: str
    ):
        clf = load_ckpt(path_to_model_ckpt)
        preprocessor = Preprocessor()
        X_val, y_val = preprocessor.load_and_preprocess_data(db, val_table, isTest=False)
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

        metrics_df = pd.DataFrame({"Id": [ _ for _ in range(len(metrics))], "Metric": metrics_list, "Value": values_list})
        print(metrics_df)
        db.insert_df(metrics_table, metrics_df)
        # metrics_df.to_csv(path_to_metrics, index=False)