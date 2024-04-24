import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from src.utils import load_ckpt
import configparser
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
from src.predict import Predictor

app = FastAPI()

from src.db import Database

db = Database()
db.create_table('tmp_test', {'ArticleId': 'UInt32', 'Text': 'String', 'Category': 'String'})
db.create_table('tmp_submission', {'ArticleId': 'UInt32', 'Category': 'String'})

config = configparser.ConfigParser()
config.read('config.ini')

# path_to_test_data = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_app_test_data'])
path_to_vectorizer_ckpt = os.path.join(cur_dir.parent.parent.parent, config['vectorizer']['path_to_vectorizer_ckpt'])
path_to_model_ckpt = os.path.join(cur_dir.parent.parent.parent, config['model']['path_to_model_ckpt'])
# path_to_submission = os.path.join(cur_dir.parent.parent.parent, config['tests']['path_to_app_submission'])

model = load_ckpt(path_to_model_ckpt)
vectorizer = load_ckpt(path_to_vectorizer_ckpt)


class InputData(BaseModel):
    X: list


@app.post("/predict/")
async def predict(input_data: InputData):
    try:
        df = pd.DataFrame(input_data.X)
        db.insert_df("tmp_test", df)

        # df.to_csv(path_to_test_data)
        predictor = Predictor()
        predictor.predict(db, "tmp_submission", "tmp_test", path_to_model_ckpt, path_to_vectorizer_ckpt)
        
        response = db.read_table("tmp_submission")
        return response.to_dict()

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")