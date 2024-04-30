from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from src.utils import load_ckpt
import configparser
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
from src.producer import Producer

app = FastAPI()

from src.db import Database
from src.vault import AnsibleVault

config = configparser.ConfigParser()
config.read('config.ini')

vault_file = os.path.join(cur_dir.parent.parent.parent, config['secrets']['vault'])
ansible_vault = AnsibleVault(vault_file)

producer = Producer(ansible_vault)
db = Database(ansible_vault)

class InputData(BaseModel):
    X: list

@app.get("/predict/{msg_id}")
async def get_prediction_by_id(msg_id: int):
    try:
        # print('tmp_predictions from app.py', db.read_table("tmp_predictions"))
        result = db.select_by_condition("tmp_predictions", f"MessageId = {msg_id}")
        # print('result from app.py', result)
        return result.to_dict()
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
async def predict(input_data: InputData):
    try:
        msg_id = producer.run(input_data.X)
        return {'msg_id': msg_id}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        producer.close()
    


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")