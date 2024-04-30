from fastapi.testclient import TestClient
import configparser
import os
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
import json
import pandas as pd
from src.app.main import db

from src.app.main import app


client = TestClient(app)


from src.db import Database
from src.vault import AnsibleVault

config = configparser.ConfigParser()
config.read('config.ini')

vault_file = os.path.join(cur_dir.parent.parent, config['secrets']['vault'])
ansible_vault = AnsibleVault(vault_file)

db = Database(ansible_vault)


def test_api():
    client.__enter__()

    config = configparser.ConfigParser()
    config.read('config.ini')
    test_path = os.path.join(cur_dir.parent, 'test_0.json')

    with open(str(test_path)) as f:
        test_json = json.load(f)

    db.create_table('tmp_queries', {'MessageId': 'UInt32', 'ArticleId': 'UInt32', 'Text': 'String', 'Category': 'String'})
    db.create_table('tmp_predictions', {'MessageId': 'UInt32', 'ArticleId': 'UInt32', 'Category': 'String'})

    y = test_json["y"]
    del test_json['y']

    response = client.post(
       f"/predict/",
       json=test_json
    )

    print('response')
    print(response.json())
    # assert response.json()['msg_id'] == 0

    msg_id = response.json()['msg_id']
    
    response = client.get(
       f"/predict/{msg_id}",
    )

    # print('msg_id from test', msg_id)

    # print('tmp_predictions from app_test.py', db.read_table("tmp_predictions"))
    # result = db.select_by_condition("tmp_predictions", f"MessageId = {msg_id}")
    # print(result)
    # print(result.to_dict())

    # print('response 2')
    # print(response.json())

    assert response.status_code == 200
    assert response.json()["Category"]["0"] == y[0]["Category"]
    db.drop_table("tmp_queries")
    db.drop_table("tmp_predictions")
    client.__exit__(None, None, None)
