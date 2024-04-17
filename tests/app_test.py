from fastapi.testclient import TestClient
import configparser
import os
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
import json
import pandas as pd

from src.app.main import app

client = TestClient(app)


def test_api():
    client.__enter__()

    config = configparser.ConfigParser()
    config.read('config.ini')
    test_path = os.path.join(cur_dir.parent, 'test_0.json')

    with open(str(test_path)) as f:
        test_json = json.load(f)

    y = test_json["y"]
    del test_json['y']

    response = client.post(
       f"/predict/",
       json=test_json
    )

    assert response.status_code == 200
    assert response.json()["Category"]["0"] == y[0]["Category"]

    client.__exit__(None, None, None)

test_api()
