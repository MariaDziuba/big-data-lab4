import kafka

from src.db import Database
from src.vault import AnsibleVault
import configparser
import os
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
import pandas as pd
from predict import Predictor
import json

class Producer:
    def __init__(
        self,
        vault: AnsibleVault
    ):
        self.topic_name = vault.get_secret('TOPIC_NAME')
        kafka_server = vault.get_secret('KAFKA_SERVER')
        self.producer = kafka.KafkaProducer(
            bootstrap_servers=kafka_server,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.msg_id = 0


    def run(self, X: list):
        self.producer.send(self.topic_name, value={'X': X, 'msg_id': self.msg_id})
        self.msg_id += 1
        return self.msg_id - 1

    def close(self):
        self.producer.close()