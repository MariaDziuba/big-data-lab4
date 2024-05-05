import kafka

from db import Database
from vault import AnsibleVault
import configparser
import os
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
import pandas as pd
from predict import Predictor
import json
from dotenv import load_dotenv
load_dotenv()


config = configparser.ConfigParser()
config.read('config.ini')

path_to_vectorizer_ckpt = os.path.join(cur_dir.parent.parent, config['vectorizer']['path_to_vectorizer_ckpt'])
path_to_model_ckpt = os.path.join(cur_dir.parent.parent, config['model']['path_to_model_ckpt'])
vault_file = os.path.join(cur_dir.parent.parent, config['secrets']['vault'])

ansible_vault = AnsibleVault(vault_file)

class KafkaConsumerError(Exception):
    def __init__(self, comment):
        self.comment = comment
    
    def __str__(self):
        return self.comment

class Consumer:
    def __init__(
        self,
        vault: AnsibleVault
    ):
        topic_name = vault.get_secret('TOPIC_NAME')
        kafka_server = vault.get_secret('KAFKA_SERVER')
        self.consumer = kafka.KafkaConsumer(
            topic_name,
            bootstrap_servers=kafka_server,
            enable_auto_commit=True,
            auto_offset_reset="earliest",
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        )
        self.db = Database(vault)
        

    def run(self, X: list, msg_id: int):
        df = pd.DataFrame(X)
        self.db.insert_df("tmp_queries", df)

        # print(self.db.table_exists("tmp_queries")['result'].iloc[0])
        # print(self.db.read_table("tmp_queries"))
        predictor = Predictor()
        predictor.predict(self.db, "tmp_predictions", "tmp_queries", path_to_model_ckpt, path_to_vectorizer_ckpt, {'MessageId': msg_id})
        # print(self.db.table_exists("tmp_predictions")['result'].iloc[0])
        # print(self.db.read_table("tmp_predictions"))


    def run_all_msg(self):
        while True:
            for msg in self.consumer:
                try:
                    X = msg.value['X']
                except KeyError:
                    raise KafkaConsumerError(f"Data for consumer was not provided: 'X' field in message is empty")
                try: 
                    msg_id = msg.value['msg_id']
                except KeyError:
                    raise KafkaConsumerError(f"Empty message id: 'msg_id' field in message is empty")
                self.run(X, msg_id)


    def close(self):
        self.connection.close()
        self.consumer.close()
        
        
def main():
    consumer = Consumer(ansible_vault)
    consumer.run_all_msg()

    # consumer.close()
if __name__ == "__main__":
    main()
