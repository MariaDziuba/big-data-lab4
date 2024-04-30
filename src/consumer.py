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
vault_pwd_file = os.path.join(cur_dir.parent.parent, config['secrets']['vault_pwd'])
vault_file = os.path.join(cur_dir.parent.parent, config['secrets']['vault'])

ansible_vault = AnsibleVault(vault_pwd_file, vault_file)

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
                self.run(msg.value['X'], msg.value['msg_id'])

    def close(self):
        self.connection.close()
        self.consumer.close()
        
        
def main():
    consumer = Consumer(ansible_vault)
    consumer.run_all_msg()

    # consumer.close()
if __name__ == "__main__":
    main()
