#!/usr/bin/env python3 -u

import os

import clickhouse_connect
import pandas as pd
from typing import Dict
from vault import AnsibleVault
import configparser
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
tmp_dir = os.path.join(cur_dir.parent.parent.parent, "tmp")


class Database():
    def __init__(self, vault: AnsibleVault):
        host = vault.get_secret('CLICKHOUSE_HOST')
        port = int(vault.get_secret('CLICKHOUSE_PORT'))
        username = vault.get_secret('CLICKHOUSE_LOGIN')
        password = vault.get_secret('CLICKHOUSE_PWD')
        # host = os.getenv('CLICKHOUSE_HOST', 'clickhouse')
        # port = int(os.getenv('CLICKHOUSE_PORT', '8123'))
        # username = os.getenv('CLICKHOUSE_LOGIN', 'default')
        # password = os.getenv('CLICKHOUSE_PWD')
        self.client = clickhouse_connect.get_client(host=host, username=username, port=port, password=password)

    def create_database(self, database_name="lab2_bd"):
        self.client.command(f"""CREATE DATABASE IF NOT EXISTS {database_name};""")

    def create_table(self, table_name: str, columns: Dict):
        cols = ""
        for k, v in columns.items():
            cols += f"`{k}` {v}, "
        id_column = list(filter(lambda i: 'Id' in i[0], columns.items()))[0][0]
        self.client.command(f"""
            CREATE TABLE IF NOT EXISTS {table_name} 
            (
                {cols}
            ) ENGINE = MergeTree
            ORDER BY {id_column};
        """)

    def insert_df(self, tablename: str, df: pd.DataFrame):
        self.client.insert_df(tablename, df)

    def read_table(self, tablename: str) -> pd.DataFrame:
        return self.client.query_df(f'SELECT * FROM {tablename}')
    
    def select_by_condition(self, tablename: str, condition: str) -> pd.DataFrame:
        # print(condition)
        # print(f"""
            # SELECT * 
            # FROM {tablename}
            # WHERE {condition};
        # """)
        return self.client.query_df(f"""
            SELECT * 
            FROM {tablename}
            WHERE {condition};
        """)

    def drop_database(self, database_name: str):
        self.client.command(f'DROP DATABASE IF EXISTS {database_name}')

    def drop_table(self, table_name: str):
        self.client.command(f'DROP TABLE IF EXISTS {table_name}')

    def table_exists(self, table_name: str):
        return self.client.query_df(f'EXISTS {table_name}')

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config.ini')

    vault_file = os.path.join(cur_dir.parent.parent, config['secrets']['vault'])
    ansible_vault = AnsibleVault(vault_file)

    db = Database(ansible_vault)

    db.drop_table("tmp_test")
    db.drop_table("tmp_val")
    db.drop_table("train")
    db.drop_table("tmp_submission")
    db.drop_table("tmp_metrics")
    db.drop_database("lab2_bd")
    # db.create_database("lab2_bd")
    # db.create_table("test1", {'nameId': 'UInt32'})
    # db.create_table("test2", {'nameId': 'UInt32', 'name3': 'UInt32'})
    # db.insert_df("test1", pd.DataFrame({"nameId": [1]}))
    # db.insert_df("test2", pd.DataFrame({"nameId": [1], "name3": [1]}))
    # print(db.read_table("test1"))
    # print(db.read_table("test2"))
    # db.drop_table("test1")
    # db.drop_table("test2")
    # db.drop_database("lab2_bd")

