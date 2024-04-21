#!/usr/bin/env python3 -u

import os

import clickhouse_connect
import pandas as pd


class Database():
    def __init__(self):
        host = os.getenv('CLICKHOUSE_HOST', 'localhost')
        port = int(os.getenv('CLICKHOUSE_PORT', '8123'))
        username = os.getenv('CLICKHOUSE_USERNAME', 'default')
        password = os.getenv('CLICKHOUSE_PASSWORD')
        self.client = clickhouse_connect.get_client(host=host, username=username, port=port, password=password)

    def create_database(self, database_name="lab2_bd"):
        self.client.command(f"""CREATE DATABASE IF NOT EXISTS {database_name};""")

    def create_table(self, table_name="test"):
        self.client.command(f"""
            CREATE TABLE IF NOT EXISTS {table_name} 
            (
                `name1` UInt32
            ) ENGINE = MergeTree
            ORDER BY name1;
        """)

    def insert_df(self, tablename: str, df: pd.DataFrame):
        self.client.insert_df(tablename, df)

    def read_table(self, tablename: str) -> pd.DataFrame:
        return self.client.query_df(f'SELECT * FROM {tablename}')

    def drop_database(self, database_name: str):
        self.client.command(f'DROP DATABASE IF EXISTS {database_name}')

    def drop_table(self, table_name: str):
        self.client.command(f'DROP TABLE IF EXISTS {table_name}')

if __name__ == '__main__':
    db = Database()
    db.create_database("lab2_bd")
    db.create_table("test")
    db.insert_df("test", pd.DataFrame({"name1": [1]}))
    print(db.read_table("test"))
    db.drop_table("test")
    db.drop_database("lab2_bd")

