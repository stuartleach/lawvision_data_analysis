# data_loader.py

import logging

import pandas as pd
from sqlalchemy import create_engine


def create_engine_connection(user, password, host, port, dbname):
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(connection_string)


def load_data(engine, query, sql_values):
    logging.info("Loading data from database...")
    data = pd.read_sql(query, engine, params=sql_values)
    logging.info("Data loaded successfully.")
    return data
