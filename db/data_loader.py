import logging
import os

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sqlalchemy import create_engine

from db import Preprocessor


def create_engine_connection(user, password, host, port, dbname):
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(connection_string)


def load_data(engine, query, sql_values):
    logging.info("Loading data from database...")
    data = pd.read_sql(query, engine, params=sql_values)
    logging.info("Data loaded successfully.")
    return data


def filter_data(data, filter_by, filter_value):
    return data[data[filter_by] == filter_value]


def load_and_preprocess_data(outputs_dir, engine, query, sql_values, filter_by, filter_value):
    data = load_data(engine, query, sql_values)
    if filter_by and filter_value:
        data = filter_data(data, filter_by, filter_value)
    preprocessor = Preprocessor()
    x, y, y_bin = preprocessor.preprocess_data(data, outputs_dir)
    x_train, y_train, x_test, y_test = split_data(x, y_bin, outputs_dir)
    return data, x_train, y_train, x_test, y_test


def create_db_connection():
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    host = os.environ.get("DB_HOST")
    port = os.environ.get("DB_PORT")
    dbname = os.environ.get("DB_NAME")
    return create_engine_connection(user, password, host, port, dbname)


def save_preprocessed_data(x, outputs_dir):
    x.to_csv(os.path.join(outputs_dir, 'X.csv'), index=False)


def save_split_data(x_train, x_test, outputs_dir):
    x_train.to_csv(os.path.join(outputs_dir, 'X_train.csv'), index=False)
    x_test.to_csv(os.path.join(outputs_dir, 'X_test.csv'), index=False)


def split_data(x, y_bin, outputs_dir):
    x_train, y_train, x_test, y_test = None, None, None, None
    try:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(x, y_bin):
            x_train = x.loc[train_index]
            y_train = y_bin.loc[train_index]
            x_test = x.loc[test_index]
            y_test = y_bin.loc[test_index]
        logging.info("Data split into training and test sets (stratified).")
        save_split_data(x_train, x_test, outputs_dir)
    except Exception as e:
        logging.error(f"Error splitting data: {e}")
        raise
    return x_train, y_train, x_test, y_test