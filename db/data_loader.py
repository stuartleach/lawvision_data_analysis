import logging
import os
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sqlalchemy import create_engine


def create_engine_connection(user, password, host, port, dbname):
    """Create a connection to the database.
    :param user:
    :param password:
    :param host:
    :param port:
    :param dbname:
    :return:
    """
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(connection_string)


def load_data(engine, query, sql_values):
    """Load data from the database.
    :param engine:
    :param query:
    :param sql_values:
    :return:
    """
    logging.info("Loading data from database...")
    data = pd.read_sql(query, engine, params=sql_values)
    logging.info("Data loaded successfully.")
    return data


def filter_data(data, filter_by, filter_value):
    """Filter data based on a column and value.
    :param data:
    :param filter_by:
    :param filter_value:
    :return:
    """
    return data[data[filter_by] == filter_value]


def create_db_connection():
    """Create a connection to the database.
    :return:
    """
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    host = os.environ.get("DB_HOST")
    port = os.environ.get("DB_PORT")
    dbname = os.environ.get("DB_NAME")
    return create_engine_connection(user, password, host, port, dbname)


def save_preprocessed_data(x_data, outputs_dir):
    """Save preprocessed data to a CSV file.
    :param x_data:
    :param outputs_dir:
    :return:
    """
    x_data.to_csv(os.path.join(outputs_dir, "X.csv"), index=False)


def save_split_data(x_train, x_test, outputs_dir):
    """Save the split data to CSV files.
    :param x_train:
    :param x_test:
    :param outputs_dir:
    :return:
    """
    x_train.to_csv(os.path.join(outputs_dir, "X_train.csv"), index=False)
    x_test.to_csv(os.path.join(outputs_dir, "X_test.csv"), index=False)


def split_data(x_bin, y_bin, outputs_dir):
    """Split the data into training and test sets.
    :param x_bin:
    :param y_bin:
    :param outputs_dir:
    :return:
    """
    x_train, y_train, x_test, y_test = None, None, None, None
    try:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index in split.split(x_bin, y_bin):
            x_train = x_bin.loc[train_index]
            y_train = y_bin.loc[train_index]
            x_test = x_bin.loc[test_index]
            y_test = y_bin.loc[test_index]
        logging.info("Data split into training and test sets (stratified).")
        save_split_data(x_train, x_test, outputs_dir)
    except Exception as exception:
        logging.error("Error splitting data: %s", exception)
        raise
    return x_train, y_train, x_test, y_test


@dataclass
class DataLoaderConfig:
    engine: any
    query: str
    sql_values: dict
    outputs_dir: str


@dataclass
class FilterConfig:
    filter_by: str
    filter_value: str
