import logging
import os

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sqlalchemy.orm import Session


def load_data(source: str = None, session: Session = None, file_path: str = None, judge_filter=None,
              county_filter=None) -> pd.DataFrame:
    """
    Load data from the specified source (either 'db' or 'csv').

    Args:
        source (str): The data source, either 'db' or 'csv'.
        session (Session, optional): The SQLAlchemy session to use for the query. Required if source is 'db'.
        file_path (str, optional): The path to the CSV file. Required if source is 'csv'.
        judge_filter (str, optional): The name of the judge to filter by. Defaults to None.
        county_filter (str, optional): The name of the county to filter by. Defaults to None.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """

    from .db.db_actions import load_data_from_db
    from .csv_tools.csv_reader import load_data_from_csv
    if source == 'db':
        if session is None:
            raise ValueError("Session is required when source is 'db'")
        return load_data_from_db(session, judge_filter, county_filter)
    elif source == 'csv':
        return load_data_from_csv(file_path)
    else:
        raise ValueError("Invalid source. Use 'db' or 'csv'.")


# Example usage
# df = load_data('db', session=my_session, judge_filter='Judge A', county_filter='County X')
# df = load_data('csv', file_path='path/to/file.csv')


def save_preprocessed_data(x_data: pd.DataFrame, outputs_dir: str):
    """
    Save preprocessed data to a CSV file.

    :param x_data: DataFrame containing the data.
    :param outputs_dir: Directory to save the CSV file.
    """
    x_data.to_csv(os.path.join(outputs_dir, "X.csv"), index=False)


def save_split_data(x_train: pd.DataFrame, x_test: pd.DataFrame, outputs_dir: str):
    """
    Save the split data to CSV files.

    :param x_train: Training data.
    :param x_test: Test data.
    :param outputs_dir: Directory to save the CSV files.
    """
    x_train.to_csv(os.path.join(outputs_dir, "X_train.csv"), index=False)
    x_test.to_csv(os.path.join(outputs_dir, "X_test.csv"), index=False)


def split_data(x_bin: pd.DataFrame, y_bin: pd.DataFrame, outputs_dir: str):
    """
    Split the data into training and test sets.

    :param x_bin: Feature data.
    :param y_bin: Target data.
    :param outputs_dir: Directory to save the split data.
    :return: x_train, y_train, x_test, y_test
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
