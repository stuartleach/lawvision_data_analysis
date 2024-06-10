import logging
import os

import pandas as pd
from alembic import command
from alembic.config import Config
from sklearn.model_selection import StratifiedShuffleSplit
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from .db_types import County, Judge, Result
from .env import BASE_QUERY
from .train import ResultObject
from .utils import tablify


def create_engine_connection(user: str, password: str, host: str, port: str, dbname: str) -> Engine:
    """
    Create a connection to the database.

    :param user: Database user.
    :param password: Database password.
    :param host: Database host.
    :param port: Database port.
    :param dbname: Database name.
    :return: SQLAlchemy Engine.
    """
    connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(connection_string)


def run_migrations():
    """
    Run Alembic migrations.
    """
    # Load the Alembic configuration
    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")


def load_data(session: Session, judge_filter=None, county_filter=None) -> pd.DataFrame:
    """
    Load data from the database with optional filters for judge and county.

    Args:
        session (Session): The SQLAlchemy session to use for the query.
        judge_filter (str, optional): The name of the judge to filter by. Defaults to None.
        county_filter (str, optional): The name of the county to filter by. Defaults to None.

    Returns:
        pd.DataFrame: The queried data as a pandas DataFrame.
    """
    logging.info("Loading data from database...")

    query = BASE_QUERY

    if judge_filter:
        logging.info(f"Applying judge filter: {judge_filter}")
        query = query.where(Judge.judge_name == judge_filter)
    if county_filter:
        logging.info(f"Applying county filter: {county_filter}")
        query = query.where(County.county_name == county_filter)

    results = session.execute(query)  # No need for fetchall() here

    data = pd.DataFrame(results.fetchall(), columns=results.keys())
    tablify(data.head(10))
    if judge_filter:
        data = data.drop(columns=['judge_name'])
    if county_filter:
        # data = data.drop(columns=['county_name', 'median_household_income'])
        print(county_filter)

    return data


def save_data(session: Session, result_object: ResultObject):
    """
    Save data to the results table.

    Args:
        session (Session): The SQLAlchemy session to use for the query.
        dataframe (pandas.DataFrame): The data to be saved to the results table.
        judge_filter (str, optional): The name of the judge to filter by. Defaults to None.
        county_filter (str, optional): The name of the county to filter by. Defaults to None.
        :param result_object:
        :param session:

    """

    model_params = result_object.model_params
    average_bail_amount = result_object.average_bail_amount
    r_squared = result_object.r_squared
    mean_squared_error = result_object.mean_squared_error
    model_type = result_object.model_type
    dataframe = result_object.dataframe
    judge_filter = result_object.judge_filter
    county_filter = result_object.county_filter

    tablify(dataframe.head(10))

    dataframe = dataframe.to_dict(orient="records")

    if judge_filter:
        model_target_type = 'judge_name'
        model_target = judge_filter
    elif county_filter:
        model_target = county_filter
        model_target_type = 'county_name'
    else:
        model_target = 'baseline'
        model_target_type = 'baseline'

    new_result = Result()

    for row in dataframe:
        row_feature = row.get('Feature')
        row_importance = row.get('Importance')
        new_result[row_feature] = row_importance  #
        new_result.__setattr__(row_feature + "_importance", row_importance)

    new_result.__setattr__('model_params', model_params)
    new_result.__setattr__('average_bail_amount', average_bail_amount)
    new_result.__setattr__('r_squared', r_squared)
    new_result.__setattr__('mean_squared_error', mean_squared_error)
    new_result.__setattr__('model_type', model_type)
    new_result.__setattr__('model_target_type', model_target_type)
    new_result.__setattr__('model_target', model_target)
    new_result.model_target = model_target  #

    session.add(new_result)

    try:
        session.commit()
        logging.info("Data saved to the results table.")
    except Exception as e:
        session.rollback()
        logging.error(f"Error saving data to the results table: {e}")


def create_db_connection() -> Engine:
    """
    Create a connection to the database.

    :return: SQLAlchemy Engine.
    """
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    host = os.environ.get("DB_HOST")
    port = os.environ.get("DB_PORT")
    dbname = os.environ.get("DB_NAME")
    return create_engine_connection(user, password, host, port, dbname)


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


def split_data(x, y, outputs_dir, test_size=0.2, random_state=None):
    """Split the data into training and test sets using stratified sampling on binned bail amounts."""
    x_train, x_test, y_train, y_test = None, None, None, None

    y_bins = pd.cut(y.copy(), bins=20, labels=False)

    try:
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        for train_index, test_index in split.split(x, y_bins):
            x_train, x_test = x.loc[train_index], x.loc[test_index]
            y_train, y_test = y.loc[train_index], y.loc[test_index]

        logging.info(f"Data split into training and test sets (stratified on bins) with test_size={test_size}.")
    except Exception as e:
        logging.error("An unexpected error occurred during data splitting: %s", e)
        raise

    save_split_data(x_train, x_test, outputs_dir)
    return x_train, y_train, x_test, y_test
