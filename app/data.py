import logging
import os

import pandas as pd
from alembic import command
from alembic.config import Config
from sklearn.model_selection import StratifiedShuffleSplit
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from .db_types import County, Judge
from .env import BASE_QUERY


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

    # Start with a basic query
    # Start with a basic query
    query = BASE_QUERY

    # Apply judge filter if provided
    if judge_filter:
        logging.info(f"Applying judge filter: {judge_filter}")
        query = query.where(Judge.judge_name == judge_filter)

    # Apply county filter if provided
    if county_filter:
        logging.info(f"Applying county filter: {county_filter}")
        query = query.where(County.county_name == county_filter)

    # Execute the query
    results = session.execute(query).fetchall()

    # Convert results to a pandas DataFrame
    df = pd.DataFrame(results, columns=[
        "gender", "ethnicity", "race", "age_at_arrest", "known_days_in_custody",
        "top_charge_at_arraign", "first_bail_set_cash", "prior_vfo_cnt",
        "prior_nonvfo_cnt", "prior_misd_cnt", "pend_nonvfo", "pend_misd",
        "pend_vfo", "judge_name", "median_household_income", "county_name",
    ])

    logging.info("Data loading complete.")
    return df


def save_data(session: Session, judge_filter=None, county_filter=None):
    """
    Save data to a db file.

    Args:
        session (Session): The SQLAlchemy session to use for the query.
        judge_filter (str, optional): The name of the judge to filter by. Defaults to None.
        county_filter (str, optional): The name of the county to filter by. Defaults to None.
    """
    data = load_data(session, judge_filter, county_filter)
    # write filters to Results table

    logging.info("Data saved to 'data.csv'.")


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
