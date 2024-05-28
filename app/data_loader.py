import logging
import os
from dataclasses import dataclass

import pandas as pd
from alembic import command
from alembic.config import Config
from sklearn.model_selection import StratifiedShuffleSplit
from sqlalchemy import Engine, create_engine, select, Numeric, cast
from sqlalchemy.orm import Session

from app.declarations import Case, County, Court, District, Judge, NYIncome, Race, Representation
from app.sql_values import SQLValues

# Load the Alembic configuration
alembic_cfg = Config("alembic.ini")


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
    command.upgrade(alembic_cfg, "head")


def load_data(session: Session, sql_values: SQLValues):
    logging.info("Loading data from database...")
    bail_threshold = 800000

    stmt = (
        select(
            Case.gender,
            Case.ethnicity,
            Race.race,
            Case.age_at_arrest,
            Case.known_days_in_custody,
            Case.top_charge_at_arraign,
            Case.first_bail_set_cash,
            Case.prior_vfo_cnt,
            Case.prior_nonvfo_cnt,
            Case.prior_misd_cnt,
            Case.pend_nonvfo,
            Case.pend_misd,
            Case.pend_vfo,
            Judge.judge_name,
            NYIncome.median_household_income
        )
        .join(Race, Case.race_id == Race.race_uuid)
        .join(County, Case.county_id == County.county_uuid)
        .join(NYIncome, County.county_name == NYIncome.county)
        .join(Judge, Case.judge_id == Judge.judge_uuid)
        .join(District, Case.district_id == District.district_uuid)
        .join(Court, Case.court_id == Court.court_uuid)
        .join(Representation, Case.representation_id == Representation.representation_uuid)
        .where(
            Case.first_bail_set_cash.isnot(None),
            cast(Case.first_bail_set_cash, Numeric) < bail_threshold,
            cast(Case.first_bail_set_cash, Numeric) > 1
        )
    )

    if sql_values["judge_names"]:
        stmt = stmt.where(Judge.judge_name.in_(sql_values.judge_names))

    if sql_values["county_names"]:
        stmt = stmt.where(County.county_name.in_(sql_values.county_names))

    stmt = stmt.limit(sql_values["limit"])

    results = session.execute(stmt).all()
    logging.info("Data loaded successfully.")
    return pd.DataFrame(results, columns=[
        "gender", "ethnicity", "race", "age_at_arrest", "known_days_in_custody",
        "top_charge_at_arraign", "first_bail_set_cash", "prior_vfo_cnt",
        "prior_nonvfo_cnt", "prior_misd_cnt", "pend_nonvfo", "pend_misd",
        "pend_vfo", "judge_name", "median_household_income"
    ])


def filter_data(data: pd.DataFrame, filter_by: str, filter_value: str) -> pd.DataFrame:
    """
    Filter data based on a column and value.

    :param data: DataFrame containing the data.
    :param filter_by: Column to filter by.
    :param filter_value: Value to filter by.
    :return: Filtered DataFrame.
    """
    return data[data[filter_by] == filter_value]


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


@dataclass
class DataLoaderConfig:
    engine: Engine
    query: str
    sql_values: SQLValues
    outputs_dir: str


@dataclass
class FilterConfig:
    filter_by: str
    filter_value: str
