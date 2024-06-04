import logging
import os

import pandas as pd

from app import COLUMNS_OF_INTEREST

BASEDIR = "../../sources/exports"

logging.basicConfig(level=logging.INFO)


def read_csv(directory: str = BASEDIR) -> pd.DataFrame:
    """
    Read a CSV file from the given directory.

    :param directory: Directory containing the CSV file.
    :return: DataFrame containing the data.
    """
    file_path = os.path.join(directory, "data.csv")
    return pd.read_csv(file_path)


def load_data_from_csv(file_path: str = "../../sources/exports/merged_data.csv") -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The data from the CSV file as a pandas DataFrame.
    """
    logging.info(f"Loading data from CSV file: {file_path}")
    dtype = {
        11: 'str',  # Change 'str' to the appropriate type if known
        99: 'str',  # Change 'str' to the appropriate type if known
    }
    df = pd.read_csv(file_path, dtype=dtype)
    # df = pd.DataFrame(df, columns=[
    #     "gender", "ethnicity", "race", "age_at_arrest", "known_days_in_custody",
    #     "top_charge_at_arraign", "first_bail_set_cash", "prior_vfo_cnt",
    #     "prior_nonvfo_cnt", "prior_misd_cnt", "pend_nonvfo", "pend_misd",
    #     "pend_vfo", "judge_name", "median_household_income", "county_name",
    # ])
    df = pd.DataFrame(df, columns=COLUMNS_OF_INTEREST)
    logging.info("Data loading from CSV complete.")
    return df
