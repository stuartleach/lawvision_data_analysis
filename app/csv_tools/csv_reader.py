# read a csv from a given directory
# convert that csv to a dataframe
# return the dataframe

import os

import pandas as pd


def read_csv(directory: str) -> pd.DataFrame:
    """
    Read a CSV file from the given directory.

    :param directory: Directory containing the CSV file.
    :return: DataFrame containing the data.
    """
    file_path = os.path.join(directory, "data.csv")
    return pd.read_csv(file_path)
# Path: app/csv_tools/csv_writer.py
