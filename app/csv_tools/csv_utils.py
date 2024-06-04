import logging

import pandas as pd

BASEDIR = "../../sources/exports"

logging.basicConfig(level=logging.INFO)


def read_columns_as_list(file_path: str) -> list:
    """
    Read column names
    """
    # logging.info(f"Reading column {column_name} from CSV file: {file_path}")
    df = pd.read_csv(file_path)
    column_data = df.head()
    # logging.info(f"Column {column_name} read successfully.")
    return column_data
