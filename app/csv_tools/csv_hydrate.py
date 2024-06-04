import logging

from app.preprocess import load_and_merge_data, load_data_from_csv

BASEDIR = "../../sources/exports"

logging.basicConfig(level=logging.INFO)


def hydrate_data():
    # Load data into DataFrames
    load_and_merge_data()


# print all columns from the merged data
def print_columns():
    df = load_data_from_csv(f'{BASEDIR}/merged_data.csv')
    logging.info(f"Columns: {len(df.columns)}")
    logging.info(f"Shape: {df.shape}")
    return df

# print_columns()
# hydrate_data()
