import logging

import pandas as pd

BASEDIR = "../../sources/exports"

logging.basicConfig(level=logging.INFO)


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The data from the CSV file as a pandas DataFrame.
    """
    logging.info(f"Loading data from CSV file: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)
    logging.info(f"Data loading from {file_path} complete. Shape: {df.shape}")
    return df


# Load data into DataFrames
cases_df = load_data_from_csv(f'{BASEDIR}/cases_data.csv')
counties_df = load_data_from_csv(f'{BASEDIR}/counties_data.csv')
courts_df = load_data_from_csv(f'{BASEDIR}/courts_data.csv')
judges_df = load_data_from_csv(f'{BASEDIR}/judges_data.csv')
races_df = load_data_from_csv(f'{BASEDIR}/races_data.csv')
representations_df = load_data_from_csv(f'{BASEDIR}/representation_data.csv')

# Print column names to verify keys
logging.info(f"cases_df columns: {cases_df.columns.tolist()}")
logging.info(f"races_df columns: {races_df.columns.tolist()}")
logging.info(f"courts_df columns: {courts_df.columns.tolist()}")
logging.info(f"judges_df columns: {judges_df.columns.tolist()}")
logging.info(f"representations_df columns: {representations_df.columns.tolist()}")
logging.info(f"counties_df columns: {counties_df.columns.tolist()}")

# Merge DataFrames
merged_df = pd.merge(cases_df, races_df, how='left', left_on='race_id', right_on='race_uuid')
logging.info(f"Merged cases with races. Shape: {merged_df.shape}")

merged_df = pd.merge(merged_df, courts_df, how='left', left_on='court_id', right_on='court_uuid',
                     suffixes=('_cases', '_courts'))
logging.info(f"Merged with courts. Shape: {merged_df.shape}")

merged_df = pd.merge(merged_df, judges_df, how='left', left_on='judge_id', right_on='judge_uuid',
                     suffixes=('', '_judges'))
logging.info(f"Merged with judges. Shape: {merged_df.shape}")

merged_df = pd.merge(merged_df, representations_df, how='left', left_on='representation_id',
                     right_on='representation_uuid', suffixes=('', '_reps'))
logging.info(f"Merged with representations. Shape: {merged_df.shape}")

merged_df = pd.merge(merged_df, counties_df, how='left', left_on='county_id', right_on='county_uuid',
                     suffixes=('', '_counties'))
logging.info(f"Merged with counties. Shape: {merged_df.shape}")

result_df = None

# Select relevant columns (if needed)
# result_df = merged_df[
#     ['internal_case_id', 'race_id', 'race_uuid', 'race', 'county_id', 'county_name', 'judge_name', 'court_name',
#      'representation_type', 'median_income']]

result_df = result_df if result_df else merged_df

# count how many rows are in the merged dataframec
logging.info(f"Rows in merged data: {result_df.shape[0]}")

# Export the final DataFrame to CSV
result_df.to_csv(f'{BASEDIR}/merged_data_1.csv')
logging.info(f"Exported merged data to {BASEDIR}/merged_data_1.csv. Shape: {result_df.shape}")
