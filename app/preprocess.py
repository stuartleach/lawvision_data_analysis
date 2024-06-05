import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy.orm import Session

from . import COLUMNS_OF_INTEREST
from .data import split_data, load_data
from .db.db_actions import create_db_connection


def load_data_from_csv(file_path: str, dtype_dict=None) -> pd.DataFrame:
    """
    Load data from a CSV file with specified data types.

    Args:
        file_path (str): The path to the CSV file.
        dtype_dict (dict): A dictionary specifying data types for columns.

    Returns:
        pd.DataFrame: The data from the CSV file as a pandas DataFrame.
    """
    logging.info(f"Loading data from CSV file: {file_path}")
    df = pd.read_csv(file_path, low_memory=False, dtype=dtype_dict)
    logging.info(f"Data loading from {file_path} complete. Shape: {df.shape}")
    return df


def convert_bail_amount(data):
    data["first_bail_set_cash"] = pd.to_numeric(data["first_bail_set_cash"], errors="coerce")
    data.rename(columns={"first_bail_set_cash": "bail_amount"}, inplace=True)
    logging.info("First few rows of data: %s", data.head())
    return data


def drop_list_columns(data):
    """
    Drop columns that contain list values.

    :param data: DataFrame containing the data
    :return: DataFrame with list columns dropped
    """
    columns_to_drop = [column for column in data.columns if data[column].apply(lambda x: isinstance(x, list)).any()]
    if columns_to_drop:
        logging.warning("Columns %s contain list values and will be dropped.", columns_to_drop)
        data = data.drop(columns=columns_to_drop)
    return data


def separate_features_and_target(data):
    """
    Separate features and target variables.

    :param data: DataFrame containing the data
    :return: DataFrames for features (x), target (y), and binned target (y_bin)
    """
    if 'bail_amount' not in data.columns or 'bail_amount_bin' not in data.columns:
        raise KeyError("'bail_amount' or 'bail_amount_bin' column not found in data.")

    x = data.drop(columns=["bail_amount", "bail_amount_bin"])
    y = data["bail_amount"]
    y_bin = data["bail_amount_bin"]
    return x, y, y_bin


def normalize_columns(columns_to_normalize, x):
    """
    Normalize the specified columns in the input data.

    :param columns_to_normalize: List of columns to normalize
    :param x: DataFrame containing the features
    :return: DataFrame with normalized columns
    """
    for column in columns_to_normalize:
        if column in x.columns:
            x[column] = StandardScaler().fit_transform(x[[column]])
            logging.info("Normalized %s column.", column)
    return x


class Preprocessing:
    def __init__(self, source, config, judge_filter=None, county_filter=None):
        self.columns_to_normalize = ["median_income", "number_of_households", "population"]
        self.filepath = "sources/exports/merged_data.csv"
        self.outputs_dir = "outputs"
        self.source = source if source in ["db", "csv"] else "csv"
        self.num_bins = 10
        self.imputation_strategy = "median"
        self.encoding_strategy = "label"
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy=self.imputation_strategy)
        self.scaler = StandardScaler()
        self.config = config
        self.total_cases = 0
        self.num_features = 0
        self.engine = create_db_connection()
        self.judge_filter = judge_filter
        self.county_filter = county_filter

    def load_and_preprocess_data(self):
        """Load and preprocess data."""
        session = Session(self.engine) if self.source == "db" else None
        data = load_data(self.source, session, self.filepath, self.judge_filter, self.county_filter)

        x, y, y_bin = self.preprocess_data(data, self.outputs_dir)

        x_column, _y_column, y_bin = separate_features_and_target(data)

        # Fit label encoders on training data
        categorical_features = ['gender', 'ethnicity', 'judge_name', 'county_name']
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder().fit(x_column[feature])
            x_column[feature] = self.label_encoders[feature].transform(x_column[feature])

        x_train, y_train, x_test, y_test = split_data(x_column, y_bin, self.outputs_dir)
        self.total_cases = len(data)
        self.num_features = x_column.shape[1]
        return data, x_train, y_train, x_test, y_test

    def preprocess_data(self, data, outputs_dir):
        """
        Preprocess the input data.

        :param data: DataFrame containing the data
        :param outputs_dir: Directory to save preprocessed data
        :return: DataFrames for features (x), target (y), and binned target (y_bin)
        """
        from .data import save_preprocessed_data
        data = convert_bail_amount(data)
        self._create_bins(data)
        data = drop_list_columns(data)
        data = self._encode_categorical_features(data)
        x, y, y_bin = separate_features_and_target(data)
        x, y, y_bin = self._handle_missing_values(x, y, y_bin)
        x = normalize_columns(self.columns_to_normalize, x)
        save_preprocessed_data(x, outputs_dir)
        return x, y, y_bin

    def _create_bins(self, data):
        """
        Create bins for the bail amount column.

        :param data: DataFrame containing the data
        :return: None
        """
        data["bail_amount_bin"] = pd.cut(data["bail_amount"], bins=self.num_bins, labels=False)
        bin_counts = data["bail_amount_bin"].value_counts()
        logging.info("Bin counts before adjustment: %s", bin_counts)
        bin_count_threshold = 2
        min_bins = 3  # Minimum number of bins to consider

        while any(bin_counts < bin_count_threshold):
            if self.num_bins <= min_bins:
                logging.warning("Too few bins to meet the threshold. Adjusting to minimum bins.")
                data["bail_amount_bin"] = pd.cut(data["bail_amount"], bins=min_bins, labels=False)
                break
            self.num_bins -= 1
            data["bail_amount_bin"] = pd.cut(data["bail_amount"], bins=self.num_bins, labels=False)
            bin_counts = data["bail_amount_bin"].value_counts()
            logging.info("Adjusted bin counts: %s", bin_counts)

        if self.num_bins < min_bins:
            logging.error("Cannot create bins with at least 2 samples each. Consider increasing the sample size.")
            raise ValueError("Cannot create bins with at least 2 samples each. Consider increasing the sample size.")

    def _encode_categorical_features(self, data):
        """
        Encode categorical features using the specified strategy.

        :param data: DataFrame containing the data
        :return: DataFrame with encoded categorical features
        """
        if self.encoding_strategy == "label":
            for column in data.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column].astype(str))
                self.label_encoders[column] = le
            logging.info("Categorical features encoded with Label Encoding.")
        elif self.encoding_strategy == "onehot":
            data = pd.get_dummies(data, columns=data.select_dtypes(include=["object"]).columns)
            logging.info("Categorical features encoded with One-Hot Encoding.")
        return data

    def _handle_missing_values(self, x, y, y_bin):
        """
        Handle missing values in the input data.

        :param x: DataFrame containing the features
        :param y: DataFrame containing the target variable
        :param y_bin: DataFrame containing the binned target variable
        :return: DataFrames for features (x), target (y), and binned target (y_bin) with missing values handled
        """
        nan_count = y.isna().sum()
        logging.info("Number of NaN values in bail_amount: %s", nan_count)
        if nan_count > 0:
            valid_indices = ~y.isna()
            x = x[valid_indices]
            y = y[valid_indices]
            y_bin = y_bin[valid_indices]
            logging.info("Removed %s rows with NaN values in bail_amount.", nan_count)
        x = pd.DataFrame(self.imputer.fit_transform(x), columns=x.columns, index=x.index)
        logging.info("Filled NaN values in features with column medians.")
        return x, y, y_bin

        # In your Preprocessing class

    def preprocess_new_data(self, data):
        data = data.copy()[COLUMNS_OF_INTEREST]  # Ensure same columns as training

        # Apply label encoding consistently
        for feature, encoder in self.label_encoders.items():
            if feature in data.columns:
                data[feature] = encoder.transform(data[feature])

                # Handle missing values or columns
        data = data.fillna(self.imputer.statistics_)  # Impute with pre-fitted imputer
        missing_cols = set(self.imputer.feature_names_in_) - set(data.columns)
        for col in missing_cols:  # Add missing columns with imputed values
            data[col] = self.imputer.statistics_[self.imputer.feature_names_in_ == col][0]

        # Normalize consistently (if applicable)
        # data[self.columns_to_normalize] = self.scaler.transform(data[self.columns_to_normalize])

        return data


def load_and_merge_data(base_dir="../../sources/exports"):
    dtype_dict = {
        'first_bail_set_cash': 'float64'
    }

    cases_df = load_data_from_csv(f'{base_dir}/cases_data.csv', dtype_dict=dtype_dict)
    counties_df = load_data_from_csv(f'{base_dir}/counties_data.csv')
    courts_df = load_data_from_csv(f'{base_dir}/courts_data.csv')
    judges_df = load_data_from_csv(f'{base_dir}/judges_data.csv')
    races_df = load_data_from_csv(f'{base_dir}/races_data.csv')
    representations_df = load_data_from_csv(f'{base_dir}/representation_data.csv')

    logging.info(f"cases_df columns: {cases_df.columns.tolist()}")
    logging.info(f"races_df columns: {races_df.columns.tolist()}")
    logging.info(f"courts_df columns: {courts_df.columns.tolist()}")
    logging.info(f"judges_df columns: {judges_df.columns.tolist()}")
    logging.info(f"representations_df columns: {representations_df.columns.tolist()}")
    logging.info(f"counties_df columns: {counties_df.columns.tolist()}")

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

    # result_df = merged_df[
    #     ['internal_case_id', 'race_id', 'race_uuid', 'race', 'county_id', 'county_name', 'judge_name', 'court_name',
    #      'representation_type', 'median_income']]

    result_df = merged_df[COLUMNS_OF_INTEREST]

    result_df.to_csv(f'{base_dir}/merged_data.csv', index=False)
    logging.info(f"Exported merged data to {base_dir}/merged_data.csv. Shape: {result_df.shape}")


# Updated `load_data_from_csv` function to include `dtype_dict`
def load_data_from_csv(file_path: str, dtype_dict=None) -> pd.DataFrame:
    logging.info(f"Loading data from CSV file: {file_path}")
    df = pd.read_csv(file_path, low_memory=False, dtype=dtype_dict)
    logging.info(f"Data loading from {file_path} complete. Shape: {df.shape}")
    return df
