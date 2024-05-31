import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sqlalchemy.orm import Session

from .data import create_db_connection, load_data, split_data


def convert_bail_amount(data):
    """
    Convert the bail amount column to numeric and rename it.

    :param data: DataFrame containing the data
    :return: DataFrame with the bail amount converted and renamed
    """
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
    def __init__(self, config, judge_filter=None, county_filter=None):
        self.columns_to_normalize = ["median_income", "number_of_households", "population"]
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
        session = Session(self.engine)
        data = load_data(session, self.judge_filter, self.county_filter)

        data = self.preprocess_data(data, self.config.outputs_dir)

        x_column, _y_column, y_bin = separate_features_and_target(data)

        # Fit label encoders on training data
        categorical_features = ['gender', 'ethnicity', 'judge_name', 'county_name']
        for feature in categorical_features:
            self.label_encoders[feature] = LabelEncoder().fit(x_column[feature])
            x_column[feature] = self.label_encoders[feature].transform(x_column[feature])

        x_train, y_train, x_test, y_test = split_data(x_column, y_bin, self.config.outputs_dir)
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
        while any(bin_counts < bin_count_threshold):
            self.num_bins -= 1
            if self.num_bins < bin_count_threshold:
                raise ValueError(
                    "Cannot create bins with at least 2 samples each. Consider increasing the sample size.")
            data["bail_amount_bin"] = pd.cut(data["bail_amount"], bins=self.num_bins, labels=False)
            bin_counts = data["bail_amount_bin"].value_counts()
            logging.info("Adjusted bin counts: %s", bin_counts)

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
            x = x[~y.isna()]
            y = y[~y.isna()]
            y_bin = y_bin[~y.isna()]
            logging.info("Removed %s rows with NaN values in bail_amount.", nan_count)
        x = pd.DataFrame(self.imputer.fit_transform(x), columns=x.columns)
        logging.info("Filled NaN values in features with column medians.")
        return x, y, y_bin

    def preprocess_new_data(self, data):
        """Apply the same preprocessing steps to new data."""
        data = data.copy()

        # Example preprocessing: label encoding for categorical features
        categorical_features = ['gender', 'ethnicity', 'judge_name', 'county_name']
        for feature in categorical_features:
            if feature in self.label_encoders:
                data[feature] = self.label_encoders[feature].transform(data[feature])
            else:
                # Fit encoder on new data if it wasn't fit during training
                self.label_encoders[feature] = LabelEncoder().fit(data[feature])
                data[feature] = self.label_encoders[feature].transform(data[feature])

        # Fill missing values with column medians
        for column in data.columns:
            if data[column].isnull().any():
                data[column].fillna(data[column].median(), inplace=True)

        return data
