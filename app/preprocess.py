import logging
import os
from typing import Tuple

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


def separate_features_and_target(data, bail_binning=False) -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features (X) and target (y) with binning."""

    x = data.drop(columns=["bail_amount"], errors='ignore').copy()  # Make a copy of dataframe

    # Create a new column for binning if requested
    if bail_binning:
        x['bail_amount_bin'] = pd.cut(data['bail_amount'], bins=10, labels=False)
        y = x.pop('bail_amount_bin')
    else:
        y = data['bail_amount']

    return x, y


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


def _handle_missing_values(x, y):
    """Handle missing values in the input data."""

    # Impute missing values in bail_amount (y)
    y.fillna(y.median(), inplace=True)

    # Drop rows where ALL columns in x are NaN
    x = x.dropna(how='all')
    y = y[x.index]  # Align y with the filtered x

    # Separate numerical and categorical columns
    numeric_cols = x.select_dtypes(include=['number'])
    categorical_cols = x.select_dtypes(exclude=['number'])

    if not numeric_cols.empty:
        imputer_numeric = SimpleImputer(strategy='median')
        numeric_cols = pd.DataFrame(imputer_numeric.fit_transform(numeric_cols),
                                    columns=numeric_cols.columns)

    # Impute categorical columns with most frequent value
    if not categorical_cols.empty:
        imputer_categorical = SimpleImputer(strategy='most_frequent')
        categorical_cols = pd.DataFrame(imputer_categorical.fit_transform(categorical_cols),
                                        columns=categorical_cols.columns)

    # Combine imputed data
    x = pd.concat([numeric_cols, categorical_cols], axis=1)

    return x, y


class Preprocessing:
    def __init__(self, config, judge_filter=None, county_filter=None):
        self.columns_to_normalize = ["median_household_income", "known_days_in_custody", "age_at_arrest",
                                     "number_of_households", "population", "prior_misd_cnt", "prior_nonvfo_cnt"
                                     ]
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
        """Load and preprocess data, printing initial values for verification."""

        session = Session(self.engine)
        data = load_data(session, self.judge_filter, self.county_filter)
        self.total_cases = len(data)

        required_columns = ["top_charge_weight_at_arraign", "first_bail_set_cash"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in loaded data.")

        x_column, y = self.preprocess_data(data, self.config.outputs_dir)
        return data, x_column, y

    def prepare_for_training(self, x_column, y):

        x_train, y_train, x_test, y_test = split_data(x_column, y, self.config.outputs_dir)

        self.num_features = x_column.shape[1]
        return x_train, y_train, x_test, y_test

    def preprocess_data(self, data, outputs_dir):
        from .data import save_preprocessed_data

        data = convert_bail_amount(data)
        bail_amount_threshold = data["bail_amount"].quantile(0.99)  # 99th percentile
        data = data[data["bail_amount"] <= bail_amount_threshold]

        data = drop_list_columns(data)

        data, y = _handle_missing_values(data, data["bail_amount"])

        severity_dictionary = {"I": 0.5, "V": 1, "BM": 2, "UM": 2.5, "AM": 3, "EF": 4, "DF": 5, "CF": 6, "BF": 7,
                               "AF": 8}
        if 'top_charge_weight_at_arraign' in data.columns:
            data["top_charge_weight_at_arraign"] = data["top_charge_weight_at_arraign"].map(
                severity_dictionary).fillna(0)

        data = self._encode_categorical_features(data)
        data["bail_amount"] = StandardScaler(with_mean=False).fit_transform(data[["bail_amount"]])
        # scale from 0 to 1
        data["bail_amount"] = data["bail_amount"] / data["bail_amount"].max()

        logging.info("Number of members in each bin: %s", data["top_charge_weight_at_arraign"].value_counts())

        x, y = separate_features_and_target(data, bail_binning=False)

        y.to_csv(os.path.join(outputs_dir, "y.csv"), index=False)

        x = normalize_columns(self.columns_to_normalize, x)

        save_preprocessed_data(x, outputs_dir)
        return x, y

    def _create_bail_bins(self, data, target="bail_amount"):
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

    def preprocess_new_data(self, data):
        """Apply the same preprocessing steps to new data."""
        data = data.copy()

        categorical_features = ['gender', 'ethnicity', 'judge_name', 'county_name', 'top_charge_weight_at_arraign']
        for feature in categorical_features:
            if feature in self.label_encoders:
                data[feature] = self.label_encoders[feature].transform(data[feature])
            else:
                self.label_encoders[feature] = LabelEncoder().fit(data[feature])
                data[feature] = self.label_encoders[feature].transform(data[feature])

        for column in data.columns:
            if data[column].isnull().any():
                data[column].fillna(data[column].median(), inplace=True)

        return data
