import logging
from typing import Tuple

import pandas as pd
import prettytable as pt
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


def separate_features_and_target(data, bail_binning=False, severity_binning=False) -> Tuple[
    pd.DataFrame, pd.Series, pd.Series]:
    """
    Separate features (X) and target variables (y and y_bin) based on binning flags.

    :param data: DataFrame containing the data
    :param bail_binning: Whether to use binned bail amount as a target
    :param severity_binning: Whether to use binned severity as a target
    :return: features (X), target variables (y and y_bin)
    """

    columns_to_drop = []

    # Conditional dropping of columns based on binning flags
    if not bail_binning:
        columns_to_drop.append("bail_amount_bin") if "bail_amount_bin" in data.columns else None
    if not severity_binning:
        columns_to_drop.extend(
            ["severity_amount", "severity_amount_bin"]) if "severity_amount" in data.columns else None

    # Check if columns exist before dropping (same as before)
    for col in columns_to_drop:
        if col not in data.columns:
            logging.warning(f"Column '{col}' not found in data. Skipping...")
            columns_to_drop.remove(col)

    if columns_to_drop:
        x = data.drop(columns=columns_to_drop)
    else:
        x = data.copy()

    y = data["bail_amount"]  # Use raw values

    # Add handling for severity target (if needed)
    y_severity = None
    if severity_binning:
        y_severity = data["severity_bin"]
    # ... (rest of the code if you want to return severity as a separate target)

    return x, y, y_severity


def normalize_columns(columns_to_normalize, x):
    """
    Normalize the specified columns in the input data.

    :param columns_to_normalize: List of columns to normalize
    :param x: DataFrame containing the features
    :return: DataFrame with normalized columns
    """
    for column in columns_to_normalize:
        if column in x.columns:
            # print numbers of members in column
            x[column] = StandardScaler().fit_transform(x[[column]])
            logging.info("Normalized %s column.", column)
    return x


class Preprocessing:
    def __init__(self, config, judge_filter=None, county_filter=None):
        self.columns_to_normalize = ["median_household_income", "known_days_in_custody", "age_at_arrest",
                                     "number_of_households", "population", "prior_misd_cnt", "prior_nonvfo_cnt",
                                     ]
        self.num_bins = 50
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

        # Check if columns exist in the loaded data
        required_columns = ["top_charge_weight_at_arraign", "first_bail_set_cash"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in loaded data.")

        # Display first 5 rows of relevant columns in a table
        logging.info("First 5 rows of relevant columns:")
        table = pt.PrettyTable()
        table.field_names = required_columns
        for _, row in data.head(5).iterrows():
            table.add_row([row[col] for col in required_columns])

        # Continue with preprocessing as usual
        x_column, y = self.preprocess_data(data, self.config.outputs_dir)
        x_train, y_train, x_test, y_test = split_data(x_column, y, self.config.outputs_dir)

        self.total_cases = len(data)
        self.num_features = x_column.shape[1]
        return data, x_train, y_train, x_test, y_test

    def preprocess_data(self, data, outputs_dir, bail_binning=False, severity_binning=True):
        """
        Preprocess the input data.
        """
        from .data import save_preprocessed_data
        data = convert_bail_amount(data)
        data = drop_list_columns(data)
        bin_dictionary = {"I": 0.5, "V": 1, "BM": 2, "UM": 2.5, "AM": 3, "EF": 4, "DF": 5, "CF": 6, "BF": 7, "AF": 8}
        unique_charges = data['top_charge_weight_at_arraign'].unique()
        unmapped_charges = [charge for charge in unique_charges if charge not in bin_dictionary and pd.notna(charge)]
        if unmapped_charges:
            raise ValueError(f"Unmapped values in 'top_charge_weight_at_arraign': {unique_charges}")
        data = self._encode_categorical_features(data)

        # Map top_charge_weight_at_arraign directly
        data["top_charge_weight_at_arraign"] = data["top_charge_weight_at_arraign"].map(bin_dictionary)

        x = data.drop(columns=["bail_amount"], errors='ignore')
        y = data["bail_amount"]

        x, y = self._handle_missing_values(x, y)
        x = normalize_columns(self.columns_to_normalize, x)
        save_preprocessed_data(x, outputs_dir)
        return x, y

    def _handle_missing_values(self, x, y):
        """Handle missing values in the input data."""
        # First impute missing values in y
        y.fillna(y.median(), inplace=True)

        # Drop rows where ALL columns in x are NaN
        x = x.dropna(how='all')
        y = y[x.index]  # Align y with the filtered x

        # Impute remaining missing values in x
        x = pd.DataFrame(self.imputer.fit_transform(x), columns=x.columns)

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

    def _handle_missing_values(self, x, y):
        """
        Handle missing values in the input data.
        """
        # Impute missing values in bail_amount
        y.fillna(y.median(), inplace=True)

        # Drop rows where ALL columns in x are NaN
        x = x.dropna(how='all')
        y = y[x.index]  # Align y with the filtered x

        # Exclude top_charge_weight_at_arraign from imputation
        x_to_impute = x.drop(columns=['top_charge_weight_at_arraign'], errors='ignore')

        # Impute remaining missing values in x (excluding 'top_charge_weight_at_arraign')
        x_to_impute = pd.DataFrame(self.imputer.fit_transform(x_to_impute), columns=x_to_impute.columns)

        # Concatenate the imputed features back with 'top_charge_weight_at_arraign'
        x = pd.concat([x_to_impute, x[['top_charge_weight_at_arraign']]], axis=1)

        return x, y

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
