import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from app.data_loader import save_preprocessed_data


def convert_bail_amount(data):
    """
    Convert the bail amount column to numeric and rename it.

    :param data:
    :return:
    """
    data["first_bail_set_cash"] = pd.to_numeric(
        data["first_bail_set_cash"], errors="coerce"
    )
    data.rename(columns={"first_bail_set_cash": "bail_amount"}, inplace=True)
    logging.info("First few rows of data: %s", data.head())
    return data


def drop_list_columns(data):
    """
    Drop columns that contain list values.

    :param data:
    :return:
    """
    columns_to_drop = [
        column
        for column in data.columns
        if data[column].apply(lambda x: isinstance(x, list)).any()
    ]
    if columns_to_drop:
        logging.warning(
            "Columns %s contain list values and will be dropped.", columns_to_drop
        )
        data = data.drop(columns=columns_to_drop)
    return data


def separate_features_and_target(data):
    """
    Separate features and target variables.

    :param data:
    :return:
    """
    x = data.drop(columns=["bail_amount", "bail_amount_bin"])
    y = data["bail_amount"]
    y_bin = data["bail_amount_bin"]
    return x, y, y_bin


def normalize_columns(columns_to_normalize, x):
    """
    Normalize the specified columns in the input data.

    :param columns_to_normalize:
    :param x:
    :return:
    """
    for column in columns_to_normalize:
        if column in x.columns:
            x[column] = StandardScaler().fit_transform(x[[column]])
            logging.info("Normalized %s column.", column)
    return x


class Preprocessor:
    def __init__(
            self,
            columns_to_normalize=None,
            num_bins=10,
            imputation_strategy="median",
            encoding_strategy="label",
    ):
        self.columns_to_normalize = columns_to_normalize or [
            "median_income",
            "number_of_households",
            "population",
        ]
        self.num_bins = num_bins
        self.imputation_strategy = imputation_strategy
        self.encoding_strategy = encoding_strategy
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy=imputation_strategy)
        self.scaler = StandardScaler()

    def preprocess_data(self, data, outputs_dir):
        """
        Preprocess the input data.

        :param data:
        :param outputs_dir:
        :return:
        """
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
        :param data:
        :return:
        """
        data["bail_amount_bin"] = pd.cut(
            data["bail_amount"], bins=self.num_bins, labels=False
        )
        bin_counts = data["bail_amount_bin"].value_counts()
        logging.info("Bin counts before adjustment: %s", bin_counts)
        bin_count_threshold = 2
        while any(bin_counts < bin_count_threshold):
            self.num_bins -= 1
            if self.num_bins < bin_count_threshold:
                raise ValueError(
                    "Cannot create bins with at least 2 samples each. Consider "
                    "increasing the sample size."
                )
            data["bail_amount_bin"] = pd.cut(
                data["bail_amount"], bins=self.num_bins, labels=False
            )
            bin_counts = data["bail_amount_bin"].value_counts()
            logging.info("Adjusted bin counts: %s", bin_counts)

    def _encode_categorical_features(self, data):
        """
        Encode categorical features using the specified strategy.

        :param data:
        :return:
        """
        if self.encoding_strategy == "label":
            for column in data.select_dtypes(include=["object"]).columns:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column].astype(str))
                self.label_encoders[column] = le
            logging.info("Categorical features encoded with Label Encoding.")
        elif self.encoding_strategy == "onehot":
            data = pd.get_dummies(
                data, columns=data.select_dtypes(include=["object"]).columns
            )
            logging.info("Categorical features encoded with One-Hot Encoding.")
        return data

    def _handle_missing_values(self, x, y, y_bin):
        """
        Handle missing values in the input data.

        :param x:
        :param y:
        :param y_bin:
        :return:
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
