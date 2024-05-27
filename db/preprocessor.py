import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from db import save_preprocessed_data


def convert_bail_amount(data):
    data['first_bail_set_cash'] = pd.to_numeric(data['first_bail_set_cash'], errors='coerce')
    data.rename(columns={'first_bail_set_cash': 'bail_amount'}, inplace=True)
    logging.info(f"First few rows of data: {data.head()}")
    return data


def drop_list_columns(data):
    columns_to_drop = [column for column in data.columns if data[column].apply(lambda x: isinstance(x, list)).any()]
    if columns_to_drop:
        logging.warning(f"Columns {columns_to_drop} contain list values and will be dropped.")
        data = data.drop(columns=columns_to_drop)
    return data


def separate_features_and_target(data):
    x = data.drop(columns=['bail_amount', 'bail_amount_bin'])
    y = data['bail_amount']
    y_bin = data['bail_amount_bin']
    return x, y, y_bin


def normalize_columns(columns_to_normalize, x):
    for column in columns_to_normalize:
        if column in x.columns:
            x[column] = StandardScaler().fit_transform(x[[column]])
            logging.info(f"Normalized {column} column.")
    return x


class Preprocessor:
    def __init__(self, columns_to_normalize=None, num_bins=10, imputation_strategy='median', encoding_strategy='label'):
        self.columns_to_normalize = columns_to_normalize or ['median_income', 'number_of_households', 'population']
        self.num_bins = num_bins
        self.imputation_strategy = imputation_strategy
        self.encoding_strategy = encoding_strategy
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy=imputation_strategy)
        self.scaler = StandardScaler()

    def preprocess_data(self, data, outputs_dir):
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
        data['bail_amount_bin'] = pd.cut(data['bail_amount'], bins=self.num_bins, labels=False)
        bin_counts = data['bail_amount_bin'].value_counts()
        logging.info(f"Bin counts before adjustment: {bin_counts}")
        while any(bin_counts < 2):
            self.num_bins -= 1
            if self.num_bins < 2:
                raise ValueError(
                    "Cannot create bins with at least 2 samples each. Consider increasing the sample size.")
            data['bail_amount_bin'] = pd.cut(data['bail_amount'], bins=self.num_bins, labels=False)
            bin_counts = data['bail_amount_bin'].value_counts()
            logging.info(f"Adjusted bin counts: {bin_counts}")

    def _encode_categorical_features(self, data):
        if self.encoding_strategy == 'label':
            for column in data.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column].astype(str))
                self.label_encoders[column] = le
            logging.info("Categorical features encoded with Label Encoding.")
        elif self.encoding_strategy == 'onehot':
            data = pd.get_dummies(data, columns=data.select_dtypes(include=['object']).columns)
            logging.info("Categorical features encoded with One-Hot Encoding.")
        return data

    def _handle_missing_values(self, x, y, y_bin):
        nan_count = y.isna().sum()
        logging.info(f"Number of NaN values in bail_amount: {nan_count}")
        if nan_count > 0:
            x = x[~y.isna()]
            y = y[~y.isna()]
            y_bin = y_bin[~y.isna()]
            logging.info(f"Removed {nan_count} rows with NaN values in bail_amount.")
        x = pd.DataFrame(self.imputer.fit_transform(x), columns=x.columns)
        logging.info("Filled NaN values in features with column medians.")
        return x, y, y_bin
