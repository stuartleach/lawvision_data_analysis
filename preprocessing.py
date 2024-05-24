# preprocessing.py

import logging
import os

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess_data(data, outputs_dir):
    # Convert first_bail_set_cash to numeric
    data['first_bail_set_cash'] = pd.to_numeric(data['first_bail_set_cash'], errors='coerce')
    data.rename(columns={'first_bail_set_cash': 'bail_amount'}, inplace=True)
    logging.info(f"First few rows of data: {data.head()}")

    # Create bins for bail_amount
    num_bins = 10
    data['bail_amount_bin'] = pd.cut(data['bail_amount'], bins=num_bins, labels=False)

    # Check the distribution of the bins
    bin_counts = data['bail_amount_bin'].value_counts()
    logging.info(f"Bin counts before adjustment: {bin_counts}")

    # Adjust the binning strategy if any bin has fewer than 2 samples
    while any(bin_counts < 2):
        num_bins -= 1
        if num_bins < 2:
            raise ValueError("Cannot create bins with at least 2 samples each. Consider increasing the sample size.")
        data['bail_amount_bin'] = pd.cut(data['bail_amount'], bins=num_bins, labels=False)
        bin_counts = data['bail_amount_bin'].value_counts()
        logging.info(f"Adjusted bin counts: {bin_counts}")

    # Drop columns that contain list values
    columns_to_drop = []
    for column in data.columns:
        if data[column].apply(lambda x: isinstance(x, list)).any():
            logging.warning(f"Column {column} contains list values and will be dropped.")
            columns_to_drop.append(column)
    data = data.drop(columns=columns_to_drop)

    # Encode categorical features with Label Encoding
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column].astype(str))
        label_encoders[column] = le
    logging.info("Categorical features encoded with Label Encoding.")

    # Separate features and target
    X = data.drop(columns=['bail_amount', 'bail_amount_bin'])
    y = data['bail_amount']
    y_bin = data['bail_amount_bin']

    # Check for NaN in y
    nan_count = y.isna().sum()
    logging.info(f"Number of NaN values in bail_amount: {nan_count}")

    # Remove rows with NaN values in the target if there are any
    if nan_count > 0:
        X = X[~y.isna()]
        y = y[~y.isna()]
        y_bin = y_bin[~y.isna()]
        logging.info(f"Removed {nan_count} rows with NaN values in bail_amount.")

    # Handle NaN values in features
    imputer = SimpleImputer(strategy='median')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    logging.info("Filled NaN values in features with column medians.")

    # Normalize columns
    columns_to_normalize = ['median_income', 'number_of_households', 'population']
    scaler = StandardScaler()
    for column in columns_to_normalize:
        if column in X.columns:
            X[column] = scaler.fit_transform(X[[column]])
            logging.info(f"Normalized {column} column.")

    # Save preprocessed data to CSV
    X.to_csv(os.path.join(outputs_dir, 'X.csv'), index=False)

    return X, y, y_bin


def split_data(X, y_bin, outputs_dir):
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(X, y_bin):
        X_train = X.loc[train_index]
        y_train = y_bin.loc[train_index]
        X_test = X.loc[test_index]
        y_test = y_bin.loc[test_index]
    logging.info("Data split into training and test sets (stratified).")

    # Save split data to CSV
    X_train.to_csv(os.path.join(outputs_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(outputs_dir, 'X_test.csv'), index=False)

    return X_train, y_train, X_test, y_test
