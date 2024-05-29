import unittest
from io import StringIO

import pandas as pd

from app.preprocess import Preprocessor, convert_bail_amount, drop_list_columns, separate_features_and_target, \
    normalize_columns


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        data_csv = """
        first_bail_set_cash,median_income,number_of_households,population,some_categorical_column
        1000,55000,200,1500,A
        2000,60000,250,1600,B
        3000,65000,300,1700,A
        NaN,70000,350,1800,C
        """
        self.data = pd.read_csv(StringIO(data_csv))

        # Initialize Preprocessor object
        self.preprocessor = Preprocessor()

    def test_convert_bail_amount(self):
        result = convert_bail_amount(self.data.copy())
        self.assertIn('bail_amount', result.columns)
        self.assertNotIn('first_bail_set_cash', result.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(result['bail_amount']))

    def test_drop_list_columns(self):
        data_with_list = self.data.copy()
        data_with_list['list_column'] = [[1, 2], [3, 4], [5, 6], [7, 8]]
        result = drop_list_columns(data_with_list)
        self.assertNotIn('list_column', result.columns)

    def test_separate_features_and_target(self):
        self.data['bail_amount_bin'] = [1, 2, 3, 4]  # Adding a bin column
        x, y, y_bin = separate_features_and_target(self.data.copy())
        self.assertNotIn('bail_amount', x.columns)
        self.assertNotIn('bail_amount_bin', x.columns)
        self.assertTrue(pd.api.types.is_numeric_dtype(y))
        self.assertTrue(pd.api.types.is_numeric_dtype(y_bin))

    def test_normalize_columns(self):
        columns_to_normalize = ['median_income', 'number_of_households', 'population']
        x = self.data.drop(columns=['first_bail_set_cash'])
        result = normalize_columns(columns_to_normalize, x)
        for column in columns_to_normalize:
            self.assertAlmostEqual(result[column].mean(), 0, delta=1e-6)
            self.assertAlmostEqual(result[column].std(), 1, delta=1e-6)

    def test_preprocess_data(self):
        self.data['bail_amount_bin'] = [1, 2, 3, 4]  # Adding a bin column
        x, y, y_bin = self.preprocessor.preprocess_data(self.data.copy(), 'dummy_output_dir')
        self.assertIn('bail_amount', self.data.columns)  # Check bail amount is present
        self.assertTrue((y_bin >= 0).all())  # Check that bin values are non-negative
        self.assertTrue(pd.api.types.is_numeric_dtype(x['median_income']))


if __name__ == '__main__':
    unittest.main()
