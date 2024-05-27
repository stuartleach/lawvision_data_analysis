from db.data_loader import load_data, split_data
from db.preprocessor import Preprocessor


class DataHandler:
    def __init__(self, engine, query, sql_values, filter_by=None, filter_value=None):
        self.engine = engine
        self.query = query
        self.sql_values = sql_values
        self.filter_by = filter_by
        self.filter_value = filter_value

    def load_and_preprocess_data(self, outputs_dir):
        data = load_data(self.engine, self.query, self.sql_values)
        if self.filter_by and self.filter_value:
            data = self._filter_data(data)
        preprocessor = Preprocessor()
        x, y, y_bin = preprocessor.preprocess_data(data, outputs_dir)
        x_train, y_train, x_test, y_test = split_data(x, y_bin, outputs_dir)
        return data, x_train, y_train, x_test, y_test

    def _filter_data(self, data):
        return data[data[self.filter_by] == self.filter_value]
