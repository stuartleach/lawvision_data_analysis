"""Module to manage the model creation."""

import logging
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

from app.utils import sanitize_metric_name


class RegressionModeler:
    """Class to manage the regression model training, evaluation, and interpretation."""

    MODEL_MAP = {
        "gradient_boosting": GradientBoostingRegressor,
        "random_forest": RandomForestRegressor,
        "hist_gradient_boosting": HistGradientBoostingRegressor,
        "ada_boost": AdaBoostRegressor,
        "bagging": BaggingRegressor,
        "extra_trees": ExtraTreesRegressor,
        "lasso": Lasso,
        "ridge": Ridge,
    }

    def __init__(self, model_type, good_hyperparameters):
        self.model_type = model_type
        self.hyperparameters = good_hyperparameters
        self.model = self._get_model()

    def _get_model(self):
        try:
            if self.model_type in self.MODEL_MAP:
                model_class = self.MODEL_MAP[self.model_type]
                return model_class(
                    random_state=42, **(self.hyperparameters.get(self.model_type, {}))
                )
            raise ValueError(f"Unknown model type: {self.model_type}")
        except Exception as e:
            logging.error("Error initializing model %s: %s", self.model_type, e)
            raise

    def train(self, x_train, y_train):
        """Train the model using the given training data."""
        try:
            self.model.fit(x_train, y_train)
            logging.info("%s model training completed.", self.model_type)
        except Exception as e:
            logging.error("Error training model %s: %s", self.model_type, e)
            raise

    def predict(self, x_test) -> list:
        """Predict the target variable using the given test data."""
        try:
            return self.model.predict(x_test)
        except Exception as e:
            logging.error("Error predicting with model %s: %s", self.model_type, e)
            raise

    def evaluate(self, x_test, y_test):
        """Evaluate the model using the given test data."""
        try:
            y_pred = self.model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logging.info("Mean Squared Error: %s", mse)
            logging.info("R-squared: %s", r2)
            return mse, r2
        except Exception as e:
            logging.error("Error evaluating model %s: %s", self.model_type, e)
            raise

    def apply(self, new_data):
        """Apply the trained model to new data and return predictions."""
        try:
            predictions = self.model.predict(new_data)
            logging.info("Predictions generated successfully.")
            return predictions
        except Exception as e:
            logging.error("Error generating predictions: %s", e)
            raise

    def log_metrics(self, mse, r2, x, outputs_dir):
        """Log the metrics and model artifacts."""
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(self.model, "model")
        self._log_feature_importances(x, outputs_dir)

    def _log_feature_importances(self, x, outputs_dir):
        importance = pd.Series(self.model.feature_importances_, index=x.columns)
        importance.sort_values(ascending=False, inplace=True)

        importance_df = importance.reset_index()
        importance_df.columns = ["Feature", "Importance"]
        importance_file_path = os.path.join(outputs_dir, "feature_importance.csv")
        importance_df.to_csv(importance_file_path, index=False)
        mlflow.log_artifact(importance_file_path)

        for feature, importance_value in importance.items():
            sanitized_feature_name = sanitize_metric_name(f"importance_{feature}")
            mlflow.log_metric(sanitized_feature_name, importance_value)

        logging.info("Feature importances: \n%s", importance)

    def save_model(self, path):
        """Save the model to the given path."""
        try:
            dump(self.model, path)
            logging.info("Model saved to %s", path)
        except Exception as e:
            logging.error("Error saving model: %s", e)
            raise

    def load_model(self, path):
        """Load the model from the given path."""
        try:
            self.model = load(path)
            logging.info("Model loaded from %s", path)
        except Exception as e:
            logging.error("Error loading model: %s", e)
            raise


class Model:
    """Unified class to manage both regression and neural network models."""

    def __init__(self, model_type, good_hyperparameters):
        self.model_type = model_type
        self.good_hyperparameters = good_hyperparameters
        self.manager = self._initialize_manager()
        self.outputs_dir = 'outputs'

    def _initialize_manager(self):
        return RegressionModeler(self.model_type, self.good_hyperparameters)

    def train(self, x_train, y_train):
        self.manager.train(x_train, y_train)

    def evaluate(self, x_test, y_test):
        return self.manager.evaluate(x_test, y_test)

    def predict(self, x_test):
        return self.manager.predict(x_test)

    def apply(self, new_data):
        return self.manager.apply(new_data)

    def get_feature_importances(self):
        return self.manager.model.feature_importances_

    def log_metrics(self, mse, r2, x, outputs_dir):
        self.manager.log_metrics(mse, r2, x, outputs_dir)

    def save_model(self, path):
        self.manager.save_model(path)

    def load_model(self, path):
        self.manager.load_model(path)
