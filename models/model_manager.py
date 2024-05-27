"""Module to manage the model creation."""

import logging
import os

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import shap
from joblib import dump, load
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score

from utils import sanitize_metric_name


class ModelManager:
    """Class to manage the model training, evaluation, and interpretation."""

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

    def __init__(self, model_type, good_hyperparameters, input_dim=None):
        self.model_type = model_type
        self.hyperparameters = good_hyperparameters
        self.input_dim = input_dim
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
        """Train the model using the given training data.
        :param x_train:
        :param y_train:
        :return:
        """
        try:
            self.model.fit(x_train, y_train)
            logging.info("%s model training completed.", self.model_type)
        except Exception as e:
            logging.error("Error training model %s: %s", self.model_type, e)
            raise

    def evaluate(self, x_test, y_test):
        """Evaluate the model using the given test data.
        :param x_test:
        :param y_test:
        :return:
        """
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

    def log_metrics(self, mse, r2, x, outputs_dir):
        """Log the metrics and model artifacts.
        :param mse:
        :param r2:
        :param x:
        :param outputs_dir:
        :return:
        """
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(self.model, "model")

        if hasattr(self.model, "feature_importances_"):
            self._log_feature_importances(x, outputs_dir)
        else:
            self._log_shap_values(x, outputs_dir)

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

    def _log_shap_values(self, x, outputs_dir):
        explainer = shap.Explainer(self.model, x)
        shap_values = explainer(x)
        shap.summary_plot(shap_values, x, show=False)
        shap_file_path = os.path.join(outputs_dir, "shap_summary.png")
        plt.savefig(shap_file_path)
        mlflow.log_artifact(shap_file_path)
        logging.info("SHAP summary plot saved as '%s'", shap_file_path)
        logging.info(
            "Model does not support feature importances, used SHAP for interpretation."
        )

    def plot_partial_dependence(self, x, features, outputs_dir):
        """Plot partial dependence for the given features.
        :param x:
        :param features:
        :param outputs_dir:
        :return:
        """
        _, ax = plt.subplots(figsize=(12, 8))
        PartialDependenceDisplay.from_estimator(self.model, x, features=features).plot(
            ax=ax
        )
        pdp_file_path = os.path.join(outputs_dir, "partial_dependence.png")
        plt.savefig(pdp_file_path)
        logging.info("Partial Dependence Plot saved as '%s'", pdp_file_path)
        plt.show()

    def save_model(self, path):
        """Save the model to the given path.
        :param path:
        :return:
        """
        try:
            dump(self.model, path)
            logging.info("Model saved to %s", path)
        except Exception as e:
            logging.error("Error saving model: %s", e)
            raise

    def load_model(self, path):
        """Load the model from the given path.
        :param path:
        :return:
        """
        try:
            self.model = load(path)
            logging.info("Model loaded from %s", path)
        except Exception as e:
            logging.error("Error loading model: %s", e)
            raise
