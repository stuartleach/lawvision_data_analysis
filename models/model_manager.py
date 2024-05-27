import logging
import os

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import shap
from joblib import dump, load
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
)
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from utils import sanitize_metric_name


class ModelManager:
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
                return model_class(random_state=42, **(self.hyperparameters.get(self.model_type, {})))
            elif self.model_type == "neural_network":
                return KerasRegressor(model=self._build_neural_network, epochs=10, batch_size=32)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        except Exception as e:
            logging.error(f"Error initializing model {self.model_type}: {e}")
            raise

    def _build_neural_network(self, layers=None, activation='relu', optimizer='adam'):
        if layers is None:
            layers = [64, 64]
        model = Sequential()
        model.add(Dense(layers[0], input_dim=self.input_dim, activation=activation))
        for layer in layers[1:]:
            model.add(Dense(layer, activation=activation))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    def train(self, x_train, y_train):
        try:
            self.model.fit(x_train, y_train)
            logging.info(f"{self.model_type} model training completed.")
        except Exception as e:
            logging.error(f"Error training model {self.model_type}: {e}")
            raise

    def evaluate(self, x_test, y_test):
        try:
            y_pred = self.model.predict(x_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            logging.info(f"R-squared: {r2}")
            return mse, r2
        except Exception as e:
            logging.error(f"Error evaluating model {self.model_type}: {e}")
            raise

    def log_metrics(self, mse, r2, x, outputs_dir):
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
        importance_df.columns = ['Feature', 'Importance']
        importance_file_path = os.path.join(outputs_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_file_path, index=False)
        mlflow.log_artifact(importance_file_path)

        for feature, importance_value in importance.items():
            sanitized_feature_name = sanitize_metric_name(f"importance_{feature}")
            mlflow.log_metric(sanitized_feature_name, importance_value)

        logging.info(f"Feature importances: \n{importance}")

    def _log_shap_values(self, x, outputs_dir):
        explainer = shap.Explainer(self.model, x)
        shap_values = explainer(x)
        shap.summary_plot(shap_values, x, show=False)
        shap_file_path = os.path.join(outputs_dir, 'shap_summary.png')
        plt.savefig(shap_file_path)
        mlflow.log_artifact(shap_file_path)
        logging.info(f"SHAP summary plot saved as '{shap_file_path}'")
        logging.info("Model does not support feature importances, used SHAP for interpretation.")

    def plot_partial_dependence(self, x, features, outputs_dir):
        fig, ax = plt.subplots(figsize=(12, 8))
        PartialDependenceDisplay.from_estimator(self.model, x, features=features).plot(ax=ax)
        pdp_file_path = os.path.join(outputs_dir, 'partial_dependence.png')
        plt.savefig(pdp_file_path)
        logging.info(f"Partial Dependence Plot saved as '{pdp_file_path}'")
        plt.show()

    def save_model(self, path):
        try:
            dump(self.model, path)
            logging.info(f"Model saved to {path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def load_model(self, path):
        try:
            self.model = load(path)
            logging.info(f"Model loaded from {path}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
