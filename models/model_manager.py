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
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.models import Sequential

import utils


class ModelManager:
    def __init__(self, model_type, good_hyperparameters, input_dim=None):
        self.model_type = model_type
        self.hyperparameters = good_hyperparameters
        self.input_dim = input_dim
        self.model = self._get_model()

    def _get_model(self):
        try:
            if self.model_type == "gradient_boosting":
                return GradientBoostingRegressor(random_state=42,
                                                 **(self.hyperparameters[
                                                        "gradient_boosting"] if self.hyperparameters else {}))
            elif self.model_type == "random_forest":
                return RandomForestRegressor(random_state=42,
                                             **(self.hyperparameters["random_forest"] if self.hyperparameters else {}))
            elif self.model_type == "hist_gradient_boosting":
                return HistGradientBoostingRegressor(random_state=42,
                                                     **(self.hyperparameters[
                                                            "hist_gradient_boosting"] if self.hyperparameters else {}))
            elif self.model_type == "ada_boost":
                return AdaBoostRegressor(random_state=42,
                                         **(self.hyperparameters["ada_boost"] if self.hyperparameters else {}))
            elif self.model_type == "bagging":
                return BaggingRegressor(random_state=42,
                                        **(self.hyperparameters["bagging"] if self.hyperparameters else {}))
            elif self.model_type == "extra_trees":
                return ExtraTreesRegressor(random_state=42,
                                           **(self.hyperparameters["extra_trees"] if self.hyperparameters else {}))
            elif self.model_type == "lasso":
                return Lasso(random_state=42, **(self.hyperparameters["lasso"] if self.hyperparameters else {}))
            elif self.model_type == "ridge":
                return Ridge(random_state=42, **(self.hyperparameters["ridge"] if self.hyperparameters else {}))
            elif self.model_type == "neural_network":
                return KerasRegressor(model=Sequential(layers=Layer(), name="my_model"), epochs=10, batch_size=32)
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

    def train(self, X_train, y_train):
        try:
            self.model.fit(X_train, y_train)
            logging.info(f"{self.model_type} model training completed.")
        except Exception as e:
            logging.error(f"Error training model {self.model_type}: {e}")
            raise

    def evaluate(self, X_test, y_test):
        try:
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logging.info(f"Mean Squared Error: {mse}")
            logging.info(f"R-squared: {r2}")
            return mse, r2
        except Exception as e:
            logging.error(f"Error evaluating model {self.model_type}: {e}")
            raise

    def log_metrics(self, mse, r2, X, outputs_dir):
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(self.model, "model")

        if hasattr(self.model, "feature_importances_"):
            self._log_feature_importances(X, outputs_dir)
        else:
            self._log_shap_values(X, outputs_dir)

    def _log_feature_importances(self, X, outputs_dir):
        importance = pd.Series(self.model.feature_importances_, index=X.columns)
        importance.sort_values(ascending=False, inplace=True)

        importance_df = importance.reset_index()
        importance_df.columns = ['Feature', 'Importance']
        importance_file_path = os.path.join(outputs_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_file_path, index=False)
        mlflow.log_artifact(importance_file_path)

        for feature, importance_value in importance.items():
            sanitized_feature_name = utils.MetricUtils.sanitize_metric_name(f"importance_{feature}")
            mlflow.log_metric(sanitized_feature_name, importance_value)

        logging.info(f"Feature importances: \n{importance}")

    def _log_shap_values(self, X, outputs_dir):
        explainer = shap.Explainer(self.model, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, show=False)
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
