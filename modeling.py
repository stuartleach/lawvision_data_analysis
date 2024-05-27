import logging
import os
import re

import mlflow
import pandas as pd
import shap
from matplotlib import pyplot as plt
from scikeras.wrappers import KerasRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor
)
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from params import HYPERPARAMETER_GRIDS


def train_model(X_train, y_train, model_type, good_hyperparameters):
    model = get_model(model_type, good_hyperparameters, input_dim=X_train.shape[1])
    model.fit(X_train, y_train)
    logging.info(f"{model_type} model training completed.")
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"R-squared: {r2}")
    return mse, r2


def log_metrics(model, mse, r2, X, outputs_dir):
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, "model")

    # Check if the model has feature_importances_ attribute
    if hasattr(model, "feature_importances_"):
        # Log feature importances
        importance = pd.Series(model.feature_importances_, index=X.columns)
        importance.sort_values(ascending=False, inplace=True)

        # Save feature importance as a CSV file and log it as an artifact
        importance_df = importance.reset_index()
        importance_df.columns = ['Feature', 'Importance']
        importance_file_path = os.path.join(outputs_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_file_path, index=False)
        mlflow.log_artifact(importance_file_path)

        # Also log individual importance values
        for feature, importance_value in importance.items():
            sanitized_feature_name = sanitize_metric_name(f"importance_{feature}")
            mlflow.log_metric(sanitized_feature_name, importance_value)

        logging.info(f"Feature importances: \n{importance}")
    else:
        # Use SHAP values to interpret the model
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap.summary_plot(shap_values, X, show=False)
        shap_file_path = os.path.join(outputs_dir, 'shap_summary.png')
        plt.savefig(shap_file_path)
        mlflow.log_artifact(shap_file_path)
        logging.info(f"SHAP summary plot saved as '{shap_file_path}'")

        logging.info("Model does not support feature importances, used SHAP for interpretation.")


def sanitize_metric_name(name):
    return re.sub(r'[^a-zA-Z0-9_\- .\/]', '_', name)


def tune_hyperparameters(X_train, y_train, model_type):
    param_grid = HYPERPARAMETER_GRIDS.get(model_type.lower())
    if param_grid is None:
        raise ValueError(f"Unknown model type: {model_type}")

    model = get_model(model_type, None, input_dim=X_train.shape[1])

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_


def feature_selection(X_train, y_train, model_type):
    model = get_model(model_type, None, input_dim=X_train.shape[1])
    selector = RFECV(estimator=model, min_features_to_select=20, step=1, cv=5, scoring='r2')
    selector = selector.fit(X_train, y_train)
    X_train_selected = selector.transform(X_train)
    logging.info(
        f"Feature selection completed using {model_type}. Number of features selected: {X_train_selected.shape[1]}")
    return X_train_selected, selector


def get_model(model_type, hyperparameters, input_dim=None):
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(random_state=42,
                                         **(hyperparameters["gradient_boosting"] if hyperparameters else {}))
    elif model_type == "random_forest":
        return RandomForestRegressor(random_state=42, **(hyperparameters["random_forest"] if hyperparameters else {}))
    elif model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(random_state=42,
                                             **(hyperparameters["hist_gradient_boosting"] if hyperparameters else {}))
    elif model_type == "ada_boost":
        return AdaBoostRegressor(random_state=42, **(hyperparameters["ada_boost"] if hyperparameters else {}))
    elif model_type == "bagging":
        return BaggingRegressor(random_state=42, **(hyperparameters["bagging"] if hyperparameters else {}))
    elif model_type == "extra_trees":
        return ExtraTreesRegressor(random_state=42, **(hyperparameters["extra_trees"] if hyperparameters else {}))
    elif model_type == "lasso":
        return Lasso(random_state=42, **(hyperparameters["lasso"] if hyperparameters else {}))
    elif model_type == "ridge":
        return Ridge(random_state=42, **(hyperparameters["ridge"] if hyperparameters else {}))
    elif model_type == "neural_network":
        return KerasRegressor(build_fn=lambda **kwargs: build_neural_network(input_dim=input_dim, **kwargs),
                              **(hyperparameters["neural_network"] if hyperparameters else {}))
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_neural_network(input_dim, layers=[64, 64], activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(layers[0], input_dim=input_dim, activation=activation))
    for layer in layers[1:]:
        model.add(Dense(layer, activation=activation))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model
