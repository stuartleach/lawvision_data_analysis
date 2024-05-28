import logging
import os
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy.orm import Session, sessionmaker

from .classes import TrainerConfig, model_config
from .env import DISCORD_AVATAR_URL, DISCORD_WEBHOOK_URL
from .env import GOOD_HYPERPARAMETERS
from .load import create_db_connection, load_data, split_data
from .model import ModelManager
from .notify import send_notification, NotificationData
from .preprocess import Preprocessor
from .utils import (
    PlotParams,
    compare_r2,
    compare_to_baseline,
    load_importance_profile,
    plot_feature_importance,
    read_previous_r2,
    save_importance_profile,
    write_current_r2,
)

load_dotenv()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=create_db_connection())


class ModelTrainer:
    """Model trainer class."""

    def __init__(self):
        self.config = TrainerConfig()
        self.session = SessionLocal()
        self.engine = create_db_connection()
        self.total_cases = 0
        self.num_features = 0
        self.ensure_outputs_dir()
        logging.info("Initialized ModelTrainer")

    def ensure_outputs_dir(self):
        """Ensure that the _outputs directory exists."""
        if not os.path.exists(self.config.outputs_dir):
            os.makedirs(self.config.outputs_dir)

    def plot_and_save_importance(self, model, plot_params: PlotParams):
        """Plot and save feature importance.

        :param model:
        :param plot_params:
        :return:
        """
        if hasattr(model, "feature_importances_"):
            importance_df = pd.read_csv(os.path.join(self.config.outputs_dir, "feature_importance.csv"))
            plot_file_path = plot_feature_importance(importance_df, self.config.outputs_dir, plot_params)
            mlflow.log_artifact(plot_file_path)

            profile_path = save_importance_profile(importance_df, self.config.baseline_profile_name,
                                                   self.config.outputs_dir)
            mlflow.log_artifact(profile_path)

            if os.path.exists(self.config.baseline_profile_name):
                baseline_df = load_importance_profile(self.config.baseline_profile_name, self.config.outputs_dir)
                comparison_df = compare_to_baseline(importance_df, baseline_df)
                comparison_path = os.path.join(self.config.outputs_dir, "comparison_to_baseline.csv")
                comparison_df.to_csv(comparison_path, index=False)
                mlflow.log_artifact(comparison_path)

            return plot_file_path
        return "No feature importances available."

    def run(self):
        """Run the model training process."""
        self.train_model()

    def get_unique_values(self, column):
        """Get unique values for a column in the dataset."""

        logging.info(f"Getting unique values for column: {column}")

        data = load_data(self.session)
        print(data.columns.tolist())
        logging.info(f"Loaded data columns: {data.columns}")
        # print(data.head().values.tolist())
        if column not in data.columns:
            logging.error(f"Column '{column}' not found in data.")
            raise KeyError(f"Column '{column}' not found in data.")
        return data[column].unique()

    def train_model(self):
        """Train model"""
        preprocessor = Preprocessing(self.config)
        _, x_train, y_train, x_test, y_test = preprocessor.load_and_preprocess_data()
        self.total_cases = preprocessor.total_cases
        self.num_features = preprocessor.num_features

        previous_r2 = read_previous_r2(self.config.previous_r2_file)
        model_r2_scores = []

        mlflow.set_experiment("LawVision Model Training")

        with mlflow.start_run(
                run_name="Model Training Run"):
            for model_type in model_config.model_types:
                with mlflow.start_run(nested=True, run_name=model_type):
                    mlflow.log_param("model_type", model_type)

                    model_manager = ModelManager(model_type=model_type, good_hyperparameters=GOOD_HYPERPARAMETERS)

                    x_train_selected, x_test_selected = x_train, x_test

                    model_manager.train(x_train_selected, y_train)

                    mse, r2 = model_manager.evaluate(x_test_selected, y_test)
                    model_r2_scores.append(r2)
                    model_manager.log_metrics(mse, r2, pd.DataFrame(x_train_selected, columns=x_train.columns),
                                              self.config.outputs_dir)

                    # Plot Partial Dependence
                    model_manager.plot_partial_dependence(pd.DataFrame(x_test_selected, columns=x_train.columns),
                                                          features=x_train.columns.tolist(),
                                                          outputs_dir=self.config.outputs_dir)

                    mlflow.log_metric("mse", mse)
                    mlflow.log_metric("r2", r2)

            average_r2 = sum(model_r2_scores) / len(model_r2_scores)
            logging.info(f"Average R-squared across all models: {average_r2}")
            mlflow.log_metric("average_r2", average_r2)

            r2_comparison = compare_r2(previous_r2, average_r2)
            write_current_r2(self.config.previous_r2_file, average_r2)

            elapsed_time = time.time() - self.config.start_time

            plot_params = PlotParams(
                r2=average_r2,
                total_cases=self.total_cases,
                r2_comparison=r2_comparison,
                elapsed_time=elapsed_time,
                model_info={"model_types": model_config.model_types, "num_features": self.num_features},
            )

            plot_file_path = self.plot_and_save_importance(model_manager.manager.model, plot_params)

            performance_data = {
                "average_r2": average_r2,
                "r2_comparison": r2_comparison,
                "total_cases": self.total_cases,
                "num_features": self.num_features,
                "time_difference": elapsed_time,
            }

            model_info = {
                "model_types": model_config.model_types,
            }

            notification_data = NotificationData(
                performance_data=performance_data,
                plot_file_path=plot_file_path,
                model_info=model_info,
            )

            send_notification(notification_data, DISCORD_WEBHOOK_URL, DISCORD_AVATAR_URL)
            logging.info("Model training completed.")


class Preprocessing:
    def __init__(self, config):
        self.config = config
        self.total_cases = 0
        self.num_features = 0
        self.engine = create_db_connection()

    def load_and_preprocess_data(self):
        """Load and preprocess data."""

        session = Session(self.engine)
        data = load_data(session)
        x_column, _y_column, y_bin = Preprocessor().preprocess_data(data, self.config.outputs_dir)
        x_train, y_train, x_test, y_test = split_data(x_column, y_bin, self.config.outputs_dir)
        self.total_cases = len(data)
        self.num_features = x_column.shape[1]
        return data, x_train, y_train, x_test, y_test
