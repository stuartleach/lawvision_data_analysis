"""
Model trainer module.
"""

import logging
import os
import time
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv

from app.config import (
    MODEL_FOR_SELECTION,
    PERFORM_FEATURE_SELECTION_FLAG,
    model_types,
    query,
    sql_values,
)
from app.params import DISCORD_AVATAR_URL, DISCORD_WEBHOOK_URL, GOOD_HYPERPARAMETERS
from db import create_db_connection, load_and_preprocess_data
from models.model_manager import ModelManager
from utils import (
    NotificationData,
    PlotParams,
    compare_r2,
    compare_to_baseline,
    load_importance_profile,
    plot_feature_importance,
    read_previous_r2,
    save_importance_profile,
    write_current_r2,
)
from utils.notify import send_notification

load_dotenv()


@dataclass
class TrainerConfig:
    """
    Data class for trainer configuration.
    """

    outputs_dir: str = "outputs"
    webhook_url: str = DISCORD_WEBHOOK_URL
    avatar_url: str = DISCORD_AVATAR_URL
    baseline_profile_name: str = "baseline"
    start_time: float = time.time()
    previous_r2_file: str = os.path.join("outputs", "previous_r2.txt")


class ModelTrainer:
    """
    Model trainer class.
    """

    def __init__(self, filter_by=None, filter_value=None):
        self.config = TrainerConfig()
        self.filter_by = filter_by
        self.filter_value = filter_value
        self.engine = create_db_connection()
        self.ensure_outputs_dir()

    def ensure_outputs_dir(self):
        """
        Ensure that the outputs directory exists.
        """
        if not os.path.exists(self.config.outputs_dir):
            os.makedirs(self.config.outputs_dir)

    def plot_and_save_importance(self, model, plot_params: PlotParams):
        """
        Plot and save feature importance.
        """
        if hasattr(model, "feature_importances_"):
            importance_df = pd.read_csv(
                os.path.join(self.config.outputs_dir, "feature_importance.csv")
            )
            plot_file_path = plot_feature_importance(
                importance_df, self.config.outputs_dir, plot_params
            )
            mlflow.log_artifact(plot_file_path)

            profile_path = save_importance_profile(
                importance_df,
                self.config.baseline_profile_name,
                self.config.outputs_dir,
            )
            mlflow.log_artifact(profile_path)

            if os.path.exists(self.config.baseline_profile_name):
                baseline_df = load_importance_profile(
                    self.config.baseline_profile_name, self.config.outputs_dir
                )
                comparison_df = compare_to_baseline(importance_df, baseline_df)
                comparison_path = os.path.join(
                    self.config.outputs_dir, "comparison_to_baseline.csv"
                )
                comparison_df.to_csv(comparison_path, index=False)
                mlflow.log_artifact(comparison_path)

            return plot_file_path
        return "No feature importances available."

    def run(self):
        """
        Run the model training process.
        """
        _, x_train, y_train, x_test, y_test = load_and_preprocess_data(
            self.config.outputs_dir,
            self.engine,
            query,
            sql_values,
            self.filter_by,
            self.filter_value,
        )
        previous_r2 = read_previous_r2(self.config.previous_r2_file)
        model_r2_scores = []

        selected_features = x_train.columns

        mlflow.set_experiment("LawVision Model Training")

        with mlflow.start_run(run_name="Model Training Run"):
            for model_type in model_types:
                with mlflow.start_run(nested=True, run_name=model_type):
                    mlflow.log_param("model_type", model_type)
                    mlflow.log_param(
                        "perform_feature_selection", PERFORM_FEATURE_SELECTION_FLAG
                    )

                    model_manager = ModelManager(
                        model_type=model_type,
                        good_hyperparameters=GOOD_HYPERPARAMETERS,
                        input_dim=x_train.shape[1],
                    )

                    x_train_selected, x_test_selected = x_train, x_test

                    model_manager.train(x_train_selected, y_train)

                    mse, r2 = model_manager.evaluate(x_test_selected, y_test)
                    model_r2_scores.append(r2)
                    model_manager.log_metrics(
                        mse,
                        r2,
                        pd.DataFrame(x_train_selected, columns=x_train.columns),
                        self.config.outputs_dir,
                    )

                    # Plot Partial Dependence
                    model_manager.plot_partial_dependence(
                        pd.DataFrame(x_test_selected, columns=x_train.columns),
                        features=selected_features,
                        outputs_dir=self.config.outputs_dir,
                    )

                    mlflow.log_metric("mse", mse)
                    mlflow.log_metric("r2", r2)

            average_r2 = sum(model_r2_scores) / len(model_r2_scores)
            logging.info("Average R-squared across all models: %s", average_r2)
            mlflow.log_metric("average_r2", average_r2)

            r2_comparison = compare_r2(previous_r2, average_r2)
            write_current_r2(self.config.previous_r2_file, average_r2)

            elapsed_time = time.time() - self.config.start_time

            plot_params = PlotParams(
                r2=average_r2,
                total_cases=self.total_cases,
                r2_comparison=r2_comparison,
                elapsed_time=elapsed_time,
                model_info={
                    "model_types": model_types,
                    "num_features": self.num_features,
                },
            )

            plot_file_path = self._plot_and_save_importance(
                model_manager.model, plot_params
            )

            performance_data = {
                "average_r2": average_r2,
                "r2_comparison": r2_comparison,
                "total_cases": self.total_cases,
                "num_features": self.num_features,
                "time_difference": elapsed_time,
            }

            model_info = {
                "model_types": model_types,
                "model_for_selection": MODEL_FOR_SELECTION,
            }

            notification_data = NotificationData(
                performance_data=performance_data,
                plot_file_path=plot_file_path,
                model_info=model_info,
            )

            send_notification(notification_data)
            logging.info("Model training completed.")
