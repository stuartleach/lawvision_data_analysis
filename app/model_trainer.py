# app/model_trainer.py

import logging
import os
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv

import db
import utils
from config import model_types, sql_values, model_for_selection, \
    perform_feature_selection, query
from models.model_manager import ModelManager
from params import GOOD_HYPERPARAMETERS, DISCORD_WEBHOOK_URL, DISCORD_AVATAR_URL
from .data_handler import DataHandler

load_dotenv()


class ModelTrainer:
    def __init__(self, filter_by=None, filter_value=None):
        self.num_features = None
        self.total_cases = None
        self.webhook_url = DISCORD_WEBHOOK_URL
        self.avatar_url = DISCORD_AVATAR_URL
        self.outputs_dir = 'outputs'
        self.start_time = time.time()
        self.previous_r2_file = os.path.join(self.outputs_dir, 'previous_r2.txt')
        self.baseline_profile_name = "baseline"
        self.baseline_profile_path = os.path.join(self.outputs_dir,
                                                  f"{self.baseline_profile_name}_importance_profile.csv")
        self._ensure_outputs_dir()
        self.engine = db.create_db_connection()
        self.filter_by = filter_by
        self.filter_value = filter_value

    def _ensure_outputs_dir(self):
        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)

    def load_and_preprocess_data(self):
        data_handler = DataHandler(self.engine, query, sql_values, self.filter_by, self.filter_value)
        data, x_train, y_train, x_test, y_test = data_handler.load_and_preprocess_data(self.outputs_dir)
        self.total_cases = len(data)
        self.num_features = x_train.shape[1]
        return x_train, y_train, x_test, y_test

    def run(self):
        x_train, y_train, x_test, y_test = self.load_and_preprocess_data()
        previous_r2 = utils.read_previous_r2(self.previous_r2_file)
        model_r2_scores = []

        selected_features = x_train.columns

        mlflow.set_experiment("LawVision Model Training")

        with mlflow.start_run(run_name="Model Training Run"):
            for model_type in model_types:
                with mlflow.start_run(nested=True, run_name=model_type):
                    mlflow.log_param("model_type", model_type)
                    mlflow.log_param("perform_feature_selection", perform_feature_selection)

                    model_manager = ModelManager(model_type=model_type, good_hyperparameters=GOOD_HYPERPARAMETERS,
                                                 input_dim=x_train.shape[1])

                    x_train_selected, x_test_selected = x_train, x_test

                    model_manager.train(x_train_selected, y_train)

                    mse, r2 = model_manager.evaluate(x_test_selected, y_test)
                    model_r2_scores.append(r2)
                    model_manager.log_metrics(mse, r2, pd.DataFrame(x_train_selected,
                                                                    columns=x_train.columns),
                                              self.outputs_dir)

                    # Plot Partial Dependence
                    model_manager.plot_partial_dependence(pd.DataFrame(x_test_selected, columns=x_train.columns),
                                                          features=selected_features, outputs_dir=self.outputs_dir)

                    mlflow.log_metric("mse", mse)
                    mlflow.log_metric("r2", r2)

            average_r2 = sum(model_r2_scores) / len(model_r2_scores)
            logging.info(f"Average R-squared across all models: {average_r2}")
            mlflow.log_metric("average_r2", average_r2)

            r2_comparison = utils.compare_r2(previous_r2, average_r2)
            utils.write_current_r2(self.previous_r2_file, average_r2)

            plot_file_path = self._plot_and_save_importance(model_manager.model, average_r2, r2_comparison)
            utils.send_notification(average_r2, r2_comparison, plot_file_path, time.time() - self.start_time)

    def _plot_and_save_importance(self, model, average_r2, r2_comparison):
        if hasattr(model, "feature_importances_"):
            importance_df = pd.read_csv(os.path.join(self.outputs_dir, 'feature_importance.csv'))
            elapsed_time = time.time() - self.start_time
            plot_file_path = utils.plot_feature_importance(importance_df, average_r2, self.total_cases,
                                                           r2_comparison,
                                                           self.outputs_dir, elapsed_time, model_types,
                                                           self.num_features,
                                                           model_for_selection)
            mlflow.log_artifact(plot_file_path)

            profile_path = utils.save_importance_profile(importance_df, self.baseline_profile_name,
                                                         self.outputs_dir)
            mlflow.log_artifact(profile_path)

            if os.path.exists(self.baseline_profile_path):
                baseline_df = utils.load_importance_profile(self.baseline_profile_name, self.outputs_dir)
                comparison_df = utils.compare_to_baseline(importance_df, baseline_df)
                comparison_path = os.path.join(self.outputs_dir, 'comparison_to_baseline.csv')
                comparison_df.to_csv(comparison_path, index=False)
                mlflow.log_artifact(comparison_path)

            return plot_file_path
