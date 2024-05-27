import logging
import os
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv

from config import model_types, tune_hyperparameters_flag, sql_values, model_for_selection, \
    perform_feature_selection, query
from data_loader import load_data, create_engine_connection
from modeling import (
    train_model,
    evaluate_model,
    log_metrics,
    tune_hyperparameters,
    feature_selection
)
from params import GOOD_HYPERPARAMETERS, DISCORD_WEBHOOK_URL, DISCORD_AVATAR_URL
from preprocessing import preprocess_data, split_data
from utils import (
    plot_feature_importance,
    write_current_r2,
    read_previous_r2,
    send_discord_notification,
    save_importance_profile,
    load_importance_profile,
    compare_to_baseline, create_message
)

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class ModelTrainer:
    def __init__(self):
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
        self.engine = self._create_db_connection()

    def _ensure_outputs_dir(self):
        if not os.path.exists(self.outputs_dir):
            os.makedirs(self.outputs_dir)

    @staticmethod
    def _create_db_connection(self):
        user = os.environ.get("DB_USER")
        password = os.environ.get("DB_PASSWORD")
        host = os.environ.get("DB_HOST")
        port = os.environ.get("DB_PORT")
        dbname = os.environ.get("DB_NAME")
        return create_engine_connection(user, password, host, port, dbname)

    def load_and_preprocess_data(self):
        data = load_data(self.engine, query, sql_values)
        x, y, y_bin = preprocess_data(data, self.outputs_dir)
        x_train, y_train, x_test, y_test = split_data(x, y_bin, self.outputs_dir)
        self.total_cases = len(data)
        self.num_features = x_train.shape[1]
        return x_train, y_train, x_test, y_test

    def run(self):
        x_train, y_train, x_test, y_test = self.load_and_preprocess_data()
        previous_r2 = read_previous_r2(self.previous_r2_file)
        model_r2_scores = []

        mlflow.set_experiment("LawVision Model Training")

        with mlflow.start_run(run_name="Model Training Run") as run:
            for model_type in model_types:
                with mlflow.start_run(nested=True, run_name=model_type):
                    mlflow.log_param("model_type", model_type)
                    mlflow.log_param("perform_feature_selection", perform_feature_selection)
                    mlflow.log_param("tune_hyperparameters_flag", tune_hyperparameters_flag)

                    if perform_feature_selection:
                        x_train_selected, selector = feature_selection(x_train, y_train, model_type)
                        x_test_selected = selector.transform(x_test)
                        selected_features = x_train.columns[selector.support_].tolist()
                        mlflow.log_param("selected_features", selected_features)
                    else:
                        x_train_selected, x_test_selected = x_train, x_test

                    if tune_hyperparameters_flag:
                        model = tune_hyperparameters(x_train_selected, y_train, model_type)
                    else:
                        model = train_model(x_train_selected, y_train, model_type, GOOD_HYPERPARAMETERS)

                    mse, r2 = evaluate_model(model, x_test_selected, y_test)
                    model_r2_scores.append(r2)
                    log_metrics(model, mse, r2, pd.DataFrame(x_train_selected,
                                                             columns=x_train.columns if not perform_feature_selection else
                                                             x_train.columns[selector.support_]),
                                self.outputs_dir)

                    mlflow.log_metric("mse", mse)
                    mlflow.log_metric("r2", r2)

            average_r2 = sum(model_r2_scores) / len(model_r2_scores)
            logging.info(f"Average R-squared across all models: {average_r2}")
            mlflow.log_metric("average_r2", average_r2)

            r2_comparison = self._compare_r2(previous_r2, average_r2)
            write_current_r2(self.previous_r2_file, average_r2)

            plot_file_path = self._plot_and_save_importance(model, average_r2, r2_comparison)
            self._send_notification(average_r2, r2_comparison, plot_file_path)

    @staticmethod
    def _compare_r2(previous_r2, average_r2):
        if previous_r2 is not None:
            if average_r2 > previous_r2:
                return f'R² increased by {(average_r2 - previous_r2):.4f}'
            elif average_r2 < previous_r2:
                return f'R² decreased by {(previous_r2 - average_r2):.4f}'
            else:
                return "R² stayed the same"
        else:
            return "No previous R² value"

    def _plot_and_save_importance(self, model, average_r2, r2_comparison):
        if hasattr(model, "feature_importances_"):
            importance_df = pd.read_csv(os.path.join(self.outputs_dir, 'feature_importance.csv'))
            elapsed_time = time.time() - self.start_time
            plot_file_path = plot_feature_importance(importance_df, average_r2, self.total_cases, r2_comparison,
                                                     self.outputs_dir, elapsed_time, model_types, self.num_features,
                                                     model_for_selection)
            mlflow.log_artifact(plot_file_path)

            profile_path = save_importance_profile(importance_df, self.baseline_profile_name, self.outputs_dir)
            mlflow.log_artifact(profile_path)

            if os.path.exists(self.baseline_profile_path):
                baseline_df = load_importance_profile(self.baseline_profile_name, self.outputs_dir)
                comparison_df = compare_to_baseline(importance_df, baseline_df)
                comparison_path = os.path.join(self.outputs_dir, 'comparison_to_baseline.csv')
                comparison_df.to_csv(comparison_path, index=False)
                mlflow.log_artifact(comparison_path)

            return plot_file_path

    def _send_notification(self, average_r2, r2_comparison, plot_file_path):
        message = create_message(average_r2, r2_comparison, self.total_cases, time.time() - self.start_time,
                                 ', '.join(model_types))
        send_discord_notification(self.webhook_url, message, plot_file_path, self.avatar_url)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
