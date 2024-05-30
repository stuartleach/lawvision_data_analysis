import logging
import os
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session

from .classes import TrainerConfig, model_config, ResultObject
from .env import DISCORD_AVATAR_URL, DISCORD_WEBHOOK_URL, GOOD_HYPERPARAMETERS
from .model import Model
from .notify import send_notification, NotificationData
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


class ModelTrainer:
    """Model trainer class."""

    def __init__(self, judge_filter=None, county_filter=None, quiet=False):
        self.model = None
        from .data import create_db_connection  # Import here to avoid circular import
        self.config = TrainerConfig()
        self.quiet = quiet
        self.session = Session(autocommit=False, autoflush=False, bind=create_db_connection())
        self.engine = create_db_connection()
        self.total_cases = 0
        self.num_features = 0
        self.judge_filter = judge_filter
        self.county_filter = county_filter
        self.ensure_outputs_dir()
        logging.info("Initialized ModelTrainer")

    def ensure_outputs_dir(self):
        """Ensure that the outputs directory exists."""
        if not os.path.exists(self.config.outputs_dir):
            os.makedirs(self.config.outputs_dir)

    def plot_and_save_importance(self, model, plot_params: PlotParams):
        """Plot and save feature importance."""
        if hasattr(model, "feature_importances_"):
            importance_df = pd.read_csv(os.path.join(self.config.outputs_dir, "feature_importance.csv"))
            plot_file_path = plot_feature_importance(importance_df, self.config.outputs_dir, plot_params,
                                                     self.judge_filter, self.county_filter)
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
        from .data import load_data  # Import here to avoid circular import
        data = load_data(self.session)
        logging.info(f"Loaded data columns: {data.columns}")

        if column not in data.columns:
            logging.error(f"Column '{column}' not found in data.")
            raise KeyError(f"Column '{column}' not found in data.")
        return data[column].unique()

    def train_model(self):
        """Train model"""
        from .data import save_data, load_data  # Import here to avoid circular import
        preprocessor = Preprocessing(self.config, self.judge_filter, self.county_filter)
        try:
            _, x_train, y_train, x_test, y_test = preprocessor.load_and_preprocess_data()
        except ValueError as e:
            logging.error(f"Error in preprocessing data: {e}")
            return

        self.total_cases = preprocessor.total_cases
        self.num_features = preprocessor.num_features

        previous_r2 = read_previous_r2(self.config.previous_r2_file)
        model_r2_scores = []

        mlflow.set_experiment("LawVision Model Training")

        with mlflow.start_run(run_name="Baseline Model Training Run"):
            for model_type in model_config.model_types:
                with mlflow.start_run(nested=True, run_name=model_type):
                    mlflow.log_param("model_type", model_type)

                    modeler = Model(model_type=model_type, good_hyperparameters=GOOD_HYPERPARAMETERS)

                    x_train_selected, x_test_selected = x_train, x_test

                    modeler.train(x_train_selected, y_train)

                    mse, r2 = modeler.evaluate(x_test_selected, y_test)
                    model_r2_scores.append(r2)
                    modeler.log_metrics(mse, r2, pd.DataFrame(x_train_selected, columns=x_train.columns),
                                        self.config.outputs_dir)

                    # Plot Partial Dependence
                    modeler.plot_partial_dependence(pd.DataFrame(x_test_selected, columns=x_train.columns),
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

            model_target_type = "judge_name" if self.judge_filter else "county_name" if self.county_filter else "baseline"
            model_params = modeler.manager.model.get_params()
            importance_df = pd.read_csv(os.path.join(self.config.outputs_dir, "feature_importance.csv"))

            df = load_data(self.session, self.judge_filter, self.county_filter)

            # Ensure 'first_bail_set_cash' is numeric
            df['first_bail_set_cash'] = pd.to_numeric(df['first_bail_set_cash'], errors='coerce')

            average_bail_amount = df['first_bail_set_cash'].mean()

            result_obj = ResultObject(
                model_type=model_type,
                model_target_type=model_target_type,
                model_target=self.judge_filter if self.judge_filter else self.county_filter if self.county_filter else "baseline",
                judge_filter=self.judge_filter,
                county_filter=self.county_filter,
                model_params=model_params,
                average_bail_amount=average_bail_amount,
                r_squared=average_r2,
                mean_squared_error=mse,
                dataframe=importance_df,
                total_cases=self.total_cases,
            )

            plot_file_path = self.plot_and_save_importance(modeler.manager.model, plot_params)

            save_data(self.session, result_obj)

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

            if not self.quiet:
                notification_data = NotificationData(
                    performance_data=performance_data,
                    plot_file_path=plot_file_path,
                    model_info=model_info,
                )

                send_notification(notification_data, DISCORD_WEBHOOK_URL, DISCORD_AVATAR_URL)
            logging.info("Model training completed.")


class Preprocessing:
    def __init__(self, config, judge_filter=None, county_filter=None):
        from .data import create_db_connection  # Import here to avoid circular import
        self.config = config
        self.total_cases = 0
        self.num_features = 0
        self.engine = create_db_connection()
        self.judge_filter = judge_filter
        self.county_filter = county_filter

    def load_and_preprocess_data(self):
        """Load and preprocess data."""
        from .preprocess import Preprocessor
        from .data import load_data, split_data  # Import here to avoid circular import
        session = Session(self.engine)
        data = load_data(session, self.judge_filter, self.county_filter)

        x_column, _y_column, y_bin = Preprocessor().preprocess_data(data, self.config.outputs_dir)
        x_train, y_train, x_test, y_test = split_data(x_column, y_bin, self.config.outputs_dir)
        self.total_cases = len(data)
        self.num_features = x_column.shape[1]
        return data, x_train, y_train, x_test, y_test


def get_judges(session):
    """Fetch the list of judges from the database."""
    result = session.execute(text("SELECT DISTINCT judge_name FROM pretrial.judges WHERE case_count > 5"))
    judges = [row[0] for row in result]
    return judges


def grade_judges_with_general_model(session, general_trainer, limit=None):
    """Grade each judge using a pre-trained general model."""
    from .data import load_data
    if limit:
        logging.info(f"Limiting the number of judges to {limit}")

    judges = get_judges(session)
    logging.info(f"Found {len(judges)} judges.")

    for judge in judges:
        logging.info(f"Grading judge: {judge}")
        judge_data = load_data(session, judge_filter=judge)
        if judge_data.empty:
            logging.warning(f"No data for judge {judge}. Skipping...")
            continue

        # Apply the pre-trained model to the judge's data
        x_judge = judge_data.drop(columns=['judge_name'])
        y_judge = judge_data['judge_name']

        mse, r2 = general_trainer.model.evaluate(x_judge, y_judge)
        feature_importances = general_trainer.model.get_feature_importances()

        # Log or save the results for the judge
        logging.info(f"Judge: {judge}, MSE: {mse}, R2: {r2}")
        logging.info(f"Feature importances for judge {judge}: {feature_importances}")


def train_model_for_each_judge(session):
    """Train a model for each judge."""
    judges = get_judges(session)
    logging.info(f"Found {len(judges)} judges.")

    for judge in judges:
        logging.info(f"Training model for judge: {judge}")
        trainer = ModelTrainer(judge_filter=judge)
        try:
            trainer.run()
            # Save or log the feature importances for the judge
            feature_importances = trainer.model.get_feature_importances()
            logging.info(f"Feature importances for judge {judge}: {feature_importances}")
        except ValueError as e:
            logging.error(f"Error in training model for judge {judge}: {e}")
