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
from .preprocess import Preprocessing
from .utils import (
    PlotParams,
    compare_r2,
    compare_to_baseline,
    load_importance_profile,
    plot_feature_importance,
    read_previous_r2,
    save_importance_profile,
    write_current_r2,
    plot_partial_dependence,
)


class ModelTrainer:
    """Model trainer class."""

    def __init__(self, judge_filter=None, county_filter=None, quiet=False):
        from .data import create_db_connection  # Import here to avoid circular import
        self.config = TrainerConfig()
        self.quiet = quiet
        self.session = Session(autocommit=False, autoflush=False, bind=create_db_connection())
        self.engine = create_db_connection()
        self.total_cases = 0
        self.num_features = 0
        self.judge_filter = judge_filter
        self.county_filter = county_filter
        self.model = None  # Initialize model as None
        self.feature_names = None  # To store feature names after training
        self.ensure_outputs_dir()
        logging.info("Initialized ModelTrainer")

    def plot_and_save_shap(self, model, x):
        logging.info("Plotting and saving SHAP values.")
        shap_values = model.explain_predictions(x)
        logging.info("SHAP values: %s", shap_values)
        model.save_shap_values(shap_values, os.path.join(self.config.outputs_dir, "shap_values.npy"))
        plot_file_path = model.get_shap_summary_plot(shap_values, x)
        mlflow.log_artifact(plot_file_path)
        return plot_file_path

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

    def predict_for_county(self, county_name):
        """
        Predict bail amounts for cases in a specified county using the trained model.

        :param county_name: Name of the county to filter the data for.
        :return: DataFrame with actual, predicted bail amounts, and their differences.
        """
        # Load and preprocess data
        preprocessor = Preprocessing(self.config, self.judge_filter, self.county_filter)
        df, X, y = preprocessor.load_and_preprocess_data()

        # Filter data for the specified county
        df_county = df[df['county_name'] == county_name]
        # X_county = X[df['county_name'] == county_name]

        print("FEATURE NAMES: ", self.feature_names)

        # Ensure feature names match those used during training
        # X_county = X_county[self.feature_names]

        # Predict bail amounts for the county's cases using the trained model
        predicted_bail_amounts = self.model.predict(df_county[self.feature_names])

        df_county['predicted_bail_amount'] = predicted_bail_amounts

        # Calculate the difference between actual and predicted bail amounts
        df_county['difference'] = df_county['bail_amount'] - df_county['predicted_bail_amount']

        return df_county[['bail_amount', 'predicted_bail_amount', 'difference']]

    def train_model(self):
        """Train model"""
        from .data import load_data  # Import here to avoid circular import
        preprocessor = Preprocessing(self.config, self.judge_filter, self.county_filter)
        try:
            data, x_column, y = preprocessor.load_and_preprocess_data()
            x_train, y_train, x_test, y_test = preprocessor.prepare_for_training(x_column, y)
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

                    self.model = Model(model_type=model_type, good_hyperparameters=GOOD_HYPERPARAMETERS)

                    x_train_selected, x_test_selected = x_train, x_test

                    self.model.train(x_train_selected, y_train)

                    # Store feature names after training
                    self.feature_names = x_train_selected.columns

                    mse, r2 = self.model.evaluate(x_test_selected, y_test)
                    model_r2_scores.append(r2)
                    self.model.log_metrics(mse, r2, pd.DataFrame(x_train_selected, columns=x_train.columns),
                                           self.config.outputs_dir)

                    plot_partial_dependence(self.model.manager.model,
                                            pd.DataFrame(x_test_selected, columns=x_train.columns),
                                            features=x_train.columns.tolist(),
                                            outputs_dir=self.config.outputs_dir)
                    # plot_file_path = self.plot_and_save_shap(self.model,
                    #                                          pd.DataFrame(x_test_selected, columns=x_train.columns))
                    # mlflow.log_artifact(plot_file_path)
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
            model_params = self.model.manager.model.get_params()
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

            plot_file_path = self.plot_and_save_importance(self.model.manager.model, plot_params)

            # don't save results data to the database for now
            # save_data(self.session, result_obj)

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

    def save_trained_data(self, path):
        """Save the trained model data to a CSV file."""
        importance_df = pd.read_csv(os.path.join(self.config.outputs_dir, "feature_importance.csv"))
        importance_df.to_csv(path, index=False)
        logging.info(f"Trained data saved to {path}")


def get_judges(session):
    """Fetch the list of judges from the database."""
    result = session.execute(text("SELECT DISTINCT judge_name FROM pretrial.judges WHERE case_count > 5"))
    judges = [row[0] for row in result]
    return judges


def get_counties(session):
    """Fetch the list of counties from the database."""
    result = session.execute(text("SELECT DISTINCT county_name FROM pretrial.counties "
                                  "WHERE counties.number_of_cases > 5"))

    counties = [row[0] for row in result]
    return counties


def grade_targets(session, trained_data, trained_model_path, target, limit=10):
    """Grade each judge or county using a pre-trained general model."""
    from .data import load_data

    targets = None

    if target == "judge":
        logging.info("Grading judges using the general model.")
        targets = get_judges(session)

    elif target == "county":
        logging.info("Grading counties using the general model.")
        targets = get_counties(session)

    if limit:
        logging.info(f"Limiting the number of targets to {limit}")

    preprocessor = Preprocessing(TrainerConfig())
    try:
        data, x_column, y = preprocessor.load_and_preprocess_data()
        x_train, y_train, x_test, y_test = preprocessor.prepare_for_training(x_column, y)
    except ValueError as e:
        logging.error(f"Error in preprocessing data: {e}")
        return

    logging.info(f"Found {len(targets)} {target} targets.")

    for tgt in targets:
        logging.info(f"Grading target: {tgt}")
        target_data = None
        if target == "judge":
            target_data = load_data(session, judge_filter=tgt)
        if target == "county":
            target_data = load_data(session, county_filter=tgt)

        if target_data.empty:
            logging.warning(f"No data for target {tgt}. Skipping...")
            continue

        # Prepare the target data for prediction
        x_target = target_data.drop(columns=['first_bail_set_cash'])

        # Preprocess the target data using the same preprocessing steps
        x_target = preprocessor.preprocess_new_data(x_target)

        # Load the pre-trained model
        trained_model = Model(model_type='random_forest', good_hyperparameters=GOOD_HYPERPARAMETERS)
        trained_model.load_model(trained_model_path)

        # Apply the pre-trained model to the target's data to generate predictions
        predictions = trained_model.apply(x_target)

        # Log the results for the target
        logging.info(f"Predictions for target {tgt}: {predictions}")

        feature_importances = trained_model.get_feature_importances()
        logging.info(f"Feature importances for target {tgt}: {feature_importances}")


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
