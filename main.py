import logging
import os
import time

import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from config import GOOD_HYPERPARAMETERS, query, model_types, tune_hyperparameters_flag, sql_values, model_for_selection, perform_feature_selection
from data_loader import load_data, create_engine_connection
from modeling import (
    train_model,
    evaluate_model,
    log_metrics,
    tune_hyperparameters,
    feature_selection
)
from preprocessing import preprocess_data, split_data
from utils import plot_feature_importance, write_current_r2, read_previous_r2, send_discord_notification

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
webhook_url = DISCORD_WEBHOOK_URL
DISCORD_AVATAR_URL = os.environ.get("DISCORD_AVATAR_URL")
avatar_url = DISCORD_AVATAR_URL

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


# Configuration parameters

def main():
    start_time = time.time()

    # Ensure the outputs directory exists
    outputs_dir = 'outputs'
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    # Database connection details
    user = os.environ.get("DB_USER")
    password = os.environ.get("DB_PASSWORD")
    host = os.environ.get("DB_HOST")
    port = os.environ.get("DB_PORT")
    dbname = os.environ.get("DB_NAME")

    # Create the connection string
    engine = create_engine_connection(user, password, host, port, dbname)

    data = load_data(engine, query, sql_values)
    X, y, y_bin = preprocess_data(data, outputs_dir)
    X_train, y_train, X_test, y_test = split_data(X, y_bin, outputs_dir)

    # Get the total number of cases used
    total_cases = len(data)
    logging.info(f"Total number of cases used: {total_cases}")

    # Path to the file storing the previous R2 value
    previous_r2_file = os.path.join(outputs_dir, 'previous_r2.txt')
    previous_r2 = read_previous_r2(previous_r2_file)

    model_r2_scores = []
    num_features = X_train.shape[1]

    mlflow.set_experiment("LawVision Model Training")

    with mlflow.start_run(run_name="Model Training Run") as run:
        for model_type in model_types:
            with mlflow.start_run(nested=True, run_name=model_type):
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("perform_feature_selection", perform_feature_selection)
                mlflow.log_param("tune_hyperparameters_flag", tune_hyperparameters_flag)

                if perform_feature_selection:
                    # Feature selection
                    X_train_selected, selector = feature_selection(X_train, y_train, model_type)
                    X_test_selected = selector.transform(X_test)
                    selected_features = X.columns[selector.support_].tolist()
                    mlflow.log_param("selected_features", selected_features)
                else:
                    X_train_selected, X_test_selected = X_train, X_test

                # Tune hyperparameters and train model
                if tune_hyperparameters_flag:
                    model = tune_hyperparameters(X_train_selected, y_train, model_type)
                else:
                    model = train_model(X_train_selected, y_train, model_type, GOOD_HYPERPARAMETERS)

                mse, r2 = evaluate_model(model, X_test_selected, y_test)
                model_r2_scores.append(r2)
                log_metrics(model, mse, r2, pd.DataFrame(X_train_selected, columns=X.columns if not perform_feature_selection else X.columns[selector.support_]), outputs_dir)  # Use selected features

                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)

        average_r2 = sum(model_r2_scores) / len(model_r2_scores)
        logging.info(f"Average R-squared across all models: {average_r2}")
        mlflow.log_metric("average_r2", average_r2)

        # Determine if R2 increased, decreased, or stayed the same
        if previous_r2 is not None:
            if average_r2 > previous_r2:
                r2_comparison = f'R² increased by {(average_r2 - previous_r2):.4f}'
            elif average_r2 < previous_r2:
                r2_comparison = f'R² decreased by {(previous_r2 - average_r2):.4f}'
            else:
                r2_comparison = "R² stayed the same"
        else:
            r2_comparison = "No previous R² value"

        # Write the current average R2 value to file for future comparisons
        write_current_r2(previous_r2_file, average_r2)

        # Plot feature importance
        if hasattr(model, "feature_importances_"):
            importance_df = pd.read_csv(os.path.join(outputs_dir, 'feature_importance.csv'))
            elapsed_time = time.time() - start_time
            plot_file_path = plot_feature_importance(importance_df, average_r2, total_cases, r2_comparison, outputs_dir,
                                                     elapsed_time, model_types, num_features, model_for_selection)

            mlflow.log_artifact(plot_file_path)

    # Send Discord notification
    message = (f"\n\n\n\n\n\n\nModel training completed.\n"
               f"Average R²:  \n **{average_r2:.4f}**\n\n"
               f"R² comparison: \n**{r2_comparison}**\n\n"
               f"Total cases: \n**{total_cases}**\n\n"
               f"Training time: \n**{time.time() - start_time:.2f}** seconds\n\n"
               f"Models used: \n**{', '.join(model_types)}**\n\n"
               f"Model used for feature selection: \n**{model_for_selection}**\n\n"
               f"Number of features: \n**{num_features}\n\n**"
               f"Bar chart: \n"
               )
    send_discord_notification(webhook_url, message, plot_file_path, avatar_url)


if __name__ == "__main__":
    main()
