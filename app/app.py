import logging

import pandas as pd
from sqlalchemy.orm import Session

from .train import ModelTrainer, grade_targets


def run_model(source="db", train=True, grade=False, trained_data_path="outputs/trained_data.csv",
              trained_model_path="outputs/trained_model.joblib"):
    from .data import create_db_connection
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    engine = create_db_connection()
    session = Session(bind=engine)
    logging.info("Running the application...")
    general_trainer = None
    if train:
        # Train the general model
        general_trainer = ModelTrainer()
        general_trainer.run()
        # Save the trained data to a CSV file
        general_trainer.save_trained_data(trained_data_path)
        # Save the trained model to a file
        general_trainer.model.save_model(trained_model_path)

    logging.info("general_trainer: %s", general_trainer)

    if grade:
        if not trained_data_path or not trained_model_path:
            raise ValueError("No trained data or model path provided for grading.")

        # Load the trained data from the CSV file
        trained_data = pd.read_csv(trained_data_path)
        print("trained_data", trained_data)
        for column in ["model_target", "model_type"]:
            print("column", column)

        for row in trained_data.iterrows():
            print("row", row)
        # Grade counties using the trained model
        grade_targets(session, trained_data, trained_model_path, "judge")

    # Close the session
    session.close()


def run_using_csv():
    data = pd.read_csv("./sources/ONYC for Web.csv")
    print("data", data.head())


def run():
    # run_model(source="db", train=True, grade=False)
    # data = pd.read_csv("data.csv")
    run_using_csv()


if __name__ == "__main__":
    run()
