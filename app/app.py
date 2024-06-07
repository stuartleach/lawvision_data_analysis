import logging

import pandas as pd
from sqlalchemy.orm import Session

from .train import ModelTrainer, grade_targets


def run_model(train=True, grade=False, trained_data_path="outputs/trained_data.csv",
              trained_model_path="outputs/trained_model.joblib"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Running the application...")
    from .data import create_db_connection

    # Create a database connection and session
    engine = create_db_connection()
    session = Session(bind=engine)

    if train:
        general_trainer = ModelTrainer(county_filter="", judge_filter="")
        general_trainer.run()
        general_trainer.save_trained_data(trained_data_path)
        general_trainer.model.save_model(trained_model_path)

    if grade:
        if not trained_data_path or not trained_model_path:
            raise ValueError("No trained data or model path provided for grading.")

        # Load the trained data from the CSV file
        trained_data = pd.read_csv(trained_data_path)

        grade_targets(session, trained_data, trained_model_path, "judge")

    # Close the session
    session.close()


def run():
    run_model(train=True, grade=False)


if __name__ == "__main__":
    run()
