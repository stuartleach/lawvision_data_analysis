import logging

import pandas as pd
from sqlalchemy.orm import Session


def run_model(source="csv", train=True, grade=False, trained_data_path="outputs/trained_data.csv",
              trained_model_path="outputs/trained_model.joblib"):
    from .train import ModelTrainer
    from .train import grade_targets
    from app import create_db_connection

    session = None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    if train:
        logging.info("Training the model...")
        if source == "db":
            logging.info("Creating a database connection...")
            engine = create_db_connection()
            session = Session(bind=engine)
            general_trainer = ModelTrainer(source="db")
            general_trainer.run()
            general_trainer.save_trained_data(trained_data_path)
            general_trainer.model.save_model(trained_model_path)
            logging.info("Running the application using db...")

        elif source == "csv":
            logging.info("No database connection required. Using CSV as the source.")
            general_trainer = ModelTrainer(source="csv")
            general_trainer.run()
            logging.info("Saving the trained data and model...")
            general_trainer.save_trained_data(trained_data_path)
            general_trainer.model.save_model(trained_model_path)
            logging.info("Running the application using csv...")
        else:
            raise ValueError("Invalid source provided. Please provide either 'db' or 'csv'.")

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
    if session:
        session.close()


def run():
    run_model(source="csv", train=True, grade=False)


if __name__ == "__main__":
    run()
