import logging

import pandas as pd
from sqlalchemy.orm import Session

from .train import ModelTrainer, grade_targets

# test

def run_model(train=True, grade=False, trained_data_path="outputs/trained_data.csv",
              trained_model_path="outputs/trained_model.joblib"):
    from .data import create_db_connection
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Running the application...")
    engine = create_db_connection()
    session = Session(bind=engine)

    if train:
        trainer = ModelTrainer(county_filter="", judge_filter="", quiet=True)
        trainer.run()
        trainer.save_trained_data(trained_data_path)
        trainer.model.save_model(trained_model_path)

        predictions = trainer.predict_for_county("Kings")
        # print(predictions[['first_bail_set_cash', 'predicted_bail_amount', 'difference']])
        predictions.to_csv("outputs/predictions.csv", index=False)

    if grade:
        if not trained_data_path or not trained_model_path:
            raise ValueError("No trained data or model path provided for grading.")
        trained_data = pd.read_csv(trained_data_path)
        grade_targets(session, trained_data, trained_model_path, "judge")
    session.close()


def run():
    run_model(train=True, grade=False)


if __name__ == "__main__":
    run()
