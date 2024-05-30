import logging

from sqlalchemy.orm import Session

from .data import create_db_connection


def run():
    from app.train import ModelTrainer, grade_judges_with_general_model
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Running the application...")

    # Create a database connection and session
    engine = create_db_connection()
    session = Session(bind=engine)

    # Train the general model
    general_trainer = ModelTrainer()
    general_trainer.run()

    # Grade judges using the general model
    grade_judges_with_general_model(session, general_trainer)

    # Close the session
    session.close()


def main():
    run()


if __name__ == "__main__":
    main()
