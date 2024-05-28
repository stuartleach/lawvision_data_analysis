"""This module is the main entry point of the application."""

import logging

from app.train import ModelFilter, ModelTrainer


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    """This method runs the application."""
    logging.info("Running the application...")
    # model_filter = ModelFilter(filter_type="county_name", filter_value="Kings")
    model_filter = ModelFilter(filter_type="", filter_value="")
    ModelTrainer(model_filter).run()


def main():
    run()


if __name__ == "__main__":
    main()

