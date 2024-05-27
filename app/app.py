"""This module is the main entry point of the application."""

import logging

from models import ModelTrainer
from models.model_trainer import ModelFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run():
    """This method runs the application.
    :return:
    """

    model_filter = ModelFilter(
        filter_type="",
        filter_value="",
    )

    ModelTrainer(model_filter).run()
