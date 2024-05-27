"""This module is the main entry point of the application."""

import logging

from models import ModelTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


def run():
    """This method runs the application.
    :return:
    """
    ModelTrainer().run()
