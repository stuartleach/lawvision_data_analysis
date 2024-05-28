"""This module is the main entry point of the application."""

import logging

from .train import ModelTrainer


def run():
    """This method runs the application."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    logging.info("Running the application...")
    ModelTrainer().run()


def main():
    run()


if __name__ == "__main__":
    main()
