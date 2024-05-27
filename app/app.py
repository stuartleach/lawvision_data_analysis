# Setup logging
import logging

from .model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


class Judge:
    def __init__(self, name, cases, average_bail_amount):
        self.name = name
        self.cases = cases
        self.average_bail_amount = average_bail_amount
        self.bail_reform_score = None


class County:
    def __init__(self, name, cases, average_bail_amount):
        self.name = name
        self.cases = cases
        self.average_bail_amount = average_bail_amount
        self.bail_reform_score = None


class App:
    def __init__(self):
        self.trainer = ModelTrainer()

    def run(self):
        self.trainer.run()
