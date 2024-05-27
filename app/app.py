# Setup logging
import logging

from models.model_trainer import ModelTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class App:
    def __init__(self):
        self.trainer = ModelTrainer()

    def run(self):
        self.trainer.run()
