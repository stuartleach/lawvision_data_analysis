from dataclasses import dataclass, field
from typing import List

from app.sql_values import SQLValues


@dataclass
class ModelType:
    """Enum-like class for model types (using strings for simplicity)."""
    GRADIENT_BOOSTING: str = "gradient_boosting"
    RANDOM_FOREST: str = "random_forest"
    HIST_GRADIENT_BOOSTING: str = "hist_gradient_boosting"
    ADA_BOOST: str = "ada_boost"
    BAGGING: str = "bagging"
    EXTRA_TREES: str = "extra_trees"


@dataclass
class ModelConfig:
    """Dataclass for model configuration."""
    model_types: List[str] = field(default_factory=lambda: [ModelType.RANDOM_FOREST])
    tune_hyperparameters: bool = False
    sql_values: SQLValues = SQLValues()  # Default SQLValues object


model_config = ModelConfig()
