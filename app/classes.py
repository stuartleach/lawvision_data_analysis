import os
import time
from dataclasses import dataclass
from typing import List

from sqlalchemy import Engine

from app.env import DISCORD_WEBHOOK_URL, DISCORD_AVATAR_URL


@dataclass
class QueryParams:
    """Dataclass for SQL query parameters."""
    limit: int = 10000000
    judge_names: List[str] = list
    county_names: List[str] = list

    def to_dict(self):
        return {
            "limit": self.limit,
            "judge_names": self.judge_names,
            "county_names": self.county_names,
        }


query_params = QueryParams(
    limit=10000000,
    judge_names=[],
    county_names=[],
)


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
    """Data class for model configuration.
    :param model_types:
    :param tune_hyperparameters:
    :param query_params:
    """

    model_types: List[str]
    tune_hyperparameters: bool
    query_params: QueryParams


model_config = ModelConfig(
    model_types=["random_forest", "gradient_boosting"],
    tune_hyperparameters=False,
    query_params=query_params,
)


@dataclass
class NotificationData:
    """Data class for notification data.

    :param performance_data:
    :param plot_file_path:
    :param model_info:
    """

    performance_data: dict
    plot_file_path: str
    model_info: dict


@dataclass
class TrainerConfig:
    """Data class for trainer configuration."""
    outputs_dir: str = "outputs"
    webhook_url: str = DISCORD_WEBHOOK_URL
    avatar_url: str = DISCORD_AVATAR_URL
    baseline_profile_name: str = "baseline"
    start_time: float = time.time()
    previous_r2_file: str = os.path.join("outputs", "previous_r2.txt")


@dataclass
class DataLoaderConfig:
    engine: Engine
    query: str
    outputs_dir: str
