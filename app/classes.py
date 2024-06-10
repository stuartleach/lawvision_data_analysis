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
    gradient_boosting: str = "gradient_boosting"
    random_forest: str = "random_forest"
    hist_gradient_boosting: str = "hist_gradient_boosting"
    ada_boost: str = "ada_boost"
    bagging: str = "bagging"
    extra_trees: str = "extra_trees"


model_type = ModelType()


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
    model_types=[model_type.gradient_boosting],
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
    quiet: bool = False
    plots: bool = True
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


@dataclass
class ResultObject:
    """Result object class."""

    def __init__(self, judge_filter, county_filter, model_type, model_target_type, model_target, model_params,
                 average_bail_amount, r_squared, mean_squared_error, dataframe, total_cases):
        self.judge_filter = judge_filter
        self.county_filter = county_filter
        self.model_type = model_type
        self.model_target_type = model_target_type
        self.model_target = model_target
        self.model_params = model_params
        self.average_bail_amount = average_bail_amount
        self.r_squared = r_squared
        self.mean_squared_error = mean_squared_error
        self.dataframe = dataframe
        self.total_cases = total_cases
