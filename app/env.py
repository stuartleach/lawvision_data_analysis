"""This file contains the parameters used in the application."""

import os

from dotenv import load_dotenv
from sqlalchemy import select, cast, Numeric, Select

from app.db_types import Case, Race, Representation, Court, NYIncome, District, County, Judge

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
DISCORD_AVATAR_URL = os.environ.get("DISCORD_AVATAR_URL")

HYPERPARAMETER_GRIDS = {
    "gradient_boosting": {
        "n_estimators": [100, 200, 300],  # Reduced range
        "learning_rate": [0.01, 0.1, 0.2],  # Reduced range
        "max_depth": [3, 5, 7],  # Reduced range
        "min_samples_split": [2, 10],  # Reduced range
        "min_samples_leaf": [1, 4],  # Reduced range
        "subsample": [0.7, 0.9],  # Reduced range
        "max_features": ["sqrt", "log2"],  # Reduced range
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True, False],
        "criterion": ["squared_error", "friedman_mse"],
    },
    "hist_gradient_boosting": {
        "max_iter": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_leaf": [10, 20, 30],
    },
    "ada_boost": {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1.0]},
    "bagging": {
        "n_estimators": [10, 50, 100],
        "max_samples": [0.5, 0.7, 1.0],
        "max_features": [0.5, 0.7, 1.0],
    },
    "extra_trees": {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "lasso": {"alpha": [0.01, 0.1, 1.0]},
    "ridge": {"alpha": [0.01, 0.1, 1.0]},
    "neural_network": {
        "layers": [[64, 64], [128, 64], [64, 32]],
        "activation": ["relu", "tanh"],
        "optimizer": ["adam", "sgd"],
        "epochs": [10, 50],
        "batch_size": [32, 64],
    },
}
# Good hyperparameters for different models
GOOD_HYPERPARAMETERS = {
    "gradient_boosting": {
        "learning_rate": 0.2,
        "max_depth": 4,  # old 4
        "min_samples_leaf": 4,  # old 4
        "n_estimators": 300,  # old 300
    },
    "random_forest": {
        "bootstrap": False,
        "criterion": "friedman_mse",
        "max_depth": None,
        "max_features": "sqrt",
        "min_samples_leaf": 2,
        "min_samples_split": 2,
        "n_estimators": 300,
    },
    "hist_gradient_boosting": {
        "max_iter": 300,
        "max_depth": None,
        "min_samples_leaf": 30,
    },
    "ada_boost": {"n_estimators": 100, "learning_rate": 1.0},
    "bagging": {"n_estimators": 100, "max_samples": 1.0, "max_features": 0.5},
    "extra_trees": {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
    },
    "lasso": {"alpha": 0.1},
    "ridge": {"alpha": 0.1},
    "neural_network": {
        "input_dim": 26,
        "layers": [64, 64],
        "activation": "relu",
        "optimizer": "adam",
    },
}

BAIL_THRESHOLD = 39000

QUERY_LIMIT = 1500

BASE_QUERY: Select = (
    select(
        Case.gender,
        Case.ethnicity,
        Race.race,
        Case.age_at_arrest,
        Case.known_days_in_custody,
        Case.top_charge_weight_at_arraign,
        Case.first_bail_set_cash,
        Case.prior_vfo_cnt,
        Case.prior_nonvfo_cnt,
        Case.prior_misd_cnt,
        Case.pend_nonvfo,
        Case.pend_misd,
        Case.pend_vfo,
        County.county_name,
        Judge.judge_name,
        NYIncome.median_household_income
    )
    .join(Race, Case.race_id == Race.race_uuid)
    .join(County, Case.county_id == County.county_uuid)
    .join(NYIncome, County.county_name == NYIncome.county)
    .join(Judge, Case.judge_id == Judge.judge_uuid)
    .join(District, Case.district_id == District.district_uuid)
    .join(Court, Case.court_id == Court.court_uuid)
    .join(Representation, Case.representation_id == Representation.representation_uuid)
    .where(
        Case.first_bail_set_cash.isnot(None),
        cast(Case.first_bail_set_cash, Numeric) < BAIL_THRESHOLD,
        Case.top_charge_weight_at_arraign.is_not(None),
        cast(Case.first_bail_set_cash, Numeric) > 1,
        cast(Case.known_days_in_custody, Numeric) < 100,  # Get rid of outliers
    )
)
