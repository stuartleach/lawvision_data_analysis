"""Configuration file for the model training pipeline."""

from dataclasses import dataclass
from typing import List

from utils import get_query


@dataclass
class SQLValues:
    """Data class for SQL values.

    :param limit:
    :param judge_names:
    :param county_names:
    """

    limit: int
    judge_names: List[str]
    county_names: List[str]


@dataclass
class ModelType:
    """Class to define the model types.

    :param gradient-boosting:
    :param random-forest:
    :param hist-gradient-boosting:
    :param ada-boost:
    :param bagging:
    :param extra-trees:
    """

    gradient_boosting: str
    random_forest: str
    hist_gradient_boosting: str
    ada_boost: str
    bagging: str
    extra_trees: str


@dataclass
class ModelConfig:
    """Data class for model configuration.

    :param model_types:
    :param model_for_selection:
    :param perform_feature_selection:
    :param tune_hyperparameters:
    :param sql_values:
    """

    model_types: List[str]
    model_for_selection: str
    perform_feature_selection: bool
    tune_hyperparameters: bool
    sql_values: SQLValues


model_type = ModelType(
    gradient_boosting="gradient_boosting",
    random_forest="random_forest",
    hist_gradient_boosting="hist_gradient_boosting",
    ada_boost="ada_boost",
    bagging="bagging",
    extra_trees="extra_trees",
)

model_config = ModelConfig(
    model_types=[model_type.gradient_boosting],
    model_for_selection=model_type.random_forest,
    perform_feature_selection=False,
    tune_hyperparameters=False,
    sql_values=SQLValues(
        limit=1000000,
        judge_names=[],
        county_names=[],
    ),
)

# Generate SQL conditions for judges and counties
JUDGE_NAMES_CONDITION = ""
if model_config.sql_values.judge_names:
    JUDGE_NAMES_LIST = ", ".join(
        [f"'{name}'" for name in model_config.sql_values.judge_names]
    )
    JUDGE_NAMES_CONDITION = f"AND j.judge_name IN ({JUDGE_NAMES_LIST})"
COUNTY_NAMES_CONDITION = ""
if model_config.sql_values.county_names:
    COUNTY_NAMES_LIST = ", ".join(
        [f"'{name}'" for name in model_config.sql_values.county_names]
    )
    COUNTY_NAMES_CONDITION = f"AND co.county_name IN ({COUNTY_NAMES_LIST})"

# SQL query template
query_template = f"""
SELECT
    c.gender,
    c.ethnicity,
    r.race,
    c.age_at_arrest,
    c.known_days_in_custody,
    c.top_charge_at_arraign,
    c.first_bail_set_cash,
    c.prior_vfo_cnt,
    c.prior_nonvfo_cnt,
    c.prior_misd_cnt,
    c.pend_nonvfo,
    c.pend_misd,
    c.pend_vfo,
    j.judge_name,
    i.median_household_income
FROM
    pretrial.cases c
JOIN
    pretrial.races r ON c.race_id = r.race_uuid
JOIN
    pretrial.counties co ON c.county_id = co.county_uuid
JOIN
    pretrial.ny_income i ON co.county_name = i.county
JOIN
    pretrial.judges j ON c.judge_id = j.judge_uuid
JOIN
    pretrial.districts d ON c.district_id = d.district_uuid
JOIN
    pretrial.courts ct ON c.court_id = ct.court_uuid
JOIN
    pretrial.representation rep ON c.representation_id = rep.representation_uuid
WHERE
    c.first_bail_set_cash IS NOT NULL
    AND c.first_bail_set_cash::numeric < 80000
    AND c.first_bail_set_cash::numeric > 1
    {JUDGE_NAMES_CONDITION}
    {COUNTY_NAMES_CONDITION}
LIMIT %(limit)s;
"""

# Generate the final SQL query
query = get_query(model_config.sql_values, query_template)
