"""
Configuration file for the model training pipeline.
"""

from utils import get_query

# Model configuration
# model_types = ["gradient_boosting", "random_forest", "hist_gradient_boosting",
#           "ada_boost", "bagging", "extra_trees", "lasso", "ridge", "neural_network"]
model_types = ["random_forest"]  # Model types to use for training
TUNE_HYPERPARAMETERS_FLAG = False  # Set to True to tune hyperparameters, False to skip
PERFORM_FEATURE_SELECTION_FLAG = (
    False  # Set to True to perform feature selection, False to skip
)
MODEL_FOR_SELECTION = "random_forest"  # Model to use for feature selection

# Data selection parameters
CASE_LIMIT = 10000000
JUDGE_NAMES = []
COUNTY_NAMES = []
sql_values = {
    "limit": CASE_LIMIT,
    "judge_names": JUDGE_NAMES,
    "county_names": COUNTY_NAMES,
}

# Generate SQL conditions for judges and counties
JUDGE_NAMES_CONDITION = ""
if JUDGE_NAMES:
    JUDGE_NAMES_LIST = ", ".join([f"'{name}'" for name in JUDGE_NAMES])
    JUDGE_NAMES_CONDITION = f"AND j.judge_name IN ({JUDGE_NAMES_LIST})"
COUNTY_NAMES_CONDITION = ""
if COUNTY_NAMES:
    COUNTY_NAMES_LIST = ", ".join([f"'{name}'" for name in COUNTY_NAMES])
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
query = get_query(sql_values, query_template)
