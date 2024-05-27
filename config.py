from utils import QueryUtils

# Model configuration
# model_types = ["gradient_boosting", "random_forest", "hist_gradient_boosting", "ada_boost", "bagging", "extra_trees",
#             "lasso", "ridge", "neural_network"]
model_types = ["random_forest"]  # Model types to use for training
tune_hyperparameters_flag = False  # Set to True to tune hyperparameters, False to skip
perform_feature_selection = False  # Set to True to perform feature selection, False to skip
model_for_selection = "random_forest"  # Model to use for feature selection

# Data selection parameters
case_limit = 10000000
judge_names = []
county_names = []
sql_values = {
    "limit": case_limit,
    "judge_names": judge_names,
    "county_names": county_names
}

# Generate SQL conditions for judges and counties
judge_names_condition = ""
if judge_names:
    judge_names_list = ", ".join([f"'{name}'" for name in judge_names])
    judge_names_condition = f"AND j.judge_name IN ({judge_names_list})"

county_names_condition = ""
if county_names:
    county_names_list = ", ".join([f"'{name}'" for name in county_names])
    county_names_condition = f"AND co.county_name IN ({county_names_list})"

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
    {judge_names_condition}
    {county_names_condition}
LIMIT %(limit)s;
"""

# Generate the final SQL query
query = QueryUtils.get_query(sql_values, query_template)
