# config.py
from utils import get_query

# Configuration parameters
# model_types = ["gradient_boosting", "random_forest", "hist_gradient_boosting", "ada_boost", "bagging", "extra_trees",
#             "lasso", "ridge", "neural_network"]

model_types = ["gradient_boosting"]  # Model types to use for training

tune_hyperparameters_flag = False  # Set to True to tune hyperparameters, False to skip
perform_feature_selection = False  # Set to True to perform feature selection, False to skip
model_for_selection = "random_forest"  # Model to use for feature selection

case_limit = 10000000
judge_names = []
county_names = []
sql_values = {
    "limit": case_limit,
    "judge_names": judge_names,
    "county_names": county_names
}

# SQL query
query_template = """
SELECT 
    c.gender,
    c.ethnicity,
    r.race,
    c.age_at_arrest,
    c.known_days_in_custody,
    c.top_charge_at_arraign,
    /*CASE 
        WHEN c.ror_at_arraign = 'Y' THEN 0 
        ELSE c.first_bail_set_cash::numeric 
    END AS*/ 
    c.first_bail_set_cash,
    c.prior_vfo_cnt,
    c.prior_nonvfo_cnt,
    c.prior_misd_cnt,
    c.pend_nonvfo,
    c.pend_misd,
    c.pend_vfo,
    -- c.rearrest_firearm,
    j.judge_name,
    i.median_household_income
    -- i.population
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

query = get_query(sql_values, query_template)
