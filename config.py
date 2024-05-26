# config.py


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

# Hyperparameter grids for different models
HYPERPARAMETER_GRIDS = {
    "gradient_boosting": {
        'n_estimators': [100, 200, 300],  # Reduced range
        'learning_rate': [0.01, 0.1, 0.2],  # Reduced range
        'max_depth': [3, 5, 7],  # Reduced range
        'min_samples_split': [2, 10],  # Reduced range
        'min_samples_leaf': [1, 4],  # Reduced range
        'subsample': [0.7, 0.9],  # Reduced range
        'max_features': ['sqrt', 'log2']  # Reduced range
    },
    "random_forest": {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False],
        'criterion': ['squared_error', 'friedman_mse']
    },
    "hist_gradient_boosting": {
        'max_iter': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_leaf': [10, 20, 30]
    },
    "ada_boost": {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1.0]
    },
    "bagging": {
        'n_estimators': [10, 50, 100],
        'max_samples': [0.5, 0.7, 1.0],
        'max_features': [0.5, 0.7, 1.0]
    },
    "extra_trees": {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "lasso": {
        'alpha': [0.01, 0.1, 1.0]
    },
    "ridge": {
        'alpha': [0.01, 0.1, 1.0]
    },
    "neural_network": {
        'layers': [[64, 64], [128, 64], [64, 32]],
        'activation': ['relu', 'tanh'],
        'optimizer': ['adam', 'sgd'],
        'epochs': [10, 50],
        'batch_size': [32, 64]
    }
}

# Good hyperparameters for different models
GOOD_HYPERPARAMETERS = {
    "gradient_boosting": {
        'learning_rate': 0.2,
        'max_depth': 4,
        'min_samples_leaf': 4,
        'n_estimators': 300,
    },
    "random_forest": {'bootstrap': False, 'criterion': 'friedman_mse', 'max_depth': None, 'max_features': 'sqrt',
                      'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 300},
    "hist_gradient_boosting": {
        "max_iter": 300,
        "max_depth": None,
        "min_samples_leaf": 30
    },
    "ada_boost": {
        'n_estimators': 100,
        'learning_rate': 1.0
    },
    "bagging": {
        'n_estimators': 100,
        'max_samples': 1.0,
        'max_features': 0.5
    },
    "extra_trees": {
        'n_estimators': 300,
        'max_depth': 20,
        'min_samples_split': 10,
        'min_samples_leaf': 4
    },
    "lasso": {
        "alpha": 0.1
    },
    "ridge": {
        "alpha": 0.1
    },
    "neural_network": {
        "input_dim": 26,
        "layers": [64, 64],
        "activation": "relu",
        "optimizer": "adam"
    }
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


def get_query(sql_values_to_interpolate):
    judge_names_condition = "AND j.judge_name = ANY(%(judge_names)s)" if sql_values_to_interpolate[
        "judge_names"] else ""
    county_names_condition = "AND co.county_name = ANY(%(county_names)s)" if sql_values_to_interpolate[
        "county_names"] else ""

    resulting_query = query_template.format(
        judge_names_condition=judge_names_condition,
        county_names_condition=county_names_condition
    )
    return resulting_query


query = get_query(sql_values)
