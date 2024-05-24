# config.py


# Configuration parameters
# model_types = ["gradient_boosting", "random_forest", "hist_gradient_boosting", "ada_boost", "bagging", "extra_trees",
#             "lasso", "ridge", "neural_network"]
model_types = ["gradient_boosting"]  # Model types to use for training
# model_types = ["lasso"]  # Model types to use for training

tune_hyperparameters_flag = True  # Set to True to tune hyperparameters, False to skip
perform_feature_selection = False  # Set to True to perform feature selection, False to skip
model_for_selection = "random_forest"  # Model to use for feature selection

case_limit = 10000000
sql_values = {"limit": case_limit}
# model_for_selection = "random_forest"  # Model to use for feature selection

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
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
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
    "gradient_boosting": {'learning_rate': 0.1, 'max_depth': 7, 'max_features': 'sqrt', 'min_samples_leaf': 4,
                          'min_samples_split': 10, 'n_estimators': 300, 'subsample': 0.9},
    "random_forest": {
        "max_depth": None,
        "min_samples_leaf": 4,
        "min_samples_split": 2,
        "n_estimators": 300
    },
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
query = """
SELECT 
    c.gender,
    c.age_at_arrest,
    -- c.top_arrest_law,
    -- c.top_charge_severity_at_arrest,
    -- c.top_charge_weight_at_arrest,
    -- c.top_charge_at_arrest_violent_felony_ind,
    c.top_charge_at_arraign,
    -- c.maintain_employment,
    -- c.maintain_housing,
    -- c.maintain_school,
    CASE 
        WHEN c.ror_at_arraign = 'Y' THEN 0 
        ELSE c.first_bail_set_cash::numeric 
    END AS first_bail_set_cash,
    c.known_days_in_custody,
    -- c.case_type,
    -- c.dat_wo_ws_prior_to_arraign,
    -- c.top_arraign_article_section,
    -- c.top_arraign_attempt_indicator,
    -- c.top_arraign_law,
    -- d.district_name,
    -- c.prior_vfo_cnt,
    -- c.prior_nonvfo_cnt,
    -- c.rearrest_firearm,
    c.pend_nonvfo,
    c.pend_misd,
    c.pend_vfo,
    -- c.ethnicity,
    co.county_name,
    -- co.median_income,
    -- i.population,
    -- i.number_of_households,
    -- j.judge_name,
    -- c.arrest_month,
    -- c.arrest_year,
    -- ct.court_name,
    -- ct.region,
    rep.representation_type,
    r.race
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
    -- AND c.age_at_arrest::numeric IS NOT NULL
    -- AND c.age_at_arrest::numeric > 0
    -- AND (co.county_name = 'Kings' OR co.county_name = 'New York' OR co.county_name = 'Bronx' OR 
       -- co.county_name = 'Queens' OR co.county_name = 'Richmond')
    AND c.first_bail_set_cash::numeric < 80000
    AND c.first_bail_set_cash NOTNULL 
LIMIT %(limit)s;
"""
