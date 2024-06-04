"""This file contains the parameters used in the application."""

import os

from dotenv import load_dotenv
from sqlalchemy import select, cast, Numeric, Select

from app.db.db_types import Case, Race, Representation, Court, NYIncome, District, County, Judge

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
        "max_depth": 4,
        "min_samples_leaf": 4,
        "n_estimators": 300,
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

BAIL_THRESHOLD = 80000

QUERY_LIMIT = 1000000

BASE_QUERY: Select = (
    select(
        Case.gender,
        Case.ethnicity,
        Race.race,
        Case.age_at_arrest,
        Case.known_days_in_custody,
        Case.top_charge_at_arraign,
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
        cast(Case.first_bail_set_cash, Numeric) > 1
    ))

COLUMNS_OF_INTEREST = [
    "gender", "ethnicity", "age_at_crime", "age_at_arrest", "top_charge_severity_at_arrest",
    "top_charge_at_arrest_violent_felony_ind", "case_type", "top_charge_at_arraign", "top_severity_at_arraign",
    "top_charge_weight_at_arraign", "top_charge_at_arraign_violent_felony_ind", "arraign_charge_category",
    "app_count_arraign_to_dispo_released", "app_count_arraign_to_dispo_detained", "app_count_arraign_to_dispo_total",
    "def_attended_sched_pretrials", "remanded_to_jail_at_arraign", "ror_at_arraign", "bail_set_and_posted_at_arraign",
    "bail_set_and_not_posted_at_arraign", "nmr_at_arraign", "release_decision_at_arraign",
    "representation_at_securing_order",
    "pretrial_supervision_at_arraign", "contact_pretrial_service_agency", "electronic_monitoring",
    "travel_restrictions",
    "passport_surrender", "no_firearms_or_weapons", "maintain_employment", "maintain_housing", "maintain_school",
    "placement_in_mandatory_program", "removal_to_hospital", "obey_order_of_protection",
    "obey_court_conditions_family_offense",
    "other_nmr", "order_of_protection", "first_bail_set_cash", "first_bail_set_credit",
    "first_insurance_company_bail_bond",
    "first_secured_surety_bond", "first_secured_app_bond", "first_unsecured_surety_bond", "first_unsecured_app_bond",
    "first_partially_secured_surety_bond", "partially_secured_surety_bond_perc", "first_partially_secured_app_bond",
    "partially_secured_app_bond_perc", "bail_made_indicator", "warrant_ordered_btw_arraign_and_dispo",
    "dat_wo_ws_prior_to_arraign", "first_bench_warrant_date", "non_stayed_wo", "num_of_stayed_wo", "num_of_row",
    "docket_status",
    "disposition_type", "disposition_detail", "dismissal_reason", "disposition_date", "most_severe_sentence",
    "top_conviction_law",
    "top_conviction_article_section", "top_conviction_attempt_indicator", "top_charge_at_conviction",
    "top_charge_severity_at_conviction", "top_charge_weight_at_conviction",
    "top_charge_at_conviction_violent_felony_ind",
    "days_arraign_remand_first_released", "known_days_in_custody", "days_arraign_bail_set_to_first_posted",
    "days_arraign_bail_set_to_first_release", "days_arraign_to_dispo", "ucmslivedate", "prior_vfo_cnt",
    "prior_nonvfo_cnt",
    "prior_misd_cnt", "pend_vfo", "pend_nonvfo", "pend_misd", "supervision", "rearrest", "rearrest_date",
    "rearrest_firearm",
    "rearrest_date_firearm", "arr_cycle_id", "race_id", "court_id", "county_id", "top_charge_id", "representation_id",
    "judge_id",
    "case_uuid", "district_id", "race_uuid", "race", "average_bail_amount_cases", "number_of_cases_cases",
    "average_known_days_in_custody", "remanded_percentage", "bail_set_percentage", "disposed_at_arraign_percentage",
    "ror_percentage", "nonmonetary_release_percentage", "court_uuid", "court_name", "court_ori", "district", "region",
    "court_type",
    "average_bail_amount_courts", "number_of_cases_courts", "judge_name", "judge_uuid", "average_bail_set",
    "unique_districts",
    "case_count", "counties", "court_names", "representation_types", "top_charges_at_arraign", "disposition_types",
    "remand_to_jail_count", "ror_count", "bail_set_and_posted_count", "bail_set_and_not_posted_count",
    "supervision_conditions",
    "average_sentence_severity", "offense_types", "rearrest_rate", "representation_uuid", "representation_type",
    "average_bail_amount", "number_of_cases", "average_known_days_in_custody_reps", "remanded_percentage_reps",
    "bail_set_percentage_reps", "disposed_at_arraign_percentage_reps", "ror_percentage_reps",
    "nonmonetary_release_percentage_reps", "county_uuid", "county_name", "average_bail_amount_counties",
    "number_of_cases_counties", "median_income", "median_id"
]
