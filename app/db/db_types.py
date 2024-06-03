import uuid

from sqlalchemy import (
    ARRAY,
    Column,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    Text,
    UniqueConstraint, JSON, Enum,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Result(Base):
    __tablename__ = "results"
    __table_args__ = {"schema": "pretrial"}
    result_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_target_type = Column(Enum("county_name", "judge_name", "baseline", name="model_target_type"), nullable=False)
    model_target = Column(Text, default="baseline", nullable=False)
    model_type = Column(Text, nullable=False)
    model_params = Column(JSON)
    average_bail_amount = Column(Numeric)
    r_squared = Column(Numeric)
    mean_squared_error = Column(Numeric)
    gender_importance = Column(Numeric)
    ethnicity_importance = Column(Numeric)
    race_importance = Column(Numeric)
    age_at_arrest_importance = Column(Numeric)
    known_days_in_custody_importance = Column(Numeric)
    top_charge_at_arraign_importance = Column(Numeric)
    first_bail_set_cash_importance = Column(Numeric)
    prior_vfo_cnt_importance = Column(Numeric)
    prior_nonvfo_cnt_importance = Column(Numeric)
    prior_misd_cnt_importance = Column(Numeric)
    pend_nonvfo_importance = Column(Numeric)
    pend_misd_importance = Column(Numeric)
    pend_vfo_importance = Column(Numeric)
    county_name_importance = Column(Numeric)
    judge_name_importance = Column(Numeric)
    median_household_income_importance = Column(Numeric)
    time_elapsed = Column(Numeric)
    created_at = Column(None)


class Judge(Base):
    __tablename__ = "judges"
    __table_args__ = {"schema": "pretrial"}
    judge_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    judge_name = Column(Text, nullable=False, unique=True)
    average_bail_set = Column(Numeric)
    unique_districts = Column(ARRAY(Text))
    case_count = Column(Integer)
    counties = Column(ARRAY(Text))
    court_names = Column(ARRAY(Text))
    representation_types = Column(ARRAY(Text))
    top_charges_at_arraign = Column(ARRAY(Text))
    disposition_types = Column(ARRAY(Text))
    remand_to_jail_count = Column(Integer)
    ror_count = Column(Integer)
    bail_set_and_posted_count = Column(Integer)
    bail_set_and_not_posted_count = Column(Integer)
    supervision_conditions = Column(ARRAY(Text))
    average_sentence_severity = Column(Numeric)
    offense_types = Column(ARRAY(Text))
    rearrest_rate = Column(Numeric)


class Court(Base):
    __tablename__ = "courts"
    __table_args__ = {"schema": "pretrial"}
    court_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    court_name = Column(Text, nullable=False, unique=True)
    court_ori = Column(Text)
    district = Column(Text)
    region = Column(Text)
    court_type = Column(Text)
    average_bail_amount = Column(Numeric)
    number_of_cases = Column(Integer)


class Crime(Base):
    __tablename__ = "crimes"
    __table_args__ = {"schema": "pretrial"}
    crime_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    top_charge_at_arrest = Column(Text, nullable=False, unique=True)
    average_bail_amount = Column(Numeric)
    number_of_cases = Column(Integer)


class Race(Base):
    __tablename__ = "races"
    __table_args__ = {"schema": "pretrial"}
    race_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    race = Column(Text, nullable=False, unique=True)
    average_bail_amount = Column(Numeric)
    number_of_cases = Column(Integer)
    average_known_days_in_custody = Column(Numeric)
    remanded_percentage = Column(Numeric)
    bail_set_percentage = Column(Numeric)
    disposed_at_arraign_percentage = Column(Numeric)
    ror_percentage = Column(Numeric)
    nonmonetary_release_percentage = Column(Numeric)


class Representation(Base):
    __tablename__ = "representation"
    __table_args__ = {"schema": "pretrial"}
    representation_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    representation_type = Column(Text, nullable=False, unique=True)
    average_bail_amount = Column(Numeric)
    number_of_cases = Column(Integer)
    average_known_days_in_custody = Column(Numeric)
    remanded_percentage = Column(Numeric)
    bail_set_percentage = Column(Numeric)
    disposed_at_arraign_percentage = Column(Numeric)
    ror_percentage = Column(Numeric)
    nonmonetary_release_percentage = Column(Numeric)


class NYIncome(Base):
    __tablename__ = "ny_income"
    __table_args__ = {"schema": "pretrial"}
    income_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    rank = Column(Integer)
    county = Column(Text, unique=True)
    per_capita_income = Column(Float)
    median_household_income = Column(Float)
    median_family_income = Column(Float)
    population = Column(Float)
    number_of_households = Column(Float)
    _2018_median_household_income = Column(Integer)


class County(Base):
    __tablename__ = "counties"
    __table_args__ = {"schema": "pretrial"}
    county_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    county_name = Column(Text, nullable=False, unique=True)
    average_bail_amount = Column(Numeric)
    number_of_cases = Column(Integer)
    median_income = Column(Integer)
    median_id = Column(UUID(as_uuid=True), ForeignKey("pretrial.ny_income.income_uuid"))


class District(Base):
    __tablename__ = "districts"
    # __table_args__ = {"schema": "pretrial"}
    district_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    district_name = Column(Text, nullable=False)
    region = Column(Text)
    county_id = Column(UUID(as_uuid=True), ForeignKey("pretrial.counties.county_uuid"))

    __table_args__ = (
        UniqueConstraint("district_name", "region", "county_id", name="_district_uc"),
        {"schema": "pretrial"}
    )


class Case(Base):
    __tablename__ = "cases"
    __table_args__ = {"schema": "pretrial"}
    case_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    internal_case_id = Column(Text, nullable=False, unique=True)
    gender = Column(Text)
    ethnicity = Column(Text)
    age_at_crime = Column(Integer)
    age_at_arrest = Column(Integer)
    offense_month = Column(Text)
    offense_year = Column(Text)
    arrest_month = Column(Text)
    arrest_year = Column(Text)
    arrest_type = Column(Text)
    top_arrest_law = Column(Text)
    top_arrest_article_section = Column(Text)
    top_arrest_attempt_indicator = Column(Text)
    top_charge_severity_at_arrest = Column(Text)
    top_charge_weight_at_arrest = Column(Text)
    top_charge_at_arrest_violent_felony_ind = Column(Text)
    case_type = Column(Text)
    first_arraign_date = Column(Text)
    top_arraign_law = Column(Text)
    top_arraign_article_section = Column(Text)
    top_arraign_attempt_indicator = Column(Text)
    top_charge_at_arraign = Column(Text)
    top_severity_at_arraign = Column(Text)
    top_charge_weight_at_arraign = Column(Text)
    top_charge_at_arraign_violent_felony_ind = Column(Text)
    arraign_charge_category = Column(Text)
    app_count_arraign_to_dispo_released = Column(Text)
    app_count_arraign_to_dispo_detained = Column(Text)
    app_count_arraign_to_dispo_total = Column(Text)
    def_attended_sched_pretrials = Column(Text)
    remanded_to_jail_at_arraign = Column(Text)
    ror_at_arraign = Column(Text)
    bail_set_and_posted_at_arraign = Column(Text)
    bail_set_and_not_posted_at_arraign = Column(Text)
    nmr_at_arraign = Column(Text)
    release_decision_at_arraign = Column(Text)
    representation_at_securing_order = Column(Text)
    pretrial_supervision_at_arraign = Column(Text)
    contact_pretrial_service_agency = Column(Text)
    electronic_monitoring = Column(Text)
    travel_restrictions = Column(Text)
    passport_surrender = Column(Text)
    no_firearms_or_weapons = Column(Text)
    maintain_employment = Column(Text)
    maintain_housing = Column(Text)
    maintain_school = Column(Text)
    placement_in_mandatory_program = Column(Text)
    removal_to_hospital = Column(Text)
    obey_order_of_protection = Column(Text)
    obey_court_conditions_family_offense = Column(Text)
    other_nmr = Column(Text)
    order_of_protection = Column(Text)
    first_bail_set_cash = Column(Text)
    first_bail_set_credit = Column(Text)
    first_insurance_company_bail_bond = Column(Text)
    first_secured_surety_bond = Column(Text)
    first_secured_app_bond = Column(Text)
    first_unsecured_surety_bond = Column(Text)
    first_unsecured_app_bond = Column(Text)
    first_partially_secured_surety_bond = Column(Text)
    partially_secured_surety_bond_perc = Column(Text)
    first_partially_secured_app_bond = Column(Text)
    partially_secured_app_bond_perc = Column(Text)
    bail_made_indicator = Column(Text)
    warrant_ordered_btw_arraign_and_dispo = Column(Text)
    dat_wo_ws_prior_to_arraign = Column(Text)
    first_bench_warrant_date = Column(Text)
    non_stayed_wo = Column(Text)
    num_of_stayed_wo = Column(Text)
    num_of_row = Column(Text)
    docket_status = Column(Text)
    disposition_type = Column(Text)
    disposition_detail = Column(Text)
    dismissal_reason = Column(Text)
    disposition_date = Column(Text)
    most_severe_sentence = Column(Text)
    top_conviction_law = Column(Text)
    top_conviction_article_section = Column(Text)
    top_conviction_attempt_indicator = Column(Text)
    top_charge_at_conviction = Column(Text)
    top_charge_severity_at_conviction = Column(Text)
    top_charge_weight_at_conviction = Column(Text)
    top_charge_at_conviction_violent_felony_ind = Column(Text)
    days_arraign_remand_first_released = Column(Text)
    known_days_in_custody = Column(Text)
    days_arraign_bail_set_to_first_posted = Column(Text)
    days_arraign_bail_set_to_first_release = Column(Text)
    days_arraign_to_dispo = Column(Text)
    ucmslivedate = Column(Text)
    prior_vfo_cnt = Column(Text)
    prior_nonvfo_cnt = Column(Text)
    prior_misd_cnt = Column(Text)
    pend_vfo = Column(Text)
    pend_nonvfo = Column(Text)
    pend_misd = Column(Text)
    supervision = Column(Text)
    rearrest = Column(Text)
    rearrest_date = Column(Text)
    rearrest_firearm = Column(Text)
    rearrest_date_firearm = Column(Text)
    arr_cycle_id = Column(Text)
    race_id = Column(UUID(as_uuid=True), ForeignKey("pretrial.races.race_uuid"))
    court_id = Column(UUID(as_uuid=True), ForeignKey("pretrial.courts.court_uuid"))
    county_id = Column(UUID(as_uuid=True), ForeignKey("pretrial.counties.county_uuid"))
    top_charge_id = Column(UUID(as_uuid=True), ForeignKey("pretrial.crimes.crime_uuid"))
    crime_id = Column(UUID(as_uuid=True), ForeignKey("pretrial.crimes.crime_uuid"))
    representation_id = Column(UUID(as_uuid=True), ForeignKey("pretrial.representation.representation_uuid"))
    judge_id = Column(UUID(as_uuid=True), ForeignKey("pretrial.judges.judge_uuid"))
    district_id = Column(UUID(as_uuid=True), ForeignKey("pretrial.districts.district_uuid"))


class Law(Base):
    __tablename__ = "laws"
    __table_args__ = {"schema": "pretrial"}
    law_uuid = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    law_ordinal = Column(Float)
    attempted_class = Column(Text)
    attempted_vf_indicator = Column(Text)
    attempted_nys_law_category = Column(Text)
    bus_driver_charge_code = Column(Float)
    sex_offender_registry_code = Column(Float)
    ncic_code = Column(Float)
    ucr_code = Column(Float)
    safis_crime_cateory_code = Column(Text)
    offense_category = Column(Text)
    jo_indicator = Column(Float)
    jd_indicator = Column(Float)
    ibr_code = Column(Text)
    maxi_law_description = Column(Text)
    law_description = Column(Text)
    mini_law_description = Column(Text)
    title = Column(Text)
    section = Column(Text)
    section13 = Column(Text)
    sub_section = Column(Text)
    sub_section13 = Column(Text)
    degree = Column(Integer)
    effective_date = Column(Float)
    repeal_date = Column(Float)
    fp_offense = Column(Text)
    unconst_date = Column(Text)
    weapon_charge = Column(Float)
    armed_vfo_charge = Column(Text)
    minors_charge = Column(Float)
    career_criminal_charge = Column(Text)
    ins_charge = Column(Text)
    non_seal_charge = Column(Text)
    sub_convict_charge = Column(Text)
    jail_charge = Column(Text)
    post_convict_charge = Column(Text)
    auto_strip_charge = Column(Text)
    full_law_description = Column(Text)
    nys_law_category = Column(Text)
    vf_indicator = Column(Text)
    class_ = Column(Text)  # Use 'class_' since 'class' is a reserved keyword
    dna_indicator = Column(Integer)
    attempted_dna_indicator = Column(Text)
    escape_charge = Column(Text)
    hate_crime = Column(Float)
    date_invalidated = Column(Float)
    terrorism_indicator = Column(Text)
    dmv_vtcode = Column(Text)
    ao_indicator = Column(Text)
    rta_fp_offense = Column(Text)
    modified_date = Column(Float)
    civil_confinement_indicator = Column(Integer)
    attempted_cci = Column(Integer)
    expanded_law_literal = Column(Text)
    sexually_motivated_ind = Column(Text)
    mcdv_charge_indicator = Column(Float)
    rdlr_indicator = Column(Text)
    spc_code = Column(Text)
    state_nics_disqualifier = Column(Text)
