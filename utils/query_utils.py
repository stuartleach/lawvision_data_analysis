class QueryUtils:
    @staticmethod
    def get_query(sql_values_to_interpolate, q_template):
        judge_names_condition = "AND j.judge_name = ANY(%(judge_names)s)" if sql_values_to_interpolate[
            "judge_names"] else ""
        county_names_condition = "AND co.county_name = ANY(%(county_names)s)" if sql_values_to_interpolate[
            "county_names"] else ""

        resulting_query = q_template.format(
            judge_names_condition=judge_names_condition,
            county_names_condition=county_names_condition
        )
        return resulting_query
