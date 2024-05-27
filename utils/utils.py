import logging
import os
import re

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


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


def sanitize_metric_name(name):
    return re.sub(r'[^a-zA-Z0-9_\- .]', '_', name)


def compare_r2(previous_r2, average_r2):
    if previous_r2 is not None:
        if average_r2 > previous_r2:
            return f'R² increased by {(average_r2 - previous_r2):.4f}'
        elif average_r2 < previous_r2:
            return f'R² decreased by {(previous_r2 - average_r2):.4f}'
        else:
            return "R² stayed the same"
    else:
        return "No previous R² value"


def read_previous_r2(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return float(file.read())
    return None


def write_current_r2(file_path, r2_value):
    with open(file_path, 'w') as file:
        file.write(str(r2_value))


def save_importance_profile(importance_df, profile_name, outputs_dir):
    profile_path = os.path.join(outputs_dir, f"{profile_name}_importance_profile.csv")
    importance_df.to_csv(profile_path, index=False)
    logging.info(f"Importance profile saved as '{profile_path}'")
    return profile_path


def load_importance_profile(profile_name, outputs_dir):
    profile_path = os.path.join(outputs_dir, f"{profile_name}_importance_profile.csv")
    if os.path.exists(profile_path):
        importance_df = pd.read_csv(profile_path)
        logging.info(f"Importance profile loaded from '{profile_path}'")
        return importance_df
    else:
        raise FileNotFoundError(f"Importance profile '{profile_path}' not found")


def compare_to_baseline(importance_df, baseline_df):
    comparison = importance_df.merge(baseline_df, on='Feature', suffixes=('_judge', '_baseline'))
    comparison['Difference'] = comparison['Importance_judge'] - comparison['Importance_baseline']
    logging.info(f"Comparison to baseline profile completed")
    return comparison


def plot_feature_importance(importance_df, r2, total_cases, r2_comparison, outputs_dir, elapsed_time, model_types,
                            num_features):
    plt.figure(figsize=(20, 32))
    plt.style.use('seaborn-darkgrid')
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='viridis')
    plt.title('Top 10 Features That Impact Bail Amount', fontsize=30)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, weight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, weight='bold')
    plt.xlabel('Importance', fontsize=20)
    plt.ylabel('Feature', fontsize=20)
    plt.tight_layout()

    # Add R-squared value, total number of cases, and R² comparison to the chart
    plt.text(0.95, 0.5, f'R² = {r2:.2f}', verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes, color='red', fontsize=64, weight='bold', backgroundcolor='white')
    plt.text(0.95, 0.05, f'Total Cases = {total_cases}', verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes, color='blue', fontsize=15, weight='bold')
    plt.text(0.95, 0.1, f'Time Elapsed = {elapsed_time:.2f} s', verticalalignment='bottom',
             horizontalalignment='right',
             transform=plt.gca().transAxes, color='green', fontsize=15, weight='bold')
    plt.text(0.95, 0.15, f'Models Used = {", ".join(model_types)}', verticalalignment='bottom',
             horizontalalignment='right',
             transform=plt.gca().transAxes, color='purple', fontsize=15, weight='bold')
    plt.text(0.95, 0.25, f'Number of Features = {num_features}', verticalalignment='bottom',
             horizontalalignment='right',
             transform=plt.gca().transAxes, color='orange', fontsize=15, weight='bold')

    r2_comparison_color = 'green' if 'increased' in r2_comparison else 'red'
    plt.text(0.95, 0.33, r2_comparison, verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes, color=r2_comparison_color, fontsize=24, weight='bold')

    # Save the plot as an image file
    plot_file_path = os.path.join(outputs_dir, 'Top_10_Features_That_Impact_Bail_Amount.png')
    plt.savefig(plot_file_path)
    logging.info(f"Top 10 feature importances chart saved as '{plot_file_path}'")
    plt.show()

    return plot_file_path