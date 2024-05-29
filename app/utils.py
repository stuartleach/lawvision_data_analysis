"""Utility functions for the bail project."""

import logging
import os
import re
from dataclasses import dataclass

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def sanitize_metric_name(name):
    """Sanitize the metric name.

    :param name:
    :return:
    """
    return re.sub(r"[^a-zA-Z0-9_\- .]", "_", name)


def compare_r2(previous_r2, average_r2):
    """Compare the R² values.

    :param previous_r2:
    :param average_r2:
    :return:
    """
    if previous_r2 is not None:
        if average_r2 > previous_r2:
            return f"R² increased by {(average_r2 - previous_r2):.4f}"
        if average_r2 < previous_r2:
            return f"R² decreased by {(previous_r2 - average_r2):.4f}"
        return "R² stayed the same"
    return "No previous R² value"


def read_previous_r2(file_path):
    """Read the previous R² value.

    :param file_path:
    :return:
    """
    if os.path.exists(file_path):
        with open(file_path, encoding="utf-8") as file:
            return float(file.read())
    return None


def write_current_r2(file_path, r2_value):
    """Write the current R² value.

    :param file_path:
    :param r2_value:
    :return:
    """
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(str(r2_value))


def save_importance_profile(importance_df, profile_name, outputs_dir):
    """Save the importance profile.

    :param importance_df:
    :param profile_name:
    :param outputs_dir:
    :return:
    """
    profile_path = os.path.join(outputs_dir, f"{profile_name}_importance_profile.csv")
    importance_df.to_csv(profile_path, index=False)
    logging.info("Importance profile saved as '%s'", profile_path)
    return profile_path


def load_importance_profile(profile_name, outputs_dir):
    """Load the importance profile.

    :param profile_name:
    :param outputs_dir:
    :return:
    """
    profile_path = os.path.join(outputs_dir, f"{profile_name}_importance_profile.csv")
    if os.path.exists(profile_path):
        importance_df = pd.read_csv(profile_path)
        logging.info("Importance profile loaded from '%s'", profile_path)
        return importance_df

    raise FileNotFoundError(f"Importance profile '{profile_path}' not found")


def compare_to_baseline(importance_df, baseline_df):
    """Compare the importance to the baseline.

    :param importance_df:
    :param baseline_df:
    :return:
    """
    comparison = importance_df.merge(
        baseline_df, on="Feature", suffixes=("_judge", "_baseline")
    )
    comparison["Difference"] = (
            comparison["Importance_judge"] - comparison["Importance_baseline"]
    )
    logging.info("Comparison to baseline profile completed")
    return comparison


@dataclass
class PlotParams:
    """Data class for plot parameters."""

    r2: float
    total_cases: int
    r2_comparison: str
    elapsed_time: float
    model_info: dict


def plot_feature_importance(importance_df, outputs_dir, plot_params: PlotParams, judge_filter: str = None,
                            county_filter: str = None):
    """Plot the feature importance.

    :param importance_df:
    :param outputs_dir:
    :param plot_params:
    :param judge_filter:
    :param county_filter:
    :return:
    """
    model_types = plot_params.model_info["model_types"]
    num_features = plot_params.model_info["num_features"]

    plt.figure(figsize=(20, 32))
    plt.style.use("seaborn-darkgrid")
    ax = sns.barplot(
        x="Importance", y="Feature", data=importance_df.head(20), palette="viridis", hue="Feature", legend=False
    )
    if judge_filter:
        plt.title(f"Top 10 Features That Impact Bail Amount for {judge_filter}", fontsize=30)

    elif county_filter:
        plt.title(f"Top 10 Features That Impact Bail Amount in {county_filter} County", fontsize=30)

    else:
        plt.title("Top 10 Features That Impact Bail Amount", fontsize=30)

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, weight="bold")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18, weight="bold")
    plt.xlabel("Importance", fontsize=20)
    plt.ylabel("Feature", fontsize=20)
    plt.tight_layout()

    plt.text(
        0.95,
        0.5,
        f"R² = {plot_params.r2:.2f}",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=plt.gca().transAxes,
        color="red",
        fontsize=64,
        weight="bold",
        backgroundcolor="white",
    )
    plt.text(
        0.95,
        0.05,
        f"Total Cases = {plot_params.total_cases}",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=plt.gca().transAxes,
        color="blue",
        fontsize=15,
        weight="bold",
    )
    plt.text(
        0.95,
        0.1,
        f"Time Elapsed = {plot_params.elapsed_time:.2f} s",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=plt.gca().transAxes,
        color="green",
        fontsize=15,
        weight="bold",
    )
    plt.text(
        0.95,
        0.15,
        f"Models Used = {', '.join(model_types)}",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=plt.gca().transAxes,
        color="purple",
        fontsize=15,
        weight="bold",
    )
    plt.text(
        0.95,
        0.25,
        f"Number of Features = {num_features}",
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=plt.gca().transAxes,
        color="orange",
        fontsize=15,
        weight="bold",
    )

    r2_comparison_color = "green" if "increased" in plot_params.r2_comparison else "red"
    plt.text(
        0.95,
        0.33,
        plot_params.r2_comparison,
        verticalalignment="bottom",
        horizontalalignment="right",
        transform=plt.gca().transAxes,
        color=r2_comparison_color,
        fontsize=24,
        weight="bold",
    )

    # Save the plot as an image file
    plot_file_path = os.path.join(
        outputs_dir, "Top_10_Features_That_Impact_Bail_Amount.png"
    )
    plt.savefig(plot_file_path)
    logging.info("Top 10 feature importances chart saved as '%s'", plot_file_path)
    plt.show()

    return plot_file_path
