import logging
import os
import re

import requests
import seaborn as sns
from matplotlib import pyplot as plt


def sanitize_metric_name(name):
    return re.sub(r'[^a-zA-Z0-9_\- .\/]', '_', name)


def read_previous_r2(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return float(file.read())
    return None


def write_current_r2(file_path, r2_value):
    with open(file_path, 'w') as file:
        file.write(str(r2_value))


def plot_feature_importance(importance_df, r2, total_cases, r2_comparison, outputs_dir, elapsed_time, model_types,
                            num_features, model_for_selection):
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
    plt.text(0.95, 0.1, f'Time Elapsed = {elapsed_time:.2f} s', verticalalignment='bottom', horizontalalignment='right',
             transform=plt.gca().transAxes, color='green', fontsize=15, weight='bold')
    plt.text(0.95, 0.15, f'Models Used = {", ".join(model_types)}', verticalalignment='bottom',
             horizontalalignment='right',
             transform=plt.gca().transAxes, color='purple', fontsize=15, weight='bold')
    # plt.text(0.95, 0.2, f'Model Used for Featured Selection = {model_for_selection }', verticalalignment='bottom',
    #          horizontalalignment='right',
    #          transform=plt.gca().transAxes, color='black', fontsize=15, weight='bold')
    plt.text(0.95, 0.25, f'Number of Features = {num_features}', verticalalignment='bottom', horizontalalignment='right',
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


def send_discord_notification(webhook_url, message, plot_file_path=None, avatar_url=None):
    data = {
        "content": message,
        "username": "LawVision AI Trainer",
    }

    if avatar_url:
        data["avatar_url"] = avatar_url

    if plot_file_path and os.path.exists(plot_file_path):
        with open(plot_file_path, 'rb') as file:
            response = requests.post(webhook_url, data=data, files={"file": ("image.png", file, "image/png")})
    else:
        response = requests.post(webhook_url, json=data)

    if response.status_code in [200, 204]:
        print("Notification sent successfully.")
    else:
        print(f"Failed to send notification. Status code: {response.status_code}")
        print(f"Response content: {response.content}")
