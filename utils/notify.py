import logging
import os

import requests


def create_message(avg_r2, r2_comparison, total_cases, time_difference, models, model, num_features):
    return (f"\n\n\n\n\n\n\nModel training completed.\n"
            f"Average R²:  \n **{avg_r2:.4f}**\n\n"
            f"R² comparison: \n**{r2_comparison}**\n\n"
            f"Total cases: \n**{total_cases}**\n\n"
            f"Training time: \n**{time_difference:.2f}** seconds\n\n"
            f"Models used: \n**{models}**\n\n"
            f"Model used for feature selection: \n**{model}**\n\n"
            f"Number of features: \n**{num_features}\n\n**"
            f"Bar chart: \n"
            )


def send_discord_webhook(webhook_url, avatar_url, message, plot_file_path=None):
    data = {
        "content": message,
        "username": "LawVision AI Trainer",
        "avatar_url": avatar_url
    }

    if avatar_url:
        data["avatar_url"] = avatar_url

    if plot_file_path and os.path.exists(plot_file_path):
        with open(plot_file_path, 'rb') as file:
            response = requests.post(webhook_url, data=data, files={"file": ("image.png", file, "image/png")})
    else:
        response = requests.post(webhook_url, json=data)

    if response.status_code in [200, 204]:
        logging.info("Notification sent successfully.")
    else:
        logging.error(f"Failed to send notification. Status code: {response.status_code}")
        logging.error(f"Response content: {response.content}")


def send_notification(webhook_url, average_r2, r2_comparison, plot_file_path, model_types, model_for_selection,
                      total_cases, num_features,
                      time_difference=None):
    if not webhook_url or webhook_url == 'None':
        logging.error("Discord webhook URL is not set.")
        return

    models = ', '.join(model_types)
    # self.num_features = len(pd.read_csv(os.path.join(self.outputs_dir, 'X.csv')).columns)
    model = model_for_selection if model_for_selection else "None"

    message = create_message(average_r2, r2_comparison, total_cases, time_difference, models, model,
                             num_features)
    send_discord_webhook(message, plot_file_path)
