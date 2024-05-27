"""
This module contains functions for sending notifications.
"""

import logging
import os
from dataclasses import dataclass

import requests

from app import DISCORD_AVATAR_URL, DISCORD_WEBHOOK_URL


@dataclass
class NotificationData:
    """
    Data class for notification data.
    """

    performance_data: dict
    plot_file_path: str
    model_info: dict


def create_message(data: NotificationData) -> str:
    """
    Create a message for the notification.
    """
    performance_data = data.performance_data
    model_info = data.model_info
    models = ", ".join(model_info["model_types"])

    return (
        f"\n\n\n\n\n\n\nModel training completed.\n"
        f"Average R²:  \n **{performance_data['average_r2']:.4f}**\n\n"
        f"R² comparison: \n**{performance_data['r2_comparison']}**\n\n"
        f"Total cases: \n**{performance_data['total_cases']}**\n\n"
        f"Training time: \n**{performance_data['time_difference']:.2f}** seconds\n\n"
        f"Models used: \n**{models}**\n\n"
        f"Model used for feature selection: \n**{model_info['model_for_selection']}**\n\n"
        f"Number of features: \n**{performance_data['num_features']}\n\n**"
        f"Bar chart: \n"
    )


def send_discord_webhook(
    webhook_url: str, avatar_url: str, message: str, plot_file_path: str = None
):
    """
    Send a Discord webhook notification.
    """
    data = {
        "content": message,
        "username": "LawVision AI Trainer",
        "avatar_url": avatar_url,
    }

    if plot_file_path and os.path.exists(plot_file_path):
        with open(plot_file_path, "rb") as file:
            response = requests.post(
                webhook_url,
                data=data,
                files={"file": ("image.png", file, "image/png")},
                timeout=10,
            )
    else:
        response = requests.post(webhook_url, json=data, timeout=10)

    if response.status_code in [200, 204]:
        logging.info("Notification sent successfully.")
    else:
        logging.error(
            "Failed to send notification. Status code: %s", response.status_code
        )
        logging.error("Response content: %s", response.content.decode("utf-8"))


def send_notification(data: NotificationData):
    """
    Send a notification with the given data.
    """
    webhook_url = DISCORD_WEBHOOK_URL
    avatar_url = DISCORD_AVATAR_URL

    if not webhook_url or webhook_url == "None":
        logging.error("Discord webhook URL is not set.")
        return

    message = create_message(data)
    send_discord_webhook(webhook_url, avatar_url, message, data.plot_file_path)
