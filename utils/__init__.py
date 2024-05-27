"""
This module contains utility functions for the project.
"""

from .notify import (
    NotificationData,
    create_message,
    send_discord_webhook,
)
from .utils import (
    PlotParams,
    compare_r2,
    compare_to_baseline,
    get_query,
    load_importance_profile,
    plot_feature_importance,
    read_previous_r2,
    sanitize_metric_name,
    save_importance_profile,
    write_current_r2,
)

__all__ = [
    "create_message",
    "send_discord_webhook",
    "get_query",
    "sanitize_metric_name",
    "read_previous_r2",
    "write_current_r2",
    "save_importance_profile",
    "load_importance_profile",
    "compare_to_baseline",
    "plot_feature_importance",
    "compare_r2",
    "compare_r2",
    "save_importance_profile",
    "load_importance_profile",
    "PlotParams",
    "NotificationData",
]
