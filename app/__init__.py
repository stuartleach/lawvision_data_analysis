"""This module is the entry point of the application."""

from .app import run
from .params import DISCORD_AVATAR_URL, DISCORD_WEBHOOK_URL, GOOD_HYPERPARAMETERS

__all__ = [
    "run",
    "GOOD_HYPERPARAMETERS",
    "DISCORD_WEBHOOK_URL",
    "DISCORD_AVATAR_URL",
]
