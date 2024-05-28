import os

from dotenv import load_dotenv

load_dotenv()

DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
DISCORD_AVATAR_URL = os.environ.get("DISCORD_AVATAR_URL")
