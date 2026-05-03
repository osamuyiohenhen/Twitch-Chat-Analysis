from dotenv import load_dotenv
import os

load_dotenv()

client_id = os.getenv("TWITCH_CLIENT_ID")
client_secret = os.getenv("TWITCH_CLIENT_SECRET")
user_token = os.getenv("TWITCH_USER_TOKEN")
refresh_token = os.getenv("TWITCH_REFRESH_TOKEN")

bot_list = [
    "fossabot",
    "nightbot",
    "streamelements",
    "potatbotat",
]  # Add more known bots
