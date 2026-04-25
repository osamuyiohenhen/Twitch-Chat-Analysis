from dotenv import load_dotenv
import os

load_dotenv()

client_id = os.getenv("TWITCH_CLIENT_ID")
client_secret = os.getenv("TWITCH_CLIENT_SECRET")
bot_list = ["fossabot", "nightbot", "streamelements", "potatbotat"] # Add more known bots