# Twitch Chat Data Scraper
# This script connects to various Twitch channels and collects chat messages,
# saving them to a CSV for later use in model training.
from twitchAPI.chat import Chat, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
import asyncio
import aiofiles
from aiocsv import AsyncWriter
import config

# Configuration
CLIENT_ID = config.client_id
CLIENT_SECRET = config.client_secret
BOT_LIST = config.bot_list
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]

# Target channels for data collection
TARGET_CHANNELS = [
    "stableronaldo",
    "fanum",
    "hasanabi",
    "yourragegaming",
    "summit1g",
    "jasontheween",
    "silky",
    "plaqueboymax",
    "jynxzi",
    "arky",
    "ludwig",
    "shroud",
    "xqc",
    "summit1g",
    "theburntpeanut",
    "yugi2x",
    "RDCgaming",
    "2xrakai",
    "sakurashymko",
]

output_file = "twitch_data_1m.csv"


# Save messages to CSV
async def save_message(message_list):
    async with aiofiles.open(output_file, mode="a", newline="", encoding="utf-8") as f:
        writer = AsyncWriter(f)
        await writer.writerow(message_list)


# Handle incoming chat messages
async def on_message(msg: ChatMessage):
    text = msg.text or ""
    user = msg.user.display_name.lower()
    # Skip bots
    if user in BOT_LIST:
        return
    # Skip commands (starting with !)
    if text.startswith("!"):
        return
    # Skip links (http or https)
    if "http" in text.lower():  # cheaper than per-word slicing
        return

    channel_name = msg.room.name

    try:
        print(f"[{channel_name}] {text}")  # Log message to console
    except Exception:
        pass  # Ignore emoji-related errors

    await save_message([channel_name, text])  # Save message to CSV


# Main execution
async def main():
    print("Authenticating...")
    twitch = await Twitch(CLIENT_ID, CLIENT_SECRET)
    auth = UserAuthenticator(twitch, USER_SCOPE)
    token, refresh_token = await auth.authenticate()
    await twitch.set_user_authentication(token, USER_SCOPE, refresh_token)
    print("Authentication successful.")

    # Initialize and start chat
    chat = await Chat(twitch)
    chat.register_event(ChatEvent.MESSAGE, on_message)
    chat.start()
    print(f"Connecting to {len(TARGET_CHANNELS)} channels...")

    # Join target channels
    try:
        await chat.join_room(TARGET_CHANNELS)
        print(f"Successfully joined: {', '.join(TARGET_CHANNELS)}")
        print(f"Scraping data to {output_file}...")
        print("Press Ctrl+C to stop.")
    except Exception as e:
        print(f"Error joining channels: {e}")

    # Keep running until interrupted
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping scraper...")
        chat.stop()
        await twitch.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Scraper stopped by user.")
