from twitchAPI.chat import Chat, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
import asyncio
import os
import aiofiles
from aiocsv import AsyncWriter
import config

# ==========================================
# CONFIGURATION
# ==========================================
CLIENT_ID = config.client_id
CLIENT_SECRET = config.client_secret
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]

# The "Big List" to get diverse slang
TARGET_CHANNELS = [
    'marlon', 'cinna', 'stableronaldo', 
    'kaicenat', 'jasontheween', 'yourragegaming',
    'yugi2x', 'arky', 'plaqueboymax'
]

OUTPUT_FILE = "twitch_data_300k.csv"

# ==========================================
# CSV HANDLING
# ==========================================
async def save_message(message_list):
    # Appends to CSV. Format is now just: [Channel, Message]
    async with aiofiles.open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = AsyncWriter(f)
        await writer.writerow(message_list)

# ==========================================
# CHAT LOGIC
# ==========================================
async def on_message(msg: ChatMessage):
    # Filter bots (Optional - add more if you see spam)
    if msg.user.display_name.lower() in ['nightbot', 'streamelements', 'fossabot']: 
        return

    channel_name = msg.room.name
    text = msg.text

    # 1. PRINT IT (So you know it's working)
    try:
        print(f"[{channel_name}] {text}")
    except Exception:
        pass # Ignore emoji printing errors in Windows terminal

    # 2. SAVE IT (Channel + Text only)
    await save_message([channel_name, text])

# ==========================================
# MAIN EXECUTION
# ==========================================
async def main():
    print("Authenticating...")
    twitch = await Twitch(CLIENT_ID, CLIENT_SECRET)
    auth = UserAuthenticator(twitch, USER_SCOPE)
    token, refresh_token = await auth.authenticate()
    await twitch.set_user_authentication(token, USER_SCOPE, refresh_token)
    print("Authentication successful.")

    # Initialize Chat
    chat = await Chat(twitch)
    chat.register_event(ChatEvent.MESSAGE, on_message)
    chat.start()
    print(f"Connecting to {len(TARGET_CHANNELS)} channels...")

    # Join all channels
    try:
        await chat.join_room(TARGET_CHANNELS)
        print(f"Successfully joined: {', '.join(TARGET_CHANNELS)}")
        print(f"Scraping data to {OUTPUT_FILE}...")
        print("Press Ctrl+C to stop.")
    except Exception as e:
        print(f"Error joining channels: {e}")

    # Keep running until user stops
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping scraper...")
        chat.stop()
        await twitch.close()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Scraper stopped by user.")