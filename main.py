# Imports
from twitchAPI.chat import Chat, EventData, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
import asyncio
from textblob import TextBlob

# Contains APP_ID & APP_SECRET
import config

# Set up Constants
CLIENT_ID = config.client_id
CLIENT_SECRET = config.client_secret
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]

TARGET_CHANNEL = 'jasontheween' # Placeholder; Enter the Twitch channel that you wish to view 

# Chat messages
async def on_message(msg: ChatMessage):
    text = msg.text
    print(f"{msg.user.display_name}: {text}")
#------------------Chat Sentiment-----------------#
    text = TextBlob(text)
    polarity, subjectivity = text.sentiment
    print(f"{polarity}, {subjectivity}")
    
async def on_ready(ready_event: EventData):
    # Connect to target channel
    await ready_event.chat.join_room(TARGET_CHANNEL)

    # Status message
    print("Bot Status: Ready\n")

async def run_bot():
    # Authenticate
    bot = await Twitch(CLIENT_ID, CLIENT_SECRET)
    auth = UserAuthenticator(bot, USER_SCOPE)
    token, refresh_token = await auth.authenticate()
    await bot.set_user_authentication(token, USER_SCOPE, refresh_token)

    # Initialize chat class
    chat = await Chat(bot)

    # Register events
    chat.register_event(ChatEvent.READY, on_ready)
    chat.register_event(ChatEvent.MESSAGE, on_message)

    # Countdown
    print('Booting up in 3...')
    await asyncio.sleep(1)
    print('2...')
    await asyncio.sleep(1)
    print('1...')
    await asyncio.sleep(1)

    # Start bot
    chat.start()

    try:
        input('\nPress "ENTER" to stop\n')
    finally:
        chat.stop()
        await bot.close()
        # Status message
        print("Bot Status: Off")

asyncio.run(run_bot())