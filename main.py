# Imports
from twitchAPI.chat import Chat, EventData, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
import asyncio

from transformers import pipeline
import torch

# Contains APP_ID & APP_SECRET & TARGET_CHANNEL
import config

# Set up Constants
CLIENT_ID = config.client_id
CLIENT_SECRET = config.client_secret
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]

CLASSIFIER = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", top_k=None)

TARGET_CHANNEL = config.channel # Placeholder; Enter the Twitch channel that you wish to view: i.e. "Twitch"

# Chat messages
async def on_message(msg: ChatMessage):
    print(f"{msg.user.display_name}: {msg.text}")
#------------------Chat Sentiment-----------------#
    sentiment = CLASSIFIER(msg.text)
    result = sentiment[0]
    dict1 = result[0]
    label1, score1 = dict1['label'], dict1['score']


    print(f"{label1}, {score1}")

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