# Imports
from twitchAPI.chat import Chat, EventData, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
import asyncio

from transformers import pipeline

# Contains APP_ID & APP_SECRET & TARGET_CHANNEL // Testing asking for channel input on creation
import config

# Set up Constants
CLIENT_ID = config.client_id
CLIENT_SECRET = config.client_secret
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]

CLASSIFIER = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None)

# TARGET_CHANNEL = config.channel # Placeholder; Enter the Twitch channel that you wish to view: i.e. "Twitch"

# Chat messages
async def on_message(msg: ChatMessage):
    print(f"{msg.user.display_name}: {msg.text}")
#------------------Chat Sentiment-----------------#
    sentiment = CLASSIFIER(msg.text)

    # Print the main sentiment analysis
    top_label, top_score = sentiment[0][0]['label'], sentiment[0][0]['score']
    print(f"Main sentiment: {top_label}, Score: {top_score:.3f}")

    # Optional: Print the second most likely sentiment
    # mid_label, mid_score = sentiment[0][1]['label'], sentiment[0][1]['score']
    # print(f"Second sentiment: {mid_label}, Score: {mid_score:.3f}")

    # Optional: Print the third most likely sentiment
    # bot_label, bot_score = sentiment[0][2]['label'], sentiment[0][2]['score']
    # print(f"Third sentiment: {bot_label}, Score: {bot_score:.3f}")

async def on_ready(ready_event: EventData):
    # Connect to target channel
    channel_name = "plaqueboymax"
    await ready_event.chat.join_room(channel_name)

    # Status message
    print("Bot Status: Ready\n")

async def run_bot():

    # Authenticate
    print("Authenicating...")
    bot = await Twitch(CLIENT_ID, CLIENT_SECRET)
    auth = UserAuthenticator(bot, USER_SCOPE)
    token, refresh_token = await auth.authenticate()
    await bot.set_user_authentication(token, USER_SCOPE, refresh_token)
    
    # Initialize chat class
    chat = await Chat(bot)
    print("Authenication done.")
    # Register events
    chat.register_event(ChatEvent.READY, on_ready)
    chat.register_event(ChatEvent.MESSAGE, on_message)

    # Countdown (optional)
    # print('Booting up in 3...', end="")
    # asyncio.sleep(1)
    # print('2...', end="")
    # asyncio.sleep(1)
    # print('1...')
    # asyncio.sleep(1)

    # Start bot
    chat.start()

    try:
        input('\nPress "ENTER" to stop\n')
    finally:
        chat.stop()
        await bot.close()
        
        # Status message
        print("\nBot Status: Off")

asyncio.run(run_bot()) # Start program