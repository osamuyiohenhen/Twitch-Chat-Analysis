# Imports
from twitchAPI.chat import Chat, EventData, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
import asyncio
# import time (diagnostic purposes)
import sys
import os

# # Data Extraction (this is for model training purposes. You can ignore this.)
# from aiocsv import AsyncWriter
# import aiofiles

from transformers import pipeline

# Contains APP_ID & APP_SECRET // Testing asking for channel input on creation
import config

# Set up Constants (imported in private config.py file)
CLIENT_ID = config.client_id
CLIENT_SECRET = config.client_secret
# Defines what the program is authorized to do. Since this is a basic program, it can simply read and write chat messages.
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]

# Current model for sentiment analysis (working on fine-tuning it in the future)
CLASSIFIER = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None)

# Various known Twitch chatbots that can be ignored
BOT_LIST = {'fossabot', 'nightbot', 'streamelements', "potatbotat"} 

async def on_message(msg: ChatMessage):
    if msg.user.display_name.lower() in BOT_LIST:
        print(f"--- Ignoring bot message from: {msg.user.display_name} ---")
        return
    elif msg.text[0] == '!':
        print(f"---Ignoring Command from: {msg.user.display_name}")
        return
    
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
    
# #-----------------Data Extraction-----------------# This is for model training purposes. You can ignore this.
#     data_for_csv = [msg.user.name, msg.text]
# #     await save_message(data_for_csv)
# async def save_message(message):
#     async with aiofiles.open("twitch_data.csv", mode='a', newline='', encoding='utf-8') as f:
#         writer = AsyncWriter(f)
#         await writer.writerow(message)

# Main function
async def main():
    # Authenticate (one time)
    print("Authenticating...")
    twitch = await Twitch(CLIENT_ID, CLIENT_SECRET)
    auth = UserAuthenticator(twitch, USER_SCOPE)
    token, refresh_token = await auth.authenticate()
    await twitch.set_user_authentication(token, USER_SCOPE, refresh_token) 
    print("Authentication successful.")

    # Initialize Chat class instance
    chat = await Chat(twitch)

    # Since this program will just be detecting messages for now, the only event that needs to be registered/detected is when a message is sent.
    chat.register_event(ChatEvent.MESSAGE, on_message)
        
    print("Initializing connection...")
    # Start the chat client
    chat.start()
    print("Connection started.")

    current_channel = None
    try:
        while True:
            # Skips this if current_channel is not set (first time run)
            if current_channel:
                await chat.leave_room(current_channel)
                await asyncio.sleep(0.3)
                print(f"Leaving channel: {current_channel}...")
                current_channel = None

            target_channel = input("\nEnter the Twitch channel you wish to connect to (or type 'q' to exit): ").lower()
            if target_channel == 'q':
                break
            
            if not target_channel:
                print("Channel name cannot be empty. Please try again.")
                continue

            try:
                print(f"\nJoining channel: {target_channel}...")
                # start_time = time.perf_counter() (diagnostic purposes)
                # After attempting to join, wait few seconds for joining channel
                await asyncio.wait_for(chat.join_room(target_channel), timeout=1.5)
                # end_time = time.perf_counter()
                # total_time = end_time - start_time
                # print(f"Time took for channel join attempt: {total_time:.2f} seconds.")

                # By here, it should be successful
                current_channel = target_channel
                await asyncio.to_thread(input, 'Press "ENTER" to switch channels or quit.\n\n')

            except asyncio.TimeoutError:
                # Runs if the join_confirmation was not set within the specified timeout frame
                print(f"\n[ERROR] Could not join channel '{target_channel}'.")
                print("The channel may not exist or you may be banned.")
    
    finally:
        print("\nStopping chat connection...")
        # Stop chat client
        chat.stop()
        print("Closing Twitch...")
        # Initialize program closing coroutine (can be possibly finnicky, so mind that)
        await twitch.close()

        print("\nProgram Status: Off")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # the program can occasionally get stuck when trying to exit the API connection. Since this would be the end of the program,
        # it should generally be okay to use the terminating function, os._exit(), however use with caution for your use case.
        print("\nProgram manually stopped.")
        os._exit(0)
    finally:
        # Exit program entirely (may have to swap out with os._exit())
        sys.exit(0) 