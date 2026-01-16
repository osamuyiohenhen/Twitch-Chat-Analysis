# Imports
from twitchAPI.chat import Chat, EventData, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
import asyncio
import time
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Data extraction utilities (used for collecting training data)
from aiocsv import AsyncWriter
import aiofiles

# Contains client_id, client_secret, and BOT_LIST
import config

# Constants
CLIENT_ID = config.client_id
CLIENT_SECRET = config.client_secret
BOT_LIST = config.bot_list

# OAuth scopes required by the application
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]

# Path to a locally saved model directory.
# NOTE: This repository does NOT auto-download the model.
# First-time users must download the model from Hugging Face manually
# and place it in this folder before running the program.
MODEL_DIR = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load tokenizer from local model directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Load sequence classification model (3 labels: Negative, Neutral, Positive)
# Note: specifying num_labels ensures the classification head has the expected output size.
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_DIR,
    num_labels=3
)

# Device selection for the pipeline (0 = first CUDA device, -1 = CPU)
device = 0 if torch.cuda.is_available() else -1

# Create a Hugging Face pipeline for text classification with explicit model/tokenizer
# top_k=None returns scores for all labels rather than only the top result.
CLASSIFIER = pipeline(
    "sentiment-analysis", 
    model=model, 
    tokenizer=tokenizer, 
    device=device, 
    top_k=None
)

raw_queue = asyncio.Queue()  # Holds raw text
results_queue = asyncio.Queue() # Holds finished predictions

# Asynchronous chat message handler
async def on_message(msg: ChatMessage):
    # Filter out bot messages, commands, and links
    if msg.user in BOT_LIST:
        return
    # Skip commands (starting with !)
    if msg.text.startswith('!'):
        return
    # Skip links (http or https)
    if "http" in msg.text.lower():  # cheaper than per-word slicing
        return

    raw_queue.put_nowait((msg.room.name, msg.text))
    # Print the incoming chat message
    # print(f"{msg.user.display_name}: {msg.text}")

import functools

# This runs the synchronous model in a separate thread so it doesn't freeze the bot
async def run_blocking_model(text):
    loop = asyncio.get_running_loop()
    
    start = time.perf_counter()
    # Use functools.partial because run_in_executor doesn't accept kwargs directly
    # 'None' uses the default ThreadPoolExecutor
    result = await loop.run_in_executor(None, functools.partial(CLASSIFIER, text))
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    # Print primary sentiment and score
    top_label, top_score = result[0][0]['label'], result[0][0]['score']
    return top_label, top_score, latency_ms

# 2. THE WORKER (Consumer)
async def worker_logic():
    print("Worker started...")
    while True:
        # Wait for a message (Non-blocking)
        channel, text = await raw_queue.get()
        
        try:
            label, score, latency_ms = await run_blocking_model(text)
            sentiment = (label, score, latency_ms)
            # Now save to CSV or whatever you need
            # print(f"[{channel}] Processed: {sentiment}")
            
        except Exception as e:
            print(f"Error: {e}")
            sentiment = ("ERROR", 0.0, 0.0)

        results_queue.put_nowait((channel, text, sentiment))
        raw_queue.task_done()

        # start = time.perf_counter()
        # sentiment = CLASSIFIER(msg.text)
        # end = time.perf_counter()
        # latency_ms = (end - start) * 1000

    # Optional: print additional label scores for inspection
    # mid_label, mid_score = sentiment[0][1]['label'], sentiment[0][1]['score']
    # print(f"Second sentiment: {mid_label}, Score: {mid_score:.3f}")
    # bot_label, bot_score = sentiment[0][2]['label'], sentiment[0][2]['score']
    # print(f"Third sentiment: {bot_label}, Score: {bot_score:.3f}")

async def writer_worker():
    while True:
        channel, text, sentiment = await results_queue.get() # Wait for work
        label, score, latency_ms = sentiment
        print(f"{channel}: {text}")
        print(f"   {label.upper()}, Score: {score:.3f} [{latency_ms:.2f} ms]")
        await save_message((channel, text, sentiment))
        results_queue.task_done()

async def save_message(message):
    # Append a single CSV row asynchronously (non-blocking file IO)
    async with aiofiles.open("twitch_chat_labelings.csv", mode='a', newline='', encoding='utf-8') as f:
        writer = AsyncWriter(f)
        await writer.writerow(message)

# Main application entrypoint
async def main():
    # Authenticate the application / user
    print("Authenticating...")
    twitch = await Twitch(CLIENT_ID, CLIENT_SECRET)
    auth = UserAuthenticator(twitch, USER_SCOPE)
    token, refresh_token = await auth.authenticate()
    await twitch.set_user_authentication(token, USER_SCOPE, refresh_token) 
    print("Authentication successful.")

    # Initialize chat client
    chat = await Chat(twitch)

    # Register message event handler
    chat.register_event(ChatEvent.MESSAGE, on_message)
        
    print("Initializing connection...")
    # Start the chat client
    chat.start()
    print("Connection started.")

    current_channel = None

    asyncio.create_task(worker_logic())
    asyncio.create_task(writer_worker())
    try:
        while True:
            # If already in a channel, leave it before joining a new one
            if current_channel:
                await chat.leave_room(current_channel)
                # await asyncio.sleep(0.3)
                print(f"Leaving channel: {current_channel}...")
                current_channel = None

            # wait until processing queues are empty before prompting
            while not raw_queue.empty() or not results_queue.empty():
                await asyncio.sleep(0.2)

            loop = asyncio.get_running_loop()
            target_channel = await loop.run_in_executor(
                None,
                lambda: input("\nEnter the Twitch channel you wish to connect to (or type 'q' to exit): ")
            )
            target_channel = target_channel.lower()

            if target_channel == 'q':
                break
            if not target_channel:
                print("Channel name cannot be empty. Please try again.")
                continue

            try:
                print(f"\nJoining channel: {target_channel}...")
                # Attempt to join the chat room with a short timeout to fail fast on invalid channels
                await asyncio.wait_for(chat.join_room(target_channel), timeout=1.5)

                # Successfully joined; block here until user presses ENTER to switch or quit
                current_channel = target_channel      

            except asyncio.TimeoutError:
                # Timeout likely means the channel doesn't exist or the bot is banned
                print(f"\n[ERROR] Could not join channel '{target_channel}'.")
                print("The channel may not exist or you may be banned.")
    
    finally:
        # Graceful shutdown sequence
        print("\nStopping chat connection...")
        chat.stop()
        print("Closing Twitch...")
        await twitch.close()
        print("\nProgram Status: Off")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Allow Ctrl+C to exit cleanly; force-exit as a last resort.
        print("\nProgram manually stopped.")
        os._exit(0)
    finally:
        # Ensure process terminates
        os._exit(0)
