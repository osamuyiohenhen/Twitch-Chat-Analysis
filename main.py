# Imports
from twitchAPI.chat import Chat, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
import asyncio
import time
import os
import functools

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
MODEL_DIR = "models/twitch-sentiment-v2"

# Load tokenizer from local model directory
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# Load sequence classification model (3 labels: Negative, Neutral, Positive)
# Note: specifying num_labels ensures the classification head has the expected output size.
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, num_labels=3)

# Device selection for the pipeline (0 = first CUDA device, -1 = CPU)
device = 0 if torch.cuda.is_available() else -1

# Create a Hugging Face pipeline for text classification with explicit model/tokenizer
# top_k=None returns scores for all labels rather than only the top result.
CLASSIFIER = pipeline(
    "sentiment-analysis", model=model, tokenizer=tokenizer, device=device, top_k=None
)

raw_queue = asyncio.Queue()  # Holds raw text
results_queue = asyncio.Queue()  # Holds finished predictions


# Asynchronous chat message handler
async def on_message(msg: ChatMessage):
    # Filter out bot messages, commands, and links
    if msg.user in BOT_LIST:
        return
    # Skip commands (starting with !)
    if msg.text.startswith("!"):
        return
    # Skip links (http or https)
    if "http" in msg.text.lower():  # cheaper than per-word slicing
        return

    raw_queue.put_nowait((msg.room.name, msg.text))
    # Print the incoming chat message
    # print(f"{msg.user.display_name}: {msg.text}")


# This runs the synchronous model in a separate thread so it doesn't freeze the bot
async def run_blocking_model(text):
    start = time.perf_counter()
    # Use functools.partial because asyncio.to_thread only accepts a callable and its arguments

    result = await asyncio.to_thread(functools.partial(CLASSIFIER, text))
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    # Print primary sentiment and score
    top_label, top_score = result[0][0]["label"], result[0][0]["score"]
    return top_label, top_score, latency_ms


# 2. The Worker (Consumer)
async def model_worker():
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

    # Optional: print additional label scores for inspection
    # mid_label, mid_score = sentiment[0][1]['label'], sentiment[0][1]['score']
    # print(f"Second sentiment: {mid_label}, Score: {mid_score:.3f}")
    # bot_label, bot_score = sentiment[0][2]['label'], sentiment[0][2]['score']
    # print(f"Third sentiment: {bot_label}, Score: {bot_score:.3f}")


async def writer_worker():
    while True:
        channel, text, sentiment = await results_queue.get()  # Wait for work
        label, score, latency_ms = sentiment
        print(f"{channel}: {text}")
        print(f"   {label.upper()}, Score: {score:.3f} [{latency_ms:.2f} ms]")
        await save_message((channel, text, sentiment))
        results_queue.task_done()


async def save_message(message):
    # Append a single CSV row asynchronously
    async with aiofiles.open(
        "twitch_chats.csv", mode="a", newline="", encoding="utf-8"
    ) as f:
        writer = AsyncWriter(f)
        await writer.writerow(message)


# Main application
async def main():
    # Start the worker and writer tasks
    asyncio.create_task(model_worker())
    asyncio.create_task(writer_worker())

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

    # Prompt user for target channel
    while True:
        target_channel = await asyncio.to_thread(
            input,
            "\nEnter a valid Twitch channel you wish to connect to (or 'q' to exit): ",
        )
        target_channel = target_channel.strip().lower()

        if target_channel == "q":
            print("\nProgram Status: Off")
            return
        elif not target_channel:
            continue
        else:
            break

    # Connect & Hold
    try:
        print(f"\nJoining channel: {target_channel}...")

        # Attempt to join
        await asyncio.wait_for(chat.join_room(target_channel), timeout=1.5)

        print(f"Connected to {target_channel}. Press Ctrl+C to exit.\n")

    except asyncio.TimeoutError:
        print(f"\n[ERROR] Could not join channel '{target_channel}'.")
        print("The channel may not exist or you may be banned.")

    while True:
        await asyncio.sleep(
            0.05
        )  # Constantly sleep to allow for the live view of chats to update

    # finally:
    #     print("\nStopping chat connection...")
    #     chat.stop()
    #     print("Closing Twitch...")
    #     await twitch.close()
    #     print("\nProgram Status: Off")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Allow Ctrl+C to exit cleanly; force-exit as a last resort.
        print("\nProgram manually stopped.")
        os._exit(0)
    finally:
        os._exit(0)
