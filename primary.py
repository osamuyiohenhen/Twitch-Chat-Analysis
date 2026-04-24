# main.py
from twitchAPI.chat import Chat, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
import asyncio
import time
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from aiocsv import AsyncWriter
import aiofiles

# Config
import config

# Globals
# We keep queues global for the workers to access easily
raw_queue = asyncio.Queue()
results_queue = asyncio.Queue()

# We keep a reference to the Streamlit UI Queue (if it exists)
# UI_QUEUE = None

HF_REPO = "muyihenhen/twitch-roberta-sentiment-v1"
LOCAL_DIR = "models/twitch-sentiment-v2"


# 1. Model Loading
def load_model():
    print("Loading model...\n")
    # Use local if it exists (faster), else use HF Cloud
    MODEL_PATH = LOCAL_DIR if os.path.exists(LOCAL_DIR) else HF_REPO

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH, num_labels=3
        )

        device = 0 if torch.cuda.is_available() else -1

        # Create the pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=None,
        )
        return classifier  # IMPORTANT: Return Classifier Object
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# --- 2. Workers ---


async def on_message(msg: ChatMessage):
    # Filter some junk messages
    if msg.user in config.bot_list or msg.text.startswith("!") or "http" in msg.text:
        return
    raw_queue.put_nowait((msg.room.name, msg.text))


# NOTE: We pass 'classifier' as an argument now
async def run_inference(classifier, text):
    start = time.perf_counter()

    # Run the classifier in a separate thread so it doesn't block Twitch
    result = await asyncio.to_thread(classifier, text)

    latency_ms = (time.perf_counter() - start) * 1000
    top_label, top_score = result[0][0]["label"], result[0][0]["score"]
    return top_label, top_score, latency_ms


# Call run_inference and contain the results
async def model_worker(classifier):
    print("Worker started...")
    while True:
        channel, text = await raw_queue.get()
        try:
            # Pass the classifier down
            label, score, latency = await run_inference(classifier, text)
            sentiment = (label, score, latency)

            # Send to writer
            results_queue.put_nowait((channel, text, sentiment))
            print(f"{[channel]} {label.upper()}, {score:.2f}: {text}")
        except Exception as e:
            print(f"Inference Error: {e}")
        raw_queue.task_done()


async def writer_worker():
    # FORCE CREATE: Always wipe the file and write a fresh header on startup
    # We use sync open() here just once to guarantee the file is ready before the loop starts
    with open("live_data.csv", mode="w", newline="", encoding="utf-8") as f:
        f.write("timestamp,channel,message,label,score,latency\n")

    while True:
        channel, text, sentiment = await results_queue.get()
        label, score, latency = sentiment
        current_time = time.time()

        # Now append data ('a' mode)
        await save_message([current_time, channel, text, label, score, latency])

        results_queue.task_done()


async def save_message(data_row):
    # Use aiofiles for non-blocking file I/O so writing won't block Twitch event loop
    async with aiofiles.open(
        "live_data.csv", mode="a", newline="", encoding="utf-8"
    ) as f:
        writer = AsyncWriter(f)
        await writer.writerow(data_row)


# --- 3. MAIN RUNNER ---


async def run_backend_async(target_channel, loaded_classifier):
    # Start Workers (Pass the classifier to the worker!)
    asyncio.create_task(model_worker(loaded_classifier))
    asyncio.create_task(writer_worker())

    # Auth
    twitch = await Twitch(config.client_id, config.client_secret)
    auth = UserAuthenticator(twitch, [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT])
    token, refresh_token = await auth.authenticate()
    await twitch.set_user_authentication(
        token, [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT], refresh_token
    )

    chat = await Chat(twitch)
    chat.register_event(ChatEvent.MESSAGE, on_message)
    chat.start()

    try:
        await chat.join_room(target_channel)
        print(f"Joined {target_channel}")
    except Exception as e:
        print(f"Failed to join: {e}")
        return

    # Keep alive
    while True:
        await asyncio.sleep(1)


# --- 4. THREAD ENTRY POINT (Call this from app.py) ---
def start_backend(target_channel, ui_queue, classifier):
    # Windows Fix: Set policy to prevent "Event loop is closed" errors in threads
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_backend_async(target_channel, classifier))
