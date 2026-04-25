from twitchAPI.chat import Chat, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
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

# Async queues for message flow
raw_queue = asyncio.Queue()
results_queue = asyncio.Queue()

HF_REPO = "muyihenhen/twitch-roberta-sentiment-v1"
LOCAL_DIR = "models/twitch-sentiment-v2"  # local filepath for model

TARGET_SCOPES = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]


def load_model():
    """Load sentiment classifier from local or HuggingFace."""
    print("Loading model...")
    MODEL_PATH = LOCAL_DIR if os.path.exists(LOCAL_DIR) else HF_REPO

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH, num_labels=3
        )
        device = 0 if torch.cuda.is_available() else -1
        classifier = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            device=device,
            top_k=None,
        )
        return classifier
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


async def on_message(msg: ChatMessage):
    """Twitch chat event handler. Filter and queue valid messages."""
    if msg.user in config.bot_list or msg.text.startswith("!") or "http" in msg.text:
        return
    raw_queue.put_nowait((msg.room.name, msg.text))


async def run_inference(classifier, text):
    """Run sentiment classifier in thread pool to avoid blocking."""
    start = time.perf_counter()
    result = await asyncio.to_thread(classifier, text)
    latency_ms = (time.perf_counter() - start) * 1000
    top_label, top_score = result[0][0]["label"], result[0][0]["score"]
    return top_label, top_score, latency_ms


async def model_worker(classifier):
    """Process messages and run inference."""
    print("Model worker started.")
    while True:
        channel, text = await raw_queue.get()
        try:
            label, score, latency = await run_inference(classifier, text)
            results_queue.put_nowait((channel, text, (label, score, latency)))
            print(f"{[channel]} {label.upper()}, {score:.2f}: {text}")
        except Exception as e:
            print(f"Inference Error: {e}")
        raw_queue.task_done()


async def writer_worker():
    """Write results to CSV file."""
    with open("live_data.csv", mode="w", newline="", encoding="utf-8") as f:
        f.write("timestamp,channel,message,label,score,latency\n")

    while True:
        channel, text, sentiment = await results_queue.get()
        label, score, latency = sentiment
        await save_message([time.time(), channel, text, label, score, latency])
        results_queue.task_done()


async def save_message(data_row):
    """Async append row to CSV."""
    async with aiofiles.open(
        "live_data.csv", mode="a", newline="", encoding="utf-8"
    ) as f:
        writer = AsyncWriter(f)
        await writer.writerow(data_row)


async def run_backend_async(target_channel, loaded_classifier):
    """Main backend: authenticate, connect to chat, and process messages."""
    asyncio.create_task(model_worker(loaded_classifier))
    asyncio.create_task(writer_worker())

    twitch = await Twitch(config.client_id, config.client_secret)
    await twitch.authenticate_app([])

    chat = await Chat(twitch)
    chat.register_event(ChatEvent.MESSAGE, on_message)
    chat.start()

    try:
        await chat.join_room(target_channel)
        print(f"Joined {target_channel}")
    except Exception as e:
        print(f"Failed to join: {e}")
        return

    while True:
        await asyncio.sleep(1)


def start_backend(target_channel, ui_queue, classifier):
    """Entry point called by run.py. Starts the async backend in a new event loop."""
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_backend_async(target_channel, classifier))
