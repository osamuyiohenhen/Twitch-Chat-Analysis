from twitchAPI.chat import Chat, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.twitch import Twitch
import asyncio
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from aiocsv import AsyncWriter
import aiofiles
import aiosqlite
import time

# Config
import config

# Async queues for message flow
raw_queue = asyncio.Queue()
results_queue = asyncio.Queue()

HF_REPO = "muyihenhen/twitch-roberta-sentiment-v1"
LOCAL_DIR = "models/twitch-sentiment-v2"  # local filepath for model
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twitch_data.db")

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

async def init_db():
    """Run this once to set up the table and WAL mode."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS chat_log (
                timestamp REAL,
                channel TEXT,
                message TEXT,
                label TEXT,
                score REAL,
                latency REAL
            )
        """)
        # The index makes querying the last 2 seconds instant
        await db.execute("CREATE INDEX IF NOT EXISTS idx_time ON chat_log(timestamp)")
        # WAL mode allows Streamlit to read while this script is writing
        await db.execute("PRAGMA journal_mode=WAL")
        await db.commit()

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
    """Write results to SQLite instead of CSV."""
    while True:
        channel, text, sentiment = await results_queue.get()
        label, score, latency = sentiment
        
        # Write directly to the DB
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                "INSERT INTO chat_log VALUES (?, ?, ?, ?, ?, ?)", 
                [time.time(), channel, text, label, score, latency]
            )
            await db.commit()
            
        results_queue.task_done()


async def run_backend_async(target_channel, loaded_classifier):
    """Main backend: authenticate, connect to chat, and process messages."""
    await init_db()
    asyncio.create_task(model_worker(loaded_classifier))
    asyncio.create_task(writer_worker())

    twitch = await Twitch(
        config.client_id, config.client_secret, authenticate_app=False
    )
    await twitch.set_user_authentication(
        config.user_token, TARGET_SCOPES, config.refresh_token
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

    while True:
        await asyncio.sleep(1)


def start_backend(target_channel, ui_queue, classifier):
    """Entry point called by run.py. Starts the async backend in a new event loop."""
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_backend_async(target_channel, classifier))
