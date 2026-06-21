from twitchAPI.chat import Chat, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent, VideoType
from twitchAPI.twitch import Twitch
import asyncio
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from torch.utils.data import Dataset

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


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


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
            batch_size=16,
        )
        return classifier
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


async def get_session_info(twitch, channel_name):
    """Get the broadcaster's user ID and current VOD ID at connect time."""
    user_id = None
    async for user in twitch.get_users(logins=[channel_name]):
        user_id = user.id
        break

    if not user_id:
        return None, None

    vod_id = None
    async for video in twitch.get_videos(user_id=user_id, video_type=VideoType.ARCHIVE):
        vod_id = video.id
        break

    stream_started_at = None
    async for stream in twitch.get_streams(user_id=[user_id]):
        stream_started_at = stream.started_at.timestamp()  # convert to unix timestamp
        break

    return user_id, vod_id, stream_started_at


async def init_db():
    """Run this once to set up the table and WAL mode."""
    async with aiosqlite.connect(DB_PATH) as db:
        # Set timeout to prevent "database is locked" errors
        await db.execute("PRAGMA busy_timeout = 5000")  # 5 second timeout
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

        await db.execute("""
            CREATE TABLE IF NOT EXISTS session_info (
                user_id TEXT,
                vod_id TEXT,
                stream_start_time REAL,
                monitor_start_time REAL
            )
        """)

        # WAL mode allows Streamlit to read while this script is writing
        await db.execute("PRAGMA journal_mode=WAL")
        await db.commit()


async def on_message(msg: ChatMessage):
    """Twitch chat event handler. Filter and queue valid messages."""
    if msg.user in config.bot_list or msg.text.startswith("!") or "http" in msg.text:
        return
    raw_queue.put_nowait((msg.room.name, msg.text))


# async def run_inference(classifier, text):
#     """Run sentiment classifier in thread pool to avoid blocking."""
#     start = time.perf_counter()
#     result = await asyncio.to_thread(classifier, text)
#     latency_ms = (time.perf_counter() - start) * 1000
#     top_label, top_score = result[0][0]["label"], result[0][0]["score"]
#     return top_label, top_score, latency_ms


# async def model_worker(classifier):
#     """Process messages and run inference."""
#     print("Model worker started.")
#     while True:
#         channel, text = await raw_queue.get()
#         asyncio.create_task(process_message(classifier, channel, text))
#         raw_queue.task_done()

# async def process_message(classifier, channel, text):
#     try:
#         label, score, latency = await run_inference(classifier, text)
#         results_queue.put_nowait((channel, text, (label, score, latency)))
#         print(f"{[channel]} {label.upper()}, {score:.2f}: {text}")
#     except Exception as e:
#         print(f"Inference Error: {e}")


async def model_worker(classifier, batch_size=16):
    """Process messages using Natural Batching.
    Instant response on low load, automatically batches on high load.
    """
    print("Model worker started.")
    while True:
        # 1. Wait for at least one message (0% CPU when chat is silent)
        channel, text = await raw_queue.get()
        batch = [(channel, text)]
        raw_queue.task_done()

        # 2. Grab any other messages that accumulated while the GPU was busy
        while len(batch) < batch_size and not raw_queue.empty():
            try:
                channel_next, text_next = raw_queue.get_nowait()
                batch.append((channel_next, text_next))
                raw_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # 3. Process batch sequentially to keep GPU operations ordered and simple
        await process_batch(classifier, batch)


async def process_batch(classifier, batch):
    """Helper to send a batch of messages to the classifier using PyTorch Datasets."""
    try:
        channels = [item[0] for item in batch]
        texts = [item[1] for item in batch]

        # Wrap texts in ListDataset so Hugging Face uses native GPU pipelining
        dataset = ListDataset(texts)

        start = time.perf_counter()
        results = await asyncio.to_thread(classifier, dataset)
        latency_ms = (time.perf_counter() - start) * 1000

        for i, result in enumerate(results):
            top_label = result[0]["label"]
            top_score = result[0]["score"]

            results_queue.put_nowait(
                (channels[i], texts[i], (top_label, top_score, latency_ms))
            )

    except Exception as e:
        print(f"Batch Inference Error: {e}")


async def writer_worker():
    """Write results to SQLite."""
    while True:
        # 1. Wait for at least one item
        channel, text, sentiment = await results_queue.get()
        label, score, latency = sentiment

        rows = [[time.time(), channel, text, label, score, latency]]
        results_queue.task_done()

        while not results_queue.empty():
            try:
                c, t, s = results_queue.get_nowait()
                lbl, scr, lat = s
                rows.append([time.time(), c, t, lbl, scr, lat])
                results_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # Bulk insert and commit once
        async with aiosqlite.connect(DB_PATH) as db:
            await db.executemany("INSERT INTO chat_log VALUES (?, ?, ?, ?, ?, ?)", rows)

            await db.commit()


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

    user_id, vod_id, stream_start = await get_session_info(twitch, target_channel)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO session_info VALUES (?, ?, ?, ?)",
            [user_id, vod_id, stream_start, time.time()],
        )
        await db.commit()

    chat = await Chat(twitch)
    chat.register_event(ChatEvent.MESSAGE, on_message)
    chat.start()

    try:
        await chat.join_room(target_channel)
        print(f"Joined {target_channel}, vod_id: {vod_id}")
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
