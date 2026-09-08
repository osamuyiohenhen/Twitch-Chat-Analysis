import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from twitchAPI.chat import Chat, ChatMessage
from twitchAPI.twitch import Twitch
from twitchAPI.type import AuthScope, ChatEvent, SortMethod, VideoType

import config
from src.db_service import BaseDatabaseService, get_db_service

logger = logging.getLogger("twitch_worker")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("twitchAPI.chat").setLevel(logging.WARNING)

HF_REPO = "muyihenhen/twitch-roberta-sentiment-v1"
LOCAL_DIR = "models/twitch-sentiment-v2"
TARGET_SCOPES = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT, AuthScope.CHANNEL_BOT]


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


async def user_auth_refresh_callback(token: str, refresh_token: str):
    """Callback function triggered when Twitch tokens are automatically refreshed silently."""
    logger.info("Twitch tokens automatically refreshed. Updating configuration...")

    config.user_token = token
    config.refresh_token = refresh_token

    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            new_lines = []
            found_token = False
            found_refresh = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("TWITCH_USER_TOKEN="):
                    new_lines.append(f'TWITCH_USER_TOKEN="{token}"\n')
                    found_token = True
                elif stripped.startswith("TWITCH_REFRESH_TOKEN="):
                    new_lines.append(f'TWITCH_REFRESH_TOKEN="{refresh_token}"\n')
                    found_refresh = True
                else:
                    new_lines.append(line)

            if not found_token:
                new_lines.append(f'TWITCH_USER_TOKEN="{token}"\n')
            if not found_refresh:
                new_lines.append(f'TWITCH_REFRESH_TOKEN="{refresh_token}"\n')

            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            logger.info("Successfully persisted refreshed tokens to .env!")
        except Exception as e:
            logger.error(f"Error saving refreshed tokens to .env: {e}")
    else:
        logger.warning(".env file not found. Refreshed tokens updated in-memory only.")


def load_model():
    """Load sentiment classifier from local directory or HuggingFace."""
    logger.info("Loading PyTorch sentiment classifier...")
    model_path = LOCAL_DIR if os.path.exists(LOCAL_DIR) else HF_REPO

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=3
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
        logger.info(
            f"Model loaded successfully on {'GPU (cuda:0)' if device == 0 else 'CPU'}."
        )
        return classifier
    except Exception as e:
        logger.error(f"Failed to load sentiment model from {model_path}: {e}")
        raise RuntimeError(f"Could not load sentiment model: {e}") from e


async def get_session_info(
    twitch: Twitch, channel_name: str
) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """Fetch broadcaster user ID, latest VOD ID, and stream start time."""
    user_id = None
    async for user in twitch.get_users(logins=[channel_name]):
        user_id = user.id
        break

    if not user_id:
        return None, None, None

    vod_id = None
    async for video in twitch.get_videos(
        user_id=user_id, video_type=VideoType.ARCHIVE, sort=SortMethod.TIME
    ):
        vod_id = video.id
        break

    stream_started_at = None
    async for stream in twitch.get_streams(user_id=[user_id]):
        stream_started_at = stream.started_at.timestamp()
        break

    return user_id, vod_id, stream_started_at


class TwitchSentimentWorker:
    """
    Manages live chat ingestion, natural batching inference,
    database persistence, and real-time subscriber broadcasting.
    """

    def __init__(
        self,
        channel_name: str,
        classifier: Any,
        db_service: Optional[BaseDatabaseService] = None,
        batch_size: int = 16,
    ):
        self.channel_name = channel_name.lower()
        self.classifier = classifier
        self.db_service = db_service or get_db_service()
        self.batch_size = batch_size

        self.raw_queue: asyncio.Queue = asyncio.Queue()
        self.results_queue: asyncio.Queue = asyncio.Queue()
        self.subscribers: Set[asyncio.Queue] = set()

        self.twitch: Optional[Twitch] = None
        self.chat: Optional[Chat] = None
        self.tasks: List[asyncio.Task] = []
        self.is_running: bool = False

        self.user_id: Optional[str] = None
        self.vod_id: Optional[str] = None
        self.stream_start: Optional[float] = None

        # Minute metrics accumulator
        self.current_minute_ts: int = int(time.time() // 60 * 60)
        self.minute_stats: Dict[str, Any] = {
            "pos_count": 0,
            "neu_count": 0,
            "neg_count": 0,
            "sum_pos_score": 0.0,
            "sum_neg_score": 0.0,
            "total": 0,
        }

    def subscribe(self) -> asyncio.Queue:
        """Register a subscriber queue (e.g. for WebSocket clients)."""
        queue = asyncio.Queue(maxsize=100)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Unregister a subscriber queue."""
        self.subscribers.discard(queue)

    def _broadcast(self, event: Dict[str, Any]) -> None:
        """Push real-time events to all active subscriber queues."""
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

    async def _on_message(self, msg: ChatMessage):
        """Twitch chat incoming message handler."""
        user = msg.user.name.lower() if msg.user and msg.user.name else ""
        text = msg.text or ""

        if user in config.bot_list or text.startswith("!") or "http" in text:
            return

        self.raw_queue.put_nowait((self.channel_name, user, text))

    async def _model_worker(self):
        """Process messages using Natural Batching."""
        logger.info(f"Model worker started for channel '{self.channel_name}'.")
        while self.is_running:
            try:
                # Wait for at least one message
                channel, user, text = await self.raw_queue.get()
                batch = [(channel, user, text)]
                self.raw_queue.task_done()

                # Grab remaining messages up to batch_size
                while len(batch) < self.batch_size and not self.raw_queue.empty():
                    try:
                        c, u, t = self.raw_queue.get_nowait()
                        batch.append((c, u, t))
                        self.raw_queue.task_done()
                    except asyncio.QueueEmpty:
                        break

                await self._process_batch(batch)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in model worker: {e}")
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch: List[Tuple[str, str, str]]):
        """Run batch inference on PyTorch sentiment classifier."""
        try:
            texts = [item[2] for item in batch]
            dataset = ListDataset(texts)

            start = time.perf_counter()
            results = await asyncio.to_thread(self.classifier, dataset)
            latency_ms = (time.perf_counter() - start) * 1000

            timestamp = time.time()
            for i, result in enumerate(results):
                channel, user, text = batch[i]
                # Result format: [{'label': 'positive', 'score': 0.98}, ...]
                top_label = result[0]["label"].lower()
                top_score = float(result[0]["score"])

                msg_data = {
                    "timestamp": timestamp,
                    "channel": channel,
                    "user": user,
                    "message": text,
                    "label": top_label,
                    "score": top_score,
                    "latency": latency_ms,
                }
                self.results_queue.put_nowait(msg_data)
                self._broadcast({"type": "chat", "data": msg_data})

        except Exception as e:
            logger.error(f"Batch inference error: {e}")

    async def _writer_worker(self):
        """Batch write sentiment results to DB and maintain timeline stats."""
        logger.info(f"Writer worker started for channel '{self.channel_name}'.")
        while self.is_running:
            try:
                first_item = await self.results_queue.get()
                rows = [first_item]
                self.results_queue.task_done()

                while not self.results_queue.empty():
                    try:
                        item = self.results_queue.get_nowait()
                        rows.append(item)
                        self.results_queue.task_done()
                    except asyncio.QueueEmpty:
                        break

                # Save to database
                await self.db_service.save_messages(rows)

                # Update minute aggregation
                for r in rows:
                    msg_minute = int(r["timestamp"] // 60 * 60)
                    if msg_minute != self.current_minute_ts:
                        # Flush previous minute
                        if self.minute_stats["total"] > 0:
                            avg_pos = (
                                self.minute_stats["sum_pos_score"]
                                / self.minute_stats["pos_count"]
                                if self.minute_stats["pos_count"] > 0
                                else 0.0
                            )
                            avg_neg = (
                                self.minute_stats["sum_neg_score"]
                                / self.minute_stats["neg_count"]
                                if self.minute_stats["neg_count"] > 0
                                else 0.0
                            )
                            await self.db_service.save_minute_metric(
                                self.channel_name,
                                self.current_minute_ts,
                                self.minute_stats["pos_count"],
                                self.minute_stats["neu_count"],
                                self.minute_stats["neg_count"],
                                avg_pos,
                                avg_neg,
                                self.minute_stats["total"],
                            )

                        # Reset accumulator
                        self.current_minute_ts = msg_minute
                        self.minute_stats = {
                            "pos_count": 0,
                            "neu_count": 0,
                            "neg_count": 0,
                            "sum_pos_score": 0.0,
                            "sum_neg_score": 0.0,
                            "total": 0,
                        }

                    label = r["label"].lower()
                    score = r["score"]
                    self.minute_stats["total"] += 1
                    if label == "positive":
                        self.minute_stats["pos_count"] += 1
                        self.minute_stats["sum_pos_score"] += score
                    elif label == "negative":
                        self.minute_stats["neg_count"] += 1
                        self.minute_stats["sum_neg_score"] += score
                    else:
                        self.minute_stats["neu_count"] += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in writer worker: {e}")
                await asyncio.sleep(0.1)

    async def start(self) -> None:
        """Connect to Twitch chat headlessly, authenticate, and launch processing."""
        if self.is_running:
            logger.warning(
                f"Worker for channel '{self.channel_name}' is already running."
            )
            return

        if not config.client_id or not config.client_secret:
            raise RuntimeError(
                "Twitch client credentials missing! Please set TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in .env."
            )

        if not config.user_token or not config.refresh_token:
            raise RuntimeError(
                "Twitch user tokens missing! Run 'python scripts/auth_twitch.py' locally to generate TWITCH_USER_TOKEN and TWITCH_REFRESH_TOKEN."
            )

        logger.info(
            f"Authenticating headlessly with Twitch API for channel '{self.channel_name}'..."
        )
        self.twitch = await Twitch(
            config.client_id, config.client_secret, authenticate_app=False
        )
        self.twitch.user_auth_refresh_callback = user_auth_refresh_callback

        try:
            await self.twitch.set_user_authentication(
                config.user_token, TARGET_SCOPES, config.refresh_token
            )
            logger.info("Twitch silent token authentication successful.")
        except Exception as e:
            logger.error(f"Headless Twitch authentication failed: {e}")
            await self.twitch.close()
            raise RuntimeError(
                f"Twitch authentication failed: {e}. Please run 'python scripts/auth_twitch.py' locally to generate fresh tokens."
            ) from e

        # Session metadata
        self.user_id, self.vod_id, self.stream_start = await get_session_info(
            self.twitch, self.channel_name
        )
        await self.db_service.save_session_info(
            user_id=self.user_id or self.channel_name,
            vod_id=self.vod_id or "",
            stream_start=self.stream_start,
            session_time=time.time(),
        )

        # Connect Chat
        self.chat = await Chat(self.twitch)
        self.chat.register_event(ChatEvent.MESSAGE, self._on_message)
        self.chat.start()

        try:
            await self.chat.join_room(self.channel_name)
            logger.info(
                f"Successfully joined Twitch room: #{self.channel_name} (VOD ID: {self.vod_id})"
            )
        except Exception as e:
            logger.error(f"Failed to join Twitch room #{self.channel_name}: {e}")
            self.chat.stop()
            await self.twitch.close()
            raise RuntimeError(f"Could not join room #{self.channel_name}: {e}") from e

        self.is_running = True
        self.tasks.append(asyncio.create_task(self._model_worker()))
        self.tasks.append(asyncio.create_task(self._writer_worker()))

    async def stop(self) -> None:
        """Gracefully disconnect from Twitch and stop background workers."""
        if not self.is_running:
            return

        logger.info(f"Stopping worker for channel '{self.channel_name}'...")
        self.is_running = False

        if self.chat:
            try:
                self.chat.stop()
            except Exception as e:
                logger.warning(f"Error stopping chat: {e}")

        if self.twitch:
            try:
                await self.twitch.close()
            except Exception as e:
                logger.warning(f"Error closing Twitch client: {e}")

        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

        # Flush pending minute metric if any
        if self.minute_stats["total"] > 0:
            avg_pos = (
                self.minute_stats["sum_pos_score"] / self.minute_stats["pos_count"]
                if self.minute_stats["pos_count"] > 0
                else 0.0
            )
            avg_neg = (
                self.minute_stats["sum_neg_score"] / self.minute_stats["neg_count"]
                if self.minute_stats["neg_count"] > 0
                else 0.0
            )
            try:
                await self.db_service.save_minute_metric(
                    self.channel_name,
                    self.current_minute_ts,
                    self.minute_stats["pos_count"],
                    self.minute_stats["neu_count"],
                    self.minute_stats["neg_count"],
                    avg_pos,
                    avg_neg,
                    self.minute_stats["total"],
                )
            except Exception as e:
                logger.error(f"Error saving final minute metric: {e}")

        logger.info(f"Worker for '{self.channel_name}' stopped.")


# Backwards compatibility helpers for run.py and existing scripts
async def run_backend_async(target_channel: str, loaded_classifier: Any):
    """Async runner maintaining compatibility with existing CLI run scripts."""
    db = get_db_service()
    await db.init_db()
    worker = TwitchSentimentWorker(target_channel, loaded_classifier, db)
    await worker.start()
    try:
        while worker.is_running:
            await asyncio.sleep(1)
    except (KeyboardInterrupt, asyncio.CancelledError):
        await worker.stop()


def start_backend(target_channel: str, ui_queue: Any, classifier: Any):
    """Sync entrypoint for running backend in a separate thread/process."""
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_backend_async(target_channel, classifier))
    finally:
        loop.close()
