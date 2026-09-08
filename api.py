"""
FastAPI Backend Application for Twitch Chat Sentiment Analysis.
Provides REST and WebSocket endpoints for real-time and historical chat sentiment.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
from primary import TwitchSentimentWorker, load_model
from src.db_service import BaseDatabaseService, get_db_service

logger = logging.getLogger("api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern async lifespan context manager.
    Initializes database and loads PyTorch model into memory once on startup.
    """
    logger.info("Initializing database connection...")
    db_service: BaseDatabaseService = get_db_service()
    try:
        await db_service.init_db()
        logger.info(f"Database ({config.DB_TYPE}) initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

    app.state.db_service = db_service

    logger.info("Loading PyTorch sentiment model into memory once on startup...")
    try:
        classifier = load_model()
        app.state.classifier = classifier
        logger.info("PyTorch model cached in application state.")
    except Exception as e:
        logger.error(f"Failed to load PyTorch model during startup: {e}")
        app.state.classifier = None

    # Channel workers dictionary: channel_name -> TwitchSentimentWorker
    app.state.workers: Dict[str, TwitchSentimentWorker] = {}

    yield

    # Shutdown logic
    logger.info("Shutting down active Twitch channel workers...")
    workers: Dict[str, TwitchSentimentWorker] = getattr(app.state, "workers", {})
    for channel_name, worker in list(workers.items()):
        try:
            await worker.stop()
        except Exception as e:
            logger.warning(f"Error stopping worker for #{channel_name}: {e}")

    await db_service.close()
    logger.info("Backend service shutdown complete.")


app = FastAPI(
    title="Twitch Chat Sentiment Analytics Engine API",
    description="Real-time and historical sentiment analysis for Twitch live streams.",
    version="2.0.0",
    lifespan=lifespan,
)

# --- CORS Configuration ---
# Allow Vite dev server (http://localhost:5173), other local frontends, and regex wildcard with credentials
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_origin_regex=r"^https?://.*$",  # Allow any origin with allow_credentials=True
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Response Models ---
class HealthResponse(BaseModel):
    status: str
    db_type: str
    model_loaded: bool
    active_channels: List[str]


class ConnectResponse(BaseModel):
    channel: str
    status: str
    message: str
    vod_id: Optional[str] = None


class DisconnectResponse(BaseModel):
    channel: str
    status: str
    message: str


class ChannelStatusResponse(BaseModel):
    channel: str
    connected: bool
    vod_id: Optional[str] = None
    stream_start: Optional[float] = None
    active_subscribers: int


class RealtimeMetricsResponse(BaseModel):
    channel: str
    window_seconds: int
    total_messages: int
    pos_ratio: float
    neg_ratio: float
    neu_ratio: float
    avg_latency_ms: float
    messages_per_second: float


class ChatMessageItem(BaseModel):
    timestamp: float
    channel: str
    message: str
    label: str
    score: float
    latency: float


class TimelineMetricItem(BaseModel):
    channel: str
    minute_timestamp: int
    pos_count: int
    neu_count: int
    neg_count: int
    avg_pos_score: float
    avg_neg_score: float
    total_messages: int


# --- Helper Functions for State Access ---
def get_current_db_service() -> BaseDatabaseService:
    """Safely obtain db_service from app.state or fallback to get_db_service singleton."""
    if hasattr(app.state, "db_service") and app.state.db_service is not None:
        return app.state.db_service
    return get_db_service()


def get_current_workers() -> Dict[str, TwitchSentimentWorker]:
    """Safely obtain active workers dictionary from app.state."""
    if not hasattr(app.state, "workers") or app.state.workers is None:
        app.state.workers = {}
    return app.state.workers


# --- REST Endpoints ---
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check and status report."""
    workers = get_current_workers()
    active = [c for c, w in workers.items() if w.is_running]
    classifier = getattr(app.state, "classifier", None)
    return HealthResponse(
        status="healthy",
        db_type=config.DB_TYPE,
        model_loaded=classifier is not None,
        active_channels=active,
    )


@app.post(
    "/api/v1/channels/{channel_name}/connect",
    response_model=ConnectResponse,
    tags=["Channel Management"],
)
async def connect_channel(channel_name: str):
    """
    Connect to a Twitch channel and start ingestion & inference in the background.
    Uses headless authentication with tokens from .env.
    """
    channel = channel_name.lower().strip()
    workers = get_current_workers()

    if channel in workers and workers[channel].is_running:
        return ConnectResponse(
            channel=channel,
            status="already_connected",
            message=f"Worker for channel #{channel} is already running.",
            vod_id=workers[channel].vod_id,
        )

    classifier = getattr(app.state, "classifier", None)
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sentiment model failed to load. Check server logs.",
        )

    db_service = get_current_db_service()
    worker = TwitchSentimentWorker(
        channel_name=channel,
        classifier=classifier,
        db_service=db_service,
    )

    try:
        await worker.start()
        workers[channel] = worker
        return ConnectResponse(
            channel=channel,
            status="connected",
            message=f"Successfully connected to #{channel}.",
            vod_id=worker.vod_id,
        )
    except Exception as e:
        logger.error(f"Error connecting to channel #{channel}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to connect to Twitch channel #{channel}: {str(e)}",
        )


@app.post(
    "/api/v1/channels/{channel_name}/disconnect",
    response_model=DisconnectResponse,
    tags=["Channel Management"],
)
async def disconnect_channel(channel_name: str):
    """Disconnect and stop the background worker for a Twitch channel."""
    channel = channel_name.lower().strip()
    workers = get_current_workers()

    if channel not in workers or not workers[channel].is_running:
        return DisconnectResponse(
            channel=channel,
            status="not_connected",
            message=f"Channel #{channel} is not currently running.",
        )

    worker = workers[channel]
    await worker.stop()
    del workers[channel]

    return DisconnectResponse(
        channel=channel,
        status="disconnected",
        message=f"Successfully disconnected #{channel}.",
    )


@app.get(
    "/api/v1/channels/{channel_name}/status",
    response_model=ChannelStatusResponse,
    tags=["Channel Management"],
)
async def get_channel_status(channel_name: str):
    """Query live worker connection status for a channel."""
    channel = channel_name.lower().strip()
    workers = get_current_workers()
    worker = workers.get(channel)

    is_running = worker.is_running if worker else False
    return ChannelStatusResponse(
        channel=channel,
        connected=is_running,
        vod_id=worker.vod_id if worker else None,
        stream_start=worker.stream_start if worker else None,
        active_subscribers=len(worker.subscribers) if worker else 0,
    )


@app.get(
    "/api/v1/channels/{channel_name}/sentiment/realtime",
    response_model=RealtimeMetricsResponse,
    tags=["Sentiment Analytics"],
)
async def get_realtime_sentiment(
    channel_name: str,
    window_seconds: int = Query(
        default=30, ge=5, le=300, description="Rolling time window in seconds"
    ),
):
    """
    Get current moving-window sentiment ratios and message throughput for a channel.
    """
    channel = channel_name.lower().strip()
    db_service: BaseDatabaseService = get_current_db_service()
    metrics = await db_service.get_realtime_metrics(
        channel, window_seconds=window_seconds
    )
    return RealtimeMetricsResponse(**metrics)


@app.get(
    "/api/v1/channels/{channel_name}/sentiment/timeline",
    response_model=List[TimelineMetricItem],
    tags=["Sentiment Analytics"],
)
async def get_timeline_sentiment(
    channel_name: str,
    start_ts: Optional[float] = Query(
        None, description="Start timestamp filter (Unix seconds)"
    ),
    end_ts: Optional[float] = Query(
        None, description="End timestamp filter (Unix seconds)"
    ),
):
    """
    Get minute-by-minute historical sentiment metrics for timeline charts.
    """
    channel = channel_name.lower().strip()
    db_service: BaseDatabaseService = get_current_db_service()
    metrics = await db_service.get_timeline_metrics(
        channel, start_ts=start_ts, end_ts=end_ts
    )
    return metrics


@app.get(
    "/api/v1/channels/{channel_name}/messages",
    response_model=List[ChatMessageItem],
    tags=["Chat Logs"],
)
async def get_recent_messages(
    channel_name: str,
    limit: int = Query(
        default=50, ge=1, le=200, description="Maximum messages to return"
    ),
):
    """
    Get the most recent classified chat messages for a channel.
    """
    channel = channel_name.lower().strip()
    db_service: BaseDatabaseService = get_current_db_service()
    messages = await db_service.get_recent_messages(channel, limit=limit)
    return messages


# --- WebSocket Real-Time Endpoint ---
@app.websocket("/ws/channels/{channel_name}")
async def websocket_channel_stream(websocket: WebSocket, channel_name: str):
    """
    WebSocket endpoint for real-time live streaming updates.
    Streams individual classified chat messages and periodic sentiment metrics snapshots.
    """
    channel = channel_name.lower().strip()
    await websocket.accept()

    workers = get_current_workers()
    worker = workers.get(channel)

    sub_queue: Optional[asyncio.Queue] = None
    if worker and worker.is_running:
        sub_queue = worker.subscribe()
    else:
        await websocket.send_json(
            {
                "type": "warning",
                "message": f"Worker for #{channel} is not running. Connect channel first via POST /api/v1/channels/{channel}/connect.",
            }
        )

    try:
        last_metric_push = 0.0
        while True:
            # If worker was started after websocket connected
            if sub_queue is None:
                worker = workers.get(channel)
                if worker and worker.is_running:
                    sub_queue = worker.subscribe()

            # Push real-time chat messages as they arrive
            if sub_queue is not None:
                try:
                    event = await asyncio.wait_for(sub_queue.get(), timeout=1.0)
                    await websocket.send_json(event)
                    sub_queue.task_done()
                except asyncio.TimeoutError:
                    pass

            # Push periodic realtime metrics snapshot every 2 seconds
            now = time.time()
            if now - last_metric_push >= 2.0:
                last_metric_push = now
                db_service = get_current_db_service()
                metrics = await db_service.get_realtime_metrics(
                    channel, window_seconds=30
                )
                await websocket.send_json({"type": "metrics", "data": metrics})

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from #{channel}")
    except Exception as e:
        logger.error(f"WebSocket error for #{channel}: {e}")
    finally:
        if worker and sub_queue is not None:
            worker.unsubscribe(sub_queue)
