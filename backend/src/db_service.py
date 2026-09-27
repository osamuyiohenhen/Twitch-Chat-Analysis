"""
Database Abstraction Layer for Twitch Chat Analysis.
Supports both SQLite (local development) and AWS DynamoDB (cloud deployment).
"""

import abc
import asyncio
import time
import os
import uuid
from typing import List, Dict, Any, Optional
import aiosqlite
import boto3
from botocore.exceptions import ClientError

import config


class BaseDatabaseService(abc.ABC):
    """Abstract Base Class for Chat Analysis Database Services."""

    @abc.abstractmethod
    async def init_db(self) -> None:
        """Initialize required tables, indexes, or resources."""
        pass

    @abc.abstractmethod
    async def save_messages(self, rows: List[Dict[str, Any]]) -> None:
        """
        Save a batch of processed messages.
        Each dict has: timestamp, channel, user, message, label, score, latency
        """
        pass

    @abc.abstractmethod
    async def save_session_info(
        self,
        user_id: str,
        vod_id: str,
        stream_start: Optional[float],
        session_time: float,
    ) -> None:
        """Save broadcaster stream session metadata."""
        pass

    @abc.abstractmethod
    async def save_minute_metric(
        self,
        channel: str,
        minute_timestamp: int,
        pos_count: int,
        neu_count: int,
        neg_count: int,
        avg_pos_score: float,
        avg_neg_score: float,
        total_messages: int,
    ) -> None:
        """Save aggregated minute-by-minute metrics."""
        pass

    @abc.abstractmethod
    async def get_realtime_metrics(
        self, channel: str, window_seconds: int = 30
    ) -> Dict[str, Any]:
        """Fetch current moving window sentiment metrics for a channel."""
        pass

    @abc.abstractmethod
    async def get_timeline_metrics(
        self,
        channel: str,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch historical minute metrics for timeline visualizations."""
        pass

    @abc.abstractmethod
    async def get_recent_messages(
        self, channel: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Fetch the most recent messages with sentiment for a channel."""
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        """Gracefully release database connections."""
        pass


class SQLiteDatabaseService(BaseDatabaseService):
    """SQLite implementation for local development."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.SQLITE_DB_PATH

    async def init_db(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.execute("PRAGMA journal_mode = WAL")
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
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_time ON chat_log(timestamp)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_channel_time ON chat_log(channel, timestamp)"
            )

            await db.execute("""
                CREATE TABLE IF NOT EXISTS session_info (
                    user_id TEXT,
                    vod_id TEXT,
                    stream_start_time REAL,
                    monitor_start_time REAL
                )
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS chat_metrics (
                    channel TEXT,
                    minute_timestamp INTEGER,
                    pos_count INTEGER,
                    neu_count INTEGER,
                    neg_count INTEGER,
                    avg_pos_score REAL,
                    avg_neg_score REAL,
                    total_messages INTEGER,
                    PRIMARY KEY (channel, minute_timestamp)
                )
            """)
            await db.commit()

    async def save_messages(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        tuples = [
            (
                r.get("timestamp", time.time()),
                r.get("channel", "").lower(),
                r.get("message", ""),
                r.get("label", "neutral"),
                float(r.get("score", 0.0)),
                float(r.get("latency", 0.0)),
            )
            for r in rows
        ]
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.executemany(
                "INSERT INTO chat_log (timestamp, channel, message, label, score, latency) VALUES (?, ?, ?, ?, ?, ?)",
                tuples,
            )
            await db.commit()

    async def save_session_info(
        self,
        user_id: str,
        vod_id: str,
        stream_start: Optional[float],
        session_time: float,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.execute(
                "INSERT INTO session_info (user_id, vod_id, stream_start_time, monitor_start_time) VALUES (?, ?, ?, ?)",
                (user_id, vod_id, stream_start, session_time),
            )
            await db.commit()

    async def save_minute_metric(
        self,
        channel: str,
        minute_timestamp: int,
        pos_count: int,
        neu_count: int,
        neg_count: int,
        avg_pos_score: float,
        avg_neg_score: float,
        total_messages: int,
    ) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.execute(
                """
                INSERT OR REPLACE INTO chat_metrics 
                (channel, minute_timestamp, pos_count, neu_count, neg_count, avg_pos_score, avg_neg_score, total_messages)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    channel.lower(),
                    minute_timestamp,
                    pos_count,
                    neu_count,
                    neg_count,
                    avg_pos_score,
                    avg_neg_score,
                    total_messages,
                ),
            )
            await db.commit()

    async def get_realtime_metrics(
        self, channel: str, window_seconds: int = 30
    ) -> Dict[str, Any]:
        cutoff = time.time() - window_seconds
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT label, score, latency
                FROM chat_log
                WHERE channel = ? AND timestamp >= ?
                """,
                (channel.lower(), cutoff),
            )
            rows = await cursor.fetchall()

        total = len(rows)
        if total == 0:
            return {
                "channel": channel,
                "window_seconds": window_seconds,
                "total_messages": 0,
                "pos_ratio": 0.0,
                "neg_ratio": 0.0,
                "neu_ratio": 0.0,
                "avg_latency_ms": 0.0,
                "messages_per_second": 0.0,
            }

        pos = sum(1 for r in rows if r["label"].lower() == "positive")
        neg = sum(1 for r in rows if r["label"].lower() == "negative")
        neu = sum(1 for r in rows if r["label"].lower() == "neutral")
        avg_latency = sum(r["latency"] for r in rows) / total

        return {
            "channel": channel,
            "window_seconds": window_seconds,
            "total_messages": total,
            "pos_ratio": round(pos / total, 4),
            "neg_ratio": round(neg / total, 4),
            "neu_ratio": round(neu / total, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "messages_per_second": round(total / max(window_seconds, 1), 2),
        }

    async def get_timeline_metrics(
        self,
        channel: str,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM chat_metrics WHERE channel = ?"
        params: List[Any] = [channel.lower()]

        if start_ts is not None:
            query += " AND minute_timestamp >= ?"
            params.append(int(start_ts))
        if end_ts is not None:
            query += " AND minute_timestamp <= ?"
            params.append(int(end_ts))

        query += " ORDER BY minute_timestamp ASC"

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        return [dict(r) for r in rows]

    async def get_recent_messages(
        self, channel: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT timestamp, channel, message, label, score, latency
                FROM chat_log
                WHERE channel = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (channel.lower(), limit),
            )
            rows = await cursor.fetchall()

        return [dict(r) for r in rows]

    async def close(self) -> None:
        pass


class DynamoDBDatabaseService(BaseDatabaseService):
    """AWS DynamoDB implementation for production deployment."""

    def __init__(
        self,
        region_name: Optional[str] = None,
        messages_table: Optional[str] = None,
        metrics_table: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        self.region_name = region_name or config.AWS_REGION
        self.messages_table_name = messages_table or config.DYNAMODB_MESSAGES_TABLE
        self.metrics_table_name = metrics_table or config.DYNAMODB_METRICS_TABLE
        self.endpoint_url = endpoint_url or config.DYNAMODB_ENDPOINT_URL

        session_kwargs = {"region_name": self.region_name}
        self._dynamodb = boto3.resource(
            "dynamodb",
            endpoint_url=self.endpoint_url,
            **session_kwargs,
        )
        self._client = boto3.client(
            "dynamodb",
            endpoint_url=self.endpoint_url,
            **session_kwargs,
        )
        self._messages_table = self._dynamodb.Table(self.messages_table_name)
        self._metrics_table = self._dynamodb.Table(self.metrics_table_name)

    async def init_db(self) -> None:
        """Create DynamoDB tables if they don't already exist."""
        await asyncio.to_thread(self._ensure_tables_exist)

    def _ensure_tables_exist(self) -> None:
        # Check / create Messages table: PK = channel (S), SK = timestamp_id (S)
        try:
            self._messages_table.load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                print(f"Creating DynamoDB table: {self.messages_table_name}")
                table = self._dynamodb.create_table(
                    TableName=self.messages_table_name,
                    KeySchema=[
                        {"AttributeName": "channel", "KeyType": "HASH"},
                        {"AttributeName": "timestamp_id", "KeyType": "RANGE"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "channel", "AttributeType": "S"},
                        {"AttributeName": "timestamp_id", "AttributeType": "S"},
                    ],
                    BillingMode="PAY_PER_REQUEST",
                )
                table.wait_until_exists()
            else:
                raise e

        # Check / create Metrics table: PK = channel (S), SK = minute_timestamp (N)
        try:
            self._metrics_table.load()
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                print(f"Creating DynamoDB table: {self.metrics_table_name}")
                table = self._dynamodb.create_table(
                    TableName=self.metrics_table_name,
                    KeySchema=[
                        {"AttributeName": "channel", "KeyType": "HASH"},
                        {"AttributeName": "minute_timestamp", "KeyType": "RANGE"},
                    ],
                    AttributeDefinitions=[
                        {"AttributeName": "channel", "AttributeType": "S"},
                        {"AttributeName": "minute_timestamp", "AttributeType": "N"},
                    ],
                    BillingMode="PAY_PER_REQUEST",
                )
                table.wait_until_exists()
            else:
                raise e

    async def save_messages(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return

        def _batch_write():
            with self._messages_table.batch_writer() as batch:
                for r in rows:
                    ts = r.get("timestamp", time.time())
                    unique_id = uuid.uuid4().hex[:8]
                    sk = f"{ts:.4f}#{unique_id}"
                    item = {
                        "channel": r.get("channel", "").lower(),
                        "timestamp_id": sk,
                        "timestamp": str(ts),
                        "message": r.get("message", ""),
                        "label": r.get("label", "neutral"),
                        "score": str(round(float(r.get("score", 0.0)), 4)),
                        "latency": str(round(float(r.get("latency", 0.0)), 2)),
                    }
                    batch.put_item(Item=item)

        await asyncio.to_thread(_batch_write)

    async def save_session_info(
        self,
        user_id: str,
        vod_id: str,
        stream_start: Optional[float],
        session_time: float,
    ) -> None:
        def _write():
            item = {
                "channel": f"SESSION#{user_id}",
                "timestamp_id": f"{session_time:.4f}#session",
                "user_id": user_id,
                "vod_id": str(vod_id) if vod_id else "",
                "stream_start": str(stream_start) if stream_start else "",
                "session_time": str(session_time),
            }
            self._messages_table.put_item(Item=item)

        await asyncio.to_thread(_write)

    async def save_minute_metric(
        self,
        channel: str,
        minute_timestamp: int,
        pos_count: int,
        neu_count: int,
        neg_count: int,
        avg_pos_score: float,
        avg_neg_score: float,
        total_messages: int,
    ) -> None:
        def _write():
            item = {
                "channel": channel.lower(),
                "minute_timestamp": minute_timestamp,
                "pos_count": pos_count,
                "neu_count": neu_count,
                "neg_count": neg_count,
                "avg_pos_score": str(round(avg_pos_score, 4)),
                "avg_neg_score": str(round(avg_neg_score, 4)),
                "total_messages": total_messages,
            }
            self._metrics_table.put_item(Item=item)

        await asyncio.to_thread(_write)

    async def get_realtime_metrics(
        self, channel: str, window_seconds: int = 30
    ) -> Dict[str, Any]:
        cutoff_ts = time.time() - window_seconds
        sk_prefix = f"{cutoff_ts:.4f}"

        def _query():
            from boto3.dynamodb.conditions import Key

            response = self._messages_table.query(
                KeyConditionExpression=Key("channel").eq(channel.lower())
                & Key("timestamp_id").gte(sk_prefix)
            )
            return response.get("Items", [])

        items = await asyncio.to_thread(_query)
        total = len(items)
        if total == 0:
            return {
                "channel": channel,
                "window_seconds": window_seconds,
                "total_messages": 0,
                "pos_ratio": 0.0,
                "neg_ratio": 0.0,
                "neu_ratio": 0.0,
                "avg_latency_ms": 0.0,
                "messages_per_second": 0.0,
            }

        pos = sum(1 for item in items if item.get("label", "").lower() == "positive")
        neg = sum(1 for item in items if item.get("label", "").lower() == "negative")
        neu = sum(1 for item in items if item.get("label", "").lower() == "neutral")
        avg_latency = sum(float(item.get("latency", 0.0)) for item in items) / total

        return {
            "channel": channel,
            "window_seconds": window_seconds,
            "total_messages": total,
            "pos_ratio": round(pos / total, 4),
            "neg_ratio": round(neg / total, 4),
            "neu_ratio": round(neu / total, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "messages_per_second": round(total / max(window_seconds, 1), 2),
        }

    async def get_timeline_metrics(
        self,
        channel: str,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        def _query():
            from boto3.dynamodb.conditions import Key

            kce = Key("channel").eq(channel.lower())
            if start_ts is not None and end_ts is not None:
                kce = kce & Key("minute_timestamp").between(int(start_ts), int(end_ts))
            elif start_ts is not None:
                kce = kce & Key("minute_timestamp").gte(int(start_ts))
            elif end_ts is not None:
                kce = kce & Key("minute_timestamp").lte(int(end_ts))

            response = self._metrics_table.query(
                KeyConditionExpression=kce,
                ScanIndexForward=True,
            )
            items = response.get("Items", [])
            # Format numbers properly
            result = []
            for it in items:
                result.append(
                    {
                        "channel": it.get("channel"),
                        "minute_timestamp": int(it.get("minute_timestamp", 0)),
                        "pos_count": int(it.get("pos_count", 0)),
                        "neu_count": int(it.get("neu_count", 0)),
                        "neg_count": int(it.get("neg_count", 0)),
                        "avg_pos_score": float(it.get("avg_pos_score", 0.0)),
                        "avg_neg_score": float(it.get("avg_neg_score", 0.0)),
                        "total_messages": int(it.get("total_messages", 0)),
                    }
                )
            return result

        return await asyncio.to_thread(_query)

    async def get_recent_messages(
        self, channel: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        def _query():
            from boto3.dynamodb.conditions import Key

            response = self._messages_table.query(
                KeyConditionExpression=Key("channel").eq(channel.lower()),
                ScanIndexForward=False,
                Limit=limit,
            )
            items = response.get("Items", [])
            result = []
            for it in items:
                result.append(
                    {
                        "timestamp": float(it.get("timestamp", 0.0)),
                        "channel": it.get("channel"),
                        "message": it.get("message"),
                        "label": it.get("label"),
                        "score": float(it.get("score", 0.0)),
                        "latency": float(it.get("latency", 0.0)),
                    }
                )
            return result

        return await asyncio.to_thread(_query)

    async def close(self) -> None:
        pass


_db_instance: Optional[BaseDatabaseService] = None


def get_db_service(db_type: Optional[str] = None) -> BaseDatabaseService:
    """Factory function returning the singleton database service instance."""
    global _db_instance
    if _db_instance is not None:
        return _db_instance

    selected_type = (db_type or config.DB_TYPE).lower()

    if selected_type == "dynamodb":
        print(
            f"[*] Initializing DynamoDB Database Service (Region: {config.AWS_REGION})..."
        )
        _db_instance = DynamoDBDatabaseService()
    else:
        print(
            f"[*] Initializing SQLite Database Service (Path: {config.SQLITE_DB_PATH})..."
        )
        _db_instance = SQLiteDatabaseService()

    return _db_instance
