"""
Integration and Unit Tests for Backend Infrastructure.
Tests:
- Database Abstraction Service (SQLite operations)
- FastAPI Endpoints (Health, CORS headers, Channels, Metrics)
- Headless Worker configuration & error handling
"""

import asyncio
import os
import sys
import tempfile
import time
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from src.db_service import SQLiteDatabaseService, get_db_service
from api import app


class TestDatabaseService(unittest.IsolatedAsyncioTestCase):
    """Test SQLite implementation of BaseDatabaseService."""

    async def asyncSetUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.tmp_dir.name, "test_twitch.db")
        self.db = SQLiteDatabaseService(db_path=self.db_path)
        await self.db.init_db()

    async def asyncTearDown(self):
        await self.db.close()
        # Allow threadpool workers to finish callbacks before event loop shuts down
        await asyncio.sleep(0.05)
        self.tmp_dir.cleanup()

    async def test_save_and_query_messages(self):
        test_messages = [
            {
                "timestamp": time.time(),
                "channel": "streamer1",
                "message": "poggers that was insane",
                "label": "positive",
                "score": 0.95,
                "latency": 15.2,
            },
            {
                "timestamp": time.time(),
                "channel": "streamer1",
                "message": "unfortunate play",
                "label": "negative",
                "score": 0.88,
                "latency": 14.8,
            },
        ]
        await self.db.save_messages(test_messages)

        recent = await self.db.get_recent_messages("streamer1", limit=10)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0]["channel"], "streamer1")

    async def test_realtime_metrics(self):
        now = time.time()
        test_messages = [
            {
                "timestamp": now,
                "channel": "streamer2",
                "message": "msg1",
                "label": "positive",
                "score": 0.9,
                "latency": 10.0,
            },
            {
                "timestamp": now,
                "channel": "streamer2",
                "message": "msg2",
                "label": "positive",
                "score": 0.9,
                "latency": 12.0,
            },
            {
                "timestamp": now,
                "channel": "streamer2",
                "message": "msg3",
                "label": "negative",
                "score": 0.8,
                "latency": 11.0,
            },
            {
                "timestamp": now,
                "channel": "streamer2",
                "message": "msg4",
                "label": "neutral",
                "score": 0.5,
                "latency": 9.0,
            },
        ]
        await self.db.save_messages(test_messages)

        metrics = await self.db.get_realtime_metrics("streamer2", window_seconds=30)
        self.assertEqual(metrics["total_messages"], 4)
        self.assertEqual(metrics["pos_ratio"], 0.5)
        self.assertEqual(metrics["neg_ratio"], 0.25)
        self.assertEqual(metrics["neu_ratio"], 0.25)

    async def test_minute_metrics_timeline(self):
        minute_ts = int(time.time() // 60 * 60)
        await self.db.save_minute_metric(
            channel="streamer3",
            minute_timestamp=minute_ts,
            pos_count=10,
            neu_count=5,
            neg_count=2,
            avg_pos_score=0.92,
            avg_neg_score=0.81,
            total_messages=17,
        )

        timeline = await self.db.get_timeline_metrics("streamer3")
        self.assertEqual(len(timeline), 1)
        self.assertEqual(timeline[0]["total_messages"], 17)
        self.assertEqual(timeline[0]["pos_count"], 10)


class TestFastAPIEndpoints(unittest.TestCase):
    """Test FastAPI REST endpoints and CORS headers."""

    @classmethod
    def setUpClass(cls):
        asyncio.run(get_db_service().init_db())
        cls.client = TestClient(app)


    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("db_type", data)
        self.assertIn("active_channels", data)

    def test_cors_headers_vite_frontend(self):
        """Test CORS headers for Vite frontend on http://localhost:5173."""
        response = self.client.options(
            "/api/v1/channels/testchannel/sentiment/realtime",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(
            response.headers.get("access-control-allow-origin"), "http://localhost:5173"
        )
        self.assertEqual(
            response.headers.get("access-control-allow-credentials"), "true"
        )

    def test_channel_status_not_connected(self):
        response = self.client.get("/api/v1/channels/inactive_streamer/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["channel"], "inactive_streamer")
        self.assertFalse(data["connected"])

    def test_realtime_metrics_empty(self):
        response = self.client.get(
            "/api/v1/channels/unknown_channel/sentiment/realtime"
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_messages"], 0)
        self.assertEqual(data["pos_ratio"], 0.0)

    def test_disconnect_inactive_channel(self):
        response = self.client.post("/api/v1/channels/not_running_channel/disconnect")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "not_connected")

    def test_websocket_connection_and_warning(self):
        """Test WebSocket client receives initial warning when channel worker is not connected."""
        with self.client.websocket_connect(
            "/ws/channels/unconnected_channel"
        ) as websocket:
            data = websocket.receive_json()
            self.assertEqual(data.get("type"), "warning")
            self.assertIn("not running", data.get("message", ""))


class TestHeadlessWorkerAuth(unittest.IsolatedAsyncioTestCase):
    """Test headless Twitch authentication requirements."""

    async def test_worker_fails_explicitly_without_tokens(self):
        """Verify that worker raises a clear RuntimeError when tokens are missing, without opening GUI."""
        from primary import TwitchSentimentWorker
        import config

        # Temporarily clear user tokens
        original_token = config.user_token
        original_refresh = config.refresh_token
        try:
            config.user_token = ""
            config.refresh_token = ""

            worker = TwitchSentimentWorker("test_channel", classifier=None)
            with self.assertRaises(RuntimeError) as ctx:
                await worker.start()

            self.assertIn("Twitch user tokens missing", str(ctx.exception))
            self.assertIn("auth_twitch.py", str(ctx.exception))
        finally:
            config.user_token = original_token
            config.refresh_token = original_refresh


if __name__ == "__main__":
    unittest.main()
