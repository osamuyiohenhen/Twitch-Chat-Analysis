import asyncio
import json
import websockets

CHANNEL = "silky"  # Change to the channel you connected in Swagger
URL = f"ws://localhost:8000/ws/channels/{CHANNEL}"


async def listen():
    print(f"Connecting to WebSocket: {URL} ...")
    async with websockets.connect(URL) as ws:
        print("Connected! Listening for live incoming data (Ctrl+C to stop)...\n")
        while True:
            raw_msg = await ws.recv()
            data = json.loads(raw_msg)

            # Print parsed WebSocket event
            event_type = data.get("type")
            if event_type == "chat":
                chat = data.get("data", {})
                label = chat.get("label", "unknown").upper()
                score = chat.get("score", 0.0)
                user = chat.get("user", "")
                text = chat.get("message", "")
                print(f"[{label} {score:.2f}] {user}: {text}")

            elif event_type == "metrics":
                metrics = data.get("data", {})
                pos = metrics.get("pos_ratio", 0.0) * 100
                neg = metrics.get("neg_ratio", 0.0) * 100
                total = metrics.get("total_messages", 0)
                print(
                    f"---> [METRICS SNAPSHOT] Last 30s: {pos:.1f}% Pos | {neg:.1f}% Neg | {total} msgs"
                )

            elif event_type == "warning":
                print(f"⚠️ Warning from server: {data.get('message')}")


if __name__ == "__main__":
    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        print("\nDisconnected from WebSocket.")
