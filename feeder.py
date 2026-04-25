import csv
import os
import time
import random

FILE_PATH = "live_data.csv"

# 1. Setup File
if not os.path.exists(FILE_PATH):
    with open(FILE_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestamp", "channel", "message", "label", "score", "latency"]
        )
    print(f"Created {FILE_PATH}")

print("--- Auto Feeder ---")
print("Press Ctrl+C to stop the spam.")

# 2. Infinite Random Loop
while True:
    try:
        # Simulate different types of chat messages
        sentiment_type = random.choice(["Positive", "Negative", "Neutral"])

        if sentiment_type == "Positive":
            score = random.uniform(0.6, 0.99)  # Random score between 0.60 and 0.99
            msg = "POGGERS"
        elif sentiment_type == "Negative":
            score = random.uniform(0.6, 0.99)
            msg = "L STREAM"
        else:
            # Neutral is usually lower confidence or around 0.5
            score = random.uniform(0.4, 0.6)
            msg = "lol"

        # 3. Write Data
        # timestamp, channel, message, label, score, latency
        row = [time.time(), "auto_bot", msg, sentiment_type, score, 15.0]

        with open(FILE_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        print(f"Sent: {sentiment_type} ({score:.2f})")

        # 4. Speed Control
        # Change this to make it faster or slower
        time.sleep(0.01)  # 100 messages per second

    except KeyboardInterrupt:
        print("\nStopping...")
        break
    except Exception as e:
        print(f"Error: {e}")
