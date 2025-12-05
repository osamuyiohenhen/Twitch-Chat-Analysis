# Prediction Testing Script for Twitch Chat Data
# This was adapted from main.py to focus on masked word prediction only.
# This was used to verify the model's performance in a new domain.
# Imports
from twitchAPI.chat import Chat, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
import asyncio
import time
import os
import random  # For random selection of words to mask
import aiofiles
from aiocsv import AsyncWriter
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline 

# Contains APP_ID & APP_SECRET
import config

# Set up Constants
CLIENT_ID = config.client_id
CLIENT_SECRET = config.client_secret
USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]

# Path to the fine-tuned model directory
MODEL_DIR = "./twitch-roberta-base" 

print("Loading Model...")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

# Load the Masked Language Model
model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR, local_files_only=True)

# Check for GPU availability
device = 0 if torch.cuda.is_available() else -1

# Initialize the prediction pipeline
PREDICTOR = pipeline(
    "fill-mask", 
    model=model, 
    tokenizer=tokenizer,
    dtype=torch.float16,
    device=device,
    top_k=3 
)

print("Model loaded successfully!")

BOT_LIST = {'fossabot', 'nightbot', 'streamelements', "potatbotat"} 

async def on_message(msg: ChatMessage):
    # Filter out bot messages, commands, and links
    if msg.user in BOT_LIST:
        return
    # Skip commands (starting with !)
    if msg.text.startswith('!'):
        return
    # Skip links (http or https)
    if "http" in msg.text.lower():  # cheaper than per-word slicing
        return
    
    state["messageCount"] += 1

    # Split message into words
    words = msg.text.split()
    
    # Skip empty messages
    if len(words) < 2:
        return
    
    # Optional: For very short messages, mask the last word
    # elif len(words) < 4:
    #     random_index = len(words) - 1
    #     real_word = words[random_index]
    #     masked_words = words.copy()
    #     masked_words[random_index] = "<mask>"
    #     masked_text = " ".join(masked_words)
    # else:

    # Randomly mask a word in the message
    random_index = random.randint(0, len(words) - 1)
    real_word = words[random_index]
    if real_word.startswith('@') and len(real_word) > 1:
        return  # Skip masking @user mentions
    masked_words = words.copy()
    masked_words[random_index] = "<mask>"
    masked_text = " ".join(masked_words)

    # Run inference
    start = time.perf_counter()
    predictions = PREDICTOR(masked_text)
    end = time.perf_counter()
    latency_ms = (end - start) * 1000

    # Save the message to a CSV file (not in use currently)
    # await save_message([msg.user.name, msg.text])

    # Print results
    print(f"\nUser ({msg.user.display_name}): {msg.text}")
    print(f"Input to AI: \"{masked_text}\" [{latency_ms:.2f} ms]")    
    print(f"AI Guesses:")

    guessed_word = False
    for i, pred in enumerate(predictions):
        clean_token = pred['token_str'].replace('Ġ', '').strip()
        score = pred['score']
        # Quick visual check: Add a star if it guessed the right word
        if clean_token.lower() == real_word.lower():
            marker = "★"
            if not guessed_word:
                state["correctWordCount"] += 1

            guessed_word = True
        else:
            marker = ""
        print(f"  {i+1}. {clean_token.strip()} ({score:.3f}) {marker}")

async def save_message(message):
    # Append message data to a CSV file
    async with aiofiles.open("twitch_data_1m.csv", mode='a', newline='', encoding='utf-8-sig') as f:
        writer = AsyncWriter(f)
        await writer.writerow(message)

state = {
    "messageCount": 0,
    "correctWordCount": 0
}

# Main function to handle Twitch authentication and chat interaction
async def main():
    print("Authenticating...")
    twitch = await Twitch(CLIENT_ID, CLIENT_SECRET)
    auth = UserAuthenticator(twitch, USER_SCOPE)
    token, refresh_token = await auth.authenticate()
    await twitch.set_user_authentication(token, USER_SCOPE, refresh_token) 
    print("Authentication successful.")

    chat = await Chat(twitch)
    chat.register_event(ChatEvent.MESSAGE, on_message)
    chat.start()
    print("Connection started.")

    current_channel = None
    try:
        while True:
            if current_channel:
                await chat.leave_room(current_channel)
                await asyncio.sleep(0.3)
                print(f"Leaving channel: {current_channel}...")
                current_channel = None

                print("Message Count in last channel:", state["messageCount"])
                print("Correct Word Count in last channel:", state["correctWordCount"])
                print("Accuracy: {:.2f}%".format((state["correctWordCount"] / state["messageCount"] * 100) if state["messageCount"] > 0 else 0.0))

            target_channel = input("\nEnter the Twitch channel you wish to connect to (or type 'q' to exit): ").lower()
            if target_channel == 'q': break
            if not target_channel: continue

            try:
                print(f"\nJoining channel: {target_channel}...")

                state["messageCount"] = 0
                state["correctWordCount"] = 0
                await asyncio.wait_for(chat.join_room(target_channel), timeout=1.5)
                current_channel = target_channel
                await asyncio.to_thread(input, 'Press "ENTER" to switch channels.\n')

            except asyncio.TimeoutError:
                print(f"\n[ERROR] Could not join channel '{target_channel}'.")
    
    finally:
        chat.stop()
        await twitch.close()
        print("\nProgram Status: Off")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        os._exit(0)
    finally:
        os._exit(0)
