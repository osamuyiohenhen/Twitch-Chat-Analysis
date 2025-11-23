# Imports
from twitchAPI.chat import Chat, ChatMessage
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.twitch import Twitch
import asyncio
import time
import os
import random  # <--- Added for random selection
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

# Make sure this points to your latest V2 folder!
MODEL_DIR = "./twitch-roberta-test" 

print("Loading Model...")

# 1. Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)

# 2. Load the MASKED LM Model
model = AutoModelForMaskedLM.from_pretrained(MODEL_DIR, local_files_only=True)

# 3. Hardware Check
device = 0 if torch.cuda.is_available() else -1

# 4. The Pipeline 
PREDICTOR = pipeline(
    "fill-mask", 
    model=model, 
    tokenizer=tokenizer, 
    device=device,
    top_k=3 
)

print("Model loaded successfully!")

BOT_LIST = {'fossabot', 'nightbot', 'streamelements', "potatbotat"} 

async def on_message(msg: ChatMessage):
    # Filter bots if you want
    # if msg.user.display_name.lower() in BOT_LIST: return

    # --- 1. PREPARE THE TEST ---
    words = msg.text.split()
    
    # If message is empty, skip
    if len(words) < 1:
        return

    # --- NEW LOGIC: RANDOM MASKING ---
    # Pick a random index from the message
    random_index = random.randint(0, len(words) - 1)
    
    # Save the actual word so we can compare (optional, for your own debugging)
    real_word = words[random_index]
    
    # Create a copy of the list so we don't mess up the original variable if needed later
    masked_words = words.copy()
    
    # Replace the random word with the mask token
    masked_words[random_index] = "<mask>"
    
    # Rejoin the list into a string
    masked_text = " ".join(masked_words)

    # --- 2. RUN INFERENCE ---
    start = time.perf_counter()
    
    predictions = PREDICTOR(masked_text)
    
    end = time.perf_counter()
    latency_ms = (end - start) * 1000
    await save_message([msg.user.name, msg.text])
    # --- 3. PRINT RESULTS ---
    print(f"\nUser ({msg.user.display_name}): {msg.text}")
    print(f"Input to AI: \"{masked_text}\" [{latency_ms:.2f} ms]")
    # print(f"Hidden Word: {real_word}") # Uncomment if you want to see explicitly what was hidden
    
    print(f"AI Guesses:")
    # Handle case where top_k=1 returns a dict, but top_k>1 returns a list
    if isinstance(predictions, dict):
        predictions = [predictions]

    for i, pred in enumerate(predictions):
        clean_token = pred['token_str'].replace('Ġ', '') 
        score = pred['score']
        
        # Quick visual check: Add a star if it guessed the right word
        marker = "★" if clean_token.strip().lower() == real_word.lower() else ""
        print(f"  {i+1}. {clean_token} ({score:.3f}) {marker}")

async def save_message(message):
    async with aiofiles.open("twitch_data_300k.csv", mode='a', newline='', encoding='utf-8') as f:
        writer = AsyncWriter(f)
        await writer.writerow(message)

# Main function
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

            target_channel = input("\nEnter the Twitch channel (or 'q'): ").lower()
            if target_channel == 'q': break
            if not target_channel: continue

            try:
                print(f"\nJoining channel: {target_channel}...")
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