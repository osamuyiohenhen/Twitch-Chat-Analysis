# Twitch Chat Sentiment Bot (WIP)

This is a Python bot in development for Twitch chat. In its current state, the bot connects to a twitch channel, retrieves chat messages, and attempts to determine if they are positive, neutral, or negative.

This project was started due to an interest in Twitch communities and a desire to experiment with natural language processing on live chat data.

## What it does (so far):

* Uses `twitchAPI` to connect to a Twitch channel's chat.
* Reads new messages as they come in.
* Uses Hugging Face `transformers` to guess the sentiment of each message.
* Prints the chat and its sentiment to your terminal.
* Runs with `asyncio` to keep things speedy.
* Uses OAuth for Twitch API access via user authentication.

## Tech Stack:

* Python 3
* `twitchAPI`
* `asyncio`
* `transformers` (Hugging Face)
* `python-dotenv` (likely used by `config.py` to load environment variables)
* `config.py` (for storing/accessing configuration like API keys and target channel)
