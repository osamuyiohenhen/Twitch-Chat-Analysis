# Real Time Twitch Chat Sentiment Engine (WIP)

This is a Python program in development for Twitch chat. In its current state, the program uses an account to connect to a twitch channel, retrieves chat messages, and attempts to determine if they are positive, neutral, or negative.

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
* `config.py` (for storing/accessing configuration like API keys and target channel)

## Getting it Running:

1.  **Clone it:**
    ```bash
    git clone [https://github.com/osamuyiohenhen/Twitch-Chat-Analysis.git](https://github.com/osamuyiohenhen/Twitch-Chat-Analysis.git)
    cd Twitch-Chat-Analysis
    ```

2.  **Make a virtual environment (good idea):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Or venv\Scripts\activate on Windows
    ```

3.  **Install the stuff:**
    Make sure you have a `requirements.txt` file (you can make one with `pip freeze > requirements.txt` after installing things).
    Then run:
    ```bash
    pip install -r requirements.txt
    ```
    *You'll definitely need `twitchAPI`, `transformers`, and `torch` (or `tensorflow` if you prefer that).*

4.  **Twitch API Keys & Config:**
    * You need to register an app on the [Twitch Developer Console](https://dev.twitch.tv/console) to get a **Client ID** and **Client Secret**.
    * Open the `config.py` file in the project.
    * **IMPORTANT: Be careful not to commit your actual secrets to GitHub if `config.py` is tracked. Ideally, `config.py` would load from a `.env` file or you'd use a `config_example.py` and have users copy it to `config.py` which is then gitignored.** For this setup, assuming you'll directly edit `config.py`:
    * Modify `config.py` to include your credentials and target channel:
        ```python
        # Inside config.py
        client_id = 'YOUR_CLIENT_ID_HERE'
        client_secret = 'YOUR_CLIENT_SECRET_HERE'
        channel = 'name_of_twitch_channel_to_join'
        # Add any other necessary configurations
        ```

## How to Use:

1.  Make sure your `config.py` file is set up correctly with your API keys and target channel.
2.  Run the main program script:
    ```bash
    python main.py
    ```
3.  The first time you run it, the script will likely guide you through an authentication process in your web browser to grant the necessary permissions.
4.  Once authenticated, the program should connect to the specified Twitch channel, and you will see chat messages and sentiment scores pop up in your terminal.
5.  Press "ENTER" in the terminal where the script is running to stop the process.

## Potential Future Additions:

* Storing sentiment data for analysis.
* Implementing user-interactive bot commands.
* Exploring additional NLP features.
* Improving configurability of settings.