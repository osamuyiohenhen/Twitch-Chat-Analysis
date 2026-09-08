"""
Standalone CLI script to interactively generate Twitch OAuth tokens.
Run this script ONCE locally to generate a paired user access token and refresh token,
which will be automatically saved to your .env file or printed to the console.
"""

import asyncio
import os
import sys

# Add parent directory to path to import config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.type import AuthScope
from dotenv import load_dotenv, set_key

TARGET_SCOPES = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT, AuthScope.CHANNEL_BOT]


def update_env_file(token: str, refresh_token: str):
    """Save tokens to .env file in root directory."""
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if not os.path.exists(env_path):
        # Create empty .env if it doesn't exist
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("# Twitch Configuration\n")

    set_key(env_path, "TWITCH_USER_TOKEN", token)
    set_key(env_path, "TWITCH_REFRESH_TOKEN", refresh_token)
    print(f"\n[+] Successfully updated {env_path} with new tokens!")


async def main():
    print("=" * 60)
    print(" Twitch Interactive OAuth Token Generator (CLI)")
    print("=" * 60)

    # Load environment variables
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv(env_path)

    client_id = os.getenv("TWITCH_CLIENT_ID")
    client_secret = os.getenv("TWITCH_CLIENT_SECRET")

    if not client_id or not client_secret:
        print(
            "\n[!] Error: TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET must be set in .env or passed as environment variables."
        )
        print("Please create or update your .env file with:")
        print("TWITCH_CLIENT_ID=your_client_id")
        print("TWITCH_CLIENT_SECRET=your_client_secret")
        sys.exit(1)

    print(
        f"\n[*] Initializing Twitch API client with Client ID: {client_id[:4]}...{client_id[-4:]}"
    )
    twitch = await Twitch(client_id, client_secret, authenticate_app=False)

    try:
        print("[*] Launching browser for Twitch user authorization...")
        print(f"[*] Requesting scopes: {[s.value for s in TARGET_SCOPES]}")
        auth = UserAuthenticator(twitch, TARGET_SCOPES)
        token, refresh_token = await auth.authenticate()

        print("\n" + "=" * 60)
        print(" Authentication Successful!")
        print("=" * 60)
        print(f"TWITCH_USER_TOKEN={token}")
        print(f"TWITCH_REFRESH_TOKEN={refresh_token}")
        print("=" * 60)

        # Update .env
        update_env_file(token, refresh_token)

        print("\nYou can now run the headless backend worker or FastAPI server!")
    except Exception as e:
        print(f"\n[!] Authentication failed: {e}")
        sys.exit(1)
    finally:
        await twitch.close()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
