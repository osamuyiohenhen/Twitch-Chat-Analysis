# get_token.py
import asyncio
from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.type import AuthScope
import config

TARGET_SCOPES = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]

async def main():
    twitch = await Twitch(config.client_id, config.client_secret, authenticate_app=False)
    auth = UserAuthenticator(twitch, TARGET_SCOPES)
    token, refresh_token = await auth.authenticate()
    print(f"Token: {token}")
    print(f"Refresh: {refresh_token}")

asyncio.run(main())