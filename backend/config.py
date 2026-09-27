from dotenv import load_dotenv
import os

load_dotenv()

client_id = os.getenv("TWITCH_CLIENT_ID")
client_secret = os.getenv("TWITCH_CLIENT_SECRET")
user_token = os.getenv("TWITCH_USER_TOKEN")
refresh_token = os.getenv("TWITCH_REFRESH_TOKEN")

bot_list = [
    "fossabot",
    "nightbot",
    "streamelements",
    "potatbotat",
]  # Add more known bots

# Database & AWS DynamoDB Configuration
DB_TYPE = os.getenv("DB_TYPE", "sqlite").lower()  # 'sqlite' or 'dynamodb'
SQLITE_DB_PATH = os.getenv(
    "SQLITE_DB_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "twitch_data.db"),
)

AWS_REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
DYNAMODB_MESSAGES_TABLE = os.getenv("DYNAMODB_MESSAGES_TABLE", "TwitchMessages")
DYNAMODB_METRICS_TABLE = os.getenv("DYNAMODB_METRICS_TABLE", "TwitchMetrics")
DYNAMODB_ENDPOINT_URL = os.getenv(
    "DYNAMODB_ENDPOINT_URL", None
)  # e.g. "http://localhost:8000" for dynamodb-local
