# Save Model to Hugging Face Hub
from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="models/twitch-sentiment-v2",
    repo_id="muyihenhen/twitch-roberta-sentiment-v1",
    repo_type="model",
)

print("Saved the model.")
