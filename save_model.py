from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="./twitch-roberta-base",
    repo_id="muyihenhen/twitch-roberta-base",
    repo_type="model",
)
