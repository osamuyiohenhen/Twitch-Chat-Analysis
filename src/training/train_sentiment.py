import pandas as pd
import os
import random
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset
import shutil

# --- CONFIG ---
# We use the Twitter Sentiment model as the base
MODEL_NAME = "./final_sentiment_model"
INPUT_FILE = "labeled_data_v2.csv"
SAVE_DIR = "./twitch-sentiment-v2"


class TwitchDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)


def main():
    # 0. Cleanup old results to avoid conflicts
    if os.path.exists("./results"):
        shutil.rmtree("./results")

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    df = df.dropna(subset=["message", "label"])

    # Remove strict duplicates to fight overfitting
    original_count = len(df)
    df = df.drop_duplicates(subset=["message"])
    print(f"Removed {original_count - len(df)} duplicate messages.")

    # Shuffle
    data = list(zip(df["message"].tolist(), df["label"].astype(int).tolist()))
    random.shuffle(data)
    texts, labels = zip(*data)

    # Split 85/15 (Since data is small, keep more for training)
    split_idx = int(0.85 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    print(f"✅ Samples: {len(train_texts)} Train | {len(val_texts)} Test")

    # 2. Tokenizer (AutoTokenizer handles the Twitter specifics)
    print(f"Loading Tokenizer ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenize
    train_encodings = tokenizer(train_texts, truncation=True, max_length=64)
    val_encodings = tokenizer(val_texts, truncation=True, max_length=64)

    train_dataset = TwitchDataset(train_encodings, train_labels)
    val_dataset = TwitchDataset(val_encodings, val_labels)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 3. Model
    print("Loading Pre-Trained Sentiment Model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=10,  # Try 10 epochs
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=20,  # Short warmup since data is small
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="epoch",  # Check score every epoch
        save_strategy="epoch",  # Save every epoch
        load_best_model_at_end=True,  # Revert to the best epoch at the end
        metric_for_best_model="eval_loss",
        no_cuda=False,
    )

    # 5. Trainer with Early Stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3)
        ],  # Stop if score gets worse 3 times in a row
    )

    print("\n🚀 STARTING TRANSFER LEARNING...")
    trainer.train()

    print(f"\n💾 Saving to {SAVE_DIR}...")
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print("✅ DONE! You now have a Twitch-Smart Sentiment Bot.")


if __name__ == "__main__":
    main()
