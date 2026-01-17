# Real-Time Twitch.tv Chat Analytics Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Accelerated-orange) 
![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

### Demo
![Demo of Twitch Chat Analyzer](demo.gif)
*A preview of the engine connecting to live channels and processing messages in real-time.*

---

## 📖 About The Project
This project extends a standard sentiment analysis tool into a real-time analytics engine capable of handling the chaotic environment of Twitch.tv chat.

It started with a simple goal: **"Can a model understand the emotional flow of a live Twitch stream?"**
However, standard models failed because:
1. They were too slow for live chat (latency).
2. They didn't understand gaming slang (e.g., "Pog", "throw", "cap"). 

This pipeline was engineered specifically to solve those constraints while moving toward accurate, real-time moderation and sentiment tracking.

## ⚙️ System Architecture

Designed for environments with **high velocity** (bursts of 50+ msgs/s) and **noisy syntax**.
It uses a hybrid architecture: training on cloud GPUs and conducting inference locally for fast performance.

### 1. Asynchronous Data Ingestion
**Challenge:**
Twitch chat is extremely bursty &mdash; A hype moment can spike traffic from 5 &rarr; 100 messages/second instantly, causing many scrapers to freeze or drop packets.

**Solution:**
An `asyncio` + `twitchAPI` ingestion pipeline to handle the WebSocket connection. This prevents blocking and ensures stability even when message volume surges.

---

### 2. The Two-Stage Model Development

**Stage 1: Domain Adaptation**
* **Problem:** Standard models think "He is cracked" means "He is broken" (Negative). In gaming, it means "He is skilled" (Positive).
* **Solution:** I trained a RoBERTa model on **1.1 million unlabeled Twitch messages** using **Masked Language Modeling (MLM)**. This "taught" the model the dialect of Twitch before it ever learned about sentiment.
* **Result:** Perplexity dropped from ~21k &rarr; ~5.5.

**Stage 2: Sentiment Fine-Tuning**
* **Problem:** Understanding slang isn't enough; we need to know if the chat is Happy, Angry, or Neutral.
* **Solution:** Fine-tuned the domain-adapted model on a labeled dataset of chat logs.
* **Result:** A classifier that correctly identifies nuances like sarcasm and hype that standard models miss.

---

### 3. Hybrid Cloud/Local Architecture
**Challenge:** Training is too heavy for a laptop, but cloud inference (API) is too slow for real-time chat.
**Solution:**
* **Training (Cloud):** Heavy MLM pre-training and Fine-tuning runs on Google Colab (A100 GPU).
* **Inference (Local):** The optimized model is deployed locally on an RTX 4050 using FP16 mixed precision.
* **Performance:** Achieves **<60ms end-to-end latency** per message.

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **ML Framework:** PyTorch, Hugging Face Transformers
* **Data & Async:** `asyncio`, `aiofiles`, `aiocsv`
* **DevOps & CI:** GitHub Actions, Ruff (Linting/Formatting)
* **API:** `twitchAPI` (OAuth2 Authentication)


## 💻 Getting Started

### 1. Clone the Repo
```bash
git clone [https://github.com/osamuyiohenhen/Twitch-Chat-Analysis.git](https://github.com/osamuyiohenhen/Twitch-Chat-Analysis.git)
cd Twitch-Chat-Analysis
```

### 2. Set up the Environment
Create a virtual environment to keep dependencies clean.
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note: If you have an NVIDIA GPU, install the CUDA version of PyTorch first.*

### 4. Twitch API Keys & Config

You need a Client ID and Secret from the Twitch Developer Console.

1. Create a file named `config.py` in the root folder.
2. Add your credentials:
```bash
# config.py
client_id = 'YOUR_TWITCH_CLIENT_ID'
client_secret = 'YOUR_TWITCH_CLIENT_SECRET'
```

### 5. Run it
The pipeline is designed to **automatically download** the latest model weights from Hugging Face if they aren't found locally.
```bash
python main.py
```
1. Authenticate when prompted.
2. Enter the channel name you want to analyze (e.g., `shroud`).
3. Watch the real-time sentiment classification in the terminal.

## 🛣️ Roadmap
* [x] **Async Scraper:** Built high-throughput chat scraper.

* [x] **Domain Adaptation (WIP):** Implemented MLM training to learn Twitch slang.

* [x] **CI/CD Pipeline:** Automated testing and linting (Ruff) via GitHub Actions.

* [x] **Fine-Tuning:** Train final sentiment classifier on labeled data.

* [ ] **Dashboard:** Build a Streamlit UI to visualize live sentiment trends.

* [ ] **Adapters:** Experiment with lightweight adapters for specific streamer communities.

## Acknowledgements & License

**License:** Apache 2.0.

**Attribution:** This model is a fine-tuned version of [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).