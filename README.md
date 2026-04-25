# Real-Time Twitch.tv Chat Analytics Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) 
![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Accelerated-orange) 
![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)
![License](https://img.shields.io/badge/License-Apache%202.0-green)

### Demo
![Demo of Twitch Chat Analyzer](proj_demo.gif)
*A preview of the engine connecting to a live channel and processing messages in real-time.*

---

## About The Project
Real-time sentiment analysis engine for Twitch chat. Standard models fail on live chat due to latency constraints and gaming slang. This project solves both of those issues.

Why it matters:
- **Too slow:** Standard models add 300+ ms per message. Users need <60 ms, especially for rooms that operate at 50+ msgs/sec.
- **Wrong dialect:** "Pog", "throw", "cap" aren't in standard datasets.
## System Design

The pipeline uses three key components:

### 1. Asynchronous Data Ingestion
Twitch chat can spike from 5 to 100 messages/second instantly. Most scrapers freeze or drop packets.

Solution: An `asyncio` + `twitchAPI` pipeline handles WebSocket connections without blocking. Stable even during traffic surges.

---

### 2. Two-Stage Model Development

**Stage 1: Domain Adaptation (MLM)**
Standard models think "He is cracked" means broken. In gaming, it means skilled. Trained RoBERTa on 1.1M unlabeled Twitch messages to "teach" it gaming dialect. Perplexity improved from ~21k to ~5.5.

**Stage 2: Sentiment Fine-Tuning**
Fine-tuned the domain-adapted model on labeled chat logs. Result: Correctly identifies nuances like sarcasm and hype.

---

### 3. Hybrid Cloud/Local Architecture
Training is compute-heavy, but cloud inference APIs are too slow for real-time chat.

- **Training:** MLM pre-training and sentiment fine-tuning on Google Colab (A100 GPU)
- **Inference:** Local deployment on RTX 4050 using FP16 mixed precision
- **Performance:** <60ms end-to-end latency per message

## Tech Stack

- **Language:** Python 3.10+
- **ML:** PyTorch, Hugging Face Transformers
- **Async:** `asyncio`, `aiofiles`, `aiocsv`
- **Dashboard:** Streamlit, Plotly
- **API:** `twitchAPI` (OAuth2)
- **DevOps:** GitHub Actions, Ruff


## Getting Started

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

### 3. Install PyTorch (GPU Optional)

If you have an NVIDIA GPU, considering installing CUDA-enabled PyTorch first to have improved inference performance:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

*If you installed CUDA PyTorch in step 3, pip will skip torch (already installed) and install everything else.*

### 4. Environment Variables

Create a `.env` file in the root folder:
```
TWITCH_CLIENT_ID=your_twitch_client_id
TWITCH_CLIENT_SECRET=your_twitch_client_secret
```

**Do not commit `.env`** — Add to `.gitignore` to keep secrets safe.

The app uses `python-dotenv` to automatically load these variables.

### 5. Run the Dashboard

```bash
streamlit run app.py
```

1. Open the sidebar and enter a Twitch channel name (e.g., `xQc`)
2. Click "Connect"
3. Watch real-time sentiment analysis in the dashboard

(Or use `feeder.py` to test with mock data if you don't want to go live)

## Roadmap

- [x] **Async Scraper:** High-throughput chat scraper
- [x] **Domain Adaptation:** MLM training on Twitch slang
- [x] **CI/CD Pipeline:** Automated testing via GitHub Actions
- [x] **Fine-Tuning:** Sentiment classifier on labeled data
- [x] **Dashboard:** Streamlit UI with live sentiment tracking
- [ ] **Cloud Deployment:** Docker containerization and cloud hosting (AWS/GCP)
- [ ] **Adapters:** Lightweight adapters for specific streamer communities

## Acknowledgements & License

**License:** Apache 2.0.

**Attribution:** This model is a modified and fine-tuned version of [cardiffnlp/twitter-roberta-base-sentiment-latest](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest).