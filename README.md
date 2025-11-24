# Real-Time Twitch.tv Chat Analytics Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Accelerated-orange) ![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)

### Demo
![Demo of Twitch Chat Analyzer](demo.gif)
*A preview of the engine connecting to live channels and processing messages in real-time.*

---

## 📖 About The Project
This project extends a standard sentiment analysis tool into a real-time analytics engine capable of handling the chaotic environment of Twitch.tv chat.

It started with a simple goal: **"Can a model understand the emotional flow of a live Twitch stream?"**
However, standard models failed because:
1. They were too slow for live chat.
2. They didn't understand gaming slang.
This pipeline was engineered specifically to solve those constraints while moving toward accurate, real-time moderation and sentiment tracking.
---

## ⚙️ System Architecture

Designed for environments with **high velocity** (bursts of 50+ msgs/s) and **noisy syntax**.
It uses a hybrid architecture: training on cloud GPUs and conducting inference locally for fast performance.

### 1. Asynchronous Data Ingestion
**Challenge:**
Twitch chat is extremely bursty &mdash; A hype moment can spike traffic from 5 &rarr; 100 messages/second instantly, causing many scrapers to freeze or drop packets.

**Solution:**
An `asyncio` + `twitchAPI` ingestion pipeline to handle the WebSocket connection. This prevents blocking and ensures stability even when message volume surges.

---

### 2. Training on Cloud, Inference on Local GPU
**Challenge:** Heavy MLM training stresses local laptop GPUs, while cloud inference adds unacceptable latency (~500ms+ round-trip) and session timeouts.

**Solution:** I implemented a hybrid architecture:
* **Cloud (Colab T4 GPU):** Handles the compute-heavy MLM pretraining using its higher VRAM and more stable throughput.
* **Local GPU (RTX 4050):** Hosts the real-time inference engine, avoiding network delays and enabling sub-60ms message processing.

This approach gives efficient training without sacrificing the real-time performance required for Twitch chat.

---

### 3. Custom Language Modeling (MLM)

**Challenge:** Standard pre-trained models misinterpret Twitch slang.
Example: *"He is cracked"* &rarr; Negative (broken) instead of Positive (skilled).

**Solution:** Implemented **Self-Supervised Domain Adaptation** with **Masked Language Modeling (MLM)** on ~300k unlabeled Twitch messages.
This teaches the model the underlying “dialect” before any supervised fine-tuning.

**Result:** Achieved a **~75% reduction in MLM training loss**, with perplexity dropping from ~21k → ~8-9 (so far), a strong indicator that the model now understands Twitch slang far better than the baseline.

---

### 4. Local Hardware Acceleration

**Challenge:**
Cloud inference introduces ~500ms round-trip latency and runtime limits &mdash; far too slow for real-time chat.

**Solution:**
Moved inference to local GPU (RTX 4050), using CUDA-accelerated PyTorch and FP16 mixed precision. Achieves **<60ms end-to-end latency** per message, enabling real-time moderation and sentiment reading.

---

## Challenges & Trade-offs
### Constraint: Data Scarcity
Labeling 100k+ messages manually is unrealistic for a solo engineer.

**Solution:**
Use **Domain Adaptation** first. By doing MLM pre-training on *unlabeled* data, the model's confusion reduced significantly on raw messages. This allows the model to require significantly fewer labeled samples for final classification.

---

## 🛠️ Tech Stack

* **Language:** Python 3
* **ML Framework:** PyTorch, Hugging Face Transformers
* **Data & Async:** `asyncio`, `aiofiles`, `aiocsv`
* **API:** `twitchAPI` (OAuth2 Authentication)
---

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
Note: If you have an NVIDIA GPU, install the CUDA version of PyTorch first.
```bash
pip install -r requirements.txt
```

### 3.5 Model Setup
This project loads the model locally and does **not** auto-download it.

Download the model from Hugging Face:
[huggingface.co/muyihenhen/twitch-roberta-base](https://huggingface.co/muyihenhen/twitch-roberta-base)

Then place the downloaded folder in the project root as:
```bash
./twitch-roberta-base/
```
That’s it &mdash; the pipeline will load it automatically.

### 4. Twitch API Keys & Config
You need to register an app on the Twitch Developer Console to get a Client ID and Client Secret.
1. Go to the file named `config_example.py`.
2. Add your credentials as shown below:
```bash
# config_example.py
# rename this file to: config.py
# Enter your Twitch Developer credentials below
# Get them from: https://dev.twitch.tv/console

client_id = 'YOUR_TWITCH_CLIENT_ID_HERE'
client_secret = 'YOUR_TWITCH_CLIENT_SECRET_HERE'
# Optional: I chose to add a bot_list for other chatbots to ignore
```
**⚠️ IMPORTANT:**  Ensure `config.py` is added to your `.gitignore`.

### How to Use
1. Ensure your Twitch credentials are set in config.py.
2. Run the main program:
```bash
python main.py
```
3. Authenticate when prompted.
4. Enter the channel name you want to analyze (e.g., `jasontheween`).
5. Watch real-time sentiment output in the terminal.
6. Press `Ctrl+C` to stop.

## 🛣️ Roadmap
* [x] **Async Scraper:** Built high-throughput chat ingestion.

* [x] **Domain Adaptation (WIP):** Implemented MLM training to learn Twitch slang.

* [ ] **Fine-Tuning:** Train final sentiment classifier on labeled data.

* [ ] **Dashboard:** Streamlit visualization to display live sentiment trends.

* [ ] **Adapters:** Experiment with LoRA adapters for channel-specific slang and contexts.

## License
This project is open-source and available under the MIT License.
