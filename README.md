# Real-Time Twitch.tv Chat Analytics Engine

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-GPU%20Accelerated-orange) ![HuggingFace](https://img.shields.io/badge/🤗%20Hugging%20Face-Transformers-yellow)

### Demo
![Demo of Twitch Chat Analyzer](demo.gif)
*A preview of the engine connecting to live channels and processing messages in real-time.*

---

## 📖 About The Project
This is a real-time NLP engine designed to read and analyze Twitch chat. Unlike standard sentiment analysis tools that are trained on Twitter or movie reviews, this project focuses on understanding **Twitch-specific context**.

Standard models often misinterpret gaming slang (e.g., classifying "He is cracked" as *Negative* because of the word "cracked"). This engine solves that by using **Domain Adaptation**—teaching the model the vocabulary of Twitch before asking it to judge sentiment.

## ⚙️ How It Works (System Design)

I built this pipeline to solve two main problems: **Latency** and **Slang Comprehension**.

### 1. Asynchronous Data Ingestion
Twitch chat is "bursty." A hype moment can generate 50+ messages per second.
* **The Solution:** The scraper uses `asyncio` and `twitchAPI` to handle the WebSocket connection. This decouples the data fetching from the processing, ensuring the bot never "freezes" even when chat moves too fast to read.

### 2. Custom Language Modeling (MLM)
I moved away from generic models like GPT-4 or stock RoBERTa for a custom approach:
* **The Architecture:** I use a **RoBERTa** (Encoder) model running locally.
* **The Training:** I implemented a **Masked Language Modeling (MLM)** stage. By training the model on ~50k unlabeled Twitch messages, it learns to predict slang words (e.g., "That play was `<mask>`" -> "Poggers").
* **The Result:** The model learns that "cracked," "goated," and "insane" are positive contextually, significantly reducing confusion compared to off-the-shelf models.

### 3. Local Hardware Acceleration
* **Why Local?** Cloud APIs are slow and expensive. This runs entirely on-premise.
* **Performance:** Optimized for NVIDIA GPUs using PyTorch/CUDA. It currently achieves **<50ms inference latency** per message on consumer hardware (RTX 4050).

### Constraint: Data Scarcity
Hand-labeling 100,000 messages for sentiment is impossible for a solo engineer.
* **Solution:** **Domain Adaptation**. By pre-training on *unlabeled* data first, I reduced the model's **Perplexity** from ~134 to ~9. The model learned the "language" of Twitch unsupervised, meaning I only need to hand-label a tiny fraction (2k messages) for the final fine-tuning stage.

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
Note: If you have an NVIDIA GPU, install the CUDA version of PyTorch first to enable hardware acceleration.
```bash
pip install -r requirements.txt
```

### 4. Twitch API Keys & Config
You need to register an app on the Twitch Developer Console to get a Client ID and Client Secret.
1. Create a file named `config.py` in the root folder.
2. Add your credentials as shown below:
```bash
# config.py
client_id = 'YOUR_TWITCH_CLIENT_ID'
client_secret = 'YOUR_TWITCH_CLIENT_SECRET'
# Add any other necessary configurations
```
**⚠️ IMPORTANT:**  Ensure `config.py` is added to your `.gitignore` file so you do not accidentally commit your credentials to GitHub.

### How to Use
1. Make sure your config.py is set up.
2. Run the main program:
```bash
python main.py
```
3. **First Run:** The script will open your browser to authenticate with Twitch.

4. **Connect:** Enter the channel name you want to analyze (e.g., `jasontheween`).
5. **View:** Watch real-time sentiment scores appear in your terminal.
6. **Stop:** Press `Ctrl+C` to exit.

## 🛣️ Roadmap
* [x] **Async Scraper:** Built high-throughput chat ingestion pipeline.

* [x] **Domain Adaptation (WIP):** Implemented MLM training to learn Twitch slang.

* [ ] **Fine-Tuning:** Train the final Classification Head on labeled sentiment data.

* [ ] **Dashboard:** Build a Streamlit frontend to visualize sentiment trends live.

* [ ] **Adapters:** Experiment with LoRA adapters for channel-specific slang and contexts.

## License
This project is open-source and available under the MIT License.