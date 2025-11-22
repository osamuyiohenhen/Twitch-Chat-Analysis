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

---

## 🛠️ Tech Stack

* **Language:** Python 3
* **ML Framework:** PyTorch, Hugging Face Transformers
* **Data & Async:** `asyncio`, `aiofiles`, `aiocsv`
* **API:** `twitchAPI` (OAuth2)

---

## 💻 Getting Started

### 1. Clone the Repo
```bash
git clone [https://github.com/osamuyiohenhen/Twitch-Chat-Analysis.git](https://github.com/osamuyiohenhen/Twitch-Chat-Analysis.git)
cd Twitch-Chat-Analysis
