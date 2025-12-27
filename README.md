# 📊 FinFuse - Real-Time Market Intelligence & LLM-Powered Financial Assistant
---
## 📌 Overview

**FinFuse** is a full-stack, AI-powered financial intelligence dashboard that combines:

* 📈 **Live stock, crypto & index market data**
* 📰 **LLM-driven financial news summarization**
* 🧠 **Retrieval-Augmented Generation (RAG) for deep market Q&A**
* 📊 **Professional trading indicators (EMA, RSI, MACD, Volume)**
* 💬 **Conversational Copilot Chat with real-time market context**

Built using **Streamlit, Plotly, yFinance, HuggingFace, and OpenAI**, this project replicates the core experience of tools like **TradingView + Bloomberg + AI Copilot** in a single unified interface.

---

## 🚀 Key Features

### 1️⃣ Market Terminal (Stocks, Crypto & Indices)

* Select **any company, US stock, Indian stock, or crypto** from a unified list
* Real-time market metrics:

  * Current Market Price
  * Open / High / Low / Close
  * Daily Change & % Change
* Interactive charts:

  * 📈 Candlestick Price Chart
  * 📊 Volume + Volume SMA
  * 📉 RSI (14)
  * 📉 MACD (12, 26, 9)

### 2️⃣ AI-Powered Financial News Engine

* Aggregates news from:

  * Finnhub
  * Yahoo Finance
  * NewsAPI
  * GNews
* Deduplicates overlapping articles
* Uses **FinBERT** for sentiment analysis
* Uses **FLAN-T5 Large** for detailed financial summaries
* Interactive **3×3 sentiment-colored news cards**

### 3️⃣ Financial Copilot Chat (LLM + RAG)

* Ask natural-language questions like:

  > “What is the current price of Infosys and what does recent news say?”
* Combines:

  * Real-time market data
  * Company profiles
  * Recent news
  * PDF-based knowledge via FAISS
* Answers generated using **OpenAI GPT + RAG pipeline**

---

## 🧱 Project Architecture

```
Financial-Copilot/
│
├── app.py                 # Main Streamlit application
├── companies.py           # Central company / asset mapping
├── data_sources.py        # Market data + news APIs
├── rag_pipeline.py        # RAG orchestration logic
├── retriever.py           # FAISS similarity search
├── indexer.py             # PDF → Embeddings → FAISS index
├── llm.py                 # OpenAI chat wrapper
├── llm_processor.py       # News summarization & sentiment
├── pdf_loader.py          # PDF chunking & loading
├── requirements.txt       # All dependencies
└── README.md              # Project documentation
```

---

## 🧠 Tech Stack

### 🖥 Frontend & Visualization

* **Streamlit** – Interactive dashboard UI
* **Plotly** – Professional trading charts
* **HTML/CSS (inline)** – Custom UI styling

### 📊 Market & Financial Data

* **yFinance** – Stocks, indices, historical OHLC data
* **CoinGecko API** – Live crypto prices
* **Finnhub API** – Company profiles & news
* **Yahoo Finance News**

### 🤖 AI & NLP

* **OpenAI GPT-4o-mini** – Conversational reasoning
* **HuggingFace Transformers**

  * FLAN-T5 Large → Financial summarization
  * FinBERT → Financial sentiment analysis
* **spaCy** – Named entity recognition (company extraction)

### 🧠 Retrieval-Augmented Generation (RAG)

* **FAISS** – Vector similarity search
* **Sentence-Transformers (MiniLM)** – Embeddings
* **LangChain** – RAG pipeline orchestration

---

## 🗂 Company & Asset Coverage

Your `companies.py` file defines a **single source of truth** for all supported assets:

* 🇮🇳 **NIFTY 50 companies**
* 🇺🇸 **US Big Tech & blue-chip stocks**
* 💰 **Popular cryptocurrencies (BTC, ETH, SOL, etc.)**

This allows:

* One dropdown → multiple asset classes
* Automatic detection of **stock vs crypto**
* Unified charting and analytics logic

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/financial-copilot.git
cd financial-copilot
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Set Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
FINNHUB_API_KEY=your_finnhub_key
NEWSAPI_KEY=your_newsapi_key
GNEWS_KEY=your_gnews_key
```

### 5️⃣ Run the App

```bash
streamlit run app.py
```

---

## 🧪 How the System Works (End-to-End)

1. **User selects a company**
2. Market data is fetched via `yfinance`
3. Indicators are calculated locally (EMA, RSI, MACD)
4. News is fetched, deduplicated, summarized, and sentiment-scored
5. Copilot chat:

   * Extracts companies from query
   * Pulls live prices + news
   * Retrieves PDF context via FAISS
   * Sends enriched prompt to OpenAI
6. Final answer is rendered with sources and live facts

---

## 📈 Use Cases

* 📊 Market trend analysis
* 🧠 AI-assisted investment research
* 📰 Financial news digestion
* 🎓 Academic & MCA project demonstration
* 💼 Interview-ready portfolio project

---

## 🔮 Future Enhancements

* Auto-refresh during market hours
* Portfolio tracking & PnL
* Buy/Sell simulation
* Options & derivatives data
* Multi-PDF financial knowledge base
* Cloud deployment (AWS / GCP)

---

## 👤 Author

**Honey J**
MCA | Financial AI & Full-Stack Development
Project built for **advanced academic + real-world finance use**
