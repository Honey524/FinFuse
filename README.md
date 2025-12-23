# 📊 FinFuse - Real-Time Market Intelligence & LLM-Powered Financial Assistant
---

## 🚀 Overview

**FinFuse Copilot** is an AI-powered financial intelligence dashboard that combines **real-time stock & crypto prices**, **financial news aggregation**, **sentiment analysis**, and a **Retrieval-Augmented Generation (RAG) based conversational assistant**.

It enables users to:

* Track live market prices with dynamic charts
* Read sentiment-colored financial news
* Ask natural language questions about markets, companies, and trends
* Get answers enriched with **live market data + PDF knowledge base**

---

## 🧠 Key Features

### 🔴 Live Market Tracking

* Real-time stock & crypto prices (India 🇮🇳 + US 🇺🇸 markets)
* Auto-refreshing price charts
* Price movement indicators (↑ ↓)

### 📰 Financial News Intelligence

* Aggregates news from multiple sources (Finnhub, NewsAPI, GNews, Yahoo Finance)
* Deduplicates similar articles
* Sentiment analysis using **FinBERT**
* One-click **LLM-generated summaries**

### 💬 Financial Copilot Chat (RAG + Live Data)

* Ask questions like:

  * *“What is the current price of Infosys?”*
  * *“How is the market sentiment today?”*
* Combines:

  * PDF-based knowledge (RAG using FAISS)
  * Live stock & crypto prices
  * Latest financial news
* Powered by OpenAI LLM

---

## 🏗️ Project Architecture

```
├── app.py                  # Streamlit dashboard (UI + interaction)
├── companies.py            # Stock, index & crypto ticker mapping
├── data_sources.py         # Market data & news aggregation layer
├── llm.py                  # OpenAI LLM interface
├── llm_processor.py        # News summarization & sentiment analysis
├── rag_pipeline.py         # Full RAG pipeline with live data
├── retriever.py            # FAISS vector search
├── indexer.py              # PDF indexing & embedding creation
├── pdf_loader.py           # PDF loading & chunking
├── requirements.txt        # Project dependencies
├── faiss_index/            # Vector database (generated)
└── README.md
```

---

## 🛠️ Tech Stack

| Layer       | Technology                       |
| ----------- | -------------------------------- |
| Frontend    | Streamlit, Plotly                |
| Backend     | Python                           |
| LLM         | OpenAI (Chat Completions API)    |
| NLP         | spaCy, Hugging Face Transformers |
| Sentiment   | FinBERT                          |
| RAG         | LangChain + FAISS                |
| Market Data | yFinance, CoinGecko              |
| News APIs   | Finnhub, NewsAPI, GNews          |
| Embeddings  | Sentence Transformers            |

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
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
FINNHUB_API_KEY=your_finnhub_key
NEWSAPI_KEY=your_newsapi_key
GNEWS_KEY=your_gnews_key
```

> ❗ **Never commit `.env` to GitHub**

---

## 📄 Index PDFs for RAG (Optional but Recommended)

If you want PDF-based question answering:

```bash
python indexer.py
```

This creates a **FAISS vector index** used by the Copilot chat.

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 💡 Example Use Cases

* 📈 Track live stock & crypto prices
* 📰 Understand market sentiment instantly
* 🤖 Ask financial questions in natural language
* 📊 Combine news + price + LLM reasoning in one place
* 🎓 Ideal for **students, analysts, and fintech demos**

---

## 📌 Future Enhancements

* User authentication
* Portfolio tracking
* Price alerts (email / push)
* Multi-PDF knowledge bases
* Deployment on AWS / GCP

---

## 👨‍💻 Author

**Honey J**
MCA | AI & Full-Stack Developer
Focused on **LLM systems, RAG, and real-time data platforms**
