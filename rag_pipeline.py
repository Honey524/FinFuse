# rag_pipeline.py

import spacy
from retriever import retrieve
from llm import chat_with_context
from data_sources import (
    get_stock_price,
    get_crypto_price,
    get_realtime_quote,
    get_company_profile,
    get_aggregated_financial_news,
)

# Load spaCy model for Named Entity Recognition
nlp = spacy.load("en_core_web_sm")

# ------------------------------------------------
# Company / Ticker Extraction
# ------------------------------------------------
def extract_companies(query):
    """
    Extract company names or tickers from the query.
    """
    doc = nlp(query)
    companies = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

    # fallback: add ticker words in uppercase that spaCy missed
    words = query.split()
    tickers = [w for w in words if w.isupper() and len(w) <= 5]

    return list(set(companies + tickers))

# ------------------------------------------------
# Fetch Live Data
# ------------------------------------------------
def fetch_live_data_for_companies(companies):
    """
    Fetch live stock prices, crypto prices, company profiles, and recent news.
    """
    live_facts = {}

    for company in companies:
        try:
            price, currency = get_stock_price(company)
            live_facts[company] = {
                "type": "stock",
                "price": price,
                "currency": currency if currency else "N/A",
                "quote": get_realtime_quote(company),
                "profile": get_company_profile(company),
                "news": get_aggregated_financial_news(company, limit=3),
            }
        except Exception as e:
            live_facts[company] = {"error": str(e)}

    # Check for cryptos
    for crypto in ["BTC", "ETH"]:
        if crypto in companies:
            try:
                live_facts[crypto] = {
                    "type": "crypto",
                    "price": get_crypto_price(crypto),
                    "currency": "USD",
                    "news": get_aggregated_financial_news(crypto, limit=3),
                }
            except Exception as e:
                live_facts[crypto] = {"error": str(e)}

    return live_facts

# ------------------------------------------------
# Full Pipeline with Dual-Mode
# ------------------------------------------------
def answer_query_real_time(query, k=3):
    """
    Full RAG pipeline with real-time data integration and fallback.
    """
    # 1️⃣ Retrieve PDF context
    docs = retrieve(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

    # 2️⃣ Extract company names
    companies = extract_companies(query)

    # 3️⃣ Fetch live data
    live_facts = fetch_live_data_for_companies(companies) if companies else {}

    # 🔹 If no companies but query mentions 'market' or 'news', fetch general market news
    if not companies and any(word in query.lower() for word in ["market", "news"]):
        general_news = get_aggregated_financial_news(limit=5)
        news_lines = []
        for n in general_news:
            headline = n.get("headline") or n.get("title")
            link = n.get("url", "")
            if headline:
                # Markdown-style link
                news_lines.append(f"- [{headline}]({link})")
        if news_lines:
            context += "\n\nMARKET NEWS:\n" + "\n".join(news_lines)


    # 4️⃣ Build final prompt
    live_context_lines = []
    for comp, info in live_facts.items():
        if info.get("type") == "stock":
            live_context_lines.append(f"{comp} stock price: {info['price']} {info['currency']}")
        elif info.get("type") == "crypto":
            live_context_lines.append(f"{comp} crypto price: {info['price']} USD")
    
    if live_context_lines or context:
        final_context = f"{context}\n\nLIVE FACTS:\n" + "\n".join(live_context_lines) if live_context_lines else context
        prompt = f"CONTEXT: {final_context}\nQUESTION: {query}"
    else:
        # 🔹 Fallback: no context, use query directly
        prompt = query

    # 5️⃣ Get answer from LLM
    answer = chat_with_context(prompt)

    return {"answer": answer, "sources": [getattr(doc, "source", "PDF") for doc in docs], "live_facts": live_facts}

# ------------------------------------------------
# Quick Test
# ------------------------------------------------
if __name__ == "__main__":
    test_queries = [
        "What is INFY price and what did the last earnings report say about revenue?",
        "Who is the president of India?",
        "Tell me about BTC price today.",
    ]

    for q in test_queries:
        resp = answer_query_real_time(q)
        print(f"\nQuery: {q}\nAnswer: {resp['answer']}\n")
