# =====================================================
# llm_processor.py (Financial News LLM Engine)
# =====================================================
import streamlit as st
from transformers import pipeline
import torch
import nltk
import warnings
import hashlib

# ------------------------------------------
# Setup
# ------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
nltk.download("punkt", quiet=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE.upper()}")

# ------------------------------------------
# Safe Pipeline Loader
# ------------------------------------------
def _safe_pipeline(task, model_name):
    try:
        return pipeline(
            task,
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
    except Exception as e:
        print(f"⚠️ Failed loading {model_name}: {e}")
        return pipeline(task, model=model_name, device=-1)

# ------------------------------------------
# Cached Models
# ------------------------------------------
@st.cache_resource
def get_summarizer():
    return _safe_pipeline("summarization", "google/flan-t5-large")

@st.cache_resource
def get_sentiment_analyzer():
    return _safe_pipeline("text-classification", "yiyanghkust/finbert-tone")

# ------------------------------------------
# 🔹 DEDUPLICATE NEWS BY NAME (TITLE / HEADLINE)
# ------------------------------------------
def deduplicate_news_by_name(articles):
    """
    Removes duplicate news articles based on headline/title.
    """
    seen = set()
    unique_articles = []

    for article in articles:
        title = (
            article.get("headline")
            or article.get("title")
            or ""
        ).strip().lower()

        description = (article.get("description") or "").strip().lower()

        # Create stable name-based signature
        signature = f"{title}::{description[:120]}"
        sig_hash = hashlib.md5(signature.encode("utf-8")).hexdigest()

        if sig_hash not in seen:
            seen.add(sig_hash)
            unique_articles.append(article)

    return unique_articles

# ------------------------------------------
# 🔹 BRIEF SUMMARY FOR ONE ARTICLE (ON CLICK)
# ------------------------------------------
def summarize_single_article(article):
    """
    Generates a DETAILED 2-paragraph financial summary
    explaining the full news clearly.
    """
    summarizer = get_summarizer()

    text = f"""
    Title: {article.get('headline') or article.get('title', '')}
    Description: {article.get('description', '')}
    """

    prompt = (
        "Summarize the following financial news in EXACTLY TWO PARAGRAPHS.\n"
        "Paragraph 1: Explain what happened, key events, companies involved, and facts.\n"
        "Paragraph 2: Explain the implications, market reaction, investor sentiment, "
        "and potential impact going forward.\n\n"
        f"{text}"
    )

    result = summarizer(
        prompt,
        max_length=220,     # enough for 2 solid paragraphs
        min_length=140,     # forces detail
        do_sample=False
    )

    return result[0]["summary_text"]


# ------------------------------------------
# 🔹 FULL MULTI-NEWS SUMMARY (OPTIONAL / REPORT)
# ------------------------------------------
def summarize_news_articles(articles):
    """
    Heavy aggregation summary (used for reports, not UI click).
    """
    if not articles:
        return "No news available."

    # ✅ Deduplicate before summarization
    articles = deduplicate_news_by_name(articles)

    summarizer = get_summarizer()

    combined_text = "\n".join([
        f"Title: {a.get('headline') or a.get('title','')}\nDescription: {a.get('description','')}"
        for a in articles
    ])

    result = summarizer(
        f"Provide a detailed financial market analysis:\n{combined_text}",
        max_length=600,
        min_length=150,
        do_sample=True,
        top_p=0.9
    )

    return refine_summary(result[0]["summary_text"])

# ------------------------------------------
# Tone Refinement (for reports)
# ------------------------------------------
def refine_summary(text):
    return (
        "**Comprehensive Financial Analysis**\n\n"
        + text.strip()
        + "\n\nOverall, this reflects market sentiment and investor behavior."
    )

# ------------------------------------------
# 🔹 SENTIMENT ANALYSIS (FAST & CORRECT)
# ------------------------------------------
def analyze_sentiment(text):
    """
    Proper FinBERT sentiment analysis with correct confidence score.
    """
    if not text or len(text.strip()) < 10:
        return {
            "sentiment": "Neutral",
            "confidence": 0.0
        }

    analyzer = get_sentiment_analyzer()
    output = analyzer(text[:512], truncation=True)[0]

    return {
        "sentiment": output["label"],          # Positive | Negative | Neutral
        "confidence": round(float(output["score"]), 3)
    }

# ------------------------------------------
# 🔹 UI-FRIENDLY PROCESSOR (ONE ARTICLE)
# ------------------------------------------
def process_single_article(article):
    """
    Used by UI when user clicks a news card.
    """
    summary = summarize_single_article(article)
    sentiment = analyze_sentiment(summary)

    return {
        "summary": summary,
        "sentiment": sentiment["sentiment"],
        "confidence": sentiment["confidence"]
    }

# ------------------------------------------
# 🔹 BACKWARD-COMPATIBILITY WRAPPER
# ------------------------------------------
def process_financial_news(articles):
    """
    Legacy function to avoid breaking older imports.
    Deduplicates news by name before processing.
    """
    if not articles:
        return {
            "summary": "No news available.",
            "sentiment": "Neutral",
            "confidence": 0.0
        }

    # ✅ Deduplicate by name
    articles = deduplicate_news_by_name(articles)

    article = articles[0]

    summary = summarize_single_article(article)
    sentiment = analyze_sentiment(summary)

    return {
        "summary": summary,
        "sentiment": sentiment["sentiment"],
        "confidence": sentiment["confidence"]
    }
