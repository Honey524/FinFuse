# =======================================================
# Financial Copilot Dashboard (Streamlit + Plotly + LLM)
# =======================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from rag_pipeline import answer_query_real_time
from data_sources import (
    get_stock_price,
    get_crypto_price,
    get_aggregated_financial_news,
    get_historical_data,
)
from companies import COMPANY_NAME_MAP
from llm_processor import (
    summarize_single_article,
    deduplicate_news_by_name,
    analyze_sentiment,
    process_financial_news,
    get_summarizer,
    get_sentiment_analyzer
)
import threading

# ------------------------
# PAGE CONFIG
# ------------------------
st.set_page_config(page_title="Financial Copilot", layout="wide")

# ------------------------
# PRELOAD MODELS IN BACKGROUND
# ------------------------
def preload_models():
    get_summarizer()
    get_sentiment_analyzer()
threading.Thread(target=preload_models, daemon=True).start()

# ------------------------
# COMPANY & CRYPTO SETUP
# ------------------------
COMPANIES = list(COMPANY_NAME_MAP.values())
TICKER_MAP = {name: ticker for ticker, name in COMPANY_NAME_MAP.items()}

# ------------------------
# SESSION STATE
# ------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ------------------------
# SIDEBAR
# ------------------------
with st.sidebar:
    st.title("Dashboard")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

    st.subheader("Quick Market Snapshot")
    snapshot_tickers = st.multiselect(
        "Select tickers to monitor",
        options=COMPANIES,
        default=["Infosys", "Tata Consultancy Services", "Bitcoin"]
    )

    for t in snapshot_tickers:
        symbol = TICKER_MAP[t]
        try:
            if t in ["BTC", "ETH"]:
                price = get_crypto_price(symbol)
                st.metric(label=f"{t} (USD)", value=price)
            else:
                price, cur = get_stock_price(symbol)
                st.metric(label=f"{t} ({cur})", value=price)
        except:
            st.warning(f"{t} data not found")

    st.subheader("⚡ Price Alerts")
    alert_ticker = st.selectbox("Ticker for alert", options=list(TICKER_MAP.keys()))
    threshold = st.number_input("Set price threshold", min_value=0.0, value=0.0)
    if st.button("Check Alert"):
        symbol = TICKER_MAP[alert_ticker]
        try:
            if alert_ticker in ["BTC", "ETH"]:
                price = get_crypto_price(symbol)
            else:
                price, cur = get_stock_price(symbol)
            if price >= threshold:
                st.success(f"{alert_ticker} crossed {threshold}! Current: {price}")
            else:
                st.info(f"{alert_ticker} below threshold. Current: {price}")
        except:
            st.error("Error fetching alert price")

# ------------------------
# MAIN PAGE
# ------------------------
st.title("Financial Market Dashboard")
tabs = st.tabs(["Live Prices & Charts", "Financial News", "Copilot Chat"])

# =======================================================
# TAB 1: LIVE MARKET TRACKER (REAL-TIME STREAMING)
# =======================================================
import time
import threading

with tabs[0]:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 20% 20%, #0b0f1a, #000000 80%);
            color: #f5f5f5;
        }
        .metric-card {
            background: rgba(25, 25, 25, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 20px rgba(0,0,0,0.5);
            border-radius: 16px;
            padding: 20px;
            backdrop-filter: blur(8px);
            color: white;
        }
        .title-text {
            text-align: center;
            font-weight: 600;
            font-size: 1.4rem;
            color: #00bfff;
        }
        .live-badge {
            background: #ff004f;
            color: white;
            border-radius: 12px;
            padding: 3px 10px;
            font-size: 0.8rem;
            margin-left: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="title-text">Live Market Tracker <span class="live-badge">REAL-TIME</span></div>', unsafe_allow_html=True)

    ticker_choice = st.selectbox("Select Ticker", options=COMPANIES, index=0)
    symbol = TICKER_MAP[ticker_choice]
    refresh_rate = st.slider("⏱ Update Interval (seconds)", 1, 15, 3)

    start_button = st.button("▶️ Start Live Tracking")
    stop_button = st.button("⏹ Stop Tracking")

    # placeholders
    price_placeholder = st.empty()
    chart_placeholder = st.empty()

    # Keep state
    if "live_tracking" not in st.session_state:
        st.session_state.live_tracking = False

    if start_button:
        st.session_state.live_tracking = True
    if stop_button:
        st.session_state.live_tracking = False

    # main live loop
    if st.session_state.live_tracking:
        st.success(f"Streaming live price updates for **{ticker_choice}** ...")

        prices, timestamps = [], []

        # start the live feed
        for _ in range(1000):  # arbitrary upper limit to prevent infinite loop
            if not st.session_state.live_tracking:
                st.warning("⏹ Live tracking stopped.")
                break

            try:
                # Get price
                if ticker_choice in ["BTC", "ETH"]:
                    price = get_crypto_price(symbol)
                    currency = "USD"
                else:
                    price, currency = get_stock_price(symbol)

                if not price:
                    raise ValueError("Price not found")

                current_time = datetime.now().strftime("%H:%M:%S")
                prices.append(price)
                timestamps.append(current_time)

                # Keep last 30 points
                if len(prices) > 30:
                    prices = prices[-30:]
                    timestamps = timestamps[-30:]

                # Compute movement
                delta = round(prices[-1] - prices[-2], 2) if len(prices) > 1 else 0
                arrow = "🟢↑" if delta > 0 else "🔴↓" if delta < 0 else "⚪→"
                delta_text = f"+{delta}" if delta > 0 else str(delta)

                # update metric card
                price_placeholder.markdown(
                    f"""
                    <div class="metric-card">
                        <h2 style="margin:0;">{ticker_choice} ({currency})</h2>
                        <h1 style="margin-top:10px; font-size:2.5rem; color:{'#39ff14' if delta>0 else '#ff4b4b' if delta<0 else '#f5f5f5'};">
                            {price:.2f} {arrow}
                        </h1>
                        <p style="font-size:0.9rem; opacity:0.8;">Change: {delta_text}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # live chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=prices,
                    mode="lines+markers",
                    line=dict(width=3, color="#00bfff"),
                    marker=dict(size=6, color="#39ff14" if delta > 0 else "#ff4b4b"),
                    name="Price"
                ))
                fig.update_layout(
                    template="plotly_dark",
                    title=dict(text=f"{ticker_choice} — Live Movement", x=0.5, font=dict(size=18, color="#00bfff")),
                    xaxis_title="Time (hh:mm:ss)",
                    yaxis_title="Price",
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
                    margin=dict(l=10, r=10, t=40, b=30),
                    height=400,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                chart_placeholder.plotly_chart(fig, use_container_width=True)

                time.sleep(refresh_rate)

            except Exception as e:
                st.error(f"⚠️ Error fetching data: {e}")
                time.sleep(refresh_rate)
                continue
# =======================================================
# TAB 2: SENTIMENT-COLORED NEWS CARDS (3x3) + LLM SUMMARY
# =======================================================
import uuid

with tabs[1]:
    st.subheader("LLM-Powered Financial Insights")

    # --------------------------------------------------
    # CSS (Compact & Clean)
    # --------------------------------------------------
    st.markdown("""
    <style>
    .news-card {
        border-radius: 14px;
        padding: 14px;
        color: white;
        height: 190px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.35);
        transition: transform 0.2s ease, box-shadow 0.3s ease;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .news-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 22px rgba(0,191,255,0.45);
    }

    .positive { background: linear-gradient(135deg, #0f9b0f, #38ef7d); }
    .neutral  { background: linear-gradient(135deg, #6d6027, #d3cbb8); }
    .negative { background: linear-gradient(135deg, #93291e, #ed213a); }

    .headline {
        font-size: 14px;
        font-weight: 600;
        line-height: 1.3;
        margin-bottom: 6px;
    }

    .meta {
        font-size: 12px;
        opacity: 0.9;
    }

    .source {
        font-size: 11px;
        margin-top: 6px;
    }

    .source a {
        color: #ffffff;
        text-decoration: underline;
        opacity: 0.85;
    }
    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------------
    # Fetch + Deduplicate + Limit (3x3)
    # --------------------------------------------------
    @st.cache_data(ttl=3600)
    def cached_news(symbol):
        raw_news = get_aggregated_financial_news(symbol, limit=20)
        unique_news = deduplicate_news_by_name(raw_news)
        return unique_news[:9]

    news_list = cached_news(symbol)

    if not news_list:
        st.warning("No news found.")
        st.stop()

    # --------------------------------------------------
    # Sentiment (FAST)
    # --------------------------------------------------
    @st.cache_data
    def cached_sentiment(text):
        return analyze_sentiment(text)

    # --------------------------------------------------
    # Session State
    # --------------------------------------------------
    if "selected_article" not in st.session_state:
        st.session_state.selected_article = None

    # --------------------------------------------------
    # 3 x 3 CARD GRID
    # --------------------------------------------------
    cols = st.columns(3)

    for idx, article in enumerate(news_list):
        headline = article.get("headline") or article.get("title", "Untitled")
        description = article.get("description", "")
        url = article.get("url", "")

        # Source normalization
        src = article.get("source")
        if isinstance(src, dict):
            source = src.get("name", "Source")
        elif isinstance(src, str):
            source = src
        else:
            source = "Source"

        sentiment_input = f"{headline}. {description}"
        sentiment_data = cached_sentiment(sentiment_input)

        sentiment = sentiment_data["sentiment"]
        confidence = int(sentiment_data["confidence"] * 100)

        if sentiment == "Positive":
            css, label = "positive", "🟢 Positive"
        elif sentiment == "Negative":
            css, label = "negative", "🔴 Negative"
        else:
            css, label = "neutral", "⚪ Neutral"

        with cols[idx % 3]:
            if st.button("Summarize", key=f"summarize_{idx}"):
                st.session_state.selected_article = article
                st.rerun()

            st.markdown(
                f"""
                <div class="news-card {css}">
                    <div>
                        <div class="headline">{headline[:80]}...</div>
                        <div class="meta">{label} · {confidence}%</div>
                    </div>
                    <div class="source">
                        🔗 <a href="{url}" target="_blank">{source}</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

# --------------------------------------------------
# LLM SUMMARY (WHITE PANEL WITH GAP)
# --------------------------------------------------
if st.session_state.selected_article:
    article = st.session_state.selected_article

    # ✅ Gap from cards
    st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)

    with st.expander("LLM-Generated News Summary", expanded=True):
        with st.spinner("Summarizing full news using LLM..."):
            summary = summarize_single_article(article)

        # ✅ White summary container
        st.markdown(
            f"""
            <div style="
                background-color: #ffffff;
                color: #000000;
                padding: 22px;
                border-radius: 14px;
                box-shadow: 0 8px 20px rgba(0,0,0,0.15);
                line-height: 1.6;
            ">
                <h3 style="margin-top:0;">📰 {article.get('headline') or article.get('title')}</h3>
                <p style="font-size:15px; white-space:pre-line;">
                    {summary}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        sentiment = analyze_sentiment(summary)

        st.markdown(
            f"""
            <div style="margin-top:15px; font-size:14px;">
                <b>Sentiment:</b> {sentiment['sentiment']} <br>
                <b>Confidence Score:</b> {int(sentiment['confidence'] * 100)}%
            </div>
            """,
            unsafe_allow_html=True
        )

        if article.get("url"):
            st.markdown(
                f"<div style='margin-top:10px;'><a href='{article['url']}' target='_blank'>🔗 Read full article</a></div>",
                unsafe_allow_html=True
            )

        if st.button("❌ Close Summary"):
            st.session_state.selected_article = None
            st.rerun()


# =====================================================
# HELPER FUNCTIONS FOR COPILOT CHAT (TAB 3)
# =====================================================

def detect_market_intent(query: str) -> bool:
    """
    Detects whether the user is asking about live prices
    or current market activity.
    """
    keywords = [
        "price", "current", "today", "now",
        "market", "trading", "stock", "crypto",
        "bitcoin", "ethereum", "share"
    ]
    return any(word in query.lower() for word in keywords)


def get_live_market_chat_response(query: str) -> str:
    """
    Fetches real-time stock / crypto prices for Copilot chat.
    """
    responses = []

    for company_name, ticker in TICKER_MAP.items():
        if company_name.lower() in query.lower():
            try:
                # Crypto
                if ticker.lower() in ["btc", "bitcoin", "eth", "ethereum"]:
                    price = get_crypto_price(ticker)
                    if price is not None:
                        responses.append(
                            f"📊 **{company_name} (Crypto)**\n"
                            f"• Current Price: **${price} USD**\n"
                            f"• Updated just now"
                        )

                # Stock
                else:
                    price, currency = get_stock_price(ticker)
                    if price is not None:
                        responses.append(
                            f"📈 **{company_name} (Stock)**\n"
                            f"• Current Price: **{price} {currency}**\n"
                            f"• Live market data"
                        )

            except Exception:
                responses.append(f"⚠️ Unable to fetch live price for {company_name}")

    return "\n\n".join(responses)


# =======================================================
# TAB 3: COPILOT CHAT — LIVE MARKET + RAG
# =======================================================
with tabs[2]:
    st.markdown(
        """
        <style>
        .main-container {
            background: radial-gradient(circle at top left, #0a0f1c, #000000 80%);
            border-radius: 16px;
            padding: 25px;
            box-shadow: 0 0 25px rgba(0,191,255,0.2);
        }
        .chat-container {
            max-height: 550px;
            overflow-y: auto;
            padding-right: 10px;
        }
        .user-msg {
            background: linear-gradient(135deg, #002b36, #004b66);
            color: #f5f5f5;
            padding: 14px 18px;
            border-radius: 18px 18px 4px 18px;
            margin: 10px 0;
            max-width: 80%;
            align-self: flex-end;
        }
        .bot-msg {
            background: linear-gradient(135deg, #111111, #1a1a1a);
            border: 1px solid rgba(0,191,255,0.3);
            color: #e5e5e5;
            padding: 14px 18px;
            border-radius: 18px 18px 18px 4px;
            margin: 10px 0;
            max-width: 85%;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown("<h2 style='color:#00bfff;text-align:center;'>💬 Financial Copilot Chat</h2>", unsafe_allow_html=True)
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # ---------------------------
    # Display Chat History
    # ---------------------------
    for chat in st.session_state.chat_history:
        st.markdown(f"""
        <div style="display:flex; flex-direction:column;">
            <div class="user-msg"><b>You</b><br>{chat['user']}</div>
            <div class="bot-msg"><b>Copilot</b><br>{chat['bot']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------
    # Chat Input
    # ---------------------------
    st.markdown("<hr style='border-color:rgba(0,191,255,0.3);margin-top:25px;'>", unsafe_allow_html=True)
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask about stocks, crypto, or markets:",
            placeholder="e.g., What is the current price of Infosys?",
        )
        submit_button = st.form_submit_button("🚀 Send")

        if submit_button and user_input.strip():
            with st.spinner("💭 Thinking..."):
                try:
                    bot_sections = []

                    # 1️⃣ Live market data
                    if detect_market_intent(user_input):
                        live_data = get_live_market_chat_response(user_input)
                        if live_data:
                            bot_sections.append("### 📊 Live Market Update\n" + live_data)

                    # 2️⃣ LLM / RAG response
                    response = answer_query_real_time(user_input)
                    llm_answer = response.get("answer", "")
                    if llm_answer:
                        bot_sections.append("### 🧠 Market Insight\n" + llm_answer)

                    # 3️⃣ Related news
                    live_facts = response.get("live_facts", {})
                    news_links = []

                    for _, info in live_facts.items():
                        for n in info.get("news", []):
                            title = n.get("headline") or n.get("title")
                            url = n.get("url", "#")
                            news_links.append(f"- [{title}]({url})")

                    if news_links:
                        bot_sections.append("### 📰 Related News\n" + "\n".join(news_links))

                    bot_answer = "\n\n".join(bot_sections)

                except Exception as e:
                    bot_answer = f"⚠️ Error fetching data: {e}"

            st.session_state.chat_history.append({
                "user": user_input,
                "bot": bot_answer
            })
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
