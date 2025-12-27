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
# TAB 1: MARKET TERMINAL (USING companies.py MAP)
# =======================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from companies import COMPANY_NAME_MAP  # 👈 YOUR FILE

with tabs[0]:

    st.subheader("Live Market Terminal")

    # ---------------------------------------------------
    # COMPANY SELECTION (NAME → SYMBOL)
    # ---------------------------------------------------
    NAME_TO_SYMBOL = {v: k for k, v in COMPANY_NAME_MAP.items()}

    company_name = st.selectbox(
        "Select Company / Asset",
        sorted(NAME_TO_SYMBOL.keys())
    )

    symbol = NAME_TO_SYMBOL[company_name]

    timeframe = st.selectbox(
        "Timeframe",
        ["1 Week", "1 Month", "3 Months", "6 Months", "1 Year"]
    )

    period_map = {
        "1 Week": "7d",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y"
    }

    interval_map = {
        "1 Week": "1d",
        "1 Month": "1d",
        "3 Months": "1d",
        "6 Months": "1d",
        "1 Year": "1wk"
    }

    # ---------------------------------------------------
    # FETCH DATA (STOCKS + CRYPTO SAFE)
    # ---------------------------------------------------
    df = yf.download(
        symbol,
        period=period_map[timeframe],
        interval=interval_map[timeframe],
        progress=False
    )

    if df.empty:
        st.error("Market data not available")
        st.stop()

    # yfinance safety
    df.columns = df.columns.get_level_values(0)

    # ---------------------------------------------------
    # MARKET METRICS
    # ---------------------------------------------------
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    change = latest["Close"] - prev["Close"]
    change_pct = (change / prev["Close"]) * 100 if prev["Close"] != 0 else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Market Price", round(latest["Close"], 2), f"{change:.2f}")
    col2.metric("Open", round(latest["Open"], 2))
    col3.metric("High", round(latest["High"], 2))
    col4.metric("Low", round(latest["Low"], 2))
    col5.metric("Change %", f"{change_pct:.2f}%")

    # ---------------------------------------------------
    # INDICATORS
    # ---------------------------------------------------
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["EMA50"] = df["Close"].ewm(span=50).mean()
    df["EMA200"] = df["Close"].ewm(span=200).mean()

    df["Volume_SMA_9"] = df["Volume"].rolling(9).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["Signal"] = df["MACD"].ewm(span=9).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]

    candle_colors = np.where(
        df["Close"].values >= df["Open"].values,
        "#26a69a",
        "#ef5350"
    )

    # ==================================================
    # 📈 PRICE CHART
    # ==================================================
    st.subheader("📈 Price Trend")

    fig_price = go.Figure()

    fig_price.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"
    ))

    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA20"], name="EMA 20"))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA50"], name="EMA 50"))
    fig_price.add_trace(go.Scatter(x=df.index, y=df["EMA200"], name="EMA 200"))

    fig_price.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig_price, use_container_width=True)

    # ==================================================
    # 📊 VOLUME CHART
    # ==================================================
    st.subheader("📊 Volume")

    fig_vol = go.Figure()

    fig_vol.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color=candle_colors
    ))

    fig_vol.add_trace(go.Scatter(
        x=df.index,
        y=df["Volume_SMA_9"],
        name="Volume SMA 9"
    ))

    fig_vol.update_layout(
        template="plotly_dark",
        height=300
    )

    st.plotly_chart(fig_vol, use_container_width=True)

    # ==================================================
    # 📉 RSI CHART
    # ==================================================
    st.subheader("📉 RSI (14)")

    fig_rsi = go.Figure()

    fig_rsi.add_trace(go.Scatter(x=df.index, y=df["RSI"]))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")

    fig_rsi.update_layout(
        template="plotly_dark",
        height=250,
        yaxis_range=[0, 100]
    )

    st.plotly_chart(fig_rsi, use_container_width=True)

    # ==================================================
    # 📉 MACD CHART
    # ==================================================
    st.subheader("📉 MACD")

    fig_macd = go.Figure()

    fig_macd.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal"))

    fig_macd.add_trace(go.Bar(
        x=df.index,
        y=df["Histogram"],
        marker_color=np.where(df["Histogram"] >= 0, "#26a69a", "#ef5350")
    ))

    fig_macd.update_layout(
        template="plotly_dark",
        height=300
    )

    st.plotly_chart(fig_macd, use_container_width=True)


# =======================================================
# TAB 2: SENTIMENT-COLORED NEWS CARDS (3x3) + LLM SUMMARY
# =======================================================

import uuid

with tabs[1]:
    st.subheader("LLM-Powered Financial Insights")

    # --------------------------------------------------
    # CSS (Compact & Clean)
    # --------------------------------------------------
    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )

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
    # Sentiment Analysis (Fast Cache)
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

        # Sentiment input
        sentiment_input = f"{headline}. {description}"
        sentiment_data = cached_sentiment(sentiment_input)

        sentiment = sentiment_data["sentiment"]
        confidence = int(sentiment_data["confidence"] * 100)

        if sentiment == "Positive":
            css_class, label = "positive", "🟢 Positive"
        elif sentiment == "Negative":
            css_class, label = "negative", "🔴 Negative"
        else:
            css_class, label = "neutral", "⚪ Neutral"

        with cols[idx % 3]:
            if st.button("Summarize", key=f"summarize_{idx}"):
                st.session_state.selected_article = article
                st.rerun()

            st.markdown(
                f"""
                <div class="news-card {css_class}">
                    <div>
                        <div class="headline">{headline[:80]}...</div>
                        <div class="meta">{label} · {confidence}%</div>
                    </div>
                    <div class="source">
                        🔗 <a href="{url}" target="_blank">{source}</a>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =====================================================
# LLM SUMMARY (WHITE PANEL WITH GAP)
# =====================================================
if st.session_state.selected_article:
    article = st.session_state.selected_article

    # Gap from cards
    st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)

    with st.expander("LLM-Generated News Summary", expanded=True):
        with st.spinner("Summarizing full news using LLM..."):
            summary = summarize_single_article(article)

        # White summary container
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
                <h3 style="margin-top:0;">
                    📰 {article.get('headline') or article.get('title')}
                </h3>
                <p style="font-size:15px; white-space:pre-line;">
                    {summary}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sentiment = analyze_sentiment(summary)

        st.markdown(
            f"""
            <div style="margin-top:15px; font-size:14px;">
                <b>Sentiment:</b> {sentiment['sentiment']}<br>
                <b>Confidence Score:</b> {int(sentiment['confidence'] * 100)}%
            </div>
            """,
            unsafe_allow_html=True,
        )

        if article.get("url"):
            st.markdown(
                f"""
                <div style="margin-top:10px;">
                    <a href="{article['url']}" target="_blank">
                        🔗 Read full article
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
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
        "bitcoin", "ethereum", "share",
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
                responses.append(
                    f"⚠️ Unable to fetch live price for {company_name}"
                )

    return "\n\n".join(responses)
# =======================================================
# TAB 3: COPILOT CHAT — LIVE MARKET + RAG
# =======================================================

with tabs[2]:

    # --------------------------------------------------
    # CSS Styling
    # --------------------------------------------------
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
        unsafe_allow_html=True,
    )

    # --------------------------------------------------
    # Main Chat Container
    # --------------------------------------------------
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    st.markdown(
        "<h2 style='color:#00bfff;text-align:center;'>💬 Financial Copilot Chat</h2>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # --------------------------------------------------
    # Display Chat History
    # --------------------------------------------------
    for chat in st.session_state.chat_history:
        st.markdown(
            f"""
            <div style="display:flex; flex-direction:column;">
                <div class="user-msg">
                    <b>You</b><br>
                    {chat['user']}
                </div>

                <div class="bot-msg">
                    <b>Copilot</b><br>
                    {chat['bot']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # Chat Input Section
    # --------------------------------------------------
    st.markdown(
        "<hr style='border-color:rgba(0,191,255,0.3); margin-top:25px;'>",
        unsafe_allow_html=True,
    )

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

                    # 1️⃣ Live Market Data
                    if detect_market_intent(user_input):
                        live_data = get_live_market_chat_response(user_input)
                        if live_data:
                            bot_sections.append(
                                "### 📊 Live Market Update\n" + live_data
                            )

                    # 2️⃣ LLM / RAG Insight
                    response = answer_query_real_time(user_input)
                    llm_answer = response.get("answer", "")

                    if llm_answer:
                        bot_sections.append(
                            "### 🧠 Market Insight\n" + llm_answer
                        )

                    # 3️⃣ Related News
                    live_facts = response.get("live_facts", {})
                    news_links = []

                    for _, info in live_facts.items():
                        for n in info.get("news", []):
                            title = n.get("headline") or n.get("title")
                            url = n.get("url", "#")
                            news_links.append(f"- [{title}]({url})")

                    if news_links:
                        bot_sections.append(
                            "### 📰 Related News\n" + "\n".join(news_links)
                        )

                    bot_answer = "\n\n".join(bot_sections)

                except Exception as e:
                    bot_answer = f"⚠️ Error fetching data: {e}"

            st.session_state.chat_history.append(
                {
                    "user": user_input,
                    "bot": bot_answer,
                }
            )

            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
