# data_sources.py

import yfinance as yf
import requests
import os
import pandas as pd
from datetime import datetime, timedelta, timezone

from companies import COMPANY_NAME_MAP

# --------------------------------------------------
# API KEYS
# --------------------------------------------------
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
GNEWS_KEY = os.getenv("GNEWS_KEY")

# ==================================================
# STOCK DATA
# ==================================================

def get_stock_price(ticker: str):
    """
    Fetch current stock price.
    Returns: (price: float | None, currency: str)
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d")

        if hist.empty:
            return None, "N/A"

        price = float(hist["Close"].iloc[-1])
        currency = stock.info.get("currency") or (
            "INR" if ticker.endswith(".NS") else "USD"
        )
        return price, currency

    except Exception:
        return None, "N/A"


def get_realtime_stock_snapshot(ticker: str):
    """
    Fetch real-time stock snapshot:
    price, change, % change
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d")

        if len(hist) < 2:
            return None

        prev_close = hist["Close"].iloc[-2]
        current = hist["Close"].iloc[-1]

        change = current - prev_close
        pct_change = (change / prev_close) * 100

        currency = stock.info.get("currency") or (
            "INR" if ticker.endswith(".NS") else "USD"
        )

        return {
            "price": round(float(current), 2),
            "change": round(float(change), 2),
            "pct_change": round(float(pct_change), 2),
            "currency": currency,
            "timestamp": datetime.utcnow()
        }

    except Exception:
        return None


def get_company_profile(ticker: str):
    """Fetch company profile using Finnhub"""
    try:
        url = f"https://finnhub.io/api/v1/stock/profile2?symbol={ticker}&token={FINNHUB_API_KEY}"
        res = requests.get(url, timeout=10)
        return res.json() if res.status_code == 200 else {}
    except Exception:
        return {}

def get_realtime_quote(ticker: str):
    """
    Backward-compatible realtime quote.
    Returns Finnhub-like structure:
    o = open
    h = high
    l = low
    c = current
    pc = previous close
    t = timestamp
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="2d")

        if hist.empty or len(hist) < 2:
            return {}

        open_price = float(hist["Open"].iloc[-1])
        high = float(hist["High"].iloc[-1])
        low = float(hist["Low"].iloc[-1])
        current = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2])

        return {
            "o": open_price,
            "h": high,
            "l": low,
            "c": current,
            "pc": prev_close,
            "t": int(datetime.utcnow().timestamp())
        }

    except Exception:
        return {}

# ==================================================
# CRYPTO DATA
# ==================================================

def get_crypto_price(symbol: str):
    """
    Fetch live crypto price in USD.
    Returns: float | None
    """
    try:
        url = (
            "https://api.coingecko.com/api/v3/simple/price"
            f"?ids={symbol.lower()}&vs_currencies=usd"
        )
        res = requests.get(url, timeout=10).json()
        return float(res[symbol.lower()]["usd"])
    except Exception:
        return None


def get_realtime_crypto_snapshot(symbol: str):
    """
    Fetch crypto snapshot: price + timestamp
    """
    price = get_crypto_price(symbol)
    if price is None:
        return None

    return {
        "price": round(price, 2),
        "currency": "USD",
        "timestamp": datetime.utcnow()
    }


# ==================================================
# HISTORICAL DATA
# ==================================================

def get_historical_data(ticker: str, crypto: bool = False, period: str = "30d"):
    try:
        if crypto:
            url = (
                f"https://api.coingecko.com/api/v3/coins/{ticker.lower()}"
                "/market_chart?vs_currency=usd&days=30"
            )
            res = requests.get(url, timeout=10).json()
            prices = res.get("prices", [])
            df = pd.DataFrame(prices, columns=["timestamp", "Close"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            return df

        stock = yf.Ticker(ticker)
        return stock.history(period="1mo")

    except Exception:
        return None


# ==================================================
# 🔥 NEW: UNIFIED MARKET SNAPSHOT (FOR COPILOT CHAT)
# ==================================================

def get_market_snapshot_by_name(name: str):
    """
    Unified market snapshot for Copilot Chat.
    Detects stock vs crypto automatically.
    """
    ticker = COMPANY_NAME_MAP.get(name)
    if not ticker:
        return None

    # Crypto
    if ticker.lower() in ["btc", "bitcoin", "eth", "ethereum"]:
        snap = get_realtime_crypto_snapshot(ticker)
        if not snap:
            return None

        return {
            "name": name,
            "type": "Crypto",
            **snap
        }

    # Stock
    snap = get_realtime_stock_snapshot(ticker)
    if not snap:
        return None

    return {
        "name": name,
        "type": "Stock",
        **snap
    }


# ==================================================
# FINANCIAL NEWS SOURCES
# ==================================================

def fetch_finnhub_news(symbol=None, limit=5):
    try:
        if symbol:
            to_date = datetime.today().date()
            from_date = to_date - timedelta(days=7)
            url = (
                "https://finnhub.io/api/v1/company-news"
                f"?symbol={symbol}&from={from_date}&to={to_date}"
                f"&token={FINNHUB_API_KEY}"
            )
        else:
            url = f"https://finnhub.io/api/v1/news?category=general&token={FINNHUB_API_KEY}"

        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return []

        return [
            {
                "title": n.get("headline") or n.get("title"),
                "url": n.get("url"),
                "summary": n.get("summary", ""),
                "source": "Finnhub",
                "published": n.get("datetime"),
            }
            for n in res.json()[:limit]
        ]

    except Exception:
        return []


def fetch_newsapi(query="finance", limit=5):
    if not NEWSAPI_KEY:
        return []

    try:
        url = (
            "https://newsapi.org/v2/everything"
            f"?q={query}&sortBy=publishedAt"
            f"&pageSize={limit}&language=en"
            f"&apiKey={NEWSAPI_KEY}"
        )
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return []

        return [
            {
                "title": a.get("title"),
                "url": a.get("url"),
                "summary": a.get("description", ""),
                "source": "NewsAPI",
                "published": a.get("publishedAt"),
            }
            for a in res.json().get("articles", [])
        ]

    except Exception:
        return []


def fetch_gnews(query="finance", limit=5):
    if not GNEWS_KEY:
        return []

    try:
        url = (
            f"https://gnews.io/api/v4/search"
            f"?q={query}&lang=en&max={limit}&token={GNEWS_KEY}"
        )
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return []

        return [
            {
                "title": a.get("title"),
                "url": a.get("url"),
                "summary": a.get("description", ""),
                "source": "GNews",
                "published": a.get("publishedAt"),
            }
            for a in res.json().get("articles", [])
        ]

    except Exception:
        return []


def fetch_yahoo_news(symbol, limit=5):
    try:
        ticker = yf.Ticker(symbol)
        return [
            {
                "title": n.get("title"),
                "url": n.get("link"),
                "summary": n.get("summary", ""),
                "source": "YahooFinance",
                "published": n.get("providerPublishTime"),
            }
            for n in ticker.news[:limit]
        ]
    except Exception:
        return []


def get_aggregated_financial_news(symbol=None, query="finance", limit=9):
    news = []
    news.extend(fetch_finnhub_news(symbol, limit))
    news.extend(fetch_newsapi(query, limit))
    news.extend(fetch_gnews(query, limit))
    if symbol:
        news.extend(fetch_yahoo_news(symbol, limit))

    seen = set()
    final = []
    for n in news:
        key = (n.get("title"), n.get("url"))
        if key not in seen:
            final.append(n)
            seen.add(key)

    def parse_date(d):
        try:
            if isinstance(d, int):
                return datetime.utcfromtimestamp(d)
            return datetime.fromisoformat(str(d).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return datetime.min

    final.sort(key=lambda x: parse_date(x.get("published")), reverse=True)
    return final[:limit]
