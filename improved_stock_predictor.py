import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lstm_model import LSTMRegressor, prepare_lstm_data, train_lstm_model, predict_lstm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import torch
import streamlit as st

nltk.download("vader_lexicon", download_dir='/tmp')
nltk.data.path.append('/tmp')

# --- CONFIG ---
NEWS_API_KEY = st.secrets["news_api_key"]
DB_PATH = "/tmp/sentiment_cache.db"

TICKER_NAME_MAP = {
    "F": "Ford Motor",
    "TSLA": "Tesla",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "META": "Meta Platforms",
    "AMZN": "Amazon",
    "NFLX": "Netflix"
}

# --- SETUP SQLITE DB ---
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS sentiment_cache (
        ticker TEXT,
        date TEXT,
        sentiment REAL,
        PRIMARY KEY (ticker, date)
    )
''')
conn.commit()

# --- FUNCTIONS ---
def fetch_stock_data(ticker, period="120d", horizon=1):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[["Close"]]
    df["Returns"] = df["Close"].pct_change()
    df["Target"] = df["Close"].shift(-horizon)
    df.dropna(inplace=True)
    return df

def fetch_sentiment(ticker, api_key, days=30):
    sid = SentimentIntensityAnalyzer()
    all_data = []
    query = TICKER_NAME_MAP.get(ticker.upper(), ticker)

    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")

        cursor.execute("SELECT sentiment FROM sentiment_cache WHERE ticker=? AND date=?", (ticker.upper(), date))
        row = cursor.fetchone()
        if row:
            all_data.append({"Date": date, "Sentiment": row[0]})
            continue

        url = (
            f"https://newsapi.org/v2/everything?q={query}&from={date}&to={date}&sortBy=publishedAt&language=en&apiKey={api_key}"
        )

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            st.warning(f"News API error on {date}: {e}")
            all_data.append({"Date": date, "Sentiment": 0})
            continue

        scores = []
        if data.get("articles"):
            for article in data["articles"]:
                title = article.get("title", "")
                sentiment = sid.polarity_scores(title)["compound"]
                scores.append(sentiment)

        daily_score = round(sum(scores) / len(scores), 3) if scores else 0
        if not scores:
            st.warning(f"No news articles found for '{query}' on {date}.")

        cursor.execute("INSERT OR REPLACE INTO sentiment_cache (ticker, date, sentiment) VALUES (?, ?, ?)", (ticker.upper(), date, daily_score))
        conn.commit()

        all_data.append({"Date": date, "Sentiment": daily_score})

    df = pd.DataFrame(all_data)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df.sort_index()

def prepare_data(df):
    for i in range(1, 6):
        df[f"Lag{i}"] = df["Returns"].shift(i)
        df[f"Sentiment_Lag{i}"] = df["Sentiment"].shift(i)
    df.dropna(inplace=True)
    return df

def train_model(df, model_name):
    if model_name == "LSTM":
        feature_cols = ["Returns", "Sentiment"]
        df_lstm = df.copy()
        X, y = prepare_lstm_data(df_lstm, feature_cols, "Target", sequence_length=10)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        model = train_lstm_model(X_train, y_train, input_size=X.shape[2], epochs=20)
        predictions = predict_lstm(model, X_test)
        return model, X_test, y_test, predictions
    else:
        df = prepare_data(df)
        X = df[[f"Lag{i}" for i in range(1, 6)] + [f"Sentiment_Lag{i}" for i in range(1, 6)]]
        y = df["Target"]
        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        elif model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "XGBoost":
            model = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, objective='reg:squarederror', random_state=42)
        else:
            raise ValueError("Unsupported model")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        return model, X_test, y_test, predictions





