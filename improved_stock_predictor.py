import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
import sqlite3
import time
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
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
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                st.warning(f"429 Too Many Requests on {date} — using fallback sentiment = 0")
                daily_score = 0
                cursor.execute("INSERT OR REPLACE INTO sentiment_cache (ticker, date, sentiment) VALUES (?, ?, ?)", (ticker.upper(), date, daily_score))
                conn.commit()
                all_data.append({"Date": date, "Sentiment": daily_score})
                continue
            else:
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
        time.sleep(1.1)  # throttle to stay under rate limit

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

def check_volatility(df, window=20, threshold=0.03):
    recent_volatility = df["Returns"].iloc[-window:].std()
    return recent_volatility, recent_volatility > threshold

# --- STREAMLIT UI ---
st.set_page_config(page_title="Stock Price Predictor + News Sentiment", layout="centered")
st.title("🧠 Stock Price Predictor with News Sentiment")

ticker = st.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL")
horizon = st.selectbox("Predict how many days ahead:", [1, 3, 5])
model_name = st.selectbox("Choose regression model:", ["Random Forest", "Linear Regression", "XGBoost"])
show_clipped = st.toggle("🔒 Clip Next Prediction to +/-15% Range", value=True)

if ticker:
    with st.spinner("Fetching stock and news sentiment data..."):
        try:
            price_df = fetch_stock_data(ticker, horizon=horizon)
            sentiment_df = fetch_sentiment(ticker, NEWS_API_KEY, days=30)
        except Exception as e:
            st.error(f"❌ Failed to fetch data: {e}")
            st.stop()

        price_df.index = price_df.index.tz_localize(None)
        sentiment_df.index = sentiment_df.index.tz_localize(None)

        df = price_df.merge(sentiment_df, how="left", left_index=True, right_index=True)
        df["Sentiment"].fillna(0, inplace=True)

        model, X_test, y_test, predictions = train_model(df, model_name)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        volatility_value, is_volatile = check_volatility(df)

    if is_volatile:
        st.warning(f"⚠️ High volatility detected (std dev = {volatility_value:.4f})")
    else:
        st.info(f"✅ Volatility is normal (std dev = {volatility_value:.4f})")

    st.subheader("📊 Model Performance")
    st.write(f"**Mean Absolute Error (MAE):** ${mae:.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

    st.subheader("📈 Actual vs Predicted Prices")
    fig, ax = plt.subplots()
    ax.plot(y_test.index, y_test, label="Actual", marker="o")
    ax.plot(y_test.index, predictions, label="Predicted", marker="x")

    if horizon == 1:
        latest_row = df.iloc[-1:]
        X_latest = latest_row[[f"Lag{i}" for i in range(1, 6)] + [f"Sentiment_Lag{i}" for i in range(1, 6)]]
        next_date = df.index[-1] + pd.Timedelta(days=1)
        next_pred = model.predict(X_latest)[0]

        recent_close = df['Close'].iloc[-1]
        lower = recent_close * 0.85
        upper = recent_close * 1.15

        clipped_pred = np.clip(next_pred, lower, upper)

        if show_clipped and clipped_pred != next_pred:
            st.warning(f"🟠 Raw prediction (${next_pred:.2f}) clipped to ${clipped_pred:.2f} to stay within 15% of recent close (${recent_close:.2f})")
            next_pred = clipped_pred

        ax.plot([next_date], [next_pred], marker='*', color='orange', markersize=12, label='Next Predicted Price')

    ax.set_title(f"{horizon}-Day Ahead Price Prediction with {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("🧠 Sentiment Trend")
    fig2, ax2 = plt.subplots()
    ax2.plot(df.index, df["Sentiment"], color='purple', label='Daily Sentiment')
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_title("Recent News Sentiment Trend")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Sentiment Score")
    ax2.grid(True)
    ax2.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    export_df = pd.DataFrame({
        "Date": y_test.index.strftime('%Y-%m-%d'),
        "Actual_Price": y_test.values,
        "Predicted_Price": predictions
    })

    st.subheader("📤 Export Predictions")
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{ticker}_{horizon}day_{model_name.replace(' ', '_').lower()}_with_sentiment.csv",
        mime="text/csv"
    )





