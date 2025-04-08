import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import streamlit as st

nltk.download("vader_lexicon")

# --- CONFIG ---
NEWS_API_KEY = st.secrets["news_api_key"]  # You must store this in .streamlit/secrets.toml

# --- FUNCTIONS ---

def fetch_stock_data(ticker, period="120d", horizon=1):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[["Close"]]
    df["Returns"] = df["Close"].pct_change()
    df["Target"] = df["Close"].shift(-horizon)
    df.dropna(inplace=True)
    return df

def fetch_sentiment(ticker, api_key, days=120):
    sid = SentimentIntensityAnalyzer()
    all_data = []

    for i in range(days):
        date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        url = (
            f"https://newsapi.org/v2/everything?q={ticker}&from={date}&to={date}&sortBy=publishedAt&language=en&apiKey={api_key}"
        )
        response = requests.get(url)
        data = response.json()

        scores = []
        if data.get("articles"):
            for article in data["articles"]:
                title = article["title"]
                sentiment = sid.polarity_scores(title)["compound"]
                scores.append(sentiment)

        daily_score = round(sum(scores) / len(scores), 3) if scores else 0
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
    X = df[[f"Lag{i}" for i in range(1, 6)] + [f"Sentiment_Lag{i}" for i in range(1, 6)]]
    y = df["Target"]
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    if model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    elif model_name == "Linear Regression":
        model = LinearRegression()
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
model_name = st.selectbox("Choose regression model:", ["Random Forest", "Linear Regression"])

if ticker:
    with st.spinner("Fetching stock and news sentiment data..."):
        try:
            price_df = fetch_stock_data(ticker, horizon=horizon)
            sentiment_df = fetch_sentiment(ticker, NEWS_API_KEY, days=130)
        except Exception as e:
            st.error(f"❌ Failed to fetch data: {e}")
            st.stop()

        # Merge price + sentiment
        price_df.index = price_df.index.tz_localize(None)
        sentiment_df.index = sentiment_df.index.tz_localize(None)
        df = price_df.merge(sentiment_df, how="left", left_index=True, right_index=True)
        df["Sentiment"].fillna(0, inplace=True)
        df = prepare_data(df)

        model, X_test, y_test, predictions = train_model(df, model_name)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        volatility_value, is_volatile = check_volatility(df)

    # Results
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
        latest_returns = df[[f"Lag{i}" for i in range(1, 6)] + [f"Sentiment_Lag{i}" for i in range(1, 6)]].iloc[-1:].values
        next_date = df.index[-1] + pd.Timedelta(days=1)
        next_pred = model.predict(latest_returns)[0]
        ax.plot([next_date], [next_pred], marker='*', color='orange', markersize=12, label='Next Predicted Price')

    ax.set_title(f"{horizon}-Day Ahead Price Prediction with {model_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Export
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
