import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import streamlit as st

# Fetch stock data
def fetch_stock_data(ticker, period="120d", horizon=1):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Close']]
    df['Returns'] = df['Close'].pct_change()
    df['Direction'] = np.where(df['Close'].shift(-horizon) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    return df

# Prepare lag features
def prepare_data(df):
    for i in range(1, 6):
        df[f'Lag{i}'] = df['Returns'].shift(i)
    df.dropna(inplace=True)
    return df

# Train model
def train_model(df):
    X = df[[f'Lag{i}' for i in range(1, 6)]]
    y = df['Direction']
    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, X_test, y_test, predictions

# Predict future direction
def predict_next_day(model, df):
    latest_features = df['Returns'].iloc[-5:].values[::-1].reshape(1, -1)
    prob = model.predict_proba(latest_features)[0]
    prediction = model.predict(latest_features)[0]
    direction = "UP" if prediction == 1 else "DOWN"
    confidence = prob[prediction] * 100
    return direction, confidence

# Volatility check
def check_volatility(df, window=20, threshold=0.03):
    recent_volatility = df['Returns'].iloc[-window:].std()
    return recent_volatility, recent_volatility > threshold

# Streamlit UI
st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("📈 Short-Term Stock Direction Predictor")

ticker = st.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL")
horizon = st.selectbox("Prediction horizon (days ahead):", [1, 3, 5])

if ticker:
    with st.spinner("Fetching and analyzing data..."):
        try:
            df = fetch_stock_data(ticker, horizon=horizon)
        except Exception as e:
            st.error(f"❌ Failed to fetch data for '{ticker}': {e}")
            st.stop()

        df = prepare_data(df)
        model, X_test, y_test, predictions = train_model(df)

        # Volatility check
        volatility_value, is_volatile = check_volatility(df)

        # Make prediction
        direction, confidence = predict_next_day(model, df)

    # Display prediction
    st.success(f"📊 {horizon}-Day Prediction: **{direction}** with **{confidence:.2f}%** confidence")

    # Display volatility warning/info
    if is_volatile:
        st.warning(f"⚠️ High volatility detected (std dev = {volatility_value:.4f}). Prediction may be less reliable.")
    else:
        st.info(f"✅ Volatility level is normal (std dev = {volatility_value:.4f}).")

    # Plot results
    st.subheader("🔍 Prediction vs Actual (Recent Days)")
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label='Actual', marker='o')
    ax.plot(predictions, label='Predicted', marker='x')
    ax.set_title(f"{horizon}-Day Ahead Prediction Accuracy for {ticker}")
    ax.set_xlabel("Days")
    ax.set_ylabel("Direction (1 = Up, 0 = Down)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)