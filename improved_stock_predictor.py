import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import streamlit as st

# Fetch data
def fetch_stock_data(ticker, period="120d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Close']]
    df['Returns'] = df['Close'].pct_change()
    df['Direction'] = np.where(df['Returns'] > 0, 1, 0)
    df.dropna(inplace=True)
    return df

# Create lag features
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



# Predict next day
def predict_next_day(model, df):
    latest_features = df['Returns'].iloc[-5:].values[::-1].reshape(1, -1)
    prob = model.predict_proba(latest_features)[0]
    prediction = model.predict(latest_features)[0]
    direction = "UP" if prediction == 1 else "DOWN"
    confidence = prob[prediction] * 100
    return direction, confidence

# Streamlit UI
st.set_page_config(page_title="Stock Predictor", layout="centered")
st.title("Short-Term Stock Direction Predictor")

ticker = st.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL")


        if ticker:
    with st.spinner("Fetching and analyzing data..."):
        try:
            df = fetch_stock_data(ticker)
        except Exception as e:
            st.error(f"❌ Failed to fetch data for '{ticker}': {e}")
            st.stop()

        df = prepare_data(df)
        model, X_test, y_test, predictions = train_model(df)
        direction, confidence = predict_next_day(model, df)

        

    st.success(f"Prediction: **{direction}** with **{confidence:.2f}%** confidence")

    st.subheader("Prediction vs Actual (Recent Days)")
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label='Actual', marker='o')
    ax.plot(predictions, label='Predicted', marker='x')
    ax.set_title(f"Prediction Accuracy for {ticker}")
    ax.set_xlabel("Days")
    ax.set_ylabel("Direction (1=Up, 0=Down)")
    ax.legend()
    ax.grid()
    st.pyplot(fig)