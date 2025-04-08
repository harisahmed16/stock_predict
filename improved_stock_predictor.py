import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Function to fetch data
def fetch_stock_data(ticker, period="300d"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Close']]
    df['Returns'] = df['Close'].pct_change()
    df['Direction'] = np.where(df['Returns'] > 0, 1, 0)
    df.dropna(inplace=True)
    return df

# Prepare dataset with additional lag features
def prepare_data(df):
    for i in range(1, 6):
        df[f'Lag{i}'] = df['Returns'].shift(i)
    df.dropna(inplace=True)
    return df

# Train Random Forest model with hyperparameter tuning
def train_model(df):
    X = df[[f'Lag{i}' for i in range(1, 6)]]
    y = df['Direction']

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    return model, X_test, y_test, predictions

# Predict next-day direction with improved confidence estimation
def predict_next_day(model, df):
    latest_features = df[['Returns']].iloc[-5:].values.flatten()[::-1].reshape(1, -1)
    prob = model.predict_proba(latest_features)[0]
    prediction = model.predict(latest_features)[0]

    direction = "UP" if prediction == 1 else "DOWN"
    confidence = prob[prediction] * 100

    print(f"\nNext-day prediction: {direction} with {confidence:.2f}% confidence.")

# Main function
import streamlit as st

def main():
    st.title("Short-Term Stock Direction Predictor")
    ticker = st.text_input("Enter stock ticker (e.g., AAPL):", value="AAPL")

    if ticker:
        df = fetch_stock_data(ticker)
        df = prepare_data(df)
        model, X_test, y_test, predictions = train_model(df)
        predict_next_day(model, df)

        st.subheader("Actual vs. Predicted Directions")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(y_test.values, label='Actual', marker='o')
        ax.plot(predictions, label='Predicted', marker='x')
        ax.set_xlabel('Days')
        ax.set_ylabel('Direction (1=Up, 0=Down)')
        ax.set_title(f'Short-term Direction Prediction for {ticker}')
        ax.legend()
        ax.grid()
        st.pyplot(fig)
if __name__ == "__main__":
    main()

    