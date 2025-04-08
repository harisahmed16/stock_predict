import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import streamlit as st

# Fetch stock data
def fetch_stock_data(ticker, period="120d", horizon=1):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    df = df[['Close']]
    df['Returns'] = df['Close'].pct_change()
    df['Target'] = df['Close'].shift(-horizon)
    df.dropna(inplace=True)
    return df

# Prepare lag features
def prepare_data(df):
    for i in range(1, 6):
        df[f'Lag{i}'] = df['Returns'].shift(i)
    df.dropna(inplace=True)
    return df

# Train model and return predictions
def train_model(df, model_name):
    X = df[[f'Lag{i}' for i in range(1, 6)]]
    y = df['Target']
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

# Volatility check
def check_volatility(df, window=20, threshold=0.03):
    recent_volatility = df['Returns'].iloc[-window:].std()
    return recent_volatility, recent_volatility > threshold

# Streamlit UI
st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 Future Stock Price Predictor")

ticker = st.text_input("Enter stock ticker (e.g., AAPL)", value="AAPL")
horizon = st.selectbox("Predict how many days ahead:", [1, 3, 5])
model_name = st.selectbox("Choose regression model:", ["Random Forest", "Linear Regression"])

if ticker:
    with st.spinner("Fetching and analyzing data..."):
        try:
            df = fetch_stock_data(ticker, horizon=horizon)
        except Exception as e:
            st.error(f"❌ Failed to fetch data for '{ticker}': {e}")
            st.stop()

        df = prepare_data(df)
        model, X_test, y_test, predictions = train_model(df, model_name)

        # Volatility check
        volatility_value, is_volatile = check_volatility(df)

        # Evaluate performance
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

    # Display volatility
    if is_volatile:
        st.warning(f"⚠️ High volatility detected (std dev = {volatility_value:.4f}). Price predictions may vary more.")
    else:
        st.info(f"✅ Volatility level is normal (std dev = {volatility_value:.4f}).")

    # Show performance metrics
    st.subheader("📊 Model Performance Metrics")
    st.write(f"**Mean Absolute Error (MAE):** ${mae:.2f}")
    st.write(f"**R² Score:** {r2:.4f}")

    # Plot actual vs predicted prices
    st.subheader("📈 Predicted vs Actual Prices")
    fig, ax = plt.subplots()
    ax.plot(y_test.index, y_test, label="Actual Price", marker='o')
    ax.plot(y_test.index, predictions, label="Predicted Price", marker='x')
    ax.set_title(f"{horizon}-Day Ahead Price Prediction using {model_name} for {ticker}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Export predictions to CSV
    export_df = pd.DataFrame({
        'Date': y_test.index.strftime('%Y-%m-%d'),
        'Actual_Price': y_test.values,
        'Predicted_Price': predictions
    })

    st.subheader("📤 Export Predictions")
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{ticker}_{horizon}day_{model_name.replace(' ', '_').lower()}_price_predictions.csv",
        mime='text/csv'
    )