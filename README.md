# 🧠 Stock Price Predictor with News Sentiment

A Streamlit app that predicts short-term stock price movement by combining historical price data with daily news sentiment using machine learning.

Live app: [https://stockpredict-lm5tdacayuqg2nypp4qdgn.streamlit.app](https://stockpredict-lm5tdacayuqg2nypp4qdgn.streamlit.app)

---

## 🚀 Features

- 📈 Predict 1, 3, or 5 days ahead using:
  - Random Forest or Linear Regression
- 🧠 Integrates news sentiment from NewsAPI (cached with SQLite)
- ⭐️ Highlights next-day predicted price with visual markers
- 🟪 Sentiment trend chart under price predictions
- 📊 MAE and R² performance metrics
- 🔒 Clipping toggle to prevent outlier predictions
- 📤 CSV export of results
- ⚠️ Volatility warning system

---

## 🛠️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/stock-predictor.git
cd stock-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your NewsAPI key
Create a `.streamlit/secrets.toml` file:
```toml
news_api_key = "your_newsapi_key_here"
```

> 🔑 Get a free key from [https://newsapi.org](https://newsapi.org)

### 4. Run the app
```bash
streamlit run improved_stock_predictor.py
```

---

## 💾 SQLite Sentiment Caching
- Caches each ticker-date sentiment combo to avoid hitting API rate limits
- Uses `/tmp/sentiment_cache.db` (Streamlit Cloud compatible)

---

## 📷 Screenshots

*Coming soon!* (Include images of your chart, sentiment trend, and clipped prediction star.)

---

## 📦 Tech Stack
- Python, Streamlit
- yFinance, scikit-learn, NLTK, NewsAPI
- SQLite (local DB)

---

## 📌 Roadmap Ideas

- Add XGBoost model
- Compare predictions with and without sentiment
- View or clear cached sentiment
- Email/Telegram alerts for daily predictions

---

## 📄 License
MIT

---

## 🙌 Credits
Built by Haris Ahmed


