import yfinance as yf
from transformers import pipeline
import matplotlib.pyplot as plt

def fetch_news_headlines(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news

        if not news:
            print("⚠️ No news found via yfinance for this ticker.")
            return []

        headlines = [item.get('title', '') for item in news if 'title' in item]
        return headlines[:10]  # Return up to 10 headlines

    except Exception as e:
        print(f"❌ Error fetching news for {ticker}: {e}")
        return []

def analyze_sentiment(headlines):
    classifier = pipeline("sentiment-analysis")
    positive = []
    negative = []

    sentiments = classifier(headlines)

    for headline, result in zip(headlines, sentiments):
        label = result['label']
        if label == 'POSITIVE':
            positive.append(headline)
        else:
            negative.append(headline)
    
    return positive, negative

def plot_sentiment(positive, negative, ticker):
    labels = ['Positive', 'Negative']
    values = [len(positive), len(negative)]

    plt.bar(labels, values)
    plt.title(f"📰 Sentiment for {ticker} News Headlines")
    plt.ylabel("Number of Headlines")
    plt.show()

def main():
    ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
    headlines = fetch_news_headlines(ticker)

    if not headlines:
        print("🚫 No headlines to analyze.")
        return

    positive, negative = analyze_sentiment(headlines)

    print("\n📈 Positive Headlines:")
    for h in positive:
        print(f"✅ {h}")

    print("\n📉 Negative Headlines:")
    for h in negative:
        print(f"❌ {h}")

    plot_sentiment(positive, negative, ticker)

if __name__ == "__main__":
    main()