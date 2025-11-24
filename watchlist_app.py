import yfinance as yf
import pandas as pd
from datetime import datetime


class Watchlist:
    def __init__(self, tickers=None):
        if tickers is None:
            tickers = []
        self.tickers = list(set([t.upper() for t in tickers]))

    def add(self, ticker):
        ticker = ticker.upper()
        if ticker not in self.tickers:
            self.tickers.append(ticker)

    def remove(self, ticker):
        ticker = ticker.upper()
        if ticker in self.tickers:
            self.tickers.remove(ticker)

    def load_data(self, period="1mo", interval="1d"):
        data = []

        for t in self.tickers:
            stock = yf.Ticker(t)

            hist = stock.history(period=period, interval=interval)
            if hist.empty:
                continue

            last_price = hist["Close"].iloc[-1]

            data.append({
                "ticker": t,
                "last_price": round(last_price, 2),
            })

        return pd.DataFrame(data)
