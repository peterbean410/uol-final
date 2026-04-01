"""
Download interval price data for an FX symbol and print the latest bar.

Usage:
    python marketdata/usecases/download-interval-price-data.py
"""

from marketdata.pricedata.price_bars import get_price_bars

if __name__ == "__main__":
    df = get_price_bars("USDJPY", time_interval=15)
    if df.empty:
        print("No data returned.")
    else:
        print(df.iloc[-1].to_string())
