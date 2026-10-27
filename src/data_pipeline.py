
"""Download, align, validate prices and compute daily log returns."""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf
from src.config import TICKERS, START_DATE, END_DATE, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, ensure_directories

def download_adjusted_prices(start: str = START_DATE, end: str = END_DATE, refresh: bool = False) -> pd.DataFrame:
    ensure_directories()
    cache = RAW_DIR / "bank_adjusted_prices_2014_2024.csv"
    if cache.exists() and not refresh:
        prices = pd.read_csv(cache, index_col=0, parse_dates=True)
        prices.index = pd.to_datetime(prices.index)
        return prices.sort_index()
    downloaded = yf.download(
        list(TICKERS.values()), start=start, end=end,
        auto_adjust=True, progress=False, group_by="column", threads=True
    )
    if downloaded.empty:
        raise RuntimeError("Yahoo Finance returned no prices. Check your internet connection or try again later.")
    # yfinance may return MultiIndex columns; with auto_adjust=True, Close is adjusted close.
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" in downloaded.columns.get_level_values(0):
            close = downloaded["Close"].copy()
        else:
            close = downloaded.xs("Close", axis=1, level=1)
    else:
        close = downloaded[["Close"]].copy()
        if len(TICKERS) == 1:
            close.columns = list(TICKERS.values())
    close = close.rename(columns={ticker: name for name, ticker in TICKERS.items()})
    missing = [name for name in TICKERS if name not in close.columns]
    if missing:
        raise RuntimeError(f"Missing ticker columns from Yahoo Finance response: {missing}. Returned: {list(close.columns)}")
    close = close[list(TICKERS)].sort_index()
    close.index = pd.to_datetime(close.index).tz_localize(None) if getattr(close.index, "tz", None) else pd.to_datetime(close.index)
    close.to_csv(cache, index_label="Date")
    return close

def build_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Inner-align common trading dates and compute log(P_t/P_(t-1))."""
    aligned = prices[list(TICKERS)].dropna(how="any").sort_index()
    aligned.to_csv(INTERIM_DIR / "aligned_adjusted_prices.csv", index_label="Date")
    returns = np.log(aligned / aligned.shift(1)).dropna()
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if returns.empty or returns.isna().any().any():
        raise ValueError("Returns are empty or contain missing values after cleaning.")
    if (returns.std() == 0).any():
        raise ValueError("A zero-variance return series was detected.")
    returns.to_csv(PROCESSED_DIR / "bank_log_returns_2014_2024.csv", index_label="Date")
    return returns

def load_or_build_returns(refresh: bool = False) -> pd.DataFrame:
    ensure_directories()
    processed = PROCESSED_DIR / "bank_log_returns_2014_2024.csv"
    if processed.exists() and not refresh:
        df = pd.read_csv(processed, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        if all(name in df.columns for name in TICKERS):
            return df[list(TICKERS)].dropna()
    prices = download_adjusted_prices(refresh=refresh)
    return build_returns(prices)
