"""Daily NSE prices from Yahoo Finance, turned into market-adjusted returns."""

from __future__ import annotations

import pandas as pd

from .tickers import INDEX_SYMBOL


def fetch_closes(symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Daily closing prices, one column per symbol plus the Nifty 50 index."""
    import yfinance as yf

    data = yf.download(
        sorted(set(symbols) | {INDEX_SYMBOL}),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    closes = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    return closes.dropna(how="all")


def abnormal_returns(closes: pd.DataFrame) -> pd.DataFrame:
    """Each stock's daily % return minus the Nifty 50's return that day.

    Subtracting the index removes market-wide moves, so what is left is the
    part of the move specific to that company, which is what company news
    should explain.
    """
    rets = closes.pct_change() * 100
    market = rets[INDEX_SYMBOL]
    return rets.drop(columns=INDEX_SYMBOL).sub(market, axis=0).iloc[1:]
