"""Join daily news sentiment with stock moves and measure how well it lines up."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .news import trading_date


def daily_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    """Average headline score per (stock, trading date), with headline counts."""
    df = scored.copy()
    df["date"] = trading_date(df["published"])
    return (
        df.groupby(["symbol", "date"])
        .agg(sentiment=("score", "mean"), headlines=("title", "count"))
        .reset_index()
    )


def align_to_sessions(daily: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    """Move dates that fall on exchange holidays to the next real session."""
    sessions = pd.DatetimeIndex(sorted(sessions))
    pos = sessions.searchsorted(daily["date"].values, side="left")
    keep = pos < len(sessions)
    out = daily.loc[keep].copy()
    out["date"] = sessions[pos[keep]]
    # Re-aggregate in case a holiday's news merged into the next session.
    out["weighted"] = out["sentiment"] * out["headlines"]
    out = out.groupby(["symbol", "date"], as_index=False)[["weighted", "headlines"]].sum()
    out["sentiment"] = out["weighted"] / out["headlines"]
    return out.drop(columns="weighted")


def join_returns(daily: pd.DataFrame, abnormal: pd.DataFrame) -> pd.DataFrame:
    """Attach the same-day and next-day abnormal return to each sentiment row."""
    long = abnormal.stack().rename("ret").reset_index()
    long.columns = ["date", "symbol", "ret"]
    nxt = abnormal.shift(-1).stack().rename("next_ret").reset_index()
    nxt.columns = ["date", "symbol", "next_ret"]
    daily = align_to_sessions(daily, abnormal.index)
    return daily.merge(long, on=["date", "symbol"], how="inner").merge(nxt, on=["date", "symbol"], how="left")


def summarise(joined: pd.DataFrame, min_headlines: int = 1) -> dict:
    """Headline numbers for the README and dashboard.

    * corr_same_day / corr_next_day: Pearson correlation of sentiment with
      the abnormal return on that day / the next session.
    * hit_rate: share of days where the sign of sentiment matched the sign
      of the same-day abnormal return (0.5 is a coin flip). Days with
      sentiment within ±0.05 of zero are left out as "no view".
    """
    df = joined[joined["headlines"] >= min_headlines]
    opinionated = df[df["sentiment"].abs() > 0.05]
    hits = np.sign(opinionated["sentiment"]) == np.sign(opinionated["ret"])

    def corr(a, b):
        ok = a.notna() & b.notna()
        return float(a[ok].corr(b[ok])) if ok.sum() >= 3 else float("nan")

    return {
        "stock_days": int(len(df)),
        "corr_same_day": corr(df["sentiment"], df["ret"]),
        "corr_next_day": corr(df["sentiment"], df["next_ret"]),
        "hit_rate": float(hits.mean()) if len(opinionated) else float("nan"),
        "opinionated_days": int(len(opinionated)),
    }


def by_bucket(joined: pd.DataFrame) -> pd.DataFrame:
    """Mean abnormal return for negative / neutral / positive news days."""
    bins = [-1.01, -0.05, 0.05, 1.01]
    labels = ["negative", "neutral", "positive"]
    df = joined.assign(bucket=pd.cut(joined["sentiment"], bins=bins, labels=labels))
    return (
        df.groupby("bucket", observed=False)
        .agg(days=("ret", "size"), mean_ret=("ret", "mean"), mean_next_ret=("next_ret", "mean"))
        .reset_index()
    )
