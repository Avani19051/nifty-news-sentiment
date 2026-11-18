"""Offline tests: no network, no model download. FinBERT is replaced by a stub."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nifty_sentiment import analysis, news, prices  # noqa: E402
from nifty_sentiment.sentiment import score_headlines  # noqa: E402
from nifty_sentiment.tickers import INDEX_SYMBOL  # noqa: E402


@pytest.mark.parametrize(
    "headline, expected",
    [
        ("TCS and Infosys lead IT rally", ["TCS.NS", "INFY.NS"]),
        ("HDFC Bank Q2 profit beats estimates", ["HDFCBANK.NS"]),
        ("ITC shares slip after GST hike", ["ITC.NS"]),
        ("Startup makes a pitch to investors", []),  # 'itc' inside 'pitch'
        ("L&T bags order worth Rs 5,000 crore", ["LT.NS"]),
        ("Rupee weakens against dollar", []),
    ],
)
def test_match_tickers(headline, expected):
    assert sorted(news.match_tickers(headline)) == sorted(expected)


def test_tag_headlines_explodes_multi_stock():
    df = pd.DataFrame({"title": ["TCS and Wipro fall", "Gold steady"], "published": [None, None]})
    tagged = news.tag_headlines(df)
    assert sorted(tagged["symbol"]) == ["TCS.NS", "WIPRO.NS"]


def test_trading_date_after_close_and_weekend():
    ts = pd.Series(
        pd.to_datetime(
            [
                "2026-11-16 10:00",  # Mon, market hours -> Mon
                "2026-11-16 16:00",  # Mon, after close -> Tue
                "2026-11-20 18:00",  # Fri, after close -> Mon
                "2026-11-21 09:00",  # Sat -> Mon
            ]
        ).tz_localize(news.IST)
    )
    got = news.trading_date(ts).dt.strftime("%Y-%m-%d").tolist()
    assert got == ["2026-11-16", "2026-11-17", "2026-11-23", "2026-11-23"]


class StubScorer:
    """Positive if 'beats' or 'jumps' is in the text, negative if 'falls', else neutral."""

    def __call__(self, texts):
        out = []
        for t in texts:
            t = t.lower()
            if "beats" in t or "jumps" in t:
                out.append([0.9, 0.05, 0.05])
            elif "falls" in t:
                out.append([0.05, 0.9, 0.05])
            else:
                out.append([0.1, 0.1, 0.8])
        return np.array(out)


def test_score_headlines_columns_and_range():
    df = pd.DataFrame({"title": ["Infosys beats estimates", "Wipro falls 3%", "TCS AGM today"]})
    s = score_headlines(df, StubScorer())
    assert list(s["label"]) == ["positive", "negative", "neutral"]
    assert s["score"].between(-1, 1).all()
    assert s.loc[0, "score"] == pytest.approx(0.85)


def _closes():
    idx = pd.bdate_range("2026-11-16", periods=4)
    return pd.DataFrame(
        {
            "INFY.NS": [100, 103, 103, 100],  # +3%, 0%, -2.9%
            "WIPRO.NS": [50, 50, 49, 49],
            INDEX_SYMBOL: [1000, 1010, 1010, 1010],  # +1%, 0%, 0%
        },
        index=idx,
    )


def test_abnormal_returns_subtracts_index():
    ab = prices.abnormal_returns(_closes())
    assert INDEX_SYMBOL not in ab.columns
    assert ab.loc["2026-11-17", "INFY.NS"] == pytest.approx(2.0)  # 3% - 1%


def test_end_to_end_join_and_summary():
    scored = score_headlines(
        pd.DataFrame(
            {
                "title": ["Infosys jumps on deal", "Wipro falls on weak guidance", "Infosys falls on downgrade"],
                "symbol": ["INFY.NS", "WIPRO.NS", "INFY.NS"],
                "published": pd.to_datetime(
                    ["2026-11-17 10:00", "2026-11-18 11:00", "2026-11-19 12:00"]
                ).tz_localize(news.IST),
            }
        ),
        StubScorer(),
    )
    daily = analysis.daily_sentiment(scored)
    joined = analysis.join_returns(daily, prices.abnormal_returns(_closes()))
    assert len(joined) == 3
    s = analysis.summarise(joined)
    assert s["hit_rate"] == 1.0  # every tone matched the move's direction
    buckets = analysis.by_bucket(joined).set_index("bucket")
    assert buckets.loc["positive", "mean_ret"] > 0 > buckets.loc["negative", "mean_ret"]


def test_holiday_news_moves_to_next_session():
    daily = pd.DataFrame(
        {"symbol": ["INFY.NS"], "date": [pd.Timestamp("2026-11-18")], "sentiment": [0.5], "headlines": [2]}
    )
    sessions = pd.DatetimeIndex(["2026-11-17", "2026-11-19"])  # 18th is a holiday
    out = analysis.align_to_sessions(daily, sessions)
    assert out["date"].iloc[0] == pd.Timestamp("2026-11-19")
