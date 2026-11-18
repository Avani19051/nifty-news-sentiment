"""Streamlit dashboard: news sentiment vs. stock moves for Nifty 50 companies.

    streamlit run app.py

Reads the files written by ``python run_pipeline.py analyse``. There is also
a "Try a headline" box that scores any text you type with FinBERT.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nifty_sentiment import analysis  # noqa: E402
from nifty_sentiment.tickers import display_name  # noqa: E402

st.set_page_config(page_title="Nifty News Sentiment", layout="wide")
st.title("Nifty 50 news sentiment")
st.caption("Headline sentiment from FinBERT (ProsusAI/finbert) compared with market-adjusted daily returns.")


@st.cache_resource
def load_scorer():
    from nifty_sentiment.sentiment import FinBertScorer

    return FinBertScorer()


with st.expander("Try a headline", expanded=False):
    text = st.text_input("Headline", "Infosys raises FY guidance after strong deal wins")
    if text:
        p = load_scorer()([text])[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Positive", f"{p[0]:.0%}")
        c2.metric("Negative", f"{p[1]:.0%}")
        c3.metric("Neutral", f"{p[2]:.0%}")

joined_path = Path("outputs/daily_sentiment_vs_returns.csv")
scored_path = Path("data/scored.csv")
if not joined_path.exists():
    st.info("No results yet. Run `python run_pipeline.py all`, then reload this page.")
    st.stop()

joined = pd.read_csv(joined_path, parse_dates=["date"])
scored = pd.read_csv(scored_path, parse_dates=["published"]) if scored_path.exists() else None
summary = json.loads(Path("outputs/summary.json").read_text())

m1, m2, m3, m4 = st.columns(4)
m1.metric("Stock-days with news", summary["stock_days"])
m2.metric("Same-day correlation", f"{summary['corr_same_day']:.2f}")
m3.metric("Next-day correlation", f"{summary['corr_next_day']:.2f}")
m4.metric("Direction hit rate", f"{summary['hit_rate']:.0%}", help="0.50 = coin flip")

st.subheader("Average move by news tone")
buckets = analysis.by_bucket(joined).set_index("bucket")
st.bar_chart(buckets[["mean_ret", "mean_next_ret"]].rename(columns={"mean_ret": "Same day %", "mean_next_ret": "Next day %"}))

st.subheader("One stock")
symbols = sorted(joined["symbol"].unique(), key=lambda s: -joined.loc[joined.symbol == s, "headlines"].sum())
pick = st.selectbox("Stock", symbols, format_func=display_name)
one = joined[joined["symbol"] == pick].set_index("date").sort_index()
st.line_chart(one[["sentiment"]].rename(columns={"sentiment": "News sentiment (-1 to 1)"}))
st.bar_chart(one[["ret"]].rename(columns={"ret": "Abnormal return %"}))

if scored is not None:
    st.subheader(f"Latest {display_name(pick)} headlines")
    rows = scored[scored["symbol"] == pick].sort_values("published", ascending=False).head(15)
    st.dataframe(rows[["published", "source", "title", "label", "score"]], hide_index=True, use_container_width=True)
