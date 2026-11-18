"""End-to-end run: collect headlines -> FinBERT scores -> join prices -> report.

    python run_pipeline.py collect            # append today's headlines to data/headlines.csv
    python run_pipeline.py analyse            # score everything collected so far and report
    python run_pipeline.py all                # both

RSS feeds only show the last day or two of news, so the dataset grows by
running ``collect`` regularly (the GitHub Action in this repo does it daily).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from nifty_sentiment import analysis, news, prices  # noqa: E402

DATA = Path("data")
HEADLINES = DATA / "headlines.csv"
SCORED = DATA / "scored.csv"
OUT = Path("outputs")


def collect() -> None:
    DATA.mkdir(exist_ok=True)
    fresh = news.fetch_headlines()
    if HEADLINES.exists():
        old = pd.read_csv(HEADLINES, parse_dates=["published"])
        fresh = pd.concat([old, fresh]).drop_duplicates("title", keep="first")
    fresh.sort_values("published").to_csv(HEADLINES, index=False)
    tagged = news.tag_headlines(fresh)
    print(f"{len(fresh)} headlines stored, {len(tagged)} stock mentions across {tagged['symbol'].nunique()} stocks")


def analyse(batch_size: int) -> None:
    from nifty_sentiment.sentiment import FinBertScorer, score_headlines

    raw = pd.read_csv(HEADLINES)
    raw["published"] = pd.to_datetime(raw["published"], utc=True).dt.tz_convert(news.IST)
    tagged = news.tag_headlines(raw)
    if tagged.empty:
        sys.exit("No headlines mention a tracked stock yet. Run `collect` a few times first.")

    scored = score_headlines(tagged, FinBertScorer(batch_size=batch_size))
    scored.to_csv(SCORED, index=False)

    daily = analysis.daily_sentiment(scored)
    start = (daily["date"].min() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    closes = prices.fetch_closes(daily["symbol"].unique().tolist(), start=start)
    joined = analysis.join_returns(daily, prices.abnormal_returns(closes))

    OUT.mkdir(exist_ok=True)
    joined.to_csv(OUT / "daily_sentiment_vs_returns.csv", index=False)
    summary = analysis.summarise(joined)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(analysis.by_bucket(joined).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["collect", "analyse", "all"])
    p.add_argument("--batch_size", type=int, default=32)
    a = p.parse_args()
    if a.step in ("collect", "all"):
        collect()
    if a.step in ("analyse", "all"):
        analyse(a.batch_size)


if __name__ == "__main__":
    main()
