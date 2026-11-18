"""Collect Indian market headlines from public RSS feeds and tag them with stocks."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import pandas as pd

from .tickers import TICKERS

IST = ZoneInfo("Asia/Kolkata")

# Public market-news RSS feeds. Any feed with <item><title><pubDate> works.
FEEDS: dict[str, str] = {
    "economic_times": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "moneycontrol": "https://www.moneycontrol.com/rss/business.xml",
    "livemint": "https://www.livemint.com/rss/markets",
    "business_standard": "https://www.business-standard.com/rss/markets-106.rss",
}

_PATTERNS = {
    symbol: re.compile(r"(?<![\w&])(" + "|".join(re.escape(a) for a in aliases) + r")(?![\w&])")
    for symbol, aliases in TICKERS.items()
}


def match_tickers(headline: str) -> list[str]:
    """Return every tracked symbol whose company is named in the headline."""
    text = headline.lower()
    return [symbol for symbol, pattern in _PATTERNS.items() if pattern.search(text)]


def _parse_date(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        try:
            dt = datetime.fromisoformat(raw)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


def fetch_headlines(feeds: dict[str, str] | None = None) -> pd.DataFrame:
    """Download the feeds and return one row per headline.

    Columns: source, published (IST), title, link.
    A feed that fails to download is skipped with a warning, so one site
    being down does not stop the run.
    """
    import feedparser  # imported here so tests don't need it

    rows = []
    for source, url in (feeds or FEEDS).items():
        parsed = feedparser.parse(url)
        if parsed.bozo and not parsed.entries:
            print(f"warning: could not read {source} ({parsed.bozo_exception})")
            continue
        for entry in parsed.entries:
            rows.append(
                {
                    "source": source,
                    "published": _parse_date(entry.get("published") or entry.get("updated")),
                    "title": (entry.get("title") or "").strip(),
                    "link": entry.get("link", ""),
                }
            )
    df = pd.DataFrame(rows, columns=["source", "published", "title", "link"])
    return df.dropna(subset=["published"]).query("title != ''").drop_duplicates("title")


def tag_headlines(df: pd.DataFrame) -> pd.DataFrame:
    """Explode headlines into one row per (headline, stock) mention.

    Headlines that name no tracked stock are dropped. A headline naming two
    stocks (e.g. "TCS and Infosys lead IT rally") counts for both.
    """
    out = df.copy()
    out["symbol"] = out["title"].map(match_tickers)
    out = out.explode("symbol").dropna(subset=["symbol"])
    return out.reset_index(drop=True)


def trading_date(published: pd.Series) -> pd.Series:
    """Map each timestamp to the trading session it can first affect.

    NSE closes at 15:30 IST. News after the close (or on a weekend) can only
    move the price on the next weekday, so it is assigned to that date.
    Exchange holidays are handled later, when dates are aligned to the
    actual price index.
    """
    ts = pd.to_datetime(published)
    after_close = (ts.dt.hour * 60 + ts.dt.minute) >= 15 * 60 + 30
    day = ts.dt.tz_localize(None).dt.normalize() + pd.to_timedelta(after_close.astype(int), unit="D")
    return day.map(lambda d: d + pd.offsets.BDay(0))
