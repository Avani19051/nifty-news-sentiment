"""Nifty 50 stocks tracked by the project, with the names news headlines use for them.

Each entry maps an NSE symbol (Yahoo Finance format, ``.NS`` suffix) to the
lower-case aliases that identify the company in a headline. Aliases are
matched as whole words, so "itc" does not match "pitch".
"""

TICKERS: dict[str, list[str]] = {
    "RELIANCE.NS": ["reliance industries", "reliance", "ril", "jio"],
    "TCS.NS": ["tcs", "tata consultancy"],
    "HDFCBANK.NS": ["hdfc bank"],
    "INFY.NS": ["infosys", "infy"],
    "ICICIBANK.NS": ["icici bank", "icici"],
    "HINDUNILVR.NS": ["hindustan unilever", "hul"],
    "ITC.NS": ["itc"],
    "SBIN.NS": ["sbi", "state bank of india"],
    "BHARTIARTL.NS": ["bharti airtel", "airtel"],
    "KOTAKBANK.NS": ["kotak mahindra bank", "kotak bank", "kotak"],
    "LT.NS": ["larsen & toubro", "larsen and toubro", "l&t"],
    "AXISBANK.NS": ["axis bank"],
    "BAJFINANCE.NS": ["bajaj finance"],
    "MARUTI.NS": ["maruti suzuki", "maruti"],
    "ASIANPAINT.NS": ["asian paints"],
    "HCLTECH.NS": ["hcl tech", "hcltech", "hcl technologies"],
    "SUNPHARMA.NS": ["sun pharma", "sun pharmaceutical"],
    "TATAMOTORS.NS": ["tata motors"],
    "TITAN.NS": ["titan"],
    "WIPRO.NS": ["wipro"],
    "ADANIENT.NS": ["adani enterprises", "adani"],
    "ULTRACEMCO.NS": ["ultratech cement", "ultratech"],
    "NTPC.NS": ["ntpc"],
    "ONGC.NS": ["ongc"],
    "POWERGRID.NS": ["power grid"],
}

INDEX_SYMBOL = "^NSEI"  # Nifty 50 index, used as the market baseline


def display_name(symbol: str) -> str:
    """Short readable name for a symbol, e.g. 'RELIANCE.NS' -> 'RELIANCE'."""
    return symbol.removesuffix(".NS")
