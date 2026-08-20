"""
Price adapter for the ATR breakout strategy.

Split/dividend-adjusted daily OHLC from Tiingo, with a yfinance fallback so the
strategy keeps working without a Tiingo token. Extracted from the trend sleeve's
strategy.py so this folder is self-contained.

Tiingo is primary because it is a real API contract; yfinance is free but
intermittently serves stale or empty responses.
"""
from __future__ import annotations

import os
import time

import pandas as pd
import yfinance as yf

PRICE_SOURCE: dict[str, str] = {}   # ticker -> "Tiingo" | "yfinance", from the most recent load


def tiingo_token() -> str | None:
    """Tiingo API token from $TIINGO_API_KEY, or a gitignored local file."""
    tok = os.environ.get("TIINGO_API_KEY")
    if tok and tok.strip():
        return tok.strip()
    for p in (".tiingo_token", os.path.join("data", "tiingo_token.txt")):
        if os.path.exists(p):
            with open(p) as f:
                t = f.read().strip()
                if t:
                    return t
    return None


def tiingo_prices(ticker: str, start: str, end: str | None = None,
                  strict: bool = False) -> pd.DataFrame | None:
    """Adjusted daily OHLC from Tiingo, or None if unavailable (no token, bad
    response, or error) so the caller can fall back to yfinance."""
    tok = tiingo_token()
    if not tok:
        if strict:
            raise RuntimeError("Tiingo API token is not configured")
        return None
    import requests
    params = {"startDate": start, "token": tok, "format": "json"}
    if end:
        params["endDate"] = end
    try:
        r = requests.get(f"https://api.tiingo.com/tiingo/daily/{ticker}/prices",
                         params=params, timeout=30)
        if r.status_code != 200:
            if strict:
                detail = ""
                try:
                    detail = str(r.json().get("detail", ""))
                except Exception:  # noqa: BLE001
                    pass
                suffix = f": {detail}" if detail else ""
                raise RuntimeError(f"Tiingo returned HTTP {r.status_code}{suffix}")
            return None
        rows = r.json()
        if not rows:
            if strict:
                raise RuntimeError(f"Tiingo returned no price history for {ticker}")
            return None
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
        out = pd.DataFrame({"Open": df["adjOpen"], "High": df["adjHigh"],
                            "Low": df["adjLow"], "Close": df["adjClose"]}, index=df.index)
        return out.sort_index().dropna()
    except Exception as exc:  # noqa: BLE001
        if strict:
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(f"Tiingo request failed for {ticker}: {exc}") from exc
        return None


def load_prices(ticker: str, start: str, end: str | None = None, retries: int = 3,
                provider: str = "auto") -> pd.DataFrame:
    """Adjusted OHLC for one ticker: Tiingo first, then yfinance with backoff."""
    if provider not in ("auto", "tiingo", "yfinance"):
        raise ValueError(f"Unknown data provider: {provider}")
    if provider == "tiingo":
        out = tiingo_prices(ticker, start, end, strict=True)
        PRICE_SOURCE[ticker] = "Tiingo"
        return out
    if provider == "auto":
        tg = tiingo_prices(ticker, start, end)
        if tg is not None and not tg.empty:
            PRICE_SOURCE[ticker] = "Tiingo"
            return tg
    last = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                PRICE_SOURCE[ticker] = "yfinance"
                return df[["Open", "High", "Low", "Close"]].dropna()
        except Exception as exc:  # noqa: BLE001
            last = exc
        time.sleep(1.5 * (attempt + 1))                    # back off on rate limits
    raise ValueError(f"No data for {ticker!r} after {retries} tries ({last})")
