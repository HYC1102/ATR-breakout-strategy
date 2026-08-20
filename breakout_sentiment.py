"""
Breakout + sentiment-ranked, slot-based swing strategy.

Idea: trade the most liquid US large caps. Each day, scan for fresh breakouts,
rank the day's breakout candidates by "sentiment", and hold the top few in a
small number of slots until each hits a stop/exit.

Phase 1 (this file, backtestable now): the ranking uses a PRICE-MOMENTUM PROXY
in place of real sentiment, so we can validate the breakout + slot + ATR-stop
machine honestly. Phase 2 swaps in a live news->LLM bullishness score.

This module is the FOUNDATION: build the universe, scan today's breakouts, and
rank the candidates. The event-driven backtester is added on top of these pieces.

Honest-data caveats:
  * Universe is TODAY's S&P 500 members -> survivorship bias (optimistic).
  * The momentum proxy is not sentiment; it only tests the mechanics.
"""
from __future__ import annotations

import io
import json
import os
import time
import pickle
import datetime as dt

import numpy as np
import pandas as pd
import requests
import yfinance as yf

CONFIG = dict(
    pool="broad",           # candidate pool: S&P 500 + Nasdaq-100 + liquid extras
    universe_size=223,      # top-N US stocks by average dollar volume
    rebuild="W",            # universe rebuild cadence: "W" weekly | "M" monthly
    adv_window=60,          # days for the ADV ranking
    entry="atr",            # entry signal: "atr" = Keltner-style ATR-band breakout | "donchian"
    breakout=20,            # Donchian high lookback (fresh N-day-high breakout) [donchian entry]
    atr_break_period=20,    # ATR-band entry: EMA basis + ATR lookback [atr entry]
    atr_break_mult=3.0,     # ATR-band entry: band width in ATRs [atr entry]
    proxy_window=63,        # momentum-proxy lookback (~3 months) [Phase 1 ranking]
    sent_weight=0.0,        # live ranking is momentum-only; sentiment overlay disabled
    atr_window=14,          # Wilder ATR
    atr_stop=1.5,           # trailing stop distance in ATRs
    intraday_stop=False,    # False = stop checked on the close, filled next open (default)
                            # | True  = resting intraday order, filled when the low touches it
    exit_low=10,            # N-day-low exit lookback
    use_exit_low=True,      # enable the N-day-low exit
    pct_stop=None,          # hard stop: exit if price <= entry*(1-pct_stop) (None = off)
    slots=5,                # concurrent positions
    sizing="slots",         # "slots" = independent 1/N slots, no rebalancing (default)
                            # | "full" = re-equalize whole book each change | "risk" = ATR risk-sized
    weight_mode="fixed",    # (full mode only) "fixed"=1/slots each | "equal"=1/held | "invvol"
    risk_frac=0.01,         # capital risked per slot to its stop (sizing="risk" only)
    regime=True,            # only OPEN positions when SPY > its 200-day MA
    regime_ma=200,
    take_profit=None,       # exit if up >= this fraction from entry (None = let winners run)
    time_stop=None,         # exit if held >= this many days (None = off)
    capital=23_000.0,
    cost_bps=0.0,           # per-side trading cost (disabled)
    bt_start="2016-01-01",  # backtest start (needs ~1y prior for warmup)
    cache_dir="data",
    cache_hours=12,         # re-download prices at most twice a day
)

# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
def _wiki_tickers(url: str) -> list[str]:
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=30).text
    for tb in pd.read_html(io.StringIO(html)):
        for col in tb.columns:
            if str(col).lower() in ("symbol", "ticker"):
                return [str(t).replace(".", "-") for t in tb[col].astype(str)
                        if str(t) not in ("nan", "")]
    return []


def sp500_tickers() -> list[str]:
    """Current S&P 500 constituents from Wikipedia (survivorship-biased)."""
    return sorted(_wiki_tickers("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"))


def nasdaq100_tickers() -> list[str]:
    """Current Nasdaq-100 constituents from Wikipedia (survivorship-biased)."""
    return sorted(_wiki_tickers("https://en.wikipedia.org/wiki/Nasdaq-100"))


def _screen(query, pages: int, custom: bool,
            min_price: float = 5.0, min_dollar_vol: float = 100e6) -> dict[str, float]:
    """Run one Yahoo screen (a predefined name, or a custom EquityQuery),
    paginating 25 at a time, and map each liquid US common stock to its dollar
    volume (price x 3-month avg share volume). Best-effort: returns what it got
    if a page fails."""
    out: dict[str, float] = {}
    for off in range(0, pages * 25, 25):
        try:
            r = (yf.screen(query, sortField="dayvolume", sortAsc=False, offset=off, count=25)
                 if custom else yf.screen(query, offset=off, count=25))
        except Exception:  # noqa: BLE001
            break
        quotes = r.get("quotes", []) if isinstance(r, dict) else []
        if not quotes:
            break
        for q in quotes:
            if q.get("quoteType") != "EQUITY":                   # drop ETFs / funds
                continue
            exch = q.get("fullExchangeName") or ""
            if "Nasdaq" not in exch and "NYSE" not in exch:      # US-listed only
                continue
            px = q.get("regularMarketPrice") or 0
            vol = q.get("averageDailyVolume3Month") or q.get("regularMarketVolume") or 0
            if px >= min_price and px * vol >= min_dollar_vol:
                out[q["symbol"]] = px * vol
    return out


def _high_volume_us_stocks() -> dict[str, float]:
    """US common stocks trading >= $100M/day, spanned by complementary Yahoo
    screens -- no hand-curated list.

    Yahoo only sorts by SHARE volume, so 'most actives' alone misses high-priced
    names that trade few shares but huge dollars (ASML ~$3.4B/day, MELI ~$1B/day).
    Price-bucketed screens fix that: a name expensive enough to be thin on shares
    lands in a small, fully-paginable price bucket. Buckets: everything (most
    actives), > $100, and > $500 (only ~60 names, but that is where MELI/ASML/NVR
    live). Union, then rank by dollar volume."""
    names = _screen("most_actives", pages=14, custom=False)
    for min_px, pages in ((100, 18), (500, 6)):
        q = yf.EquityQuery("and", [
            yf.EquityQuery("gt", ["intradayprice", min_px]),
            yf.EquityQuery("gt", ["dayvolume", 100_000]),
            yf.EquityQuery("eq", ["region", "us"]),
        ])
        names.update(_screen(q, pages=pages, custom=True))
    return names


def discover_extras(cap: int = 200) -> list[str]:
    """High-volume US stocks OUTSIDE the S&P 500, discovered WEEKLY and ranked by
    dollar volume -- entirely from Yahoo's screener, no hand-curated list.

    Cached per ISO week in data/extras_YYYYWww.json (the workflow commits it, so
    it is genuinely weekly on the live system). On a total screen failure it
    returns [] and broad_universe() degrades to the S&P 500 alone (or the last
    cached week, if present). Still survivorship-biased -- today's live names."""
    yr, wk, _ = dt.date.today().isocalendar()
    cache = os.path.join(CONFIG["cache_dir"], f"extras_{yr}W{wk:02d}.json")
    if os.path.exists(cache):
        try:
            with open(cache) as f:
                return json.load(f)
        except Exception:  # noqa: BLE001
            pass
    sp = set(sp500_tickers())
    dv = _high_volume_us_stocks()
    non_sp = sorted((t for t in dv if t not in sp), key=dv.get, reverse=True)[:cap]
    if non_sp:                                       # only cache a successful screen
        os.makedirs(CONFIG["cache_dir"], exist_ok=True)
        try:
            with open(cache, "w") as f:
                json.dump(non_sp, f)
        except Exception:  # noqa: BLE001
            pass
    return non_sp


def broad_universe() -> list[str]:
    """Current S&P 500 + Nasdaq-100 + weekly-discovered liquid US extras.

    A true all-US-stocks-by-volume universe needs a full-market feed; this is the
    best free proxy. The 60-day-ADV cut in build_universe() then keeps the top
    ``universe_size`` names.
    """
    return sorted(set(sp500_tickers()) | set(nasdaq100_tickers()) | set(discover_extras()))


def download_prices(tickers: list[str], period: str = "400d",
                    pool: str | None = None) -> dict[str, pd.DataFrame]:
    """Batch-download daily OHLCV (auto-adjusted), cached to a pickle.

    `pool` names the cache file and MUST describe `tickers`. Callers that pass a
    ticker list other than the configured pool have to say so, otherwise e.g. an
    S&P-500-only pull lands in prices_broad_<period>.pkl and a later broad run
    silently reuses it, quietly dropping the non-index names."""
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    cache = os.path.join(CONFIG["cache_dir"],
                         f"prices_{pool or CONFIG['pool']}_{period}.pkl")
    if os.path.exists(cache):
        age_h = (time.time() - os.path.getmtime(cache)) / 3600
        if age_h < CONFIG["cache_hours"]:
            with open(cache, "rb") as f:
                return pickle.load(f)

    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False,
                      group_by="ticker", threads=True)
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = raw[t][["Open", "High", "Low", "Close", "Volume"]].dropna()
        except (KeyError, TypeError):
            continue
        if len(df) > CONFIG["adv_window"]:
            out[t] = df
    # Only cache a plausibly complete pull. A rate-limited or empty yfinance
    # response would otherwise be pickled and re-served for `cache_hours`,
    # leaving every run in that window with no prices and no retry.
    if len(out) >= max(1, len(tickers) // 2):
        with open(cache, "wb") as f:
            pickle.dump(out, f)
    else:
        print(f"  warning: only {len(out)}/{len(tickers)} tickers returned data; "
              f"not caching (will retry on the next run)")
    return out


def build_universe(prices: dict[str, pd.DataFrame], asof: pd.Timestamp | None = None):
    """Top-N tickers by trailing average dollar volume as of `asof`."""
    adv = {}
    for t, df in prices.items():
        d = df.loc[:asof] if asof is not None else df
        if len(d) < CONFIG["adv_window"]:
            continue
        dollar = (d["Close"] * d["Volume"]).tail(CONFIG["adv_window"]).mean()
        if np.isfinite(dollar):
            adv[t] = dollar
    ranked = sorted(adv, key=adv.get, reverse=True)
    return ranked[: CONFIG["universe_size"]]


# --------------------------------------------------------------------------- #
# Signals / indicators
# --------------------------------------------------------------------------- #
def atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Wilder ATR."""
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def is_breakout(df: pd.DataFrame, n: int) -> bool:
    """True if today triggers a fresh breakout.

    entry="atr": today's High pierces yesterday's Keltner-style upper band
    (EMA basis + mult*ATR).  entry="donchian": today's Close makes a fresh
    N-day high.  Chosen by CONFIG['entry'].
    """
    if CONFIG.get("entry", "donchian") == "atr":
        N, K = CONFIG["atr_break_period"], CONFIG["atr_break_mult"]
        if len(df) < N + 2:
            return False
        band = (df["Close"].ewm(span=N, adjust=False).mean() + K * atr(df, N)).shift(1)
        return df["High"].iloc[-1] > band.iloc[-1]
    if len(df) < n + 1:
        return False
    prior_high = df["High"].iloc[-(n + 1):-1].max()
    return df["Close"].iloc[-1] > prior_high


def momentum_proxy(df: pd.DataFrame, n: int) -> float:
    """Phase-1 stand-in for sentiment: trailing n-day total return."""
    if len(df) < n + 1:
        return np.nan
    return df["Close"].iloc[-1] / df["Close"].iloc[-(n + 1)] - 1.0


_VADER = None
_SENT_CACHE = None


def _vader():
    global _VADER
    if _VADER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER = SentimentIntensityAnalyzer()
    return _VADER


def _sent_cache():
    global _SENT_CACHE
    if _SENT_CACHE is None:
        try:
            with open(os.path.join(CONFIG["cache_dir"], "sentiment.pkl"), "rb") as f:
                _SENT_CACHE = pickle.load(f)
        except Exception:  # noqa: BLE001
            _SENT_CACHE = {}
    return _SENT_CACHE


def sentiment_score(ticker: str, df: pd.DataFrame | None = None) -> float:
    """Real news sentiment in [0, 100]: average VADER compound over the ticker's
    recent headlines (yfinance), mapped -1..1 -> 0..100. Neutral 50 when there is
    no news. Cached per (day, ticker) to disk. (Swap _vader() for an LLM call to
    upgrade to news->LLM scoring.)"""
    key = (str(dt.date.today()), ticker)
    cache = _sent_cache()
    if key in cache:
        return cache[key]
    score, titles = 50.0, []
    try:
        for it in (yf.Ticker(ticker).news or [])[:10]:
            c = it.get("content", {}) if isinstance(it, dict) else {}
            title = c.get("title") or (it.get("title") if isinstance(it, dict) else "")
            if title:
                titles.append(title)
        if titles:
            va = _vader()
            avg = sum(va.polarity_scores(t)["compound"] for t in titles) / len(titles)
            score = round((avg + 1) / 2 * 100, 1)               # -1..1 -> 0..100
    except Exception:  # noqa: BLE001
        pass
    cache[key] = score
    try:
        os.makedirs(CONFIG["cache_dir"], exist_ok=True)
        with open(os.path.join(CONFIG["cache_dir"], "sentiment.pkl"), "wb") as f:
            pickle.dump(cache, f)
    except Exception:  # noqa: BLE001
        pass
    return score


def _pct_rank(vals: pd.Series) -> pd.Series:
    """0..100 percentile rank (ties averaged)."""
    if len(vals) <= 1:
        return pd.Series(50.0, index=vals.index)
    return vals.rank(pct=True) * 100.0


def rank_breakouts(prices: dict[str, pd.DataFrame], universe: list[str],
                   asof: pd.Timestamp | None = None,
                   use_sentiment: bool = True) -> pd.DataFrame:
    """Today's fresh breakouts in `universe`, scored by a MOMENTUM index and a
    SENTIMENT score, combined into a single rank (sent_weight controls the mix)."""
    rows = []
    w = CONFIG.get("sent_weight", 0.0)
    score_sentiment = use_sentiment and w > 0
    for t in universe:
        df = prices.get(t)
        if df is not None and asof is not None:
            df = df.loc[:asof]
        if df is None or df.empty or not is_breakout(df, CONFIG["breakout"]):
            continue
        a = atr(df, CONFIG["atr_window"]).iloc[-1]
        px = float(df["Close"].iloc[-1])
        if not np.isfinite(a) or a <= 0:
            continue
        rows.append(dict(ticker=t, close=px, atr=float(a),
                         stop=px - CONFIG["atr_stop"] * float(a),
                         mom_ret=momentum_proxy(df, CONFIG["proxy_window"]),
                         sentiment=sentiment_score(t, df) if score_sentiment else 50.0))
    cols = ["ticker", "close", "atr", "stop", "mom_ret", "sentiment",
            "momentum", "combined"]
    if not rows:
        return pd.DataFrame(columns=cols)
    d = pd.DataFrame(rows)
    d["momentum"] = _pct_rank(d["mom_ret"]).values          # 0..100 momentum index
    d["combined"] = (1 - w) * d["momentum"] + w * d["sentiment"]
    return d.sort_values("combined", ascending=False).reset_index(drop=True)[cols]


# --------------------------------------------------------------------------- #
# Today's ranked candidates
# --------------------------------------------------------------------------- #
def scan_candidates(prices: dict[str, pd.DataFrame], universe: list[str],
                    capital: float) -> pd.DataFrame:
    """Fresh breakouts in the universe, ranked by the momentum proxy, with
    ATR-based size and initial stop for `capital` at `risk_frac` risk."""
    rows = []
    for t in universe:
        df = prices[t]
        if not is_breakout(df, CONFIG["breakout"]):
            continue
        a = atr(df, CONFIG["atr_window"]).iloc[-1]
        px = float(df["Close"].iloc[-1])
        if not np.isfinite(a) or a <= 0:
            continue
        stop_dist = CONFIG["atr_stop"] * a
        shares = (CONFIG["risk_frac"] * capital) / stop_dist
        rows.append(dict(
            ticker=t, close=px, proxy=momentum_proxy(df, CONFIG["proxy_window"]),
            atr=a, stop=px - stop_dist, shares=shares,
            dollars=shares * px, pct=shares * px / capital,
        ))
    cols = ["ticker", "close", "proxy", "atr", "stop", "shares", "dollars", "pct"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return (pd.DataFrame(rows).sort_values("proxy", ascending=False)
            .reset_index(drop=True)[cols])


# --------------------------------------------------------------------------- #
# Phase-1 event-driven backtest (momentum proxy in place of sentiment)
# --------------------------------------------------------------------------- #
def build_panels(prices: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Vectorised, date-aligned indicator panels (index=dates, cols=tickers)."""
    F = {f: pd.DataFrame({t: df[f] for t, df in prices.items()})
         for f in ["Open", "High", "Low", "Close", "Volume"]}
    close, high, low = F["Close"], F["High"], F["Low"]
    pc = close.shift(1)
    tr = np.maximum(high - low, np.maximum((high - pc).abs(), (low - pc).abs()))
    if CONFIG.get("entry", "donchian") == "atr":
        N, K = CONFIG["atr_break_period"], CONFIG["atr_break_mult"]
        band = close.ewm(span=N, adjust=False).mean() + K * tr.ewm(alpha=1 / N, adjust=False).mean()
        breakout = high > band.shift(1)
    else:
        breakout = close > high.rolling(CONFIG["breakout"]).max().shift(1)
    P = dict(
        open=F["Open"], close=close, high=high, low=low,
        breakout=breakout,
        proxy=close.pct_change(CONFIG["proxy_window"], fill_method=None),
        atr=tr.ewm(alpha=1 / CONFIG["atr_window"], adjust=False).mean(),
        adv=(close * F["Volume"]).rolling(CONFIG["adv_window"]).mean(),
    )
    return P


def _metrics(eq: pd.Series) -> dict:
    r = eq.pct_change().dropna(); ny = len(r) / 252
    dd = eq / eq.cummax() - 1
    return dict(cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / ny) - 1 if ny else 0,
                sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() else 0,
                vol=r.std() * np.sqrt(252), maxdd=dd.min(), end=eq.iloc[-1])


def spy_regime(index) -> pd.Series:
    """Boolean SPY>MA series aligned to `index` (precompute once for speed).

    SPY is fetched with a full MA warm-up BEFORE index[0]. Starting the download
    at index[0] would leave the rolling mean NaN for the first `regime_ma` rows;
    `spy > NaN` is False, so the regime filter would silently block every entry
    over roughly the first year of any backtest."""
    ma_n = CONFIG.get("regime_ma", 200)
    # ~1.45 calendar days per trading day, plus slack for holidays.
    warmup = pd.Timedelta(days=int(ma_n * 1.6) + 15)
    spy = yf.download("SPY", start=str((pd.Timestamp(index[0]) - warmup).date()), end=None,
                      auto_adjust=True, progress=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    ma = spy.rolling(ma_n).mean()
    if ma.reindex(index).isna().any():
        print(f"  warning: SPY {ma_n}-day MA is short of history at the start of "
              f"the window; regime is treated as risk-off there")
    return (spy > ma).reindex(index).ffill().fillna(False)


def backtest(prices=None, capital=None, P=None, regime_full=None,
             start=None, end=None) -> dict:
    """Daily slot-based simulation. Signals on the close, filled at the next open.
    Pass precomputed `P`/`regime_full` and a `start`/`end` window for speed."""
    capital = capital or CONFIG["capital"]
    if P is None:
        P = build_panels(prices)
    dates = P["close"].loc[(start or CONFIG["bt_start"]):end].index
    cost = CONFIG["cost_bps"] / 1e4
    stop_k = CONFIG["atr_stop"]
    intraday = CONFIG.get("intraday_stop", False)
    exlo = P["low"].rolling(CONFIG["exit_low"]).min().shift(1)   # swept param -> compute here
    atr_prev = P["atr"].shift(1)                                 # ATR known at day's open (no look-ahead)

    # point-in-time universe (top-N by ADV), rebuilt weekly or monthly
    weekly = CONFIG.get("rebuild", "M") == "W"
    uni_by_period = {}
    for d in dates:
        key = (d.isocalendar()[0], d.isocalendar()[1]) if weekly else (d.year, d.month)
        if key not in uni_by_period:
            adv = P["adv"].loc[:d].iloc[-1].dropna()
            uni_by_period[key] = set(adv.nlargest(CONFIG["universe_size"]).index)

    sizing = CONFIG.get("sizing", "risk")

    # optional regime filter: only OPEN new positions when SPY > its N-day MA
    regime_ok = None
    if CONFIG.get("regime"):
        regime_ok = (regime_full.reindex(dates).ffill().fillna(False)
                     if regime_full is not None else spy_regime(dates))

    def target_weights(held, at_row):
        """Target weights over the held set."""
        if not held:
            return {}
        wm = CONFIG.get("weight_mode")
        if wm == "fixed":                               # 1/slots each; empty slots -> cash
            return {t: 1.0 / CONFIG["slots"] for t in held}
        if wm == "invvol":                              # inverse ATR% (risk-parity-ish)
            iv = {t: (P["close"].loc[day].get(t, np.nan) / at_row.get(t, np.nan))
                  for t in held}
            iv = {t: v for t, v in iv.items() if np.isfinite(v) and v > 0}
            s = sum(iv.values())
            if s > 0:
                return {t: v / s for t, v in iv.items()}
        return {t: 1.0 / len(held) for t in held}       # equal weight over held

    cash = capital
    pos: dict[str, dict] = {}                  # ticker -> {shares, hi}
    pend_sell: list[str] = []; pend_buy: list[dict] = []; pend_tgt = None
    eq_curve = []; deploy = []; trades = []; holds = []

    for i, day in enumerate(dates):
        op, cl = P["open"].loc[day], P["close"].loc[day]

        # 1) execute yesterday's signals at today's open
        if sizing == "full":
            if pend_tgt is not None:                     # rebalance whole book to targets
                eq_open = cash + sum(p["shares"] * op.get(t, np.nan) for t, p in pos.items()
                                     if np.isfinite(op.get(t, np.nan)))
                for t in set(pos) | set(pend_tgt):
                    price = op.get(t, np.nan)
                    if not np.isfinite(price) or price <= 0:
                        continue                         # untradeable today, leave as is
                    cur = pos[t]["shares"] * price if t in pos else 0.0
                    tgt = pend_tgt.get(t, 0.0) * eq_open
                    delta = tgt - cur
                    if abs(delta) < 1e-9:
                        continue
                    cash -= delta + abs(delta) * cost
                    trades.append(dict(date=day, ticker=t, side="BUY" if delta > 0 else "SELL",
                                       px=float(price), shares=abs(delta) / price, value=abs(delta)))
                    if tgt > 1e-9:
                        if t in pos:                     # rebalance existing: keep entry/hi
                            pos[t]["shares"] = tgt / price
                        else:                            # new position
                            pos[t] = dict(shares=tgt / price, hi=float(cl.get(t, price)),
                                          entry=float(price), entry_i=i)
                    else:
                        pos.pop(t, None)
                pend_tgt = None
        else:                                            # independent slots ("slots" / "risk")
            for t in pend_sell:                          # a slot's stock exits -> full sell
                if t in pos and np.isfinite(op.get(t, np.nan)):
                    sh = pos[t]["shares"]; val = sh * op[t]
                    cash += val * (1 - cost)
                    trades.append(dict(date=day, ticker=t, side="SELL", px=float(op[t]),
                                       shares=sh, value=val))
                    del pos[t]
            for o in pend_buy:                           # freed slot's cash buys the next breakout
                t = o["ticker"]
                if t in pos or len(pos) >= CONFIG["slots"]:
                    continue
                price = op.get(t, np.nan)
                if not np.isfinite(price) or price <= 0:
                    continue
                if sizing == "slots":                    # 1/N of equity, capped by available cash
                    budget = min(o["equity"] / CONFIG["slots"], cash / (1 + cost))
                else:                                    # ATR risk-sized
                    budget = min((CONFIG["risk_frac"] * o["equity"]) / (stop_k * o["atr"]) * price,
                                 cash / (1 + cost))
                shares = budget / price
                if shares <= 0:
                    continue
                cash -= shares * price * (1 + cost)
                pos[t] = dict(shares=shares, hi=float(cl.get(t, price)),
                              entry=float(price), entry_i=i)
                trades.append(dict(date=day, ticker=t, side="BUY", px=float(price),
                                   shares=shares, value=shares * price))

        # 1b) intraday resting ATR stop: filled same day when the low touches the
        #     stop level (or at the open if it gaps below). Levels use info through
        #     the prior close (no look-ahead); skip names opened today.
        if intraday:
            lo_d, ap_d = P["low"].loc[day], atr_prev.loc[day]
            for t in list(pos):
                p = pos[t]
                a = ap_d.get(t, np.nan)
                if p["entry_i"] >= i or not np.isfinite(a):
                    continue
                level = p["hi"] - stop_k * a
                low_t = lo_d.get(t, np.nan)
                if np.isfinite(low_t) and low_t <= level:
                    opn = op.get(t, np.nan)
                    fill = min(level, opn) if np.isfinite(opn) else level
                    sh = p["shares"]; val = sh * fill
                    cash += val * (1 - cost)
                    trades.append(dict(date=day, ticker=t, side="SELL", px=float(fill),
                                       shares=sh, value=val))
                    del pos[t]

        # 2) mark to market
        equity = cash + sum(p["shares"] * cl.get(t, np.nan) for t, p in pos.items()
                            if np.isfinite(cl.get(t, np.nan)))
        eq_curve.append(equity)
        deploy.append((equity - cash) / equity if equity > 0 else 0)
        holds.append((day, cash, {t: p["shares"] * cl.get(t, np.nan) for t, p in pos.items()}))

        # 3) update trailing highs
        for t, p in pos.items():
            c = cl.get(t, np.nan)
            if np.isfinite(c):
                p["hi"] = max(p["hi"], float(c))

        if i == len(dates) - 1:
            break

        # 4) exits (signalled on today's close, executed next open)
        atr_d, exlo_d = P["atr"].loc[day], exlo.loc[day]
        tp, tstop = CONFIG.get("take_profit"), CONFIG.get("time_stop")
        ps, use_lo = CONFIG.get("pct_stop"), CONFIG.get("use_exit_low", True)
        pend_sell = []
        for t, p in pos.items():
            c = cl.get(t, np.nan)
            if not np.isfinite(c):
                pend_sell.append(t); continue
            a = atr_d.get(t, np.nan)
            stop = p["hi"] - stop_k * a if np.isfinite(a) else -np.inf
            hit = (not intraday) and c < stop                     # 1) trailing ATR stop (close-based)
            if use_lo and c < exlo_d.get(t, -np.inf):             # 2) N-day-low exit
                hit = True
            if ps and c <= p["entry"] * (1 - ps):                 # 3) hard % stop from entry
                hit = True
            if tp and c >= p["entry"] * (1 + tp):                 # take-profit
                hit = True
            if tstop and (i - p["entry_i"]) >= tstop:             # time stop
                hit = True
            if hit:
                pend_sell.append(t)

        # 5) entries: rank the day's fresh breakouts, fill free slots
        ukey = ((day.isocalendar()[0], day.isocalendar()[1]) if weekly
                else (day.year, day.month))
        uni = uni_by_period[ukey]
        free = CONFIG["slots"] - (len(pos) - len(pend_sell))
        bo, px, at = P["breakout"].loc[day], P["proxy"].loc[day], P["atr"].loc[day]
        picks = []
        if regime_ok is not None and not bool(regime_ok.get(day, False)):
            free = 0                                     # risk-off: no new entries
        if free > 0:
            cand = [(t, px[t]) for t in uni
                    if bool(bo.get(t, False)) and t not in pos
                    and np.isfinite(px.get(t, np.nan)) and np.isfinite(at.get(t, np.nan))
                    and at.get(t, 0) > 0]
            rmode = CONFIG.get("rank_mode", "proxy")
            if rmode == "random":
                np.random.shuffle(cand)
            else:
                cand.sort(key=lambda x: x[1], reverse=(rmode != "anti"))
            picks = [t for t, _ in cand[:free]]

        if sizing == "full":
            new_held = (set(pos) - set(pend_sell)) | set(picks)
            pend_tgt = target_weights(new_held, at) if new_held != set(pos) else None
        else:
            pend_buy = [dict(ticker=t, atr=float(at[t]), equity=equity) for t in picks]

    eq = pd.Series(eq_curve, index=dates)
    tdf = pd.DataFrame(trades)
    return dict(equity=eq, trades=tdf, metrics=_metrics(eq),
                avg_deploy=float(np.mean(deploy)), holds=holds)


def run_backtest():
    cap = CONFIG["capital"]
    print("Fetching broad-universe list + full history (cached)...")
    tickers = broad_universe()
    prices = download_prices(tickers, period="12y", pool="broad")
    print(f"  {len(prices)} tickers; running backtest from {CONFIG['bt_start']}...")

    res = backtest(prices, cap)
    m = res["metrics"]; eq = res["equity"]; tr = res["trades"]

    # SPY benchmark, rebased to capital
    spy = yf.download("SPY", start=str(eq.index[0].date()), auto_adjust=True,
                      progress=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    spy = spy.reindex(eq.index).ffill()
    spy_eq = cap * spy / spy.iloc[0]
    sm = _metrics(spy_eq)

    n_buys = int((tr["side"] == "BUY").sum()) if not tr.empty else 0
    yrs = len(eq) / 252
    print(f"\n=== Breakout + momentum-proxy, {CONFIG['slots']} slots, "
          f"{CONFIG['risk_frac']*100:.0f}% risk/slot  ({eq.index[0].date()} -> {eq.index[-1].date()}) ===")
    print(f"{'':<14}{'CAGR':>8}{'Sharpe':>8}{'vol':>7}{'maxDD':>8}{'end $'+f'{cap/1000:.0f}k':>11}")
    print(f"{'strategy':<14}{m['cagr']*100:>7.1f}%{m['sharpe']:>8.2f}{m['vol']*100:>6.0f}%"
          f"{m['maxdd']*100:>7.0f}%{m['end']:>11,.0f}")
    print(f"{'SPY buy&hold':<14}{sm['cagr']*100:>7.1f}%{sm['sharpe']:>8.2f}{sm['vol']*100:>6.0f}%"
          f"{sm['maxdd']*100:>7.0f}%{sm['end']:>11,.0f}")
    print(f"\ntrades: {n_buys} buys (~{n_buys/yrs:.0f}/yr) | "
          f"avg deployment implied by ~{CONFIG['risk_frac']*100:.0f}% risk x {CONFIG['slots']} slots")


def main():
    cap = CONFIG["capital"]
    print("Fetching broad-universe list + prices (cached)...")
    tickers = broad_universe()
    prices = download_prices(tickers, pool="broad")
    print(f"  {len(prices)}/{len(tickers)} tickers with data")

    universe = build_universe(prices)
    asof = max(df.index[-1] for df in prices.values())
    print(f"Universe: top {len(universe)} by {CONFIG['adv_window']}-day ADV "
          f"(as of {asof.date()})")

    cand = scan_candidates(prices, universe, cap)
    print(f"\n{len(cand)} fresh {CONFIG['breakout']}-day breakouts today "
          f"(ranked by {CONFIG['proxy_window']}-day momentum proxy):\n")
    if cand.empty:
        print("  (no breakouts today)")
        return
    print(f"{'rank':>4} {'tkr':<6}{'close':>9}{'proxy%':>8}{'ATR':>7}"
          f"{'stop':>9}{'shares':>8}{'$size':>9}{'wgt':>6}")
    for i, r in cand.iterrows():
        flag = "  <- would BUY" if i < CONFIG["slots"] else ""
        print(f"{i+1:>4} {r.ticker:<6}{r.close:>9.2f}{r.proxy*100:>8.1f}"
              f"{r.atr:>7.2f}{r.stop:>9.2f}{r.shares:>8.1f}{r.dollars:>9,.0f}"
              f"{r.pct*100:>5.1f}%{flag}")
    top = cand.head(CONFIG["slots"])
    print(f"\nTop {len(top)} would fill the {CONFIG['slots']} slots: "
          f"${top.dollars.sum():,.0f} deployed ({top.pct.sum()*100:.0f}% of "
          f"${cap:,.0f}), rest cash. Each risks ~{CONFIG['risk_frac']*100:.0f}% to its stop.")


if __name__ == "__main__":
    import sys
    if "--backtest" in sys.argv:
        run_backtest()
    else:
        main()
