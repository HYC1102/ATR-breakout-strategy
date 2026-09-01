"""
Forward paper-trading engine for the breakout strategy, with persistence.

State (positions, cash, every trade + its sentiment, daily equity) is stored in
data/paper_breakout.json and mirrored to CSVs. Run it daily (e.g. via the
dashboard): it executes the previous session's planned orders at the latest open,
marks the book, checks stops, and plans the next session's orders — ranking fresh
breakouts by the live momentum + news-sentiment score. Nothing is recomputed
retroactively, so the log is a genuine forward record.
"""
from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd

import breakout_sentiment as bs

STATE_PATH = os.path.join("data", "paper_breakout.json")
TRADES_CSV = os.path.join("data", "paper_trades.csv")
EQUITY_CSV = os.path.join("data", "paper_equity.csv")


def load_state():
    try:
        with open(STATE_PATH) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return None


def save_state(st):
    os.makedirs("data", exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(st, f, indent=2, default=str)
    if st["trades"]:
        pd.DataFrame(st["trades"]).to_csv(TRADES_CSV, index=False)
    if st["equity"]:
        pd.DataFrame(st["equity"]).to_csv(EQUITY_CSV, index=False)


def init_state(start_date, capital):
    return dict(start_date=str(start_date), capital=float(capital), cash=float(capital),
                positions={}, trades=[], equity=[], pending=[], last_close=None)


def _cost():
    return bs.CONFIG["cost_bps"] / 1e4


def _equity(positions, cash, price_row):
    """Book value. A missing quote falls back to the last known mark (then entry)
    rather than dropping the position from equity, which would read as a total
    loss on that name for the day and a matching bounce when data returns."""
    v = cash
    for tk, p in positions.items():
        px = price_row.get(tk, np.nan)
        if np.isfinite(px):
            p["last_px"] = float(px)
        else:
            px = p.get("last_px", p["entry"])
        v += p["shares"] * px
    return v


def stop_level(p, ticker, prices, asof=None):
    """Live exit trigger for a held position: higher of the 1.5-ATR trail and the
    N-day low. Returns (level, binding-rule).

    The N-day low is taken through YESTERDAY (`.shift(1)`), matching the
    backtest's `exlo`. Including today's bar would make the level <= today's low
    <= today's close, so the `close < level` breach test in _plan() could never
    fire and the N-day-low exit would be dead.

    `asof` truncates the history to that date. It matters when advance() has more
    than one new day to process (first run after start_date, or after a missed
    run): without it every back-filled day is judged against the LATEST bar's
    stop, which is look-ahead and can retro-fire exits that never happened."""
    df = prices[ticker]
    if asof is not None:
        df = df.loc[:asof]
    atr = float(bs.atr(df, bs.CONFIG["atr_window"]).iloc[-1])
    atr_stop = p["hi"] - bs.CONFIG["atr_stop"] * atr
    low = float(df["Low"].rolling(bs.CONFIG["exit_low"]).min().shift(1).iloc[-1])
    if not np.isfinite(low):                      # not enough history for the low yet
        return atr_stop, f"{bs.CONFIG['atr_stop']}-ATR trail"
    return ((atr_stop, f"{bs.CONFIG['atr_stop']}-ATR trail") if atr_stop >= low
            else (low, f"{bs.CONFIG['exit_low']}-day low"))


def _plan(st, asof, prices, P, regime, use_sentiment=True):
    """Orders to place at the NEXT open, decided on the `asof` close.

    Returns (pending, cand): `cand` is the ranked breakout frame when one was
    computed, else None, so the caller can reuse it instead of rescanning."""
    cl = P["close"].loc[asof]
    pending, sells = [], set()
    for tk, p in st["positions"].items():
        c = cl.get(tk, np.nan)
        if not np.isfinite(c):
            continue                       # transient data gap -> HOLD, never force-sell
        try:
            lvl, rule = stop_level(p, tk, prices, asof)
        except Exception:  # noqa: BLE001
            continue                       # can't compute a stop (missing history) -> hold
        if np.isfinite(lvl) and c < lvl:   # only a genuine, finite stop breach exits
            pending.append(dict(side="SELL", ticker=tk, reason=rule))
            sells.add(tk)
    free = bs.CONFIG["slots"] - (len(st["positions"]) - len(sells))
    reg = bool(regime.get(asof, False)) if regime is not None else True
    cand = None
    if reg and free > 0:
        # Estimate next-open sizing from today's close so the dashboard can
        # provide an actionable share count.  _execute() repeats this sizing
        # with the actual next-open prices, so gaps can change the final fill.
        cost = _cost()
        estimated_equity = _equity(st["positions"], st["cash"], cl)
        estimated_cash = float(st["cash"])
        for tk in sells:
            p = st["positions"].get(tk)
            mark = cl.get(tk, np.nan)
            if p and np.isfinite(mark):
                sale_value = p["shares"] * mark
                estimated_cash += sale_value * (1 - cost)
                estimated_equity -= sale_value * cost

        # Price frames must be truncated to the decision date when catching up
        # after missed sessions; otherwise the latest bar leaks into old plans.
        cand = bs.rank_breakouts(prices, bs.build_universe(prices, asof), asof=asof,
                                 use_sentiment=use_sentiment)
        held = set(st["positions"])
        for _, r in cand.iterrows():
            if r.ticker in held or r.ticker in sells:
                continue
            price_hint = float(r.close)
            budget = min(estimated_equity / bs.CONFIG["slots"],
                         estimated_cash / (1 + cost))
            if budget <= 0:
                break
            shares_hint = budget / price_hint if price_hint > 0 else 0.0
            order = dict(side="BUY", ticker=r.ticker,
                         price_hint=round(price_hint, 2),
                         shares_hint=round(shares_hint, 3),
                         value_hint=round(budget, 2),
                         cost_hint=round(budget * cost, 2),
                         momentum=round(float(r.momentum), 1))
            if bs.CONFIG.get("sent_weight", 0.0) > 0:
                order.update(sentiment=round(float(r.sentiment), 1),
                             combined=round(float(r.combined), 1))
            pending.append(order)
            estimated_cash -= budget * (1 + cost)
            if len([o for o in pending if o["side"] == "BUY"]) >= free:
                break
    return pending, cand


def _execute(st, d, prices, P):
    """Fill the pending orders at day d's open."""
    op, cl, c = P["open"].loc[d], P["close"].loc[d], _cost()
    for o in [x for x in st["pending"] if x["side"] == "SELL"]:
        tk = o["ticker"]; p = st["positions"].get(tk); px = op.get(tk, np.nan)
        if p and np.isfinite(px):
            val = p["shares"] * px
            st["cash"] += val * (1 - c)
            st["trades"].append(dict(date=str(d.date()), side="SELL", ticker=tk,
                                     shares=round(p["shares"], 3), price=round(float(px), 2),
                                     value=round(val, 2), reason=o.get("reason", ""),
                                     momentum="", sentiment="", combined=""))
            del st["positions"][tk]
    eq = _equity(st["positions"], st["cash"], op)                # equity after sells, for sizing
    for o in [x for x in st["pending"] if x["side"] == "BUY"]:
        tk = o["ticker"]
        if tk in st["positions"] or len(st["positions"]) >= bs.CONFIG["slots"]:
            continue
        px = op.get(tk, np.nan)
        if not np.isfinite(px) or px <= 0:
            continue
        budget = min(eq / bs.CONFIG["slots"], st["cash"] / (1 + c))
        shares = budget / px
        if shares <= 0:
            continue
        st["cash"] -= shares * px * (1 + c)
        st["positions"][tk] = dict(shares=shares, entry=round(float(px), 2),
                                   entry_date=str(d.date()), hi=float(cl.get(tk, px)))
        st["trades"].append(dict(date=str(d.date()), side="BUY", ticker=tk,
                                 shares=round(shares, 3), price=round(float(px), 2),
                                 value=round(shares * px, 2), reason="breakout",
                                 momentum=o.get("momentum"), sentiment=o.get("sentiment"),
                                 combined=o.get("combined")))


def advance(st, prices, P, regime):
    """Process any new trading days since last run, then plan the next session.

    Returns (state, asof, cand) -- see _plan() for `cand`."""
    close = P["close"]; asof = close.index[-1]
    start = pd.Timestamp(st["start_date"]); processed = False; cand = None
    if asof >= start:
        last = pd.Timestamp(st["last_close"]) if st.get("last_close") else None
        for d in close.loc[start:asof].index:
            if last is not None and d <= last:
                continue
            _execute(st, d, prices, P)
            st["equity"].append(dict(date=str(d.date()),
                                     value=round(_equity(st["positions"], st["cash"], close.loc[d]), 2)))
            for tk, p in st["positions"].items():
                cc = close.loc[d].get(tk, np.nan)
                if np.isfinite(cc):
                    p["hi"] = max(p["hi"], float(cc))
            st["last_close"] = str(d.date())
            # Current news cannot be reconstructed for a missed historical
            # session. Use the documented neutral score instead of leaking
            # today's headlines backward into a retroactive decision.
            st["pending"], cand = _plan(st, d, prices, P, regime,
                                         use_sentiment=(d == asof))
            processed = True
    if not processed:
        st["pending"], cand = _plan(st, asof, prices, P, regime)
    return st, asof, cand
