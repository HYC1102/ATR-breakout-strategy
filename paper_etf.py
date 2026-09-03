"""
Forward paper-trading engine for the ETF breakout strategy.

Deliberately separate from paper_trade.py (the 223-stock account) because the
sizing rule differs. paper_trade.py runs independent 1/N slots and leaves the
rest in cash; this one:

  * re-equalises across whatever it currently holds (2 names -> 50% each), and
  * parks anything left over in a DEFAULT HOLDING (VT) rather than in cash,
    which in practice means "hold VT on the days nothing is breaking out".

Keeping the two engines apart means the live stock record in
data/paper_breakout.json cannot be disturbed by changes made for this strategy.

State lives in data/paper_etf.json and is mirrored to CSVs. Run daily via
breakout_dashboard.py: it fills the previous session's plan at the latest open,
marks the book, checks stops, and plans the next session.

Two deliberate departures from etf_breakout.py's backtest, both documented on
the dashboard:
  * a rebalancing band (`rebal_band`) suppresses dust orders below a fraction
    of equity. The backtest re-equalises exactly, which generated ~14% of its
    orders at under 1% of equity -- untradeable at this account size.
  * the book is only rebalanced when the SET of holdings changes, not daily.

Costs are OFF (`cost_bps=0`), matching the stock account, so both tabs report
gross returns. Turnover on this strategy is high (~264 orders/yr in backtest),
so the gross figure flatters it more than it flatters the stock tab.
"""
from __future__ import annotations

import os
import json

import numpy as np
import pandas as pd

import breakout_sentiment as bs
import etf_breakout as eb
import paper_trade as pt

STATE_PATH = os.path.join("data", "paper_etf.json")
TRADES_CSV = os.path.join("data", "paper_etf_trades.csv")
EQUITY_CSV = os.path.join("data", "paper_etf_equity.csv")

DEFAULT_ASSET = "VT"
UNIVERSE = list(eb.UNIVERSE)

CFG = dict(
    sizing="full", weight_mode="equal", slots=5, regime=True, regime_ma=200,
    rank_mode="proxy", sent_weight=0.0,
    entry="atr", atr_break_period=20, atr_break_mult=3.0,
    proxy_window=63, atr_window=14, adv_window=60,
    atr_stop=1.5, exit_low=10, use_exit_low=True,
    pct_stop=None, take_profit=None, time_stop=None,
    universe_size=len(UNIVERSE), rebuild="W", pool="etf",
    cost_bps=0.0,             # gross, matching the stock account
    default_asset=DEFAULT_ASSET,
    rebal_band=0.005,          # skip rebalancing trades under 0.5% of equity
)


def apply_config():
    """Make this strategy's parameters the active ones. The two accounts share
    breakout_sentiment.CONFIG, so whoever runs last wins -- always call this
    immediately before touching ETF state."""
    bs.CONFIG.update(CFG)
    return CFG


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
                positions={}, trades=[], equity=[], pending=[], target={},
                last_close=None)


def _cost():
    return bs.CONFIG["cost_bps"] / 1e4


def _mark(positions, cash, price_row):
    """Book value, carrying the last known mark through a data gap."""
    v = cash
    for tk, p in positions.items():
        px = price_row.get(tk, np.nan)
        if np.isfinite(px):
            p["last_px"] = float(px)
        else:
            px = p.get("last_px", p["entry"])
        v += p["shares"] * px
    return v


def _targets(st, asof, prices, P, regime):
    """Target weights for the NEXT open, decided on the `asof` close.

    Equal weight across the names that will be held; anything unallocated (i.e.
    when nothing is held at all) goes to the default asset."""
    cl = P["close"].loc[asof]
    dflt = bs.CONFIG.get("default_asset")
    held = {t for t in st["positions"] if t != dflt}

    sells, reasons = set(), {}
    for tk in held:
        c = cl.get(tk, np.nan)
        if not np.isfinite(c):
            continue                        # data gap -> hold, never force-sell
        try:
            lvl, rule = pt.stop_level(st["positions"][tk], tk, prices, asof)
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(lvl) and c < lvl:
            sells.add(tk); reasons[tk] = rule

    free = bs.CONFIG["slots"] - (len(held) - len(sells))
    reg = bool(regime.get(asof, False)) if regime is not None else True
    cand, picks = None, []
    if reg and free > 0:
        cand = bs.rank_breakouts(prices, [t for t in UNIVERSE if t in prices],
                                 asof=asof, use_sentiment=False)
        for _, r in cand.iterrows():
            if r.ticker in held or r.ticker in sells or r.ticker == dflt:
                continue
            picks.append(r)
            if len(picks) >= free:
                break

    new_held = (held - sells) | {r.ticker for r in picks}
    # Reserve the round-trip cost so the book does not run a small cash overdraft.
    scale = 1.0 - _cost()
    tgt = {t: scale / len(new_held) for t in new_held} if new_held else {}
    residual = scale - sum(tgt.values())
    if dflt and residual > 1e-9:
        tgt[dflt] = tgt.get(dflt, 0.0) + residual

    # Only rebalance when the SET of holdings changes, matching the backtest.
    # Re-equalising every day instead would add ~17% more orders for a ~0.5%
    # difference in outcome -- not a trade worth making at this account size.
    if set(tgt) == set(st["positions"]):
        return {}, reasons, {r.ticker: r for r in picks}, cand
    return tgt, reasons, {r.ticker: r for r in picks}, cand


def _display_orders(st, tgt, reasons, picks, cl):
    """Human-readable version of the rebalance, estimated at the latest close.
    _execute() re-derives the real sizes at the next open."""
    if not tgt:
        return []
    eq = _mark(st["positions"], st["cash"], cl)
    band = bs.CONFIG.get("rebal_band", 0.0) * eq
    dflt = bs.CONFIG.get("default_asset")
    orders = []
    for tk in sorted(set(st["positions"]) | set(tgt)):
        price = cl.get(tk, np.nan)
        if not np.isfinite(price):
            price = st["positions"].get(tk, {}).get("last_px", np.nan)
        if not np.isfinite(price) or price <= 0:
            continue
        cur = st["positions"][tk]["shares"] * price if tk in st["positions"] else 0.0
        want = tgt.get(tk, 0.0) * eq
        delta = want - cur
        if abs(delta) < max(band, 1.0):
            continue
        r = picks.get(tk)
        o = dict(side="BUY" if delta > 0 else "SELL", ticker=tk,
                 price_hint=round(float(price), 2),
                 shares_hint=round(abs(delta) / price, 3),
                 value_hint=round(abs(delta), 2),
                 target_weight=round(tgt.get(tk, 0.0) * 100, 1))
        if tk == dflt:
            o["reason"] = "default holding (nothing breaking out)"
        elif tk in reasons:
            o["reason"] = f"exit — {reasons[tk]}"
        elif r is not None:
            o["reason"] = "breakout"
            o["momentum"] = round(float(r.momentum), 1)
        else:
            o["reason"] = "rebalance to equal weight"
        orders.append(o)
    order = {"SELL": 0, "BUY": 1}
    return sorted(orders, key=lambda o: (order[o["side"]], -o["value_hint"]))


def _execute(st, d, prices, P):
    """Rebalance the book to st['target'] at day d's open."""
    tgt = st.get("target") or {}
    if not tgt:
        return
    op, cl, c = P["open"].loc[d], P["close"].loc[d], _cost()
    eq = _mark(st["positions"], st["cash"], op)
    band = bs.CONFIG.get("rebal_band", 0.0) * eq
    dflt = bs.CONFIG.get("default_asset")

    for tk in sorted(set(st["positions"]) | set(tgt)):
        price = op.get(tk, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue                       # untradeable today -> leave as is
        cur_sh = st["positions"][tk]["shares"] if tk in st["positions"] else 0.0
        cur = cur_sh * price
        want = tgt.get(tk, 0.0) * eq
        delta = want - cur
        if abs(delta) < max(band, 1.0):
            continue
        st["cash"] -= delta + abs(delta) * c
        side = "BUY" if delta > 0 else "SELL"
        st["trades"].append(dict(date=str(d.date()), side=side, ticker=tk,
                                 shares=round(abs(delta) / price, 3),
                                 price=round(float(price), 2),
                                 value=round(abs(delta), 2),
                                 reason=("default holding" if tk == dflt else
                                         "rebalance" if cur_sh > 0 and want > 1e-9 else
                                         "breakout" if side == "BUY" else "exit")))
        if want > 1e-9:
            new_sh = want / price
            if tk in st["positions"]:
                st["positions"][tk]["shares"] = new_sh
            else:
                st["positions"][tk] = dict(shares=new_sh, entry=round(float(price), 2),
                                           entry_date=str(d.date()),
                                           hi=float(cl.get(tk, price)) if np.isfinite(
                                               cl.get(tk, np.nan)) else float(price))
        else:
            st["positions"].pop(tk, None)
    st["target"] = {}


def advance(st, prices, P, regime):
    """Process any new sessions since the last run, then plan the next one.

    Returns (state, asof, cand)."""
    close = P["close"]; asof = close.index[-1]
    # Same split re-basing as the stock account; the parking asset is included
    # since it is a real position with stored shares.
    for tk, factor in pt.reconcile_splits(st, P, asof):
        print(f"  split adjustment: {tk} {factor:g}-for-1")
    start = pd.Timestamp(st["start_date"]); processed = False; cand = None
    dflt = bs.CONFIG.get("default_asset")
    if asof >= start:
        last = pd.Timestamp(st["last_close"]) if st.get("last_close") else None
        for d in close.loc[start:asof].index:
            if last is not None and d <= last:
                continue
            _execute(st, d, prices, P)
            st["equity"].append(dict(date=str(d.date()),
                                     value=round(_mark(st["positions"], st["cash"],
                                                       close.loc[d]), 2)))
            for tk, p in st["positions"].items():
                if tk == dflt:
                    continue                       # parking asset is not stop-managed
                cc = close.loc[d].get(tk, np.nan)
                if np.isfinite(cc):
                    p["hi"] = max(p["hi"], float(cc))
            st["last_close"] = str(d.date())
            tgt, reasons, picks, cand = _targets(st, d, prices, P, regime)
            st["target"] = tgt
            st["pending"] = _display_orders(st, tgt, reasons, picks, close.loc[d])
            processed = True
    if not processed:
        tgt, reasons, picks, cand = _targets(st, asof, prices, P, regime)
        st["target"] = tgt
        st["pending"] = _display_orders(st, tgt, reasons, picks, close.loc[asof])
    return st, asof, cand
