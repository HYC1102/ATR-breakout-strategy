"""Does the breakout/rotation edge survive a universe chosen WITHOUT hindsight?

The ETF universe in etf_breakout.py was written down in 2026 naming AI, nuclear,
quantum, GLP-1 and spot bitcoin. Those are known winners, so a momentum system
run over them is flattered by the universe itself. This script re-runs the
identical rules over a universe defined by a rule you could have written in 2006
and could not have tuned:

    every S&P sector SPDR + the major liquid asset classes

It is exhaustive (all 11 sectors, not a chosen few) and structural (equity /
international / bonds / gold / REITs / commodities), so no constituent is there
because it went up. If the rotation still adds return over simply holding the
default asset, that is evidence about the RULES. If it collapses, the thematic
result was measuring hindsight.

  python universe_test.py
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

import breakout_sentiment as bs
import etf_breakout as eb

warnings.filterwarnings("ignore")

SECTORS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
ASSETS  = ["SPY", "EFA", "EEM", "AGG", "TLT", "GLD", "IYR", "DBC"]
HONEST  = SECTORS + ASSETS
DEFAULT = "SPY"          # the parking / default holding (exists since 1993)

CFG = dict(eb.CFG); CFG.update(universe_size=len(HONEST), pool="honest")


def load():
    prices = bs.download_prices(HONEST, period="max", pool="honest")
    missing = [t for t in HONEST if t not in prices]
    if missing:
        print(f"  no data for {missing}")
    return eb.clean_calendar(prices)


def run(prices, universe, start, end=None, capital=100_000.0, **ov):
    bs.CONFIG.update(CFG); bs.CONFIG.update(universe_size=len(universe)); bs.CONFIG.update(ov)
    tradable = {t: d for t, d in prices.items() if t in universe}
    P = bs.build_panels(tradable)
    spy = prices.get("SPY", prices.get("VT"))
    reg = eb.regime_series(spy, bs.CONFIG.get("regime_ma", 200))
    return bs.backtest(capital=capital, P=P, regime_full=reg, start=start, end=end), P


def deployed(res):
    eq = res["equity"]
    return 1 - pd.Series([h[1] for h in res["holds"]], index=eq.index) / eq


def row(name, eq, note=""):
    m = bs._metrics(eq)
    return (name, m["cagr"], m["sharpe"], m["vol"], m["maxdd"], m["end"], note)


def show(rows, cap=100_000.0):
    w = max(len(r[0]) for r in rows) + 2
    print(f"{'':<{w}}{'CAGR':>8}{'Sharpe':>8}{'vol':>7}{'maxDD':>8}"
          f"{'end $'+f'{cap/1000:.0f}k':>12}   notes")
    for n, c, s, v, dd, e, note in rows:
        print(f"{n:<{w}}{c*100:>7.1f}%{s:>8.2f}{v*100:>6.0f}%{dd*100:>7.0f}%{e:>12,.0f}   {note}")


EW = dict(sizing="full", weight_mode="equal")


def block(prices, universe, start, end, default, label, cap=100_000.0):
    base, _ = run(prices, universe, start, end, cap)
    ewr,  _ = run(prices, universe, start, end, cap, **EW)
    dfl,  _ = run(prices, universe, start, end, cap, default_asset=default, **EW)
    d = base["equity"].index
    px = prices[default]["Close"].reindex(d).ffill()
    rows = [
        row(f"{default} buy & hold", cap * px / px.iloc[0], "the thing to beat"),
        row("equal-weight hold (all names)", eb.buy_hold(prices, universe, d, cap)),
        row("breakout, 1/5 fixed, cash idle", base["equity"],
            f"{deployed(base).mean()*100:.0f}% deployed"),
        row("breakout, eq-wt held, cash idle", ewr["equity"],
            f"{deployed(ewr).mean()*100:.0f}% deployed"),
        row(f"breakout, eq-wt held, {default} default", dfl["equity"],
            f"{int((dfl['trades']['side']=='BUY').sum())} buys"),
    ]
    print(f"\n=== {label}  ({d[0].date()} -> {d[-1].date()}, {len(d)/252:.1f}y) ===")
    show(rows, cap)
    return dfl, base


def main():
    print("Loading hindsight-free universe (19 names)...")
    hp = load()
    print("Loading thematic universe (29 names)...")
    tp = eb.load()

    block(hp, HONEST, "2007-01-01", None, DEFAULT,
          "HINDSIGHT-FREE universe: 11 sector SPDRs + 8 asset classes")
    block(hp, HONEST, "2019-01-01", None, DEFAULT,
          "HINDSIGHT-FREE universe, same window as the thematic test")
    block(tp, eb.UNIVERSE, "2019-01-01", None, "VT",
          "THEMATIC universe (chosen in 2026)")

    # split the honest run to see stability
    print("\n=== hindsight-free: stability across four disjoint periods ===")
    rows = []
    for lo, hi in [("2007-01-01", "2011-12-31"), ("2012-01-01", "2016-12-31"),
                   ("2017-01-01", "2021-12-31"), ("2022-01-01", None)]:
        r, _ = run(hp, HONEST, lo, hi, 100_000.0, default_asset=DEFAULT, **EW)
        d = r["equity"].index
        px = hp[DEFAULT]["Close"].reindex(d).ffill()
        m, b = bs._metrics(r["equity"]), bs._metrics(100_000.0 * px / px.iloc[0])
        rows.append((f"{lo[:4]}-{hi[:4] if hi else '26'}", m["cagr"], m["sharpe"],
                     m["maxdd"], b["cagr"], b["sharpe"], b["maxdd"]))
    print(f"{'period':<12}{'strategy':>10}{'Sh':>6}{'DD':>7}   {'SPY':>9}{'Sh':>6}{'DD':>7}   edge")
    for p, c, s, dd, bc, bsh, bdd in rows:
        print(f"{p:<12}{c*100:>9.1f}%{s:>6.2f}{dd*100:>6.0f}%   {bc*100:>8.1f}%"
              f"{bsh:>6.2f}{bdd*100:>6.0f}%   {(c-bc)*100:>+5.1f}pp")


if __name__ == "__main__":
    main()
