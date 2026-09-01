"""
ATR-breakout strategy run over a fixed ETF universe instead of US large caps.

Reuses the engine in breakout_sentiment.py unchanged -- same ATR-band entry,
same 1.5-ATR trail + 10-day-low exit, same 5 independent slots, same next-open
fills and 2bps costs. The ONLY change is the tradable universe: ~29 ETFs
(core/diversifier sleeves, thematic sleeves, sector tilts) in place of the
top-223 US stocks by dollar volume.

  python etf_breakout.py                 # headline run + benchmarks
  python etf_breakout.py --variants      # also sweep slots / regime / exits

READ THE CAVEATS printed at the end. The thematic sleeves were chosen in 2026
with knowledge of which themes worked, so the universe itself carries heavy
selection bias -- which is exactly why the equal-weight-hold benchmark below
matters more than the SPY comparison.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

import breakout_sentiment as bs

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Universe
# --------------------------------------------------------------------------- #
SLEEVES = {
    "core / diversifier": ["VT", "BND", "TLT", "GLDM"],
    "thematic":           ["SMH", "IGV", "BOTZ", "ICLN", "URA", "CIBR", "XBI", "ITA",
                           "IBIT", "BLOK", "FIW", "PAVE", "FINX", "LIT", "COPX",
                           "QTUM", "OZEM", "ARKX", "ESPO", "MOO", "INDA"],
    "sector tilt":        ["XLF", "XLE", "XLP", "XLY"],
}
UNIVERSE = [t for s in SLEEVES.values() for t in s]
BENCHMARKS = ["SPY", "ACWI"]

# Live-dashboard settings, transplanted verbatim except for the universe size.
CFG = dict(sizing="slots", slots=5, regime=True, rank_mode="proxy", atr_stop=1.5,
           exit_low=10, use_exit_low=True, pct_stop=None, take_profit=None,
           time_stop=None, universe_size=len(UNIVERSE), rebuild="W", pool="etf",
           entry="atr", atr_break_period=20, atr_break_mult=3.0, proxy_window=63,
           atr_window=14, cost_bps=2.0, capital=100_000.0)


def clean_calendar(prices, min_coverage: float = 0.8, verbose: bool = True):
    """Drop dates where fewer than `min_coverage` of the ALIVE names reported.

    yfinance intermittently returns a date for a handful of tickers and not the
    rest. Left in, such a date shows most of the book as un-marked; with only a
    few slots that reads as a huge phantom loss and an equal phantom gain the
    next day. Genuine holidays are absent for everyone and so are unaffected."""
    idx = sorted(set().union(*(d.index for d in prices.values())))
    idx = pd.DatetimeIndex(idx)
    alive = pd.DataFrame({t: (idx >= d.index[0]) & (idx <= d.index[-1])
                          for t, d in prices.items()}, index=idx)
    have = pd.DataFrame({t: idx.isin(d.index) for t, d in prices.items()}, index=idx)
    n_alive = alive.sum(axis=1)
    frac = (alive & have).sum(axis=1) / n_alive.replace(0, np.nan)
    bad = idx[(frac < min_coverage) & (n_alive > 0)]
    if len(bad) and verbose:
        print(f"  dropped {len(bad)} low-coverage date(s): "
              f"{', '.join(str(d.date()) for d in bad[:5])}"
              f"{' ...' if len(bad) > 5 else ''}")
    keep = idx.difference(bad)
    return {t: d[d.index.isin(keep)] for t, d in prices.items()}


def load(period: str = "max"):
    """Full history for the ETF universe + benchmarks (cached in data/)."""
    prices = bs.download_prices(UNIVERSE + BENCHMARKS, period=period, pool="etf")
    missing = [t for t in UNIVERSE if t not in prices]
    if missing:
        print(f"  warning: no data for {missing} -- excluded from the universe")
    return clean_calendar(prices)


def regime_series(spy: pd.DataFrame, ma_n: int = 200) -> pd.Series:
    """SPY > its N-day MA, computed over SPY's FULL history so the MA is already
    warm at the backtest start (a MA warming up inside the window reads as NaN
    -> False -> no entries at all for the first N sessions)."""
    c = spy["Close"]
    return c > c.rolling(ma_n).mean()


def buy_hold(prices, tickers, dates, capital) -> pd.Series:
    """Equal-weight, monthly-rebalanced hold of `tickers`, using whichever of them
    exist on each date. This is the benchmark that matters: it holds the SAME
    hand-picked universe, so beating it is evidence about the TIMING rules rather
    than about the themes having been chosen with hindsight."""
    px = pd.DataFrame({t: prices[t]["Close"] for t in tickers if t in prices})
    px = px.reindex(dates).ffill()
    r = px.pct_change(fill_method=None)
    # equal weight across names that are alive (rebalanced monthly)
    alive = px.notna()
    w = alive.div(alive.sum(axis=1).replace(0, np.nan), axis=0)
    w = w.groupby([dates.year, dates.month]).transform("first")   # hold weights within a month
    port_r = (r * w).sum(axis=1, min_count=1).fillna(0.0)
    return capital * (1 + port_r).cumprod()


def summarise(name, eq, extra=""):
    m = bs._metrics(eq)
    return dict(name=name, cagr=m["cagr"], sharpe=m["sharpe"], vol=m["vol"],
                maxdd=m["maxdd"], end=m["end"], extra=extra)


def table(rows, cap):
    w = max(len(r["name"]) for r in rows) + 2
    out = [f"{'':<{w}}{'CAGR':>8}{'Sharpe':>8}{'vol':>7}{'maxDD':>8}"
           f"{'end $' + f'{cap/1000:.0f}k':>12}   notes"]
    for r in rows:
        out.append(f"{r['name']:<{w}}{r['cagr']*100:>7.1f}%{r['sharpe']:>8.2f}"
                   f"{r['vol']*100:>6.0f}%{r['maxdd']*100:>7.0f}%{r['end']:>12,.0f}"
                   f"   {r['extra']}")
    return "\n".join(out)


def run(prices, start, end=None, capital=None, **overrides):
    """One backtest with CFG + overrides applied."""
    bs.CONFIG.update(CFG); bs.CONFIG.update(overrides)
    cap = capital or CFG["capital"]
    tradable = {t: d for t, d in prices.items() if t in UNIVERSE}
    P = bs.build_panels(tradable)
    reg = regime_series(prices["SPY"], bs.CONFIG.get("regime_ma", 200))
    res = bs.backtest(capital=cap, P=P, regime_full=reg, start=start, end=end)
    return res, P


def coverage(prices, dates):
    """How many of the universe actually existed at each point -- the honest
    answer to 'what was tradable back then'."""
    rows = []
    for y in sorted({d.year for d in dates}):
        d = [x for x in dates if x.year == y][0]
        n = sum(1 for t in UNIVERSE if t in prices and prices[t].index[0] <= d)
        rows.append((y, n))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--capital", type=float, default=100_000.0)
    ap.add_argument("--variants", action="store_true", help="sweep slots/regime/exit")
    ap.add_argument("--robust", action="store_true", help="parameter sweep + split sample")
    a = ap.parse_args()

    print("Loading ETF history (cached in data/prices_etf_max.pkl)...")
    prices = load()
    cap = a.capital

    res, P = run(prices, a.start, capital=cap)
    eq, tr = res["equity"], res["trades"]
    dates = eq.index

    print(f"\nUniverse: {len([t for t in UNIVERSE if t in prices])} ETFs "
          f"({', '.join(f'{k}: {len(v)}' for k, v in SLEEVES.items())})")
    print("Names available by year:  " +
          "  ".join(f"{y}:{n}" for y, n in coverage(prices, dates)))

    rows = [summarise("ATR breakout (5 slots)", eq,
                      f"{int((tr['side']=='BUY').sum())} buys, "
                      f"{res['avg_deploy']*100:.0f}% avg deployed")]
    rows.append(summarise("Equal-weight hold (same 29 ETFs)",
                          buy_hold(prices, UNIVERSE, dates, cap), "same universe, no timing"))
    for b in BENCHMARKS:
        if b in prices:
            s = prices[b]["Close"].reindex(dates).ffill()
            rows.append(summarise(f"{b} buy & hold", cap * s / s.iloc[0], ""))
    if "VT" in prices and "BND" in prices:
        rows.append(summarise("60/40 VT+BND", buy_hold(prices, ["VT"]*3 + ["BND"]*2, dates, cap),
                              "approx, monthly rebal"))

    print(f"\n=== ETF ATR-breakout, {bs.CONFIG['slots']} slots  "
          f"({dates[0].date()} -> {dates[-1].date()}, {len(dates)/252:.1f}y) ===")
    print(table(rows, cap))

    # calendar-year returns
    yr = eq.resample("YE").last()
    yr = pd.concat([eq.iloc[:1], yr]).pct_change().dropna()
    bh = buy_hold(prices, UNIVERSE, dates, cap)
    bh_yr = pd.concat([bh.iloc[:1], bh.resample("YE").last()]).pct_change().dropna()
    print("\ncalendar year   strategy   eq-wt hold")
    for d, v in yr.items():
        b = bh_yr.get(d, float("nan"))
        print(f"  {d.year:<12}{v*100:>7.1f}%   {b*100:>9.1f}%")

    # which ETFs actually made the money
    if not tr.empty:
        pnl = {}
        for t, g in tr.groupby("ticker"):
            buys = g[g.side == "BUY"]["value"].sum()
            sells = g[g.side == "SELL"]["value"].sum()
            open_val = 0.0
            held = g[g.side == "BUY"]["shares"].sum() - g[g.side == "SELL"]["shares"].sum()
            if held > 1e-6 and t in prices:
                open_val = held * float(prices[t]["Close"].reindex(dates).ffill().iloc[-1])
            pnl[t] = (sells + open_val - buys, int((g.side == "BUY").sum()))
        top = sorted(pnl.items(), key=lambda kv: -kv[1][0])
        print(f"\nP&L by ETF (top 8 / bottom 5 of {len(pnl)} traded):")
        for t, (v, n) in top[:8] + [("...", (float("nan"), 0))] + top[-5:]:
            print(f"  {t:<6}" + ("" if t == "..." else f"${v:>+10,.0f}   {n} entries"))

    if a.variants:
        print("\n=== variants ===")
        vrows = []
        for label, ov in [("baseline (5 slots, regime on)", {}),
                          ("3 slots", dict(slots=3)),
                          ("8 slots", dict(slots=8)),
                          ("no SPY regime filter", dict(regime=False)),
                          ("no 10-day-low exit", dict(use_exit_low=False)),
                          ("wider trail (2.5 ATR)", dict(atr_stop=2.5)),
                          ("tighter band (2.0 ATR entry)", dict(atr_break_mult=2.0)),
                          ("random ranking (control)", dict(rank_mode="random"))]:
            r, _ = run(prices, a.start, capital=cap, **ov)
            vrows.append(summarise(label, r["equity"],
                                   f"{int((r['trades']['side']=='BUY').sum())} buys"))
        print(table(vrows, cap))

    if a.robust:
        print("\n=== robustness ===")
        print("\nATR-trail sweep (is 2.5 a peak or a plateau? a spike = curve-fit):")
        srows = []
        for k in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0):
            r, _ = run(prices, a.start, capital=cap, atr_stop=k)
            srows.append(summarise(f"trail = {k} ATR", r["equity"],
                                   f"{int((r['trades']['side']=='BUY').sum())} buys"))
        print(table(srows, cap))

        print("\nSplit sample (same rules, two disjoint halves):")
        halves = [("2019-01-01", "2022-12-31"), ("2023-01-01", None)]
        rrows = []
        for lo, hi in halves:
            lbl = f"{lo[:4]}-{hi[:4] if hi else 'now'}"
            for name, ov in [("baseline", {}), ("2.5 ATR trail", dict(atr_stop=2.5)),
                             ("random rank", dict(rank_mode="random"))]:
                r, _ = run(prices, lo, end=hi, capital=cap, **ov)
                rrows.append(summarise(f"{lbl}  {name}", r["equity"]))
            d = pd.date_range(lo, hi or dates[-1], freq="D")
            sub_dates = dates[(dates >= pd.Timestamp(lo)) &
                              (dates <= (pd.Timestamp(hi) if hi else dates[-1]))]
            rrows.append(summarise(f"{lbl}  eq-wt hold",
                                   buy_hold(prices, UNIVERSE, sub_dates, cap)))
        print(table(rrows, cap))

        print("\nRanking value: momentum vs random, averaged over 12 random seeds")
        base, _ = run(prices, a.start, capital=cap)
        seeds = []
        for sd in range(12):
            np.random.seed(sd)
            r, _ = run(prices, a.start, capital=cap, rank_mode="random")
            seeds.append(bs._metrics(r["equity"]))
        print(f"  momentum rank : CAGR {bs._metrics(base['equity'])['cagr']*100:5.1f}%  "
              f"Sharpe {bs._metrics(base['equity'])['sharpe']:.2f}")
        print(f"  random rank   : CAGR {np.mean([m['cagr'] for m in seeds])*100:5.1f}%  "
              f"Sharpe {np.mean([m['sharpe'] for m in seeds]):.2f}   "
              f"(mean of 12; CAGR range {min(m['cagr'] for m in seeds)*100:.1f}"
              f"-{max(m['cagr'] for m in seeds)*100:.1f}%)")

    print(f"""
--- caveats ---------------------------------------------------------------
* SELECTION BIAS is the dominant caveat. This universe was written down in
  2026 naming AI, nuclear, quantum, GLP-1 and spot bitcoin as sleeves. Those
  are known winners. Any backtest over them is flattered by the choice of
  universe, not just by the rules. Compare against the equal-weight-hold row,
  not against SPY: that row holds the same biased universe, so the difference
  between them is the part attributable to the breakout/stop TIMING.
* Young funds: IBIT (2024-01), OZEM (2024-05), ARKX (2021-03) exist for only
  part of the window; QTUM/GLDM/BLOK/ESPO start 2018. Early years trade a
  much smaller universe -- see the by-year coverage line above.
* Liquidity: FINX, ESPO, OZEM and FIW run under ~$10M/day. 2bps per side is
  optimistic for those; they are fine at small size and not at large.
* No dividend/distribution timing effects beyond yfinance's auto-adjust, and
  no tax. TLT/BND income matters for the diversifier sleeve in particular.
* Only ~{len(UNIVERSE)} names and 5 slots: with this few candidates the ranking step
  has far less to choose between than the 223-stock version it came from.
---------------------------------------------------------------------------""")


if __name__ == "__main__":
    main()
