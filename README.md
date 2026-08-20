# ATR Breakout Strategy

A self-contained, slot-based swing trader on liquid US large caps. Split out of
the main repo so it runs independently of the Diversified Trend Sleeve
(`../strategy.py`, `../dashboard.py`) — nothing here imports that code.

All commands below assume you are **inside this folder** (paths are relative to
the working directory).

## Files

| File | Role |
|---|---|
| `breakout_sentiment.py` | The strategy: universe, ATR-band entry, ranking, event-driven backtest |
| `paper_trade.py` | Persistent forward paper-trade (state in `data/paper_breakout.json`) |
| `prices.py` | Tiingo-first price adapter with a yfinance fallback |
| `breakout_dashboard.py` | Advances the paper account and renders `breakout_dashboard.html` |
| `data/` | Price caches, weekly universe snapshots, sentiment cache, paper-trade state |

## How it works

**Universe** — S&P 500 + Nasdaq-100 + liquid extras, narrowed to the top
**223 names by 60-day average dollar volume**, rebuilt weekly.

**Entry** (`entry="atr"`) — today's *High* pierces yesterday's Keltner-style
upper band, `EMA(20) + 3.0 × ATR(20)`. A `donchian` mode (close above the prior
20-day high) is available. New positions only open when **SPY > its 200-day MA**;
exits fire in any regime.

**Ranking** — more breakouts fire than there are slots, so candidates are ranked
by the percentile of their **63-day trailing return** and the strongest momentum
names win. Live trading and the backtest now use the same momentum-only ranking.
The VADER sentiment utilities remain in the code but are disabled
(`sent_weight=0`) and are not called during candidate selection.

If the paper account catches up after a missed run, historical price signals are
evaluated strictly as of each session.

**Sizing** — **5 concurrent slots**, each entry takes 1/5 of equity and is then
left alone (no rebalancing). Empty slots sit in cash.

**Exit** — whichever hits first: a trailing ATR stop at
`highest close since entry − 1.5 × ATR(14)`, or a close below the **prior
10-day low**. Both are evaluated on the close and filled at the next open.
Setting `intraday_stop=True` switches the ATR stop to a resting intraday order
(filled when the low touches the level); it is **off** by default, so headline
backtest numbers are close-based. Take-profit, time-stop and hard % stop exist
but are off.

Signals fire on the close and fill at the **next open** — no look-ahead.
Transaction costs are not modeled (`cost_bps=0`).

## Running

```bash
pip install -r requirements.txt
# For contributors/testing instead: pip install -r requirements-dev.txt

python breakout_sentiment.py --backtest        # historical simulation
python breakout_dashboard.py --capital 8800    # advance the paper account -> breakout_dashboard.html
pytest                                         # offline regression suite
```

`prices.py` reads a Tiingo token from `$TIINGO_API_KEY`, or from a gitignored
`.tiingo_token` / `data/tiingo_token.txt`. Without one it falls back to yfinance.

## Automatic dashboard refresh

`.github/workflows/dashboard.yml` runs at **4:30 PM America/New_York**, Monday
through Friday. It runs the regression suite, advances the paper account once,
renders `breakout_dashboard.html`, and commits the dashboard plus JSON/CSV paper
state to the default branch. The workflow can also be run manually from the
GitHub Actions page.

Repository contents require write permission for GitHub Actions; the workflow
declares that permission explicitly. For Tiingo-backed fills and position marks,
add a repository Actions secret named `TIINGO_API_KEY`. If it is absent, the
existing yfinance fallback is used.

## Honest-data caveats

* The universe uses **today's** index membership → survivorship bias, so
  backtested returns are optimistic.
* Live trading and the backtest both rank on price momentum; the disabled
  sentiment utility is retained only for possible future research.
* Price and sentiment caches are local, regenerable artifacts and are ignored by
  Git. The small JSON/CSV paper-account record remains versioned deliberately.
* The test suite rejects any tracked file larger than 10 MiB, preventing market
  data caches or other large artifacts from being added to future commits.
