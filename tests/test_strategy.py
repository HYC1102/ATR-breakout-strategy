import numpy as np
import pandas as pd

import breakout_dashboard as dashboard
import breakout_sentiment as bs
import paper_trade as pt


def price_frame(dates, breakout_on_last=False):
    close = np.full(len(dates), 100.0)
    high = np.full(len(dates), 101.0)
    if breakout_on_last:
        high[-1] = 150.0
    return pd.DataFrame(
        {"Open": close, "High": high, "Low": close - 1, "Close": close,
         "Volume": np.full(len(dates), 2_000_000)},
        index=dates,
    )


def test_rank_breakouts_respects_asof_and_can_avoid_live_sentiment(monkeypatch):
    dates = pd.bdate_range("2026-01-01", periods=80)
    prices = {"TEST": price_frame(dates, breakout_on_last=True)}
    sentiment_calls = []
    monkeypatch.setattr(bs, "sentiment_score", lambda *args: sentiment_calls.append(args) or 99.0)

    historical = bs.rank_breakouts(prices, ["TEST"], asof=dates[-2], use_sentiment=False)
    current = bs.rank_breakouts(prices, ["TEST"], asof=dates[-1], use_sentiment=True)

    assert historical.empty
    assert current["ticker"].tolist() == ["TEST"]
    assert current.loc[0, "sentiment"] == 50.0
    assert sentiment_calls == []


def test_advance_uses_neutral_sentiment_for_missed_sessions(monkeypatch):
    dates = pd.bdate_range("2026-08-17", periods=3)
    close = pd.DataFrame({"TEST": [100.0, 101.0, 102.0]}, index=dates)
    plans = []

    monkeypatch.setattr(pt, "_execute", lambda *args: None)

    def fake_plan(st, day, prices, panels, regime, use_sentiment=True):
        plans.append((day, use_sentiment))
        return [], None

    monkeypatch.setattr(pt, "_plan", fake_plan)
    state = pt.init_state(dates[0].date(), 1_000)
    pt.advance(state, {}, {"close": close}, pd.Series(True, index=dates))

    assert plans == [(dates[0], False), (dates[1], False), (dates[2], True)]


def test_paper_plan_does_not_see_a_future_breakout():
    dates = pd.bdate_range("2026-01-01", periods=80)
    prices = {"TEST": price_frame(dates, breakout_on_last=True)}
    panels = bs.build_panels(prices)
    state = pt.init_state(dates[0].date(), 1_000)
    regime = pd.Series(True, index=dates)

    pending, candidates = pt._plan(
        state, dates[-2], prices, panels, regime, use_sentiment=False
    )

    assert pending == []
    assert candidates.empty


def test_paper_plan_includes_estimated_buy_quantity_and_cost(monkeypatch):
    day = pd.Timestamp("2026-08-19")
    close = pd.DataFrame({"TEST": [100.0]}, index=[day])
    candidate = pd.DataFrame([
        {"ticker": "TEST", "close": 100.0, "momentum": 80.0}
    ])
    monkeypatch.setattr(bs, "rank_breakouts", lambda *args, **kwargs: candidate)
    monkeypatch.setattr(bs, "build_universe", lambda *args, **kwargs: ["TEST"])
    monkeypatch.setitem(bs.CONFIG, "slots", 5)
    monkeypatch.setitem(bs.CONFIG, "cost_bps", 0.0)
    monkeypatch.setitem(bs.CONFIG, "sent_weight", 0.0)

    state = pt.init_state(day.date(), 1_000)
    pending, _ = pt._plan(
        state, day, {}, {"close": close}, pd.Series(True, index=[day]),
        use_sentiment=False,
    )

    assert pending == [{"side": "BUY", "ticker": "TEST", "price_hint": 100.0,
                        "shares_hint": 2.0, "value_hint": 200.0,
                        "cost_hint": 0.0, "momentum": 80.0}]


def test_broad_universe_explicitly_includes_all_three_sources(monkeypatch):
    monkeypatch.setattr(bs, "sp500_tickers", lambda: ["SPYNAME", "SHARED"])
    monkeypatch.setattr(bs, "nasdaq100_tickers", lambda: ["NDXNAME", "SHARED"])
    monkeypatch.setattr(bs, "discover_extras", lambda: ["EXTRA"])

    assert bs.broad_universe() == ["EXTRA", "NDXNAME", "SHARED", "SPYNAME"]


def test_dashboard_is_standalone_and_uses_configured_strategy_labels():
    state = {
        "capital": 1_000.0,
        "start": "2026-08-17",
        "asof": pd.Timestamp("2026-08-18"),
        "started": True,
        "value": 1_010.0,
        "positions": [{"ticker": "HELD", "shares": 2.0, "entry": 100.0,
                       "price": 183.0, "stop": 160.0, "stop_rule": "10-day low",
                       "room": 0.14375, "value": 366.0, "ret": 0.83,
                       "pnl": 166.0, "source": "Tiingo",
                       "asof_date": pd.Timestamp("2026-08-18").date()}],
        "pending": [{"side": "BUY", "ticker": "TEST", "price_hint": 100.0,
                     "shares_hint": 2.0, "value_hint": 200.0,
                     "cost_hint": 0.0, "momentum": 80.0}],
        "trades": [],
        "equity": [{"date": "2026-08-17", "value": 1_000.0},
                   {"date": "2026-08-18", "value": 1_010.0}],
        "cand": pd.DataFrame(),
        "spy_line": [1_000.0, 1_005.0],
        "m": {"ret": 0.01, "day_pnl": 10.0, "day_ret": 0.01,
              "sharpe": 1.0, "vol": 0.1, "maxdd": 0.0,
              "win": np.nan, "avg_hold": np.nan, "n_closed": 0},
    }

    page = dashboard.html(state)

    assert f"{bs.CONFIG['slots']}-slot swing" in page
    assert f"top {bs.CONFIG['slots']} fill the available slots" in page.lower()
    assert "EMA(20) + 3-ATR breakout" in page
    assert "Chart.js" not in page
    assert "cdnjs.cloudflare.com" not in page
    assert "Today&rsquo;s P&amp;L" in page
    assert "+$10" in page
    assert "Est. shares" in page
    assert "~2.000" in page
    assert "Est. total value" in page
    assert "~$200.00" in page
    assert "Est. cost" not in page
    assert ">$+166<" not in page
    assert ">+166<" in page
    assert 'id="eqTip0"' in page          # ids are suffixed per panel since the tabs landed
    assert "ctx.arc" in page


# --------------------------------------------------------------------------- #
# Split / price-basis handling
#
# APH went 2-for-1 ex 2026-09-03. Ahead of the ex-date yfinance restated its
# price HISTORY onto the post-split basis while still serving recent bars
# pre-split, leaving a 2x step in the middle of the series. momentum_proxy read
# 160.08 / ~74 - 1 = +116% against a true 63-day move of +8-11%, ATR came out
# ~5.4% of price against a normal ~2%, and that was enough to rank APH #2 and
# queue a buy on a signal that did not exist.
# --------------------------------------------------------------------------- #
def _restated_series(n=120, step_at=-40, level=160.0, factor=2.0):
    """A series whose older half sits on the post-split basis and whose recent
    half does not -- exactly the shape yfinance served for APH."""
    dates = pd.bdate_range("2026-03-16", periods=n)
    close = np.full(n, level, dtype=float)
    close[:step_at] = level / factor                 # history already halved
    close += np.linspace(0, 2.0, n)                  # mild drift so it is not flat
    return pd.DataFrame(
        {"Open": close, "High": close * 1.004, "Low": close * 0.996, "Close": close,
         "Volume": np.full(n, 4_000_000)},
        index=dates,
    )


def test_basis_break_flags_a_split_restated_series():
    restated = _restated_series()
    clean = price_frame(pd.bdate_range("2026-03-16", periods=120))

    assert bs.basis_break(restated, 63) is True          # size alone, unverified
    assert bs.basis_break(clean, 63) is False
    # the artefact is exactly what inflated APH's momentum
    assert bs.momentum_proxy(restated, 63) > 0.8


def test_basis_break_needs_a_split_to_blame(monkeypatch):
    """A large move is only a basis change when a reported split explains it.
    MRNA really did move +177% on 2026-08-19 with no split, and must stay
    tradable; APH's identical-looking step had a split against it."""
    import types
    restated = _restated_series()
    step_date = restated["Close"].pct_change().abs().idxmax()

    def with_splits(series):
        bs._SPLITS_CACHE.clear()
        monkeypatch.setattr(bs.yf, "Ticker",
                            lambda sym: types.SimpleNamespace(splits=series))

    with_splits(pd.Series([2.0], index=pd.DatetimeIndex([step_date])))
    assert bs.basis_break(restated, 63, ticker="APH") is True

    with_splits(pd.Series(dtype=float))                  # no splits reported
    assert bs.basis_break(restated, 63, ticker="MRNA") is False

    # a split reported far from the step does not explain it
    with_splits(pd.Series([2.0], index=pd.DatetimeIndex(["2024-06-12"])))
    assert bs.basis_break(restated, 63, ticker="APH") is False
    bs._SPLITS_CACHE.clear()


def test_rank_breakouts_excludes_split_restated_names(capsys):
    dates = pd.bdate_range("2026-03-16", periods=120)
    restated = _restated_series()
    restated.loc[restated.index[-1], "High"] = 400.0        # force a breakout too

    clean = price_frame(dates, breakout_on_last=True)

    bs.CONFIG.update(entry="atr", atr_break_period=20, atr_break_mult=3.0,
                     breakout=20, proxy_window=63, atr_window=14,
                     atr_stop=1.5, sent_weight=0.0)
    step_date = restated["Close"].pct_change().abs().idxmax()
    bs._SPLITS_CACHE.clear()
    bs._SPLITS_CACHE["APH"] = pd.Series([2.0], index=pd.DatetimeIndex([step_date]))
    bs._SPLITS_CACHE["GOOD"] = pd.Series(dtype=float)
    ranked = bs.rank_breakouts({"APH": restated, "GOOD": clean},
                               ["APH", "GOOD"], use_sentiment=False)
    bs._SPLITS_CACHE.clear()

    assert "APH" not in list(ranked["ticker"]), "split-restated name must not be tradable"
    assert "GOOD" in list(ranked["ticker"])
    assert "split-restated" in capsys.readouterr().out


def test_reconcile_splits_rebases_a_held_position(monkeypatch):
    """A confirmed split scales shares up and entry/hi down, so the stop stays
    meaningful instead of force-exiting the position at a fabricated loss."""
    asof = pd.Timestamp("2026-09-03")
    P = {"close": pd.DataFrame({"APH": [80.0]}, index=[asof])}
    st = {"last_close": str(asof.date()),
          "positions": {"APH": {"shares": 10.0, "entry": 150.0, "hi": 160.0,
                                "last_px": 160.0}}}

    monkeypatch.setattr(pt, "_split_factor", lambda *a, **k: 2.0)
    applied = pt.reconcile_splits(st, P, asof)

    assert applied == [("APH", 2.0)]
    p = st["positions"]["APH"]
    assert p["shares"] == 20.0                     # share count doubles
    assert p["entry"] == 75.0 and p["hi"] == 80.0   # per-share levels halve
    assert p["shares"] * p["last_px"] == 1_600.0    # book value unchanged
    assert st["splits"] == [{"date": "2026-09-03", "ticker": "APH", "factor": 2.0}]

    # and the stop is now sane rather than stranded above the price
    assert p["hi"] - 1.5 * 1.7 < 80.0


def test_reconcile_splits_ignores_a_real_move_with_no_split(monkeypatch):
    """A genuine -33% day also gives a ~1.5 ratio, which is a real 3-for-2 split
    ratio. Without calendar confirmation the position must be left alone."""
    asof = pd.Timestamp("2026-09-03")
    P = {"close": pd.DataFrame({"BIO": [100.0]}, index=[asof])}
    st = {"last_close": str(asof.date()),
          "positions": {"BIO": {"shares": 10.0, "entry": 140.0, "hi": 150.0,
                                "last_px": 150.0}}}

    monkeypatch.setattr(pt, "_split_factor", lambda *a, **k: None)   # calendar: no split
    assert pt.reconcile_splits(st, P, asof) == []
    assert st["positions"]["BIO"] == {"shares": 10.0, "entry": 140.0,
                                      "hi": 150.0, "last_px": 150.0}


def test_split_factor_requires_calendar_confirmation(monkeypatch):
    """_split_factor accepts only a ratio the vendor's calendar actually reports
    near the date, and returns None whenever the lookup is unhelpful."""
    import types

    del types
    bs._SPLITS_CACHE.clear()
    bs._SPLITS_CACHE["APH"] = pd.Series([2.0], index=pd.DatetimeIndex(["2026-09-03"]))

    assert pt._split_factor("APH", pd.Timestamp("2026-09-03"), 2.0) == 2.0
    # ratio that does not match the reported factor
    assert pt._split_factor("APH", pd.Timestamp("2026-09-03"), 1.5) is None
    # split reported far from the date
    assert pt._split_factor("APH", pd.Timestamp("2026-12-01"), 2.0) is None
    # a ratio too close to 1 is never a basis change
    assert pt._split_factor("APH", pd.Timestamp("2026-09-03"), 1.02) is None
    # no split information at all -> leave the position alone
    bs._SPLITS_CACHE["APH"] = pd.Series(dtype=float)
    assert pt._split_factor("APH", pd.Timestamp("2026-09-03"), 2.0) is None
    bs._SPLITS_CACHE.clear()
