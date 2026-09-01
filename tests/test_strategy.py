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
