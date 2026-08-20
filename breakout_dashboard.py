"""
Live dashboard for the breakout / slot-based swing strategy (separate from the
Diversified Trend Sleeve dashboard). Drives off the forward paper-trading state in
paper_trade.py (data/paper_breakout.json) so it tracks a genuine forward record:

  * Today's actions  orders to place at the next open (exits + sentiment-ranked buys)
  * Breakouts today  ranked by a MOMENTUM index + news SENTIMENT (combined)
  * Positions        current book with live stop trigger + cushion
  * Trades           the full forward log (with the sentiment captured at entry)
  * Measures         return / Sharpe / drawdown / win rate since the start date

Generates breakout_dashboard.html. Run daily to advance the paper account.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd
import yfinance as yf

import breakout_sentiment as bs
import paper_trade as pt
import prices as px                 # Tiingo-first price adapter (local)

CSS = """
:root{--bg:#fbfbfa;--card:#fff;--ink:#1a1a19;--mut:#6b6a66;--line:#e6e4dd;
--blue:#2a78d6;--green:#1baf7a;--amber:#c98500;--red:#c0392b}
*{box-sizing:border-box;margin:0}
.dash{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);
color:var(--ink);padding:28px;line-height:1.5;max-width:960px;margin:0 auto}
h1{font-size:22px;font-weight:600}h2{font-size:16px;font-weight:600;margin:26px 0 12px}
.sub{color:var(--mut);font-size:13px;margin-top:2px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin:16px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .l{font-size:12px;color:var(--mut)}.card .v{font-size:21px;font-weight:600;margin-top:3px}
table{width:100%;border-collapse:collapse;font-size:13.5px;background:var(--card);
border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:8px 11px;text-align:left;border-bottom:1px solid var(--line)}
th{font-size:12px;color:var(--mut);font-weight:500;background:#f6f5f1}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.buy{color:var(--green);font-weight:600}.sell{color:var(--red);font-weight:600}
.pos{color:var(--green)}.neg{color:var(--red)}.amber{color:var(--amber)}
.hold{background:#eef7f2}
.pill{display:inline-block;background:#eef4fb;color:var(--blue);font-size:12px;padding:2px 9px;border-radius:20px}
.tag{font-size:11px;color:var(--amber);background:#fdf6e9;padding:1px 7px;border-radius:10px}
.act{background:#f2f7fd;border:1px solid #d7e6f7;border-radius:10px;padding:14px 16px;margin:14px 0}
.foot{color:var(--mut);font-size:12px;margin-top:24px;border-top:1px solid var(--line);padding-top:12px}
.tg{background:#fff;border:1px solid var(--line);border-radius:8px;padding:5px 13px;font-size:13px;
cursor:pointer;margin-right:6px;color:var(--mut)}
.tg.on{background:var(--blue);color:#fff;border-color:var(--blue)}
.leg{display:inline-flex;gap:14px;font-size:12px;color:var(--mut);margin-left:6px}
.leg span{display:inline-flex;align-items:center;gap:5px}
.sw{width:13px;height:3px;display:inline-block}
"""


def episodes(trades):
    """Closed round-trips from the trade log (for win rate / avg hold)."""
    ep, closed = {}, []
    for t in sorted(trades, key=lambda x: x["date"]):
        e = ep.setdefault(t["ticker"], dict(cost=0.0, proceeds=0.0, shares=0.0,
                                            entry=t["date"], exit=t["date"]))
        if t["side"] == "BUY":
            e["cost"] += t["value"]; e["shares"] += t["shares"]
        else:
            e["proceeds"] += t["value"]; e["shares"] -= t["shares"]; e["exit"] = t["date"]
        if e["shares"] <= 1e-6 and e["cost"] > 0:
            closed.append(dict(ticker=t["ticker"], pnl=e["proceeds"] - e["cost"],
                               entry=e["entry"], exit=e["exit"]))
            del ep[t["ticker"]]
    return closed


def overlay_tiingo(prices, P, names) -> list[str]:
    """Replace yfinance OHLC with Tiingo's (more reliable) values for `names` --
    the held positions and pending buys -- in BOTH the prices dict and the P
    panels, so marks, stops and fills use Tiingo. The bulk 223-name breakout scan
    (detection + ranking) stays on yfinance. Per-cell fallback to yfinance where
    Tiingo lacks a date; a no-op (returns []) if there is no Tiingo token."""
    names = [t for t in dict.fromkeys(names) if t]        # de-dupe, keep order
    if not names:
        return []
    idx = P["close"].index
    start = str((idx[-1] - pd.Timedelta(days=400)).date())  # enough for stops + recent marks
    used = []
    for tk in names:
        try:
            tg = px.tiingo_prices(tk, start)
        except Exception:  # noqa: BLE001
            tg = None
        if tg is None or tg.empty:
            continue
        for field, col in (("open", "Open"), ("close", "Close"),
                           ("high", "High"), ("low", "Low")):
            if tk in P[field].columns:
                P[field][tk] = tg[col].reindex(idx).fillna(P[field][tk])
        if tk in prices:
            df = prices[tk].copy()
            for col in ("Open", "High", "Low", "Close"):
                df[col] = tg[col].reindex(df.index).fillna(df[col])
            prices[tk] = df
        used.append(tk)
    return used


def build(capital: float, start: str):
    bs.CONFIG.update(sizing="slots", slots=5, regime=True, rank_mode="proxy",
                     sent_weight=0.0,
                     cost_bps=0.0,
                     atr_stop=1.5, exit_low=10, use_exit_low=True, pct_stop=None,
                     take_profit=None, time_stop=None, universe_size=223,
                     rebuild="W", pool="broad")

    prices = bs.download_prices(bs.broad_universe(), period="3y")
    P = bs.build_panels(prices)
    regime = bs.spy_regime(P["close"].index)

    st = pt.load_state() or pt.init_state(start, capital)
    # Route the decision-critical few (held + pending buys) through Tiingo, so
    # marks/stops/fills use reliable prices while the 223-name scan stays on
    # yfinance (Tiingo's free tier can't do 223 queries).
    overlay = set(st["positions"]) | {o["ticker"] for o in st.get("pending", [])
                                      if o.get("side") == "BUY"}
    tiingo_names = overlay_tiingo(prices, P, overlay)
    if tiingo_names:
        print(f"  Tiingo prices for {len(tiingo_names)} name(s): {', '.join(tiingo_names)}")
    st, asof, cand = pt.advance(st, prices, P, regime)
    pt.save_state(st)

    started = len(st["equity"]) > 0
    value = st["equity"][-1]["value"] if started else st["capital"]

    # Positions with live stop + cushion. A held name can be missing from the
    # current download (delisted, renamed, or a short/failed batch pull); _plan()
    # holds it rather than force-selling, so render it here too instead of dying
    # on a KeyError. NB: do not bind the name `px` in this scope -- it is the
    # `prices` module, used further down for the SPY benchmark.
    positions = []
    for tk, p in st["positions"].items():
        col = P["close"][tk] if tk in P["close"].columns else None
        mark = float(col.loc[asof]) if col is not None else np.nan
        if col is not None and not np.isfinite(mark):
            last_ok = col.last_valid_index()               # fall back to the last real close
            mark = float(col.loc[last_ok]) if last_ok is not None else np.nan
        stale = not np.isfinite(mark)
        if stale:
            mark = float(p["entry"])                       # nothing to mark with -> flat at entry
        try:
            stop, rule = pt.stop_level(p, tk, prices)
        except Exception:  # noqa: BLE001
            stop, rule = 0.0, "no data"
        if not np.isfinite(stop):
            stop, rule = 0.0, "no data"
        last_dt = prices[tk]["Close"].last_valid_index() if tk in prices else None
        positions.append(dict(ticker=tk, shares=p["shares"], entry=p["entry"],
                              entry_date=p["entry_date"], price=mark, value=p["shares"] * mark,
                              pnl=p["shares"] * (mark - p["entry"]),
                              ret=mark / p["entry"] - 1 if p["entry"] else 0.0,
                              stop=stop, stop_rule=rule,
                              room=mark / stop - 1 if stop > 0 else 0,
                              source="no data" if stale else
                                     ("Tiingo" if tk in tiingo_names else "yfinance"),
                              asof_date=last_dt.date() if last_dt is not None else None))
    positions.sort(key=lambda x: -x["value"])

    # measures from the forward equity log
    m = dict(ret=0.0, day_pnl=0.0, day_ret=0.0, sharpe=float("nan"),
             maxdd=0.0, vol=float("nan"), win=float("nan"),
             avg_hold=float("nan"), n_closed=0)
    if started:
        eq = pd.Series([e["value"] for e in st["equity"]],
                       index=pd.to_datetime([e["date"] for e in st["equity"]]))
        r = eq.pct_change().dropna()
        closed = episodes(st["trades"])
        wins = [c for c in closed if c["pnl"] > 0]
        previous_value = eq.iloc[-2] if len(eq) >= 2 else st["capital"]
        day_pnl = value - previous_value
        m = dict(ret=value / st["capital"] - 1,
                 day_pnl=day_pnl,
                 day_ret=day_pnl / previous_value if previous_value else 0.0,
                 sharpe=(r.mean() / r.std() * np.sqrt(252)) if len(r) > 1 and r.std() else float("nan"),
                 vol=r.std() * np.sqrt(252) if len(r) > 1 else float("nan"),
                 maxdd=(eq / eq.cummax() - 1).min(),
                 win=len(wins) / len(closed) if closed else float("nan"),
                 avg_hold=np.mean([(pd.Timestamp(c["exit"]) - pd.Timestamp(c["entry"])).days
                                   for c in closed]) if closed else float("nan"),
                 n_closed=len(closed))

    # SPY benchmark, rebased to the account's starting capital, aligned to the
    # equity log's dates -- for the chart's Strategy-vs-SPY comparison.
    spy_line = []
    if started:
        try:
            spy_full = px.load_prices("SPY", st["start_date"])["Close"]
            dates = pd.to_datetime([e["date"] for e in st["equity"]])
            spy_close = spy_full.reindex(spy_full.index.union(dates)).ffill().reindex(dates)
            if spy_close.notna().all() and spy_close.iloc[0] > 0:
                spy_line = list((st["capital"] * spy_close / spy_close.iloc[0]).round(2))
        except Exception:  # noqa: BLE001
            spy_line = []

    if cand is None:            # risk-off or no free slot -> _plan skipped the scan
        cand = bs.rank_breakouts(prices, bs.build_universe(prices))
    return dict(capital=st["capital"], start=st["start_date"], asof=asof, started=started,
                value=value, positions=positions, pending=st["pending"], trades=st["trades"],
                equity=st["equity"], cand=cand, m=m, spy_line=spy_line)


def html(s) -> str:
    cap, m = s["capital"], s["m"]
    slots = bs.CONFIG["slots"]
    momentum_only = bs.CONFIG.get("sent_weight", 0.0) == 0
    if bs.CONFIG.get("entry", "atr") == "atr":
        entry_label = (f"EMA({bs.CONFIG['atr_break_period']}) + "
                       f"{bs.CONFIG['atr_break_mult']:g}-ATR breakout")
    else:
        entry_label = f"{bs.CONFIG['breakout']}-day Donchian breakout"
    rc = lambda x: "pos" if x >= 0 else "neg"
    signed_money = lambda x: f'{"+" if x >= 0 else "-"}${abs(x):,.0f}'
    roomcls = lambda r: "neg" if r < 0.03 else ("amber" if r < 0.08 else "pos")
    start_d = dt.date.fromisoformat(s["start"])

    # header cards
    if s["started"]:
        cards = [("Account value", f"${s['value']:,.0f}", ""),
                 ("Today&rsquo;s P&amp;L", signed_money(m.get("day_pnl", 0.0)),
                  rc(m.get("day_pnl", 0.0))),
                 ("Return", f"{m['ret']*100:+.1f}%", rc(m["ret"])),
                 ("Sharpe", f"{m['sharpe']:.2f}" if np.isfinite(m['sharpe']) else "—", ""),
                 ("Max drawdown", f"{m['maxdd']*100:.0f}%", rc(m["maxdd"])),
                 ("Win rate", f"{m['win']*100:.0f}%" if np.isfinite(m['win']) else "—", "")]
        status = f"as of {s['asof']:%d %b %Y}"
    else:
        cards = [("Starting capital", f"${cap:,.0f}", ""),
                 ("Status", "starts tomorrow", ""),
                 ("Start date", f"{start_d:%d %b}", "")]
        status = f"forward test begins {start_d:%A %d %b %Y}"

    cards_html = "".join(f'<div class="card"><div class="l">{l}</div>'
                         f'<div class="v {cls}">{v}</div></div>' for l, v, cls in cards)

    # TODAY'S ACTIONS (top)
    pend = s["pending"]
    if pend:
        rows = ""
        for o in pend:
            if o["side"] == "SELL":
                rows += (f'<tr><td class="sell">SELL</td><td><b>{o["ticker"]}</b></td>'
                         f'<td colspan="{4 if momentum_only else 6}" style="color:var(--mut)">'
                         f'exit — {o.get("reason","")}</td></tr>')
            else:
                rows += (f'<tr><td class="buy">BUY</td><td><b>{o["ticker"]}</b></td>'
                         f'<td class="n">~${o.get("price_hint",0):,.2f}</td>'
                         f'<td class="n">~{o.get("shares_hint",0):,.3f}</td>'
                         f'<td class="n">~${o.get("value_hint",0):,.2f}</td>'
                         f'<td class="n"><b>{o.get("momentum","")}</b></td>'
                         + ('' if momentum_only else
                            f'<td class="n">{o.get("sentiment","")}</td>'
                            f'<td class="n"><b>{o.get("combined","")}</b></td>')
                         + '</tr>')
        when = "at the open on " + (f"{start_d:%a %d %b}" if not s["started"] else "the next session")
        score_headers = ("<th class=\"n\">Momentum</th>" if momentum_only else
                         "<th class=\"n\">Mom</th><th class=\"n\">Sent</th>"
                         "<th class=\"n\">Combined</th>")
        actions = (f'<div class="act"><b>Today&rsquo;s actions</b> &mdash; place {when}:'
                   f'<table style="margin-top:8px"><tr><th>Action</th><th>Ticker</th>'
                   f'<th class="n">~Price</th><th class="n">Est. shares</th>'
                   f'<th class="n">Est. total value</th>'
                   f'{score_headers}</tr>{rows}</table>'
                   '<p class="sub" style="margin-top:7px">Buy quantities are estimates using '
                   'the latest close; actual fills are sized again at the next open.</p></div>')
    else:
        actions = ('<div class="act"><b>Today&rsquo;s actions</b> &mdash; none. '
                   'Hold current positions; no qualifying breakouts (or risk-off regime).</div>')

    # today's breakouts
    c = s["cand"]
    if len(c):
        held = {p["ticker"] for p in s["positions"]}
        crows = ""
        for i, r in c.head(15).iterrows():
            tag = ' <span class="tag" style="color:var(--green);background:#eef7f2">top pick</span>' if i < 3 else ""
            if r.ticker in held:
                tag = ' <span class="tag">held</span>'
            cls = ' class="hold"' if r.ticker in held else ""
            crows += (f'<tr{cls}><td>{i+1}</td><td><b>{r.ticker}</b>{tag}</td>'
                      f'<td class="n">${r.close:,.2f}</td><td class="n">{r.mom_ret*100:+.0f}%</td>'
                      f'<td class="n"><b>{r.momentum:.0f}</b></td>'
                      + ('' if momentum_only else
                         f'<td class="n">{r.sentiment:.0f}</td>'
                         f'<td class="n"><b>{r.combined:.0f}</b></td>')
                      + f'<td class="n">${r.stop:,.2f}</td></tr>')
        score_headers = ("" if momentum_only else
                         '<th class="n">Sentiment</th><th class="n">Combined</th>')
        cand_html = (f'<table><tr><th>#</th><th>Ticker</th><th class="n">Price</th>'
                     f'<th class="n">3-mo mom</th><th class="n">Momentum idx</th>'
                     f'{score_headers}<th class="n">ATR stop</th></tr>{crows}</table>')
    else:
        cand_html = '<p class="sub">No fresh breakouts in the universe today.</p>'

    # positions
    if s["positions"]:
        def prow(p):
            src_cell = (f'{p["source"]}<br><span style="font-size:10px;color:var(--mut)">'
                       f'{p["asof_date"]:%d %b %y}</span>' if p["asof_date"] else p["source"])
            return (f'<tr><td><b>{p["ticker"]}</b></td><td class="n">{p["shares"]:.1f}</td>'
                   f'<td class="n">${p["entry"]:,.2f}</td><td class="n">${p["price"]:,.2f}</td>'
                   f'<td class="n">${p["stop"]:,.2f}<br><span style="font-size:10px;color:var(--mut)">'
                   f'{p["stop_rule"]}</span></td>'
                   f'<td class="n {roomcls(p["room"])}">{p["room"]*100:+.1f}%</td>'
                   f'<td class="n">${p["value"]:,.0f}</td>'
                   f'<td class="n {rc(p["ret"])}">{p["ret"]*100:+.1f}%</td>'
                   f'<td class="n {rc(p["pnl"])}">{p["pnl"]:+,.0f}</td>'
                   f'<td>{src_cell}</td></tr>')
        prows = "".join(prow(p) for p in s["positions"])
        pos_html = (f'<table><tr><th>Ticker</th><th class="n">Shares</th><th class="n">Entry</th>'
                    f'<th class="n">Price</th><th class="n">Stop</th><th class="n">Room</th>'
                    f'<th class="n">Value</th><th class="n">Return</th><th class="n">P&amp;L $</th>'
                    f'<th>Source / as of</th></tr>{prows}</table>')
    else:
        pos_html = '<p class="sub">No open positions yet.</p>'

    # trade log (most recent first) with sentiment
    if s["trades"]:
        trows = ""
        for t in sorted(s["trades"], key=lambda x: x["date"], reverse=True)[:20]:
            sent = t.get("sentiment", "")
            trows += (f'<tr><td>{t["date"]}</td>'
                      f'<td class="{"buy" if t["side"]=="BUY" else "sell"}">{t["side"]}</td>'
                      f'<td><b>{t["ticker"]}</b></td><td class="n">{t["shares"]:.1f}</td>'
                      f'<td class="n">${t["price"]:,.2f}</td><td class="n">${t["value"]:,.0f}</td>'
                      f'<td class="n">{sent if sent not in (None,"") else "&mdash;"}</td>'
                      f'<td style="color:var(--mut)">{t.get("reason","")}</td></tr>')
        trades_html = (f'<table><tr><th>Date</th><th>Action</th><th>Ticker</th><th class="n">Shares</th>'
                       f'<th class="n">Price</th><th class="n">$ value</th><th class="n">Sent</th>'
                       f'<th>Note</th></tr>{trows}</table>')
    else:
        trades_html = '<p class="sub">No trades yet — the log fills from the first session.</p>'

    # equity chart (once there is a track record): value/return toggle, vs SPY
    chart = ""
    if len(s["equity"]) >= 2:
        ed = json.dumps([e["date"][5:] for e in s["equity"]])
        ev = json.dumps([e["value"] for e in s["equity"]])
        spy = s.get("spy_line") or []
        has_spy = len(spy) == len(s["equity"])
        spy_j = json.dumps(spy)
        spy_var = f",SPY={spy_j}" if has_spy else ""
        spy_series = ',{name:"SPY",values:SPY,color:"#898781",dash:[4,3]}' if has_spy else ""
        spy_legend = ('<span><span class="sw" style="background:#898781"></span>SPY</span>'
                      if has_spy else "")
        # Dependency-free canvas chart keeps the generated dashboard genuinely
        # standalone and usable without access to a CDN.
        chart = f'''
<div style="margin:6px 0 10px">
<button id="bV" class="tg on">Account value ($)</button>
<button id="bR" class="tg">Return (%)</button>
<span class="leg"><span><span class="sw" style="background:#2a78d6"></span>Strategy</span>
{spy_legend}</span></div>
<div style="position:relative;height:220px"><canvas id="eq" style="width:100%;height:100%"></canvas>
<div id="eqTip" style="display:none;position:absolute;pointer-events:none;background:#fff;
border:1px solid #d7d5ce;border-radius:7px;padding:6px 8px;font:12px -apple-system,sans-serif;
box-shadow:0 2px 8px #0002;white-space:nowrap;z-index:2"></div></div>
<script>(()=>{{
const ED={ed},EV={ev}{spy_var};
const base=[{{name:"Strategy",values:EV,color:"#2a78d6",dash:[]}}{spy_series}];
const cv=document.getElementById("eq"),ctx=cv.getContext("2d"),tip=document.getElementById("eqTip");
let mode="value",hover=-1,chartState=null;
const rb=a=>a.map(v=>(v/a[0]-1)*100);
function draw(){{
 const dpr=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
 cv.width=w*dpr;cv.height=h*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
 const series=base.map(s=>({{...s,values:mode==="return"?rb(s.values):s.values}}));
 const vals=series.flatMap(s=>s.values),lo=Math.min(...vals),hi=Math.max(...vals);
 const pad={{l:62,r:12,t:10,b:25}},pw=w-pad.l-pad.r,ph=h-pad.t-pad.b;
 const span=hi-lo||1,x=i=>pad.l+(ED.length<2?0:i/(ED.length-1))*pw;
 const y=v=>pad.t+(hi-v)/span*ph;
 ctx.clearRect(0,0,w,h);ctx.font="11px -apple-system,sans-serif";ctx.fillStyle="#898781";
 ctx.strokeStyle="#e6e4dd";ctx.lineWidth=1;
 for(let i=0;i<5;i++){{const v=lo+span*i/4,yy=y(v);ctx.beginPath();ctx.moveTo(pad.l,yy);ctx.lineTo(w-pad.r,yy);ctx.stroke();
  const label=mode==="return"?v.toFixed(1)+"%":"$"+Math.round(v).toLocaleString();ctx.fillText(label,2,yy+4);}}
 const ticks=Math.min(6,ED.length);for(let i=0;i<ticks;i++){{const j=Math.round(i*(ED.length-1)/(ticks-1||1));ctx.fillText(ED[j],Math.min(x(j)-16,w-45),h-5);}}
 for(const s of series){{
  ctx.strokeStyle=s.color;ctx.lineWidth=s.name==="Strategy"?2:1.5;ctx.setLineDash(s.dash);
  ctx.beginPath();s.values.forEach((v,i)=>{{i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v));}});ctx.stroke();
  ctx.setLineDash([]);ctx.fillStyle=s.color;
  s.values.forEach((v,i)=>{{ctx.beginPath();ctx.arc(x(i),y(v),hover===i?4:2.5,0,Math.PI*2);ctx.fill();}});
 }}
 ctx.setLineDash([]);
 chartState={{series,pad,pw,w,h,x}};
}}
function setMode(m){{mode=m;draw();document.getElementById("bV").classList.toggle("on",m==="value");document.getElementById("bR").classList.toggle("on",m==="return");}}
document.getElementById("bV").onclick=()=>setMode("value");document.getElementById("bR").onclick=()=>setMode("return");
cv.addEventListener("mousemove",e=>{{
 if(!chartState)return;
 const rect=cv.getBoundingClientRect(),mx=e.clientX-rect.left;
 hover=Math.max(0,Math.min(ED.length-1,Math.round((mx-chartState.pad.l)/chartState.pw*(ED.length-1))));
 draw();
 const values=chartState.series.map(s=>{{
  const v=s.values[hover],shown=mode==="return"?(v>=0?"+":"")+v.toFixed(2)+"%":"$"+v.toLocaleString(undefined,{{minimumFractionDigits:2,maximumFractionDigits:2}});
  return `<span style="color:${{s.color}}">●</span> ${{s.name}}: <b>${{shown}}</b>`;
 }}).join("<br>");
 tip.innerHTML=`<b>${{ED[hover]}}</b><br>${{values}}`;tip.style.display="block";
 tip.style.left=Math.max(4,Math.min(chartState.x(hover)+10,chartState.w-tip.offsetWidth-4))+"px";
 tip.style.top="8px";
}});
cv.addEventListener("mouseleave",()=>{{hover=-1;tip.style.display="none";draw();}});
new ResizeObserver(draw).observe(cv.parentElement);draw();
}})();</script>'''

    meas = ""
    if s["started"]:
        # A one-day-old account has <2 returns, so sharpe/vol/win are NaN --
        # show an em-dash rather than a literal "nan", as the header cards do.
        def num(v, spec, scale=1.0, suffix=""):
            return format(v * scale, spec) + suffix if np.isfinite(v) else "&mdash;"
        card = lambda l, v, cls="": (f'<div class="card"><div class="l">{l}</div>'
                                     f'<div class="v {cls}">{v}</div></div>')
        meas = ('<h2>Strategy measures</h2><div class="cards">'
                + card("Total return", num(m["ret"], "+.1f", 100, "%"), rc(m["ret"]))
                + card("Sharpe", num(m["sharpe"], ".2f"))
                + card("Volatility", num(m["vol"], ".0f", 100, "%"))
                + card("Max drawdown", num(m["maxdd"], ".0f", 100, "%"))
                + card("Round-trips", str(m["n_closed"]))
                + card("Win rate", num(m["win"], ".0f", 100, "%"))
                + '</div>')

    return f"""<!doctype html><meta charset="utf-8"><title>Breakout strategy</title>
<style>{CSS}</style><div class="dash">
<h1>Breakout Momentum &mdash; {slots}-slot swing <span class="pill">forward paper test</span></h1>
<p class="sub">Top-{bs.CONFIG['universe_size']} US stocks by dollar volume (weekly) &nbsp;·&nbsp; {entry_label} &nbsp;·&nbsp;
ranked by momentum only &nbsp;·&nbsp; 1.5-ATR trail + 10-day-low exit &nbsp;|&nbsp;
{status} &nbsp;·&nbsp; <span class="pill">${cap:,.0f} account</span></p>

{actions}
<div class="cards">{cards_html}</div>
{chart}

<h2>Today&rsquo;s breakouts &mdash; momentum ranked</h2>
<p class="sub">Breakouts are ranked solely by the percentile of their trailing
{bs.CONFIG['proxy_window']}-day return. The top {slots} fill the available slots.</p>
{cand_html}

<h2>Current positions</h2>
<p class="sub">Stop = live exit trigger (higher of the 1.5-ATR trail and the 10-day low).
Room = cushion to that stop.</p>
{pos_html}

<h2>Trade log (forward)</h2>
{trades_html}

{meas}

<p class="foot">Generated {dt.datetime.now():%Y-%m-%d %H:%M} by breakout_dashboard.py.
Forward record stored in data/paper_breakout.json (+ paper_trades.csv, paper_equity.csv).
Universe = S&amp;P 500 + Nasdaq-100 + liquid extras, top {bs.CONFIG['universe_size']} by dollar volume,
weekly. Survivorship-aware; a research tool, not investment advice.</p>
</div>"""


def next_business_day():
    d = dt.date.today() + dt.timedelta(days=1)
    while d.weekday() >= 5:                                   # skip Sat/Sun
        d += dt.timedelta(days=1)
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capital", type=float, default=8800.0)
    p.add_argument("--start", default=None, help="Forward-test start (default: next business day).")
    p.add_argument("--out", default="breakout_dashboard.html")
    a = p.parse_args()
    start = a.start or next_business_day().isoformat()
    print("Building breakout dashboard (advancing paper account)...")
    s = build(a.capital, start)
    with open(a.out, "w") as f:
        f.write(html(s))
    n_pend = len(s["pending"])
    print(f"Wrote {a.out}  (value ${s['value']:,.0f}, {n_pend} orders queued, "
          f"{len(s['trades'])} trades logged)")


if __name__ == "__main__":
    main()
