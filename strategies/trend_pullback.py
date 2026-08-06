"""strategies/trend_pullback.py — Candidate A: H1 trend filter + M5 pullback entry.

GPT Phase-9.1 direction: test a DIFFERENT edge family — not another EMA
parameterization (that's curve fitting). Candidate A:

  H1: trend filter   — EMA(20) slope over the last 6 H1 bars (persistent trend)
  M5: entry trigger  — pullback: M5 price touches/crosses back through the H1
                       EMA level in the trend direction
  Exit: ATR-based    — trailing stop = max(entry, latest H1 EMA) - 2.0 * ATR(H1)
                       (longs), symmetric for shorts

Design intent (from the Phase-8 finding): reduce turnover + spread impact vs
the tick-scalper; trade WITH the higher-timeframe trend so entries align with
persistent direction rather than microstructure noise.

Contract: pure signal generation — build_signals(ticks) -> {minute_bucket_key:
'BUY'|'SELL'|None}. Execution is done by the walk-forward harness (realistic
fills, costs, risk), NOT here. No state leaks between calls.
"""
from __future__ import annotations

import statistics
from typing import Optional


def ema(vals: list[float], n: int) -> list[float]:
    k = 2.0 / (n + 1)
    out = [0.0] * len(vals)
    if not vals:
        return out
    e = vals[0]
    out[0] = e
    for i in range(1, len(vals)):
        e = vals[i] * k + e * (1 - k)
        out[i] = e
    return out


def _bars(ticks: list[dict], span_min: int) -> list[dict]:
    """Aggregate ticks into fixed-span bars (mid price)."""
    buckets: dict[int, list] = {}
    for t in ticks:
        b = int(t["time_sec"] // (span_min * 60))
        buckets.setdefault(b, []).append(t)
    out = []
    for b in sorted(buckets):
        mids = [x["mid"] for x in buckets[b]]
        out.append({
            "bucket": b * span_min * 60,      # bar START epoch-second
            "open": mids[0],
            "high": max(mids),
            "low": min(mids),
            "close": mids[-1],
            "spread_pts": min(
                (x.get("spread", x.get("ask", 0.0) - x.get("bid", 0.0))
                 / (x.get("point", 1e-5) or 1e-5)) for x in buckets[b]),
        })
    return out


def _atr(h1_bars: list[dict], n: int = 14) -> list[float]:
    """True-range-based ATR over H1 bars (mid approximation)."""
    trs = []
    for i in range(1, len(h1_bars)):
        prev_c = h1_bars[i - 1]["close"]
        h, l = h1_bars[i]["high"], h1_bars[i]["low"]
        trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    out = [0.0] * len(h1_bars)
    for i in range(len(h1_bars)):
        window = trs[max(0, i - n):i]
        out[i] = statistics.fmean(window) if window else 0.0
    return out


def build_signals(ticks: list[dict], max_allowed_spread_pts: int = 40) -> dict:
    """Return {minute_bucket_key: 'BUY'|'SELL'} for Candidate A.

    State machine per M5 bar:
      - trend = H1 EMA(20) slope over last 6 H1 bars (>0 long, <0 short)
      - flat (no trend) -> no signal
      - entry when the M5 close recrosses the H1 EMA level in trend dir after
        touching it (pullback confirmation)
      - exit via ATR trailing stop handled by the harness; here we only flip
        direction when the trend itself reverses (trail re-anchors).
    """
    m5 = _bars(ticks, 5)
    h1 = _bars(ticks, 60)
    if not m5 or not h1:
        return {}
    h1_close = [b["close"] for b in h1]
    h1_ema = ema(h1_close, 20)
    h1_atr = _atr(h1)

    # map each M5 bar to its containing H1 bar index + H1 EMA/ATR at that time
    m5_with_ctx = []
    hi = 0
    for b in m5:
        while hi < len(h1) - 1 and h1[hi + 1]["bucket"] <= b["bucket"]:
            hi += 1
        m5_with_ctx.append({
            **b,
            "h1_ema": h1_ema[hi] if hi < len(h1_ema) else b["close"],
            "h1_atr": h1_atr[hi] if hi < len(h1_atr) else 0.0,
            "h1_slope": (h1_ema[hi] - h1_ema[max(0, hi - 6)]) if hi < len(h1_ema) else 0.0,
        })

    sig: dict = {}
    pos = 0  # 1 long / -1 short / 0 flat
    for i in range(1, len(m5_with_ctx)):
        b = m5_with_ctx[i]
        if b["spread_pts"] > max_allowed_spread_pts:
            continue
        trend = 1 if b["h1_slope"] > 0 else (-1 if b["h1_slope"] < 0 else 0)
        if trend == 0:
            continue
        # pullback confirmation: M5 price dipped to/through the H1 EMA then
        # closed back on the trend side
        if trend == 1:
            touch = b["low"] <= b["h1_ema"]
            confirm = b["close"] > b["h1_ema"]
        else:
            touch = b["high"] >= b["h1_ema"]
            confirm = b["close"] < b["h1_ema"]
        if pos != trend and touch and confirm:
            sig[b["bucket"] // 60] = "BUY" if trend == 1 else "SELL"
            pos = trend
        elif pos == trend and trend == 1 and b["close"] < b["h1_ema"]:
            # trend still up but M5 broke below the level -> exit long
            sig[b["bucket"] // 60] = "SELL"
            pos = 0
        elif pos == -trend and trend == -1 and b["close"] > b["h1_ema"]:
            sig[b["bucket"] // 60] = "BUY"
            pos = 0
    return sig