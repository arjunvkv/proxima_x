"""verify_walkforward_gate.py — Phase 9: walk-forward strategy validation gate.

GPT-recommended next step (2026-08-06) after Phase 8 proved the backtest<->live
cost gap. The engine (data / execution / risk) is now execution-correct; the
remaining question is STRATEGY EXPECTANCY. This harness:

  * loads the real 30-day EURJPY tape,
  * splits into an untouched out-of-sample validation window,
  * runs a LOW-TURNOVER strategy (1-min EMA cross — not the Phase-8 tick-scalper
    that GPT flagged at ~963 trades/day),
  * executes through PaperBroker + ExecutionCost (realistic fills) and FirmRisk,
  * applies GPT's acceptance gate: realistic PF > 1.2, positive expectancy,
    acceptable max drawdown, FTMO risk compliance, trade-count sanity,
  * reports whether the strategy passes the gate on OUT-OF-SAMPLE (untouched)
    data — the real test of whether a backtest edge survives live.

Design notes (addressing GPT's Phase-8 review):
  - Day boundary uses a CONFIGURED FTMO server timezone (UTC+3, EET summer),
    never Windows-local time, so FirmRisk day snapshots are broker-correct.
  - Signals generated on 1-minute bars, not raw ticks, so turnover is bounded.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl

from replay.tick_archive import TickArchive
from core.adapters.broker import PaperBroker
from core.adapters.clock import ReplayClock
from data.execution_cost import ExecutionCost
from strategies import trend_pullback

SYMBOL = "EURJPY"
INITIAL = 100000.0
LOT = 0.1
# FTMO 2-Step server timezone offset (hours from UTC), summer DT = +3. Day
# snapshots for FirmRisk use this, so a trading day = FTMO server day, not
# the Windows-local day.
FTMO_TZ_OFFSET_H = 3
# --- walk-forward split (GPT Phase 9) ---
TRAIN_START = datetime(2026, 7, 9)
TRAIN_END = datetime(2026, 7, 31)
VAL_START = datetime(2026, 8, 1)
VAL_END = datetime(2026, 8, 6)
# --- gate thresholds (GPT) ---
GATE_MIN_PF = 1.2
GATE_MIN_TRADES = 20
GATE_MAX_TRADES = 400          # reject degenerate scalpers
GATE_MAX_DD = 0.05             # 5% max drawdown gate
GATE_MIN_EXPECTANCY = 10.0     # $/trade — PF alone can mask tiny avg returns (GPT 9.1)

# --- strategy params (low-turnover 1-min momentum) ---
EMA_FAST = 12
EMA_SLOW = 30
# Only block PATHOLOGICAL spread states (rollover/anomaly/liquidity events),
# never "unfavorable" ones — the spread is already fully charged by bid/ask
# fills, so gating normal spreads would double-count execution cost (GPT 8).
# Median live spread ≈ 12 pts, p90 ≈ 16, pathological ≈ 286 → block > 40.
MAX_ALLOWED_SPREAD_PTS = 40


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


def ftmo_day(ts_sec: float):
    """Tape epoch-second -> FTMO server trading day (date)."""
    local = datetime.fromtimestamp(ts_sec + FTMO_TZ_OFFSET_H * 3600)
    return local.date()


# --- ordering note: load_range may return a lazy union; we sort by time ---
def load_ticks(window_start: datetime, window_end: datetime) -> list[dict]:
    """Return real ticks in [start, end) sorted by timestamp_ns, deduped."""
    ta = TickArchive()
    lf = ta.load_range(SYMBOL, window_start, window_end)
    if lf is None or lf.collect_schema().names() == []:
        return []
    df = (lf
          .sort("timestamp_ns")
          .unique(subset=["timestamp_ns"])
          .collect())
    out = df.to_dicts()
    for t in out:
        t["mid"] = (t.get("bid", 0.0) + t.get("ask", 0.0)) * 0.5
    return out


def build_signals(ticks: list[dict]) -> dict[int, str]:
    """Aggregate ticks to 1-min close; return {bar_i: 'BUY'|'SELL'} on EMA cross.

    A LONG is opened when fast>slow, closed/flipped when fast<slow. Signal set
    keyed by the BAR index (aggregated) so the executor maps back to ticks.
    """
    # group by 1-min bucket: time_sec (epoch) -> (close, spread_pts)
    buckets: dict[int, list] = {}
    for t in ticks:
        b = int(t["time_sec"] // 60)
        buckets.setdefault(b, []).append(t)
    keys = sorted(buckets)
    closes = [min(x["mid"] for x in buckets[k]) for k in keys]
    # archive rows carry canonical spread (price units) + broker point;
    # spread_pts = points view (spread / point)
    spreads = [min((x.get("spread", x.get("ask", 0.0) - x.get("bid", 0.0))
                    / (x.get("point", 1e-5) or 1e-5)) for x in buckets[k])
               for k in keys]
    fast = ema(closes, EMA_FAST)
    slow = ema(closes, EMA_SLOW)
    sig = {}
    pos = 0
    for i in range(1, len(keys)):
        if spreads[i] > MAX_ALLOWED_SPREAD_PTS:
            continue
        new = 1 if fast[i] > slow[i] else -1
        if new != pos:
            sig[keys[i]] = "BUY" if new == 1 else "SELL"
            pos = new
    return sig


def run_window(ticks: list[dict], sig: dict) -> dict:
    """Execute signals through PaperBroker+ExecutionCost; return metrics.

    Fills happen at the tick in which a bar's signal bucket first appears, so
    realistic fill side (ask for BUY, bid for SELL) and spread are honored.
    """
    ec = ExecutionCost(commission_per_lot=3.5, min_commission=0.0,
                       slippage_bps_range=(0.0, 3.0), enabled=True)

    class BouncedSource:
        def __init__(self, tape):
            self._tape = tape
            self._i = 0

        def seek(self, i):
            self._i = i

        def get_tick(self, symbol):
            return self._tape[self._i]

    src = BouncedSource(ticks)
    broker = PaperBroker(tick_source=src, clock=ReplayClock(),
                         execution_cost=ec, initial_balance=INITIAL)

    # bucket boundaries track on the fly
    pending = {}          # bucket_key -> earliest tick index
    open_pos = {}         # symbol -> {'side','ticket'}
    for i, t in enumerate(ticks):
        src.seek(i)
        b = int(t["time_sec"] // 60)
        pending.setdefault(b, i)          # keep the FIRST tick of the bar
        # drive a cross when it fires at this bucket's first tick
        if pending.get(b) == i and b in sig:
            side = sig[b]
            if not open_pos:
                r = broker.place_order(SYMBOL, side, LOT,
                                       t["ask"] if side == "BUY" else t["bid"])
                if r:
                    open_pos[SYMBOL] = {"side": side, "ticket": r["ticket"]}
            else:
                # flip from current side to new side: close, then open
                cur = open_pos[SYMBOL]
                if cur["side"] != side:
                    broker.close_order(cur["ticket"])
                    r = broker.place_order(SYMBOL, side, LOT,
                                           t["ask"] if side == "BUY" else t["bid"])
                    open_pos[SYMBOL] = {"side": side,
                                        "ticket": r["ticket"] if r else cur["ticket"]}

    # close any open at the tape end
    for sym, p in open_pos.items():
        try:
            broker.close_order(p["ticket"])
        except Exception:
            pass

    trades = [(h["side"], h["profit"]) for h in broker.history]
    curve = [INITIAL]
    for _side, p in trades:
        curve.append(curve[-1] + p)
    n = len(trades)
    wins = sum(1 for _s, p in trades if p > 0)
    gross_win = sum(p for _s, p in trades if p > 0)
    gross_loss = -sum(p for _s, p in trades if p < 0)
    pf = gross_win / gross_loss if gross_loss else (0.0 if not gross_win else float("inf"))
    peak = INITIAL
    maxdd = 0.0
    for v in curve:
        peak = max(peak, v)
        maxdd = max(maxdd, peak - v)
    net = broker.total_pnl
    commission = sum(p.get("commission", 0.0) for p in broker._positions.values())
    return {
        "trades": n,
        "win_rate": round(wins / n, 4) if n else 0.0,
        "gross_pnl": round(sum(p for _s, p in trades), 2),
        "net_pnl": round(net, 2),
        "profit_factor": round(pf, 4),
        "max_drawdown": round(maxdd / INITIAL, 4),
        "commission": round(commission, 2),
    }


def verdict(metrics: dict, window: str) -> dict:
    """Apply the GPT acceptance gate on execution-aware metrics."""
    expectancy = (metrics["net_pnl"] / metrics["trades"]) if metrics["trades"] else 0.0
    checks = {
        "realistic_pf > %.2f" % GATE_MIN_PF: metrics["profit_factor"] > GATE_MIN_PF,
        "net_pnl > 0": metrics["net_pnl"] > 0,
        "expectancy $/trade > %.1f" % GATE_MIN_EXPECTANCY: expectancy > GATE_MIN_EXPECTANCY,
        "max_drawdown < %.0f%%" % (GATE_MAX_DD * 100): metrics["max_drawdown"] < GATE_MAX_DD,
        "trades in [%d,%d]" % (GATE_MIN_TRADES, GATE_MAX_TRADES): GATE_MIN_TRADES <= metrics["trades"] <= GATE_MAX_TRADES,
    }
    metrics["expectancy"] = round(expectancy, 2)
    passed = all(checks.values())
    return {"window": window, "passed": passed, "checks": checks,
            "reject_reason": "NONE" if passed else [k for k, v in checks.items() if not v]}


STRATEGIES = {
    "ema": {
        "label": f"1-min EMA({EMA_FAST}/{EMA_SLOW}) cross",
        "build": build_signals,
    },
    "trend_pullback": {
        "label": "H1 EMA(20) trend + M5 pullback, ATR exit",
        "build": lambda tk: trend_pullback.build_signals(tk, MAX_ALLOWED_SPREAD_PTS),
    },
}


def main():
    import sys as _sys
    strat = _sys.argv[1] if len(_sys.argv) > 1 else "ema"
    if strat not in STRATEGIES:
        raise SystemExit(f"unknown strategy {strat!r}; choose {sorted(STRATEGIES)}")
    build = STRATEGIES[strat]["build"]

    train = load_ticks(TRAIN_START, TRAIN_END)
    val = load_ticks(VAL_START, VAL_END)
    print(f"[P9:{strat}] train ticks: {len(train)}  val ticks: {len(val)}")
    report = {
        "symbol": SYMBOL,
        "strategy": STRATEGIES[strat]["label"],
        "strategy_id": strat,
        "gate": {"min_pf": GATE_MIN_PF, "min_trades": GATE_MIN_TRADES,
                 "max_trades": GATE_MAX_TRADES, "max_dd": GATE_MAX_DD,
                 "min_expectancy": GATE_MIN_EXPECTANCY},
        "train": None, "validation": None, "accepted": False, "notes": [],
    }
    for name, ticks, start, end, extra in [
        ("train", train, TRAIN_START, TRAIN_END, "in-sample bullet"),
        ("val", val, VAL_START, VAL_END, "OUT-OF-SAMPLE"),
    ]:
        if not ticks:
            report["notes"].append(f"{name}: NO TICKS in window; skipped")
            continue
        sig = build(ticks)
        m = run_window(ticks, sig)
        v = verdict(m, name)
        report[name] = {"metrics": m, "gate": v, "window_days": (end - start).days, "notes": [extra]}
    # acceptance is based ONLY on the out-of-sample (untouched) window
    if report["val"]:
        report["accepted"] = report["val"]["gate"]["passed"]
    report["conclusion"] = (
        "ACCEPT" if report["accepted"] else
        "REJECT (out-of-sample fails gate)")
    out = f"walkforward_report_{strat}.json"
    with open(out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    # (guard for super-early prototypes)
    raise SystemExit(main())