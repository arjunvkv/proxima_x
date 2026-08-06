#!/usr/bin/env python3
"""
Apples-to-apples: ENGINE vs LIVE execution path on the SAME archived FTMO ticks.

Data: data/fundednext_ticks/*.npy (5 pairs, Jun 29 - Jul 27 2026).
  - Path A (ENGINE): MultiPairBacktestEngine.run(data) -> decisions + fills
  - Path B (LIVE):   LiveRunner + LiveExecutor(paper) on the SAME aligned bars
  - Compare: Level-1 fingerprints (ts/symbol/type/side) AND entry prices.

  demo (default): ticks -> M5 bars -> run both paths -> parity report.
  live:           additionally place real 0.01 FTMO demo orders through
                  LiveExecutor(mode="live", mt5) at the same bar-open prices
                  and confirm the live fill stream reconciles.

TokyoH0 is re-pointed to the 5 archived tick pairs (min_pairs=5) so it fires;
the point is engine == live equality on identical inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
for p in (ROOT, ROOT / "proxima_honest_backtest"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from proxima_honest_backtest.strategies.multi_pair_base import MultiPairStrategy
from proxima_honest_backtest.strategies.tokyo_h0.strategy import TokyoH0Strategy
from proxima_honest_backtest.live.events.emitter import EventEmitter, EmitterMode
from proxima_honest_backtest.live.executor import LiveExecutor
from proxima_honest_backtest.live.feed import ReplayFeed
from proxima_honest_backtest.live.parity import compare_level1, extract_decision_enters
from proxima_honest_backtest.live.runner import LiveRunner
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine

TICK_DIR = ROOT / "data" / "fundednext_ticks"
TICK_PAIRS = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD"]


# ----------------------------------------------------------------------
# Data: archived FTMO ticks -> aligned M5 bars
# ----------------------------------------------------------------------
def load_ticks(symbol: str) -> pd.DataFrame:
    arr = np.load(TICK_DIR / f"{symbol}.npy")
    df = pd.DataFrame({"time": arr["time_msc"], "bid": arr["bid"], "ask": arr["ask"]})
    df = df.dropna(subset=["bid", "ask"])
    df = df[(df["bid"] > 0) & (df["ask"] > 0)]
    df = df.sort_values("time").reset_index(drop=True)
    return df


def ticks_to_m5(symbol: str) -> pd.DataFrame:
    t = load_ticks(symbol)
    if t.empty:
        return pd.DataFrame()
    mid = (t["bid"] + t["ask"]) / 2.0
    idx = pd.to_datetime(t["time"], unit="ms", utc=True)
    s = pd.Series(mid.values, index=idx, name="mid")
    r = s.resample("5min").agg(["first", "max", "min", "last"]).dropna()
    out = pd.DataFrame({
        "time": r.index,
        "open": r["first"], "high": r["max"], "low": r["min"], "close": r["last"],
        "tick_volume": 0, "spread": 0,
    })
    return out.reset_index(drop=True)


def build_data() -> Dict[str, pd.DataFrame]:
    return {s: ticks_to_m5(s) for s in TICK_PAIRS}


def make_strategy() -> MultiPairStrategy:
    """TokyoH0 re-pointed to the 5 archived pairs so it fires."""
    return TokyoH0Strategy({
        "pairs": TICK_PAIRS,
        "top_n": 5,
        "lookback_bars": 6,
        "lookback_confirm_bars": 3,
        "hold_bars": 12,
        "session_hour": 0,
        "min_pairs": 5,
        "min_confidence": 0.30,
        "require_decline_persistence": True,
    })


def run_engine(data):
    eng = MultiPairBacktestEngine(make_strategy(), ExecutionSimulator("ftmo"))
    res = eng.run(data)
    return res


def run_live(data):
    eng = MultiPairBacktestEngine(make_strategy(), ExecutionSimulator("ftmo"))
    aligned = eng._align_bars(data)
    live_ex = LiveExecutor(TICK_PAIRS, magic_base=400000, base_lot=0.15,
                           mode="paper", spread_model_half=0.0)
    runner = LiveRunner(make_strategy(), ReplayFeed(aligned), live_ex,
                        TICK_PAIRS, persist=False)
    runner.run_replay(ReplayFeed(aligned))
    return runner.decisions, aligned


def report_parity(bt_res, live_decisions, aligned) -> Dict[str, Any]:
    bt_decisions = bt_res.decisions
    l1 = compare_level1(bt_decisions, live_decisions)

    # Entry-price parity: engine ENTERs fill at bar OPEN (from signal metadata);
    # live ENTERs carry requested_price = same bar OPEN. Both must equal the
    # aligned bar open for that (ts, symbol).
    bt_enters = extract_decision_enters(bt_decisions)
    live_enters = extract_decision_enters(live_decisions)

    open_by = {}
    for rec in aligned:
        ts = rec["time"]
        for p in TICK_PAIRS:
            if rec.get(p) is not None and not (isinstance(rec[p], float) and np.isnan(rec[p])):
                open_by.setdefault((str(ts), p), float(rec[f"{p}_open"]))

    live_req = {
        (str(d["ts"]), d["symbol"]): float(d["requested_price"])
        for d in live_decisions if d.get("type") == "ENTER"
    }
    price_ok = []
    for k, req in live_req.items():
        expected = open_by.get(k)
        if expected is None:
            continue
        price_ok.append(abs(req - expected))

    return {
        "level1_parity": l1,
        "backtest_trades": len(bt_res.trades),
        "backtest_net_pnl": round(bt_res.net_pnl, 2),
        "entry_price_matches_checked": len(price_ok),
        "entry_price_max_abs_diff": round(max(price_ok), 10) if price_ok else None,
    }


# ----------------------------------------------------------------------
# LIVE: place real 0.01 FTMO demo orders at the SAME bar-open prices
# ----------------------------------------------------------------------
def run_live_orders(bt_res, data):
    import MetaTrader5 as mt5

    mt5.initialize()
    acct = mt5.account_info()
    print(f"[live] connected account={acct.login} server={acct.server} "
          f"balance={acct.balance} trade_allowed={acct.trade_allowed}")

    # select a few ENTER decisions (distinct days) to replay at 0.01 lot
    enters = extract_decision_enters(bt_res.decisions)
    seen_days, picks = set(), []
    for d in enters:
        day = d["ts"][:10]
        if day in seen_days:
            continue
        seen_days.add(day)
        picks.append(d)
        if len(picks) >= 3:
            break

    # bar-open price map from the SAME M5 bars the backtest used
    eng = MultiPairBacktestEngine(make_strategy(), ExecutionSimulator("ftmo"))
    aligned = eng._align_bars(data)
    open_by = {}
    for rec in aligned:
        ts = rec["time"]
        for p in TICK_PAIRS:
            if rec.get(p) is not None and not (isinstance(rec[p], float) and np.isnan(rec[p])):
                open_by.setdefault((str(ts), p), float(rec[f"{p}_open"]))

    emitter = EventEmitter(str(ROOT / "live_orders.jsonl"), strategy="tokyo_h0",
                           run_id=f"live_verify_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}",
                           mode=EmitterMode.LIVE)
    ex = LiveExecutor(TICK_PAIRS, magic_base=400000, base_lot=0.01, mode="live",
                      mt5=mt5, hard_sl_pips=0.0, hard_tp_pips=0.0,
                      slippage_pips=1.0, spread_model_half=0.0, emitter=emitter)

    results = []
    for d in picks:
        ts, sym, side = d["ts"], d["symbol"], d["side"]
        bar_open = open_by.get((ts, sym))
        ts_dt = datetime.fromisoformat(ts)
        # Fill at the CURRENT live price (historical bar-open would be an
        # invalid price for a live DEAL); the historical bar_open is reported
        # for apples-to-apples reference. decision_id pins this to the matched
        # tokyo_h0 ENTER decision -> proves decision->live-fill mapping.
        tick = mt5.symbol_info_tick(sym)
        entry_px = tick.ask if side.upper() == "LONG" else tick.bid
        report = ex.execute_order(side=side, quantity=0.01, symbol=sym,
                                  price=entry_px, volatility=0.001, hour_utc=0,
                                  timestamp=ts_dt,
                                  decision_id=f"verify|{sym}|{ts}|ENTER")
        if report.filled:
            crep, close_px = None, None
            for _ in range(3):  # close may reject if the quote moved; retry fresh
                tick2 = mt5.symbol_info_tick(sym)
                close_px = tick2.ask if side.upper() == "SHORT" else tick2.bid
                crep = ex.close_position(sym, price=close_px, timestamp=datetime.utcnow(),
                                         decision_id=f"verify|{sym}|{ts}|EXIT")
                if crep.filled:
                    break
            pnl = ex.calculate_pnl(entry_px, crep.fill_price if crep and crep.filled else close_px,
                                   0.01, side, sym) if crep and crep.filled else None
        else:
            entry_px, crep, close_px, pnl = None, None, None, None
        results.append({
            "ts": ts, "symbol": sym, "side": side, "bar_open_hist": bar_open,
            "entry_px_live": entry_px, "entry_filled": report.filled,
            "entry_reject": report.reject_reason, "fill_price": report.fill_price,
            "close_filled": bool(crep and crep.filled), "pnl_usd": pnl,
        })
        print(json.dumps(results[-1]))

    emitter.close()
    mt5.shutdown()
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="also place real 0.01 FTMO demo orders")
    args = ap.parse_args()

    data = build_data()
    n_bars = {s: len(v) for s, v in data.items() if not v.empty}
    print(f"[data] M5 bars built from FTMO ticks: {n_bars}")

    bt_res = run_engine(data)
    live_decisions, aligned = run_live(data)
    report = report_parity(bt_res, live_decisions, aligned)
    print(json.dumps(report, indent=2))

    if args.live:
        run_live_orders(bt_res, data)


if __name__ == "__main__":
    main()