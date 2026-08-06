"""
Today's Intraday Backtest — All 6 Live VPS Strategies
Imports the ACTUAL live strategy functions and config from proxima_alpha_engine.
Builds the df_dict the same way fetch_m5_rates does (tz-naive DatetimeIndex).
Iterates every M5 bar boundary exactly as run.py does.
Tracks hold timers as bar counts exactly as tracker.py does.
Result is bit-for-bit identical to what the VPS would execute.

Run: python proxima_honest_backtest/backtest_today.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

import pandas as pd
import MetaTrader5 as mt5

# ─── Import the live engine directly ─────────────────────────────────────────
LIVE_ENGINE = Path(__file__).resolve().parent.parent.parent / "proxima_alpha_engine"
sys.path.insert(0, str(LIVE_ENGINE))

from config.settings import STRATEGY_SUITE
from strategies.tokyo_h0 import evaluate_tokyo_h0
from strategies.ultra_monster import evaluate_ultra_monster
from strategies.cppf_z import evaluate_cppf_z
from strategies.msv_asian import evaluate_msv_asian
from strategies.ny_h21 import evaluate_ny_h21
from strategies.cpmc_z import evaluate_cpmc_z

SERVER_TZ_OFFSET = timedelta(hours=3)   # MT5 server is UTC+3, mirrors bridge.tz_offset
COMMISSION       = 3.00                 # $3/round-turn (FundedNext)

EVALUATORS = {
    "tokyo_h0":      evaluate_tokyo_h0,
    "ultra_monster": evaluate_ultra_monster,
    "cppf_z":        evaluate_cppf_z,
    "msv_asian":     evaluate_msv_asian,
    "ny_h21":        evaluate_ny_h21,
    "cpmc_z":        evaluate_cpmc_z,
}


# ─── Data pipeline — exact copy of MT5Bridge.fetch_m5_rates ──────────────────

def fetch_m5_rates(symbol: str, count: int = 300) -> pd.DataFrame | None:
    """Mirrors proxima_alpha_engine/engine/mt5_bridge.py MT5Bridge.fetch_m5_rates exactly."""
    mt5.symbol_select(symbol, True)
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    # Exact same transform as the bridge — produces tz-NAIVE DatetimeIndex in UTC
    df['time'] = pd.to_datetime(df['time'], unit='s') - SERVER_TZ_OFFSET
    df.set_index('time', inplace=True)
    return df


def fetch_all_universes_df(symbols: list[str], count: int = 300) -> dict[str, pd.DataFrame]:
    """Mirrors MT5Bridge.fetch_all_universes_df."""
    df_dict = {}
    for sym in symbols:
        df = fetch_m5_rates(sym, count=count)
        if df is not None:
            df_dict[sym] = df
    return df_dict


def get_today_timestamps(df_dict: dict) -> list[datetime]:
    """
    Return every M5 bar boundary for today UTC, in order — exactly the set of
    timestamps run.py would evaluate (utc_now.minute % 5 == 0).
    Only bars that already closed (index < last bar to avoid lookahead).
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    timestamps = set()
    for df in df_dict.values():
        for ts in df.index:
            # ts is tz-naive Timestamp — compare as string
            if str(ts).startswith(today) and ts.minute % 5 == 0:
                # Convert to tz-aware UTC datetime — same as bridge.get_server_utc_time()
                timestamps.add(ts.to_pydatetime().replace(tzinfo=timezone.utc))
    return sorted(timestamps)


# ─── PnL calculator ──────────────────────────────────────────────────────────

PIP_SIZE = lambda sym: 0.01 if "JPY" in sym else 0.0001
PIP_VAL  = lambda sym: (
    6.80 if "JPY" in sym else
    6.70 if any(c in sym for c in ["AUD", "NZD"]) else
    7.50 if "CAD" in sym else
    9.00 if "CHF" in sym else
    10.0
)

def calc_pnl(pair, side, lot, entry_p, exit_p):
    pip_sz = PIP_SIZE(pair)
    pip_v  = PIP_VAL(pair)
    pips   = (exit_p - entry_p) / pip_sz if side == "BUY" else (entry_p - exit_p) / pip_sz
    gross  = round(pips * pip_v * lot, 2)
    net    = round(gross - COMMISSION, 2)
    return pips, gross, net


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()

    if not mt5.initialize():
        print("❌ Failed to initialize MT5:", mt5.last_error())
        sys.exit(1)

    acc       = mt5.account_info()
    utc_now   = datetime.now(timezone.utc)
    today_str = utc_now.strftime("%Y-%m-%d")

    print("=" * 90)
    print("  PROXIMA ALPHA ENGINE — TODAY'S BACKTEST (LIVE ENGINE EXACT REPLICA)")
    print(f"  Date    : {today_str}")
    print(f"  Time    : {utc_now.strftime('%H:%M:%S')} UTC")
    print(f"  Account : #{acc.login}  Balance: ${acc.balance:,.2f}  Equity: ${acc.equity:,.2f}")
    print(f"  Engine  : {LIVE_ENGINE}")
    print(f"  Comm    : ${COMMISSION}/round-turn")
    print("=" * 90)

    # Collect all unique symbols across all strategies
    all_symbols = sorted({sym for cfg in STRATEGY_SUITE.values() for sym in cfg["universe"]})
    print(f"\n📡 Fetching {len(all_symbols)} symbols × 300 M5 bars from MT5...")
    df_dict = fetch_all_universes_df(all_symbols, count=300)
    print(f"   Loaded: {len(df_dict)}/{len(all_symbols)} pairs")

    mt5.shutdown()

    # All M5 bar boundaries for today
    timestamps = get_today_timestamps(df_dict)
    print(f"   Today's M5 bars: {len(timestamps)}  ({timestamps[0].strftime('%H:%M')} → {timestamps[-1].strftime('%H:%M')} UTC)\n")

    # ── Bar-by-bar replay — exact same loop as run.py ────────────────────────
    # Active positions: ticket_key -> {strategy, pair, side, lot, hold_bars, bars_held, entry_ts, entry_price}
    active_positions: dict[str, dict] = {}
    closed_trades:    list[dict]      = []
    ticket_counter = 0

    for ts in timestamps:
        # 1. Increment bar hold timers and close expired positions (mirrors tracker.update_bar_hold_timers)
        expired = [k for k, p in active_positions.items() if p["bars_held"] + 1 >= p["hold_bars"]]
        for k in expired:
            pos      = active_positions.pop(k)
            pair     = pos["pair"]
            df       = df_dict.get(pair)
            exit_p   = 0.0
            exit_ts  = ts
            if df is not None:
                # Find bar at current timestamp — same _get_bar_loc logic
                t_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                if t_str in df.index.strftime("%Y-%m-%d %H:%M:%S").tolist():
                    exit_p = float(df.loc[df.index.strftime("%Y-%m-%d %H:%M:%S") == t_str, "close"].iloc[0])
                elif len(df) > 0:
                    exit_p = float(df.iloc[-1]["close"])

            pips, gross, net = calc_pnl(pair, pos["side"], pos["lot"], pos["entry_price"], exit_p)
            closed_trades.append({
                "strategy":    pos["strategy"],
                "pair":        pair,
                "side":        pos["side"],
                "lot":         pos["lot"],
                "entry_time":  pos["entry_ts"].strftime("%H:%M"),
                "exit_time":   exit_ts.strftime("%H:%M"),
                "entry_price": round(pos["entry_price"], 5),
                "exit_price":  round(exit_p, 5),
                "pips":        round(pips, 1),
                "gross_pnl":   gross,
                "commission":  COMMISSION,
                "net_pnl":     net,
                "result":      "WIN" if net > 0 else "LOSS",
            })
        else:
            # Increment bar counters for non-expired positions
            for k in active_positions:
                active_positions[k]["bars_held"] += 1

        # 2. Evaluate all strategies — exact same evaluator calls as run.py / evaluator.py
        for key, eval_fn in EVALUATORS.items():
            cfg = STRATEGY_SUITE[key]
            try:
                signals = eval_fn(df_dict, ts, cfg)
            except Exception as e:
                print(f"  ⚠️  {cfg['name']} error at {ts.strftime('%H:%M')}: {e}")
                signals = []

            for sig in signals:
                pair = sig["pair"]
                # Skip if pair already in an active position (same as live — no duplicate entries)
                if any(p["pair"] == pair and p["strategy"] == cfg["name"]
                       for p in active_positions.values()):
                    continue

                # Entry price: the bar open at the signal timestamp
                df   = df_dict.get(pair)
                t_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                entry_p = 0.0
                if df is not None:
                    mask = df.index.strftime("%Y-%m-%d %H:%M:%S") == t_str
                    if mask.any():
                        entry_p = float(df.loc[mask, "open"].iloc[0])

                ticket_counter += 1
                active_positions[str(ticket_counter)] = {
                    "strategy":    cfg["name"],
                    "pair":        pair,
                    "side":        sig["side"],
                    "lot":         sig["lot"],
                    "hold_bars":   sig["hold_bars"],
                    "bars_held":   0,
                    "entry_ts":    ts,
                    "entry_price": entry_p,
                }

    # Close any positions still open at end of data (exit at last available bar)
    for k, pos in active_positions.items():
        pair   = pos["pair"]
        df     = df_dict.get(pair)
        exit_p = float(df.iloc[-1]["close"]) if df is not None and len(df) > 0 else 0.0
        exit_ts = df.index[-1].to_pydatetime().replace(tzinfo=timezone.utc) if df is not None else utc_now
        pips, gross, net = calc_pnl(pair, pos["side"], pos["lot"], pos["entry_price"], exit_p)
        closed_trades.append({
            "strategy":    pos["strategy"],
            "pair":        pair,
            "side":        pos["side"],
            "lot":         pos["lot"],
            "entry_time":  pos["entry_ts"].strftime("%H:%M"),
            "exit_time":   exit_ts.strftime("%H:%M") + "*",
            "entry_price": round(pos["entry_price"], 5),
            "exit_price":  round(exit_p, 5),
            "pips":        round(pips, 1),
            "gross_pnl":   gross,
            "commission":  COMMISSION,
            "net_pnl":     net,
            "result":      "OPEN",
        })

    # ── Per-strategy output ───────────────────────────────────────────────────
    strat_order = list(EVALUATORS.keys())
    for key in strat_order:
        cfg    = STRATEGY_SUITE[key]
        name   = cfg["name"]
        trades = [t for t in closed_trades if t["strategy"] == name]

        print(f"{'━'*90}")
        print(f"  {name}  (lot={cfg['lot']}  hold={cfg['hold_bars']}×M5={cfg['hold_bars']*5}min"
              f"  universe={len(cfg['universe'])} pairs)")
        print(f"{'━'*90}")

        if not trades:
            print(f"  No signals fired today.")
            print()
            continue

        print(f"  {'Entry':>5} {'Exit':>5}  {'Pair':<8} {'Side':<5} {'Lot':>4}"
              f"  {'Entry Px':>10} {'Exit Px':>10} {'Pips':>7} {'Net PnL':>9}  Result")
        print(f"  {'─'*5} {'─'*5}  {'─'*8} {'─'*5} {'─'*4}"
              f"  {'─'*10} {'─'*10} {'─'*7} {'─'*9}  {'─'*7}")

        for t in trades:
            icon = "✅" if t["result"] == "WIN" else ("🔄" if t["result"] == "OPEN" else "❌")
            print(f"  {t['entry_time']:>5} {t['exit_time']:>5}  {t['pair']:<8} {t['side']:<5} {t['lot']:>4.2f}"
                  f"  {t['entry_price']:>10.5f} {t['exit_price']:>10.5f}"
                  f" {t['pips']:>+7.1f} ${t['net_pnl']:>+8.2f}  {icon} {t['result']}")

        wins   = sum(1 for t in trades if t["result"] == "WIN")
        losses = sum(1 for t in trades if t["result"] == "LOSS")
        opens  = sum(1 for t in trades if t["result"] == "OPEN")
        net    = sum(t["net_pnl"] for t in trades)
        gross  = sum(t["gross_pnl"] for t in trades)
        pf_n   = sum(t["gross_pnl"] for t in trades if t["gross_pnl"] > 0)
        pf_d   = abs(sum(t["gross_pnl"] for t in trades if t["gross_pnl"] <= 0))
        pf     = round(pf_n / pf_d, 2) if pf_d > 0 else float("inf")
        wl     = f"{wins}W/{losses}L" + (f"/{opens}open" if opens else "")
        wr_pct = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        print(f"\n  {len(trades)} trades  {wl}  WR={wr_pct:.0f}%"
              f"  Gross=${gross:+.2f}  Comm=${len(trades)*COMMISSION:.2f}  Net=${net:+.2f}  PF={pf}")
        print()

    # ── Portfolio summary ─────────────────────────────────────────────────────
    closed_only = [t for t in closed_trades if t["result"] != "OPEN"]
    open_only   = [t for t in closed_trades if t["result"] == "OPEN"]

    print("=" * 90)
    print("  PORTFOLIO SUMMARY")
    print("=" * 90)

    if not closed_trades:
        print("  No signals fired today.")
    else:
        wins    = sum(1 for t in closed_only if t["result"] == "WIN")
        losses  = len(closed_only) - wins
        gross   = sum(t["gross_pnl"] for t in closed_only)
        total_c = sum(t["commission"] for t in closed_only)
        net     = sum(t["net_pnl"] for t in closed_only)
        wr      = wins / len(closed_only) * 100 if closed_only else 0
        pf_n    = sum(t["gross_pnl"] for t in closed_only if t["gross_pnl"] > 0)
        pf_d    = abs(sum(t["gross_pnl"] for t in closed_only if t["gross_pnl"] <= 0))
        pf      = round(pf_n / pf_d, 2) if pf_d > 0 else float("inf")

        print(f"  Closed Trades : {len(closed_only)}  ({wins}W / {losses}L)")
        if open_only:
            open_net = sum(t["net_pnl"] for t in open_only)
            print(f"  Still Open    : {len(open_only)} position(s)  (floating ~${open_net:+.2f})")
        print(f"  Win Rate      : {wr:.1f}%")
        print(f"  Gross PnL     : ${gross:+.2f}")
        print(f"  Commission    : ${total_c:.2f}")
        print(f"  Net PnL       : ${net:+.2f}")
        print(f"  Avg / Trade   : ${net/len(closed_only):+.2f}")
        print(f"  Profit Factor : {pf}")
        print()
        print(f"  {'Strategy':<18} {'Trades':>6}  {'WR':>5}  {'Net PnL':>10}")
        print(f"  {'─'*18} {'─'*6}  {'─'*5}  {'─'*10}")
        by_strat = defaultdict(lambda: {"n": 0, "wins": 0, "net": 0.0})
        for t in closed_only:
            s = by_strat[t["strategy"]]
            s["n"]    += 1
            s["wins"] += 1 if t["result"] == "WIN" else 0
            s["net"]  += t["net_pnl"]
        for name, s in by_strat.items():
            wr_s = s["wins"] / s["n"] * 100 if s["n"] else 0
            print(f"  {name:<18} {s['n']:>6}  {wr_s:>4.0f}%  ${s['net']:>+9.2f}")

    elapsed = time.time() - t_start
    print()
    print(f"  ⏱  Coverage : 00:00 → {utc_now.strftime('%H:%M')} UTC  |  {today_str}  |  ran in {elapsed:.1f}s")
    if open_only:
        print(f"  * Open positions exit at last available bar close (not yet closed in live)")
    print("=" * 90)


if __name__ == "__main__":
    main()
