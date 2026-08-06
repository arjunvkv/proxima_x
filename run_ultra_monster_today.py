"""
Run today's Ultra Monster backtest (M5, 2026-08-04 00:00 UTC → now)
and compare vs the 2 live trades recorded on the VPS dashboard.

LIVE TRADES (from /api/predictive_radar):
  1. EURUSD SELL  entry=1.15053 @ 04:00  exit=1.15033 @ 04:03  +$24.00  WIN
  2. AUDCAD BUY   entry=0.98193 @ 16:30 (Aug-03) ...
"""

import sys
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

# ── config (mirrors settings.py) ─────────────────────────────────────────────
UNIVERSE = ["EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","USDCHF",
            "EURJPY","GBPJPY","EURAUD"]
LOOKBACK_BARS   = 12
MIN_RANGE_PIPS  = 6.0
TRIGGERS        = [0, 30]          # minute-of-hour that fires
HOLD_BARS       = 3                # 15-min hold
LOT             = 1.20
INITIAL_BAL     = 100_000.0
PIP_VALUE_USD   = {"JPY": 6.30, "DEFAULT": 10.0}   # per 0.01-lot per pip at 1.2 lot

def pip_size(sym):
    return 0.01 if "JPY" in sym else 0.0001

def pip_val(sym):
    """Approx USD per pip for 1.2-lot position."""
    return LOT * (PIP_VALUE_USD["JPY"] if "JPY" in sym else PIP_VALUE_USD["DEFAULT"])

# ── MT5 data fetch ────────────────────────────────────────────────────────────
def fetch_m5(sym, date_from, date_to):
    rates = mt5.copy_rates_range(sym, mt5.TIMEFRAME_M5, date_from, date_to)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df.set_index("time", inplace=True)
    return df

# ── backtest engine ──────────────────────────────────────────────────────────
def run_backtest(bars_dict, date_from, date_to):
    """Exact replication of evaluate_ultra_monster logic."""
    # Build a unified timeline of every M5 bar in the window
    all_times = set()
    for df in bars_dict.values():
        all_times.update(df.index.tolist())
    all_times = sorted(all_times)

    trades = []
    open_trade = None   # only 1 trade at a time (single best pair, no concurrent)

    for ts in all_times:
        if ts < date_from or ts > date_to:
            continue

        # ── close any open trade ────────────────────────────────────────────
        if open_trade is not None:
            exit_ts_target = open_trade["entry_bar"] + timedelta(minutes=5 * HOLD_BARS)
            if ts >= exit_ts_target:
                sym = open_trade["pair"]
                df  = bars_dict.get(sym)
                if df is not None and ts in df.index:
                    exit_price = df.loc[ts, "close"]
                    entry_price = open_trade["entry_price"]
                    side_mult   = 1 if open_trade["side"] == "BUY" else -1
                    pips = (exit_price - entry_price) / pip_size(sym) * side_mult
                    pnl  = pips * pip_val(sym)
                    open_trade["exit_time"]  = ts
                    open_trade["exit_price"] = exit_price
                    open_trade["pips"]       = round(pips, 1)
                    open_trade["net_pnl"]    = round(pnl, 2)
                    open_trade["is_win"]     = pnl > 0
                    trades.append(open_trade)
                    open_trade = None

        # ── signal evaluation at :00 and :30 ────────────────────────────────
        if ts.minute not in TRIGGERS:
            continue
        if open_trade is not None:
            continue   # already in a trade

        candidates = []
        for sym in UNIVERSE:
            df = bars_dict.get(sym)
            if df is None or len(df) == 0:
                continue
            if ts not in df.index:
                continue
            loc = df.index.get_loc(ts)
            if loc < LOOKBACK_BARS:
                continue

            window = df.iloc[loc - LOOKBACK_BARS : loc]
            rh = window["high"].max()
            rl = window["low"].min()
            rng_pips = (rh - rl) / pip_size(sym)

            if rng_pips < MIN_RANGE_PIPS:
                continue

            c = df.iloc[loc]["close"]
            if c > rh:
                candidates.append((sym, "BUY", rng_pips, rh, rl))
            elif c < rl:
                candidates.append((sym, "SELL", rng_pips, rh, rl))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[2], reverse=True)
        sym, side, rng, rh, rl = candidates[0]
        entry_price = bars_dict[sym].iloc[bars_dict[sym].index.get_loc(ts)]["open"]

        open_trade = {
            "entry_time":  ts,
            "entry_bar":   ts,
            "entry_price": entry_price,
            "pair":        sym,
            "side":        side,
            "range_pips":  round(rng, 1),
        }

    return pd.DataFrame(trades)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    if not mt5.initialize():
        print("❌ MT5 initialize() failed:", mt5.last_error())
        return

    # Today window: 2026-08-04 00:00 UTC → now
    date_from = datetime(2026, 8, 4, 0, 0, 0)
    date_to   = datetime.utcnow()

    # Also load yesterday afternoon for the Aug-03 trades
    date_from_ext = datetime(2026, 8, 3, 15, 0, 0)

    print(f"\n=== ULTRA MONSTER BACKTEST ===")
    print(f"Window : {date_from_ext.strftime('%Y-%m-%d %H:%M')} → {date_to.strftime('%Y-%m-%d %H:%M')} UTC")
    print(f"Pairs  : {', '.join(UNIVERSE)}")
    print(f"Params : lookback={LOOKBACK_BARS}bars, min_range={MIN_RANGE_PIPS}p, hold={HOLD_BARS}bars, lot={LOT}\n")

    bars_dict = {}
    for sym in UNIVERSE:
        df = fetch_m5(sym, date_from_ext, date_to)
        if df.empty:
            print(f"  ⚠  {sym}: no data")
        else:
            bars_dict[sym] = df
            print(f"  ✓  {sym}: {len(df)} bars  ({df.index[0]} → {df.index[-1]})")

    print()
    df_trades = run_backtest(bars_dict, date_from_ext, date_to)

    if df_trades.empty:
        print("No trades generated in this window.")
    else:
        print(f"Total trades: {len(df_trades)}\n")
        print(f"{'#':<4} {'Pair':<8} {'Side':<5} {'Entry Time':<22} {'Entry':<10} {'Exit Time':<22} {'Exit':<10} {'Pips':>6} {'PnL':>9} {'W/L'}")
        print("-" * 110)
        for i, row in df_trades.iterrows():
            wl   = "✅ WIN" if row["is_win"] else "❌ LOSS"
            pnl  = f"+${row['net_pnl']:.2f}" if row["net_pnl"] > 0 else f"-${abs(row['net_pnl']):.2f}"
            print(f"{i+1:<4} {row['pair']:<8} {row['side']:<5} {str(row['entry_time']):<22} {row['entry_price']:<10.5f} {str(row['exit_time']):<22} {row['exit_price']:<10.5f} {row['pips']:>6.1f} {pnl:>9} {wl}")

        wins  = df_trades["is_win"].sum()
        total = len(df_trades)
        net   = df_trades["net_pnl"].sum()
        wr    = wins / total * 100 if total else 0

        print("-" * 110)
        print(f"\n{'Total Trades':<20}: {total}")
        print(f"{'Wins':<20}: {wins}  ({wr:.1f}% WR)")
        print(f"{'Net PnL':<20}: {'+'if net>=0 else ''}${net:.2f}")

    # ── Compare vs live trades ─────────────────────────────────────────────
    print("\n" + "="*110)
    print("COMPARISON: LIVE (VPS) vs BACKTEST")
    print("="*110)
    live_trades = [
        {"pair":"EURUSD","side":"SELL","entry_time":"2026-08-04 04:00","entry_price":1.15053,
         "exit_time":"2026-08-04 04:03","exit_price":1.15033,"pips":2.0,"net_pnl":24.00,"is_win":True},
        {"pair":"AUDCAD","side":"BUY","entry_time":"2026-08-03 16:30","entry_price":0.98193,
         "exit_time":"2026-08-03 16:33","exit_price":0.98172,"pips":-2.1,"net_pnl":-17.95,"is_win":False},
    ]
    print(f"\n{'#':<4} {'Source':<12} {'Pair':<8} {'Side':<5} {'Entry Time':<22} {'Entry':<10} {'PnL':>9} {'W/L'}")
    print("-" * 85)
    for i, t in enumerate(live_trades, 1):
        wl  = "✅ WIN" if t["is_win"] else "❌ LOSS"
        pnl = f"+${t['net_pnl']:.2f}" if t["net_pnl"] > 0 else f"-${abs(t['net_pnl']):.2f}"
        print(f"{i:<4} {'LIVE':12} {t['pair']:<8} {t['side']:<5} {t['entry_time']:<22} {t['entry_price']:<10.5f} {pnl:>9} {wl}")

    # Find matching backtest rows
    print()
    for live in live_trades:
        et = pd.Timestamp(live["entry_time"])
        match = df_trades[
            (df_trades["pair"]      == live["pair"]) &
            (df_trades["side"]      == live["side"]) &
            (df_trades["entry_time"] >= et - timedelta(minutes=5)) &
            (df_trades["entry_time"] <= et + timedelta(minutes=5))
        ] if not df_trades.empty else pd.DataFrame()

        if not match.empty:
            row = match.iloc[0]
            pnl_diff = row["net_pnl"] - live["net_pnl"]
            print(f"  {live['pair']} {live['side']} @ {live['entry_time']}:")
            print(f"    LIVE     → entry={live['entry_price']:.5f}  pips={live['pips']:.1f}  pnl=${live['net_pnl']:.2f}")
            print(f"    BACKTEST → entry={row['entry_price']:.5f}  pips={row['pips']:.1f}  pnl=${row['net_pnl']:.2f}")
            print(f"    DIFF     → pnl_diff={pnl_diff:+.2f}  {'✅ MATCH' if abs(pnl_diff) < 1.0 else '⚠ DIVERGENCE'}\n")
        else:
            print(f"  {live['pair']} {live['side']} @ {live['entry_time']}: ⚠ NO MATCHING BACKTEST TRADE\n")

    mt5.shutdown()


if __name__ == "__main__":
    main()
