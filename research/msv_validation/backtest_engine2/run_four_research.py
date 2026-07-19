"""
4 parallel research directions in a single script (MT5 data loads once).
1. All-28-pair universe
2. Sydney Open (22:00 UTC)
3. Dealer Capitulation on MT5
4. QAI Adaptive Exit
"""
import sys, os, time, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from collections import deque, Counter
import warnings
warnings.filterwarnings('ignore')

project_root = str(Path(__file__).resolve().parents[3])
cd_root = os.path.join(project_root, "currency_decomposition")
sys.path.insert(0, cd_root)

HOLD = 3
LOOKBACK = 3
TOP_N = 3
MAX_POS = 3
COSTS_BP = 0.5
STOP_BP = 20

np.random.seed(42)


def load_mt5(days=200):
    print("=" * 70)
    print("MT5 DATA LOAD")
    print("=" * 70)
    t0 = time.time()
    import MetaTrader5 as mt5
    if not mt5.initialize():
        print("  MT5 init FAILED")
        return None, []

    from config.settings import BASE_CURRENCY_MAP
    all_pairs = list(BASE_CURRENCY_MAP.keys())

    end = datetime.now()
    start = end - timedelta(days=days)
    all_data = {}
    for pair in all_pairs:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, start, end)
        if rates is not None and len(rates) > 200:
            all_data[pair] = rates
    available = [p for p in all_pairs if p in all_data]
    N = min(len(v) for v in all_data.values()) if all_data else 0
    print(f"  Pairs: {len(available)}/{len(all_pairs)}, Bars: {N:,d} ({N/288:.0f}d, {time.time()-t0:.1f}s)")
    print(f"  Missing: {[p for p in all_pairs if p not in all_data]}")
    mt5.shutdown()
    return all_data, available


def basic_stats(trades, label):
    if not trades:
        print(f"  {label}: 0 trades")
        return {}
    pnls = [t["pnl"] for t in trades]
    n = len(pnls)
    wr = sum(1 for x in pnls if x > 0) / n * 100
    mu = float(np.mean(pnls))
    med = float(np.median(pnls))
    std = float(np.std(pnls)) if n > 1 else 0
    t_stat = mu / (std / max(np.sqrt(n), 1e-8)) if std > 0 else 0
    print(f"  {label}: n={n:,d}, WR={wr:.1f}%, Mean={mu:+.2f}bp, Med={med:+.2f}bp, t={t_stat:.2f}")
    return {"n": n, "wr": wr, "mean": mu, "t": t_stat}


def month_wr(trades):
    if not trades:
        return
    by_month = {}
    for t in trades:
        m = t["dt"].strftime("%Y-%m")
        by_month.setdefault(m, []).append(t["pnl"])
    wrs = []
    for m in sorted(by_month.keys()):
        v = by_month[m]
        wr_m = sum(1 for x in v if x > 0) / len(v) * 100
        wrs.append(wr_m)
    if wrs:
        print(f"  Monthly WRs: {', '.join(f'{w:.0f}%' for w in wrs)}")
        print(f"  Range: {min(wrs):.1f}% – {max(wrs):.1f}%, All > 60%: {all(w>60 for w in wrs)}")


# ====================================================================
# 1. ALL-28-PAIR UNIVERSE
# ====================================================================
def test_all28(all_data, avail_pairs):
    print("\n" + "=" * 70)
    print("TEST 1: ALL-28-PAIR UNIVERSE")
    print("=" * 70)

    N = min(len(v) for v in all_data.values() if v is not None)
    positions = {}

    # Baseline: first 15 pairs only
    first15 = avail_pairs[:15]
    trades15 = run_strategy(all_data, first15, N, hour=0, minute=0)
    basic_stats(trades15, "15-pair baseline")
    month_wr(trades15)
    pair_stats(trades15, "15-pair")

    # Full universe: all 28 pairs
    trades28 = run_strategy(all_data, avail_pairs, N, hour=0, minute=0)
    basic_stats(trades28, "28-pair full")
    month_wr(trades28)
    pair_stats(trades28, "28-pair")

    # Test: what if we still take only 3 positions but from 28 pairs?
    print(f"\n  WR delta: {trades28 and len(trades28):.0f}t vs {trades15 and len(trades15):.0f}t")

    # Isolation: test non-EUR pairs only (pairs not starting with EUR)
    non_eur = [p for p in avail_pairs if not p.startswith("EUR")]
    if non_eur:
        trades_ne = run_strategy(all_data, non_eur, N, hour=0, minute=0)
        basic_stats(trades_ne, "Non-EUR pairs only")


def run_strategy(all_data, pairs, N, hour, minute, costs_bp=COSTS_BP,
                 hold=HOLD, lookback=LOOKBACK, top_n=TOP_N, max_pos=MAX_POS):
    positions = {}
    trades = []

    for idx in range(max(lookback, 0), min(N - hold, len(next(iter(all_data.values()))))):
        dt = datetime.fromtimestamp(float(all_data[pairs[0]][idx]["time"]), tz=timezone.utc)
        if dt.hour != hour or dt.minute != minute:
            continue

        for p in list(positions.keys()):
            if idx >= positions[p]:
                del positions[p]
        if len(positions) >= max_pos:
            continue

        pair_moves = []
        for p in pairs:
            if p in positions:
                continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for p, mag, ret in pair_moves[:top_n]:
            if p in positions or len(positions) >= max_pos:
                break
            if ret > 0:
                continue
            if idx + 1 + hold >= N:
                continue
            spread_cost = costs_bp / 10000
            entry = float(all_data[p][idx + 1]["open"])
            exit_ = float(all_data[p][idx + hold]["close"])
            gross = (exit_ / entry - 1) if entry > 0 else 0
            net_pnl = gross - spread_cost
            trades.append({
                "pnl": net_pnl * 10000, "won": net_pnl > 0,
                "idx": idx, "pair": p, "dt": dt,
                "gross_bp": gross * 10000,
            })
            positions[p] = idx + hold
    return trades


def pair_stats(trades, label):
    if not trades:
        return
    by_pair = {}
    for t in trades:
        by_pair.setdefault(t["pair"], []).append(t["pnl"])
    print(f"\n  Pair stats ({label}):")
    print(f"  {'Pair':<10s} {'n':>4s} {'WR':>6s} {'Mean(bp)':>10s}")
    for pair in sorted(by_pair.keys()):
        v = by_pair[pair]
        pnls = v
        mu = float(np.mean(pnls))
        wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
        print(f"  {pair:<10s} {len(pnls):>4d}  {wr:>5.1f}%  {mu:>+9.2f}bp")


# ====================================================================
# 2. SYDNEY OPEN (22:00 UTC)
# ====================================================================
def test_sydney(all_data, avail_pairs):
    print("\n" + "=" * 70)
    print("TEST 2: SYDNEY OPEN (22:00 UTC)")
    print("=" * 70)

    N = min(len(v) for v in all_data.values() if v is not None)
    first15 = avail_pairs[:15]

    # Test each candidate hour
    for hr in [21, 22, 23, 0]:
        trades = run_strategy(all_data, first15, N, hour=hr, minute=0)
        label = f"H{hr:02d}:00"
        basic_stats(trades, label)
        if trades:
            month_wr(trades)

    # Sydney Open (22:00) with different lookback/hold
    print(f"\n  Sydney Open (22:00 UTC) parameter sweeps:")
    for look in [3, 6, 12]:
        for hd in [1, 2, 3, 6]:
            trades = run_strategy(all_data, first15, N, hour=22, minute=0,
                                  lookback=look, hold=hd)
            if trades and len(trades) > 10:
                pnls = [t["pnl"] for t in trades]
                wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
                print(f"    lookback={look}, hold={hd}: n={len(trades):>4d}, WR={wr:>5.1f}%, Mean={np.mean(pnls):+.2f}bp")


# ====================================================================
# 3. DEALER CAPITULATION ON MT5 M5 DATA
# ====================================================================
def test_dealer_cap(all_data, avail_pairs):
    print("\n" + "=" * 70)
    print("TEST 3: DEALER CAPITULATION (MT5 M5 approximation)")
    print("=" * 70)

    N = min(len(v) for v in all_data.values() if v is not None)
    all_trades = []

    for p in avail_pairs:
        data = all_data[p]
        prices = np.array([float(r["close"]) for r in data])
        spreads = np.array([float(r["spread"]) for r in data],
                            dtype=np.float64)
        highs = np.array([float(r["high"]) for r in data])
        lows = np.array([float(r["low"]) for r in data])
        pnl_series = []

        for i in range(50, N - 2):
            dt = datetime.fromtimestamp(float(data[i]["time"]), tz=timezone.utc)

            # Rolling median spread (24 bars = 2 hours)
            window_spreads = spreads[max(0, i - 24):i + 1]
            med_spread = np.median(window_spreads)
            cur_spread = spreads[i]

            # Range anomaly
            window_ranges = np.diff(
                np.column_stack([lows[max(0, i - 24):i + 1],
                                 highs[max(0, i - 24):i + 1]]),
                axis=1).flatten() if False else \
                highs[max(0, i - 24):i + 1] - lows[max(0, i - 24):i + 1]
            med_range = np.median(window_ranges)
            cur_range = highs[i] - lows[i]

            # Spread anomaly: spread > 2x rolling median
            if med_spread < 1:
                continue
            spread_ratio = cur_spread / med_spread
            if spread_ratio < 2.0:
                continue

            # Range anomaly: range > 2x rolling median
            if med_range < 1e-6:
                continue
            range_ratio = cur_range / med_range
            if range_ratio < 1.5:
                continue

            # Price moved significantly — go counter-trend
            ret_1b = (prices[i] / prices[i - 1] - 1) if prices[i - 1] > 0 else 0
            if abs(ret_1b) < 0.0003:
                continue

            # Enter next bar, hold 1-2 bars
            for hd in [1, 2]:
                if i + 1 + hd >= N:
                    continue
                entry = prices[i + 1]
                exit_ = prices[i + hd]
                if entry <= 0:
                    continue
                # Counter-trend: go AGAINST the move
                direction = -1 if ret_1b > 0 else 1
                gross = direction * (exit_ / entry - 1)
                net_pnl = gross - (COSTS_BP / 10000)
                won = net_pnl > 0

                pnl_series.append({
                    "pnl": net_pnl * 10000, "won": won,
                    "pair": p, "dt": dt,
                    "spread_ratio": spread_ratio,
                    "range_ratio": range_ratio,
                    "ret_1b": ret_1b * 10000,
                    "hold": hd,
                    "hour": dt.hour,
                })
                all_trades.append(pnl_series[-1])
                break  # one hold test per bar

    if pnl_series and len(pnl_series) > 0:
        pnls = [t["pnl"] for t in all_trades]
        wr = sum(1 for x in pnls if x > 0) / len(pnls) * 100
        print(f"  All pairs: n={len(pnls)}, WR={wr:.1f}%, Mean={np.mean(pnls):+.2f}bp")

        # By spread ratio threshold
        for thresh, label in [(2.0, "2-3x"), (3.0, "3-5x"), (5.0, ">5x")]:
            mask = [t["spread_ratio"] >= thresh for t in all_trades]
            if thresh < 5.0 and thresh > 2.0:
                mask = [(t["spread_ratio"] >= thresh) and (t["spread_ratio"] < 5.0) for t in all_trades]
            n = sum(mask)
            if n < 5: continue
            vals = [t["pnl"] for i, t in enumerate(all_trades) if mask[i]]
            wr_m = sum(1 for x in vals if x > 0) / n * 100
            print(f"    Spread {label:<6s}: n={n:>3d}, WR={wr_m:.1f}%, Mean={np.mean(vals):+.2f}bp")

        # By pair
        by_pair = {}
        for t in all_trades:
            by_pair.setdefault(t["pair"], []).append(t["pnl"])
        print(f"\n  By pair:")
        for pair in sorted(by_pair.keys()):
            v = by_pair[pair]
            wr_p = sum(1 for x in v if x > 0) / len(v) * 100
            print(f"    {pair:<10s}: n={len(v):>3d}, WR={wr_p:.1f}%, Mean={np.mean(v):+.2f}bp")

        # By hour
        by_hour = {}
        for t in all_trades:
            h = t["hour"]
            by_hour.setdefault(h, []).append(t["pnl"])
        print(f"\n  By hour:")
        for h in sorted(by_hour.keys()):
            v = by_hour[h]
            wr_h = sum(1 for x in v if x > 0) / len(v) * 100
            print(f"    H{h:02d}: n={len(v):>3d}, WR={wr_h:.1f}%, Mean={np.mean(v):+.2f}bp")

    else:
        print("  No trades found")


# ====================================================================
# 4. QAI ADAPTIVE EXIT
# ====================================================================
def test_qai_exit(all_data, avail_pairs):
    print("\n" + "=" * 70)
    print("TEST 4: QAI ADAPTIVE EXIT")
    print("=" * 70)

    N = min(len(v) for v in all_data.values() if v is not None)
    first15 = avail_pairs[:15]

    # Baseline: fixed 15-min hold
    trades_fixed = run_strategy(all_data, first15, N, hour=0, minute=0, hold=3)
    basic_stats(trades_fixed, "Fixed 15min hold")

    # QAI exit: check after 1 bar (5min), decide whether to hold or exit
    trades_qai = run_strategy_qai(all_data, first15, N)
    basic_stats(trades_qai, "QAI adaptive exit")

    # Compare
    if trades_fixed and trades_qai:
        fp = [t["pnl"] for t in trades_fixed]
        qp = [t["pnl"] for t in trades_qai]
        print(f"\n  Comparison:")
        print(f"    Fixed trades: {len(fp)}, Mean={np.mean(fp):+.2f}bp, WR={sum(1 for x in fp if x>0)/len(fp)*100:.1f}%")
        print(f"    QAI trades:   {len(qp)}, Mean={np.mean(qp):+.2f}bp, WR={sum(1 for x in qp if x>0)/len(qp)*100:.1f}%")


def run_strategy_qai(all_data, pairs, N, costs_bp=COSTS_BP,
                     lookback=LOOKBACK, top_n=TOP_N, max_pos=MAX_POS):
    """QAI adaptive exit: check after 1 bar, hold up to 6 bars max."""
    positions = {}
    trades = []

    for idx in range(max(lookback, 0), min(N - 6, len(next(iter(all_data.values()))))):
        dt = datetime.fromtimestamp(float(all_data[pairs[0]][idx]["time"]), tz=timezone.utc)
        if dt.hour != 0 or dt.minute != 0:
            continue

        for p in list(positions.keys()):
            if idx >= positions[p]:
                del positions[p]
        if len(positions) >= max_pos:
            continue

        pair_moves = []
        for p in pairs:
            if p in positions:
                continue
            cur = float(all_data[p][idx]["close"])
            bf = float(all_data[p][idx - lookback]["close"])
            ret = (cur / bf - 1) if bf > 0 else 0
            pair_moves.append((p, abs(ret), ret))
        pair_moves.sort(key=lambda x: x[1], reverse=True)

        for p, mag, ret in pair_moves[:top_n]:
            if p in positions or len(positions) >= max_pos:
                break
            if ret > 0:
                continue
            if idx + 1 + 6 >= N:
                continue
            spread_cost = costs_bp / 10000
            entry = float(all_data[p][idx + 1]["open"])

            # QAI: check price after each bar, decide when to exit
            # We look at each subsequent bar
            best_exit_bar = idx + 1
            best_pnl = -999

            for hold_bar in range(1, 7):  # 1 to 6 bars (5-30 min)
                exit_idx = idx + 1 + hold_bar
                if exit_idx >= N:
                    break
                exit_ = float(all_data[p][exit_idx]["close"])
                gross = (exit_ / entry - 1) if entry > 0 else 0
                net_pnl = gross - spread_cost * hold_bar

                # First bar check: if we lost money, consider early exit
                if hold_bar == 1:
                    if net_pnl < -0.0005:  # -0.05% first bar, cut
                        best_exit_bar = exit_idx
                        best_pnl = net_pnl * 10000
                        break
                    # If we made money, take it
                    if net_pnl > 0.0008:  # +0.08% first bar, take profit
                        best_exit_bar = exit_idx
                        best_pnl = net_pnl * 10000
                        break

                if net_pnl * 10000 > best_pnl:
                    best_pnl = net_pnl * 10000
                    best_exit_bar = exit_idx

            final_exit = float(all_data[p][best_exit_bar]["close"])
            final_gross = (final_exit / entry - 1) if entry > 0 else 0
            hold_bars = best_exit_bar - (idx + 1)
            final_net = final_gross - spread_cost * hold_bars

            trades.append({
                "pnl": final_net * 10000, "won": final_net > 0,
                "idx": idx, "pair": p, "dt": dt,
                "gross_bp": final_gross * 10000,
                "hold_bars": hold_bars,
            })
            positions[p] = best_exit_bar
    return trades


# ====================================================================
# MAIN
# ====================================================================
def main():
    t_main = time.time()

    all_data, avail_pairs = load_mt5(days=200)
    if not all_data or len(avail_pairs) < 10:
        print("ERROR: Not enough data")
        return

    first15 = avail_pairs[:15]
    print(f"\nBaseline check: {len(avail_pairs)} pairs, {len(first15)} in first 15")
    N = min(len(v) for v in all_data.values())

    # Test 1: All 28 pairs
    test_all28(all_data, avail_pairs)

    # Test 2: Sydney Open
    test_sydney(all_data, avail_pairs)

    # Test 3: Dealer Capitulation
    test_dealer_cap(all_data, avail_pairs)

    # Test 4: QAI exit
    test_qai_exit(all_data, avail_pairs)

    print(f"\n{'='*70}")
    print(f"ALL 4 TESTS COMPLETE ({time.time()-t_main:.1f}s)")


if __name__ == '__main__':
    main()
