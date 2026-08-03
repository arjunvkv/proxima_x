#!/usr/bin/env python3
"""Inter-Session Arbitrage (#9) — sweep + full validation."""
import sys, time, json, math, random
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.inter_session.strategy import InterSessionStrategy, ALL_PAIRS
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from data.providers.mt5_provider import MT5Provider

BROKERS = ["exness","ftmo","fundednext","fusionmarkets","dukascopy"]
MONTHS = [(2026, m) for m in range(1, 8)]
ENTRY_HOURS = [8, 16]
LOOKBACKS = [6, 12, 24]
HOLDS = [6, 12, 24]
TOP_N = [3, 5]


def load_and_align(tf="m5"):
    provider = MT5Provider()
    raw = {}
    for p in ALL_PAIRS:
        frames = [provider.load_rates(p, y, m, tf) for y, m in MONTHS]
        frames = [f for f in frames if not f.empty]
        if frames:
            d = pd.concat(frames, ignore_index=True)
            d.sort_values("time", inplace=True)
            d.reset_index(drop=True, inplace=True)
            raw[p] = d
    pieces = []
    for pair, df in raw.items():
        sub = df.set_index("time")[["close","open","high","low","tick_volume","spread"]]
        sub.columns = [pair, f"{pair}_open", f"{pair}_high", f"{pair}_low", f"{pair}_volume", f"{pair}_spread"]
        pieces.append(sub)
    aligned = pd.concat(pieces, axis=1, sort=True)
    aligned.sort_index(inplace=True)
    aligned.ffill(inplace=True); aligned.bfill(inplace=True)
    aligned.reset_index(inplace=True); aligned.rename(columns={"index": "time"}, inplace=True)
    return raw, aligned.to_dict("records")


def run_cfg(data, pre_align, hours, lb, hold, top_n, broker):
    s = InterSessionStrategy({"entry_hours": hours, "lookback_bars": lb, "hold_bars": hold, "top_n": top_n})
    e = MultiPairBacktestEngine(s, ExecutionSimulator(broker))
    return e.run(data, pre_aligned=pre_align)


def run_p0(pre_align):
    """P0: Check if session transitions show reversion signal."""
    print("=" * 70)
    print("PHASE 0: Session Transition Reversion Analysis")
    print("=" * 70)

    pair_data = defaultdict(lambda: {"events_8":0, "wins_8":0, "ret_8":0.0, "events_16":0, "wins_16":0, "ret_16":0.0})

    for row_idx, row in enumerate(pre_align):
        ts = row["time"]
        hour = ts.hour if hasattr(ts, "hour") else 0
        today = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)

        if hour == 8:
            key = "8"
            for pair in ALL_PAIRS:
                close = row.get(pair)
                if close is None or np.isnan(close):
                    continue
                # Find close 12 bars ago
                prev_idx = row_idx - 12
                if prev_idx >= 0:
                    prev_row = pre_align[prev_idx]
                    prev_close = prev_row.get(pair)
                    if prev_close is not None and not np.isnan(prev_close):
                        ret = (close - prev_close) / prev_close
                        pair_data[pair]["events_8"] += 1
                        # Forward return over hold=12 (60 min)
                        fwd_idx = row_idx + 12
                        if fwd_idx < len(pre_align):
                            fwd_row = pre_align[fwd_idx]
                            fwd_close = fwd_row.get(pair)
                            if fwd_close is not None and not np.isnan(fwd_close):
                                entry_open = row.get(f"{pair}_open", close)
                                if ret < 0:
                                    fwd_ret = (fwd_close - entry_open) / entry_open
                                    pair_data[pair]["ret_8"] += fwd_ret
                                    if fwd_ret > 0:
                                        pair_data[pair]["wins_8"] += 1

        if hour == 16:
            key = "16"
            for pair in ALL_PAIRS:
                close = row.get(pair)
                if close is None or np.isnan(close):
                    continue
                prev_idx = row_idx - 12
                if prev_idx >= 0:
                    prev_row = pre_align[prev_idx]
                    prev_close = prev_row.get(pair)
                    if prev_close is not None and not np.isnan(prev_close):
                        ret = (close - prev_close) / prev_close
                        pair_data[pair]["events_16"] += 1
                        fwd_idx = row_idx + 12
                        if fwd_idx < len(pre_align):
                            fwd_row = pre_align[fwd_idx]
                            fwd_close = fwd_row.get(pair)
                            if fwd_close is not None and not np.isnan(fwd_close):
                                entry_open = row.get(f"{pair}_open", close)
                                if ret < 0:
                                    fwd_ret = (fwd_close - entry_open) / entry_open
                                    pair_data[pair]["ret_16"] += fwd_ret
                                    if fwd_ret > 0:
                                        pair_data[pair]["wins_16"] += 1

    for hour_label, event_key, win_key, ret_key in [("08 UTC", "events_8", "wins_8", "ret_8"),
                                                     ("16 UTC", "events_16", "wins_16", "ret_16")]:
        print(f"\n--- Entry at {hour_label} (LONG most declined over prior 60min) ---")
        print(f"{'Pair':>6s}  {'Events':>7s}  {'WR':>5s}  {'AvgRet%':>8s}")
        for pair in sorted(ALL_PAIRS, key=lambda p: pair_data[p][win_key]/max(pair_data[p][event_key],1), reverse=True):
            d = pair_data[pair]
            if d[event_key] < 10:
                continue
            wr = d[win_key] / d[event_key]
            ret = d[ret_key] / d[event_key] * 100
            print(f"{pair:>6s}  {d[event_key]:>7d}  {wr:>4.1%}  {ret:>+7.3f}%")

    return pair_data


def main():
    t0 = time.time()
    print("Loading & aligning M5 data...")
    raw, pre_align = load_and_align()
    print(f"  {len(raw)} pairs, {len(pre_align):,} aligned bars ({time.time()-t0:.1f}s)")

    pair_data = run_p0(pre_align)

    # Phase 1: Sweep on Exness
    print(f"\n{'='*70}")
    print("PHASE 1: Parameter sweep on Exness")
    print(f"{'='*70}")

    phases = ["8_only", "16_only", "8_and_16"]
    phase1 = []
    total_cfgs = len(phases) * len(LOOKBACKS) * len(HOLDS) * len(TOP_N)
    cfg_idx = 0

    t1 = time.time()
    for phase in phases:
        hours = {"8_only": [8], "16_only": [16], "8_and_16": [8, 16]}[phase]
        for lb in LOOKBACKS:
            for hold in HOLDS:
                for n in TOP_N:
                    cfg_idx += 1
                    r = run_cfg(raw, pre_align, hours, lb, hold, n, "exness")
                    elapsed = time.time() - t1; t1 = time.time()
                    entry = {"phase": phase, "hours": hours, "lb": lb, "hold": hold, "n": n,
                             "trades": r.n_trades, "net_pnl": r.net_pnl,
                             "wr": r.win_rate, "pf": r.profit_factor}
                    phase1.append(entry)
                    print(f"  [{cfg_idx:>2d}/{total_cfgs}] {phase:>10s} lb={lb:>2d} h={hold:>2d} n={n:>2d} | "
                          f"T={r.n_trades:>3d} Net=${r.net_pnl:>+8.2f} WR={r.win_rate*100:>5.1f}% "
                          f"PF={r.profit_factor:>5.2f} ({elapsed:.1f}s)")

    phase1.sort(key=lambda x: x["net_pnl"], reverse=True)
    print(f"\n--- Exness Top 10 ---")
    for i, r in enumerate(phase1[:10]):
        print(f"  #{i+1} {r['phase']:>10s} lb={r['lb']:>2d} hold={r['hold']:>2d} n={r['n']:>2d} | "
              f"T={r['trades']:>3d} Net=${r['net_pnl']:>+8.2f} "
              f"WR={r['wr']*100:>5.1f}% PF={r['pf']:>5.2f}")

    best = phase1[0]
    print(f"\n--- Best config: {best['phase']} lb={best['lb']} hold={best['hold']} n={best['n']} ---")

    # Phase 2: Multi-broker validation
    print(f"\n{'='*70}")
    print(f"PHASE 2: Multi-broker validation (best config)")
    print(f"{'='*70}")
    phase2 = []
    for broker in BROKERS:
        r = run_cfg(raw, pre_align, best["hours"], best["lb"], best["hold"], best["n"], broker)
        entry = {**{k: best[k] for k in ["phase","lb","hold","n"]},
                 "broker": broker, "trades": r.n_trades, "net_pnl": r.net_pnl,
                 "wr": r.win_rate, "pf": r.profit_factor, "dd": r.max_drawdown_pct,
                 "avg_win": r.avg_win, "avg_loss": r.avg_loss, "recon": r.reconciliation_pass}
        phase2.append(entry)
        print(f"  {broker:>15s} | T={r.n_trades:>3d} Net=${r.net_pnl:>+8.2f} "
              f"WR={r.win_rate*100:>5.1f}% PF={r.profit_factor:>5.2f} "
              f"DD={r.max_drawdown_pct:>5.2f}% Recon={'PASS' if r.reconciliation_pass else 'FAIL'}")

    # Phase 3: Sign-permutation test (best config on Exness)
    print(f"\n{'='*70}")
    print(f"PHASE 3: Sign-permutation test (10,000 shuffles)")
    print(f"{'='*70}")
    base = run_cfg(raw, pre_align, best["hours"], best["lb"], best["hold"], best["n"], "exness")
    base_sharpe = base.sharpe
    # Filter to only EXIT trades (pnl != 0) — entries have pnl=0
    exit_trades = [t for t in base.trades if t.pnl != 0]
    print(f"  Observed Sharpe: {base_sharpe:.2f} ({len(exit_trades)} exit trades)")

    pnl_series = np.array([t.pnl for t in exit_trades])
    count_exceed = 0
    N_PERM = 10000
    for i in range(N_PERM):
        sign = np.random.choice([1, -1], size=len(pnl_series))
        shuffled = pnl_series * sign
        s = float(np.mean(shuffled) / (np.std(shuffled) + 1e-12)) * math.sqrt(252 * 288 / len(shuffled))
        if s >= base_sharpe:
            count_exceed += 1
    p_value = (count_exceed + 1) / (N_PERM + 1)

    # Phase 4: Walk-forward (sequential windows on exit trades)
    print(f"\n{'='*70}")
    print(f"PHASE 4: Walk-forward (5 windows)")
    print(f"{'='*70}")
    n = len(exit_trades)
    window_size = max(n // 5, 3)
    wf_results = []
    for w in range(5):
        start = w * window_size
        mid = start + int(window_size * 0.7)
        end = min(start + window_size, n)
        if mid >= end or start >= n:
            break
        train = pnl_series[start:mid]
        test = pnl_series[mid:end]
        if len(train) < 3 or len(test) < 3:
            break
        train_s = float(np.mean(train) / (np.std(train) + 1e-12)) * math.sqrt(252 * 288 / len(train))
        test_s = float(np.mean(test) / (np.std(test) + 1e-12)) * math.sqrt(252 * 288 / len(test))
        wf_results.append({"window": w+1, "train_n": len(train), "test_n": len(test),
                           "train_sharpe": round(train_s, 2), "test_sharpe": round(test_s, 2)})
        print(f"  Window {w+1}: train={len(train)} trades (IS Sh={train_s:.2f}) "
              f"→ test={len(test)} trades (OOS Sh={test_s:.2f})")

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Best config: {best['phase']} lb={best['lb']} hold={best['hold']} n={best['n']}")
    print(f"  Exness: ${best['net_pnl']:.2f} net ({best['trades']} trades, {best['wr']*100:.1f}% WR)")
    print(f"  Sign-permutation p={p_value:.4f} ({count_exceed}/{N_PERM} exceed)")
    print(f"  Walk-forward: {len(wf_results)} windows")
    for wf in wf_results:
        print(f"    W{wf['window']}: IS Sh={wf['train_sharpe']} OOS Sh={wf['test_sharpe']}")
    survivors = [r for r in phase2 if r["net_pnl"] > 0]
    print(f"  Broker survival: {len(survivors)}/{len(BROKERS)} positive Net PnL")

    out = Path(__file__).parent / "sweep_results.json"
    with open(out, "w") as f:
        json.dump({
            "p0": {pair: dict(d) for pair, d in pair_data.items()},
            "phase1": phase1,
            "phase2": phase2,
            "validation": {
                "base_sharpe": round(base_sharpe, 4),
                "perm_p_value": round(p_value, 4),
                "perm_exceed": count_exceed,
                "perm_total": N_PERM,
                "walkforward": wf_results,
                "broker_survivors": len(survivors),
                "broker_total": len(BROKERS),
            },
            "total_sec": round(time.time() - t0, 1),
        }, f, indent=2, default=str)
    print(f"\nSaved to {out}")
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
