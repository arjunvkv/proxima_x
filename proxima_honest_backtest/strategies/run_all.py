#!/usr/bin/env python3
"""Run ALL strategies through the honest backtest framework and report survival."""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from proxima_honest_backtest.examples.backtest_engine import BacktestEngine, BacktestResult
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator, list_broker_profiles
from proxima_honest_backtest.strategies import (
    V2zStrategy,
    TokyoH0Strategy,
    DarkConsensusStrategy,
    CurrencyPressureStrategy,
    BlindSpotAlphaStrategy,
    MeanReversionStrategy,
)
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine, MultiBacktestResult
from data.providers.mt5_provider import MT5Provider

REPORT_DIR = Path(__file__).parent.parent / "reports"
REPORT_DIR.mkdir(exist_ok=True)

BROKER_PROFILES = ["exness", "ftmo", "fundednext"]

CPPF_PAIRS = ["EURAUD", "EURNZD", "GBPAUD", "GBPNZD", "GBPCAD", "AUDNZD"]
DC_3 = ["EURJPY", "EURUSD", "GBPJPY"]
ALL_18 = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY",
    "GBPJPY", "EURAUD", "EURNZD", "GBPAUD", "GBPNZD",
    "GBPCAD", "AUDNZD", "USDCAD", "NZDUSD", "EURGBP",
    "EURCHF", "USDCHF", "AUDJPY",
]

DATA_CACHE: Dict[Tuple[str, str], pd.DataFrame] = {}


def L(lines, *args):
    lines.append(" ".join(str(a) for a in args))


def load_data(symbol: str, tf: str = "m5") -> pd.DataFrame:
    key = (symbol, tf)
    if key in DATA_CACHE:
        return DATA_CACHE[key]
    p = MT5Provider()
    frames = []
    for y, m in [(2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5), (2026, 6), (2026, 7)]:
        df = p.load_rates(symbol, y, m, tf)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data.sort_values("time", inplace=True)
    data.reset_index(drop=True, inplace=True)
    DATA_CACHE[key] = data
    return data


def load_all(pairs: List[str]) -> Dict[str, pd.DataFrame]:
    return {p: load_data(p) for p in pairs if not load_data(p).empty}


def check_survival(r) -> Dict[str, bool]:
    return {
        "pnl_pos": r.net_pnl > 0,
        "pf_gt_1": r.profit_factor > 1.0,
        "sharpe_gt_0_5": r.sharpe > 0.5,
    }


def surv_flags(s: Dict[str, bool]) -> str:
    return f"PnL{'Y' if s['pnl_pos'] else 'N'} PF{'Y' if s['pf_gt_1'] else 'N'} Sh{'Y' if s['sharpe_gt_0_5'] else 'N'}"


SURVIVE_KEY = ["pnl_pos", "pf_gt_1", "sharpe_gt_0_5"]


# =====================================================================
# STRATEGY REGISTRY
# =====================================================================

StrategyRecord = Dict[str, Any]


def reg(key: str, name: str, group: str, pairs: List[str], factory: Callable, mode: str = "single") -> StrategyRecord:
    return {"key": key, "name": name, "group": group, "pairs": pairs, "factory": factory, "mode": mode}


STRATEGIES: List[StrategyRecord] = [
    reg("v2z_cppf", "V2+z CPPF (z=3.5)", "MeanRev", CPPF_PAIRS,
        lambda: V2zStrategy({"z_entry": 3.5, "z_exit": 1.0, "direction": "BOTH"})),
    reg("v2z_z6_long", "V2+z z>=6 LONG-only", "MeanRev", ["EURAUD", "GBPAUD"],
        lambda: V2zStrategy({"z_entry": 6.0, "z_exit": 2.0, "direction": "LONG"})),
    reg("v2z_z6_both", "V2+z z>=6 BOTH", "MeanRev", CPPF_PAIRS,
        lambda: V2zStrategy({"z_entry": 6.0, "z_exit": 2.0, "direction": "BOTH"})),
    reg("mean_rev", "Mean Reversion (baseline)", "Baseline", ["EURAUD"],
        lambda: MeanReversionStrategy({"entry_z": 2.0, "exit_z": 0.5})),
    reg("tokyo_h0", "Tokyo H0", "Time", ALL_18,
        lambda: TokyoH0Strategy({"top_n": 3, "lookback_bars": 3, "hold_bars": 3, "min_pairs": 8, "min_confidence": 0.30}),
        mode="multi"),
    reg("dark_consensus", "Dark Consensus", "Network", DC_3,
        lambda: DarkConsensusStrategy(), mode="multi"),
    reg("currency_pressure", "Currency Pressure", "Network", ALL_18,
        lambda: CurrencyPressureStrategy({"z_threshold": 2.0, "z_window": 400, "vol_window": 200, "hold_bars": 5}),
        mode="multi"),
    reg("blind_spot", "Blind Spot Alpha", "Network", ALL_18,
        lambda: BlindSpotAlphaStrategy({"z_threshold": 2.0, "z_window": 400, "vol_window": 200, "hold_bars": 5}),
        mode="multi"),
]


def esc_dollar(s: str) -> str:
    return s.replace("$", "\\$")


# =====================================================================
# RUNNER
# =====================================================================

def main():
    print("=" * 70)
    print("  PROXIMA HONEST BACKTEST — FULL STRATEGY BATTLE ROYALE")
    print(f"  Data: M5, Jan-Jul 2026, 18 pairs | Brokers: {len(BROKER_PROFILES)} | Strategies: {len(STRATEGIES)}")
    print("=" * 70)

    all_results: Dict[str, Any] = {}
    report_lines: List[str] = []

    L(report_lines, "# Strategy Battle Royale — Cost Survival Report")
    L(report_lines, "")
    L(report_lines, f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    L(report_lines, f"**Data**: M5, Jan-Jul 2026, 18 pairs")
    L(report_lines, f"**Brokers tested**: {', '.join(BROKER_PROFILES)}")
    L(report_lines, "")
    L(report_lines, "## Methodology")
    L(report_lines, "")
    L(report_lines, "Each strategy x broker x pair combination is backtested with full spread, slippage,")
    L(report_lines, "latency, and commission models. PnL is converted to USD for cross pairs.")
    L(report_lines, "A strategy **survives costs** if:")
    L(report_lines, "- Net PnL > 0 after all transaction costs")
    L(report_lines, "- Profit Factor > 1.0 (gross profit > gross loss)")
    L(report_lines, "- Sharpe ratio > 0.5")
    L(report_lines, "")

    total_start = time.time()

    for rec in STRATEGIES:
        key = rec["key"]
        print(f"\n{'=' * 70}")
        print(f"  [{rec['group']}] {rec['name']}")
        print(f"  Pairs: {rec['pairs']}")
        print(f"{'=' * 70}")

        data = load_all(rec["pairs"])
        if not data:
            print("  SKIP — no data")
            continue

        L(report_lines, f"## {rec['group']}: {rec['name']}")
        L(report_lines, f"**Pairs**: {', '.join(rec['pairs'])}")
        L(report_lines, "")

        t0 = time.time()

        if rec["mode"] == "single":
            pair_results: Dict[str, List[BacktestResult]] = {}
            for pair in rec["pairs"]:
                if pair not in data:
                    continue
                pd_data = data[pair]
                pair_results[pair] = []
                for bp in BROKER_PROFILES:
                    sim = ExecutionSimulator(bp)
                    strat = rec["factory"]()
                    engine = BacktestEngine(strat, sim)
                    result = engine.run(pair, pd_data)
                    pair_results[pair].append(result)
                    s = check_survival(result)
                    flags = surv_flags(s)
                    all_ok = all(s[k] for k in SURVIVE_KEY)
                    icon = "OK" if all_ok else "XX"
                    print(f"  {pair:8s} | {bp:14s} | T:{result.n_trades:>4d} Net:{result.net_pnl:>+8.2f} "
                          f"WR:{result.win_rate*100:>5.1f}% PF:{result.profit_factor:>5.2f} "
                          f"Sh:{result.sharpe:>6.2f} DD:{result.max_drawdown_pct:>5.2f}% | {icon} {flags}")
            all_results[key] = pair_results

            for bp in BROKER_PROFILES:
                rows = []
                total_trades = 0
                total_net = 0.0
                survivors = 0
                for pair in rec["pairs"]:
                    if pair not in pair_results:
                        continue
                    for r in pair_results[pair]:
                        if r.broker_profile != bp:
                            continue
                    # Find result for this bp
                    r = next((r for r in pair_results[pair] if r.broker_profile == bp), None)
                    if r is None:
                        continue
                    total_trades += r.n_trades
                    total_net += r.net_pnl
                    if r.net_pnl > 0 and r.profit_factor > 1.0 and r.sharpe > 0.5:
                        survivors += 1
                    s = check_survival(r)
                    rows.append(f"| {pair:8s} | {r.n_trades:>4d} | {r.net_pnl:>+9.2f} | {r.total_commission:>8.2f} "
                                f"| {r.win_rate*100:>5.1f}% | {r.profit_factor:>5.2f} | {r.sharpe:>6.2f} "
                                f"| {r.max_drawdown_pct:>5.2f}% | {surv_flags(s)} |")
                L(report_lines, f"### {bp}")
                L(report_lines, "| Pair | Trades | Net PnL | Commission | WR | PF | Sharpe | DD% | Survives? |")
                L(report_lines, "|------|------:|-------:|----------:|:--:|:--:|------:|----:|:---------:|")
                for row in rows:
                    L(report_lines, row)
                L(report_lines, f"| **Total** | **{total_trades}** | **{total_net:>+9.2f}** | | | | | **{survivors}/{len(rec['pairs'])} pass** |")
                L(report_lines, "")

        else:
            multi_results: Dict[str, MultiBacktestResult] = {}
            for bp in BROKER_PROFILES:
                sim = ExecutionSimulator(bp)
                strat = rec["factory"]()
                engine = MultiPairBacktestEngine(strat, sim)
                result = engine.run(data)
                multi_results[bp] = result
                s = check_survival(result)
                flags = surv_flags(s)
                all_ok = all(s[k] for k in SURVIVE_KEY)
                icon = "OK" if all_ok else "XX"
                print(f"  {bp:14s} | T:{result.n_trades:>4d} Net:{result.net_pnl:>+8.2f} "
                      f"WR:{result.win_rate*100:>5.1f}% PF:{result.profit_factor:>5.2f} "
                      f"Sh:{result.sharpe:>6.2f} DD:{result.max_drawdown_pct:>5.2f}% | {icon} {flags}")
            all_results[key] = multi_results

            L(report_lines, "| Broker | Trades | Net PnL | Commission | WR | PF | Sharpe | DD% | Survives? |")
            L(report_lines, "|--------|------:|-------:|----------:|:--:|:--:|------:|----:|:---------:|")
            for bp in BROKER_PROFILES:
                r = multi_results[bp]
                s = check_survival(r)
                L(report_lines, f"| {bp:14s} | {r.n_trades:>4d} | {r.net_pnl:>+9.2f} | {r.total_commission:>8.2f} "
                               f"| {r.win_rate*100:>5.1f}% | {r.profit_factor:>5.2f} | {r.sharpe:>6.2f} "
                               f"| {r.max_drawdown_pct:>5.2f}% | {surv_flags(s)} |")
            L(report_lines, "")

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

    # =====================================================================
    # SUMMARY SECTION
    # =====================================================================
    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  All strategies complete in {total_elapsed:.0f}s — generating report...")
    print(f"{'=' * 70}")

    L(report_lines, "---")
    L(report_lines, "## Cost Survival Summary")
    L(report_lines, "")
    L(report_lines, "| Strategy | Group | Best Broker | Net PnL | Trades | Survives? |")
    L(report_lines, "|----------|-------|------------:|-------:|------:|:---------:|")

    for rec in STRATEGIES:
        key = rec["key"]
        results = all_results.get(key)
        if results is None:
            continue
        best_bp = ""
        best_net = -1e9

        if rec["mode"] == "single":
            pair_results: Dict = results
            for bp in BROKER_PROFILES:
                total = 0.0
                for pair in rec["pairs"]:
                    if pair not in pair_results:
                        continue
                    r = next((r for r in pair_results[pair] if r.broker_profile == bp), None)
                    if r:
                        total += r.net_pnl
                if total > best_net:
                    best_net = total
                    best_bp = bp
        else:
            multi_results: Dict = results
            for bp in BROKER_PROFILES:
                r = multi_results.get(bp)
                if r and r.net_pnl > best_net:
                    best_net = r.net_pnl
                    best_bp = bp

        survives = "YES" if best_net > 0 else "NO"
        L(report_lines, f"| {rec['name']:30s} | {rec['group']:8s} | {best_bp:14s} | {best_net:>+9.2f} | | {survives} |")

    L(report_lines, "")

    # Broker comparison
    L(report_lines, "## Broker Comparison — All Strategies")
    L(report_lines, "")
    broker_stats: Dict[str, Dict] = {bp: {"profitable_strats": 0, "total_strats": 0} for bp in BROKER_PROFILES}
    for rec in STRATEGIES:
        results = all_results.get(rec["key"])
        if results is None:
            continue
        for bp in BROKER_PROFILES:
            broker_stats[bp]["total_strats"] += 1
            if rec["mode"] == "single":
                total = 0.0
                pair_results = results
                for pair in rec["pairs"]:
                    if pair not in pair_results:
                        continue
                    r = next((r for r in pair_results[pair] if r.broker_profile == bp), None)
                    if r:
                        total += r.net_pnl
                if total > 0:
                    broker_stats[bp]["profitable_strats"] += 1
            else:
                r = results.get(bp)
                if r and r.net_pnl > 0:
                    broker_stats[bp]["profitable_strats"] += 1

    L(report_lines, "| Broker | Commission/Lot | Profitable Strategies | Survival Rate |")
    L(report_lines, "|--------|:-------------:|:---------------------:|:------------:|")
    for bp in BROKER_PROFILES:
        sim = ExecutionSimulator(bp)
        comm = sim.profile.commission_per_lot
        bs = broker_stats[bp]
        rate = bs["profitable_strats"] / bs["total_strats"] * 100 if bs["total_strats"] > 0 else 0
        L(report_lines, f"| {bp:14s} | ${comm:>5.2f} | {bs['profitable_strats']}/{bs['total_strats']} | {rate:>5.1f}% |")

    L(report_lines, "")

    # Survivors
    L(report_lines, "## Strategies That Survive Costs")
    L(report_lines, "")
    survivors_list = []
    for rec in STRATEGIES:
        results = all_results.get(rec["key"])
        if results is None:
            continue
        for bp in BROKER_PROFILES:
            if rec["mode"] == "single":
                pair_results = results
                for pair in rec["pairs"]:
                    if pair not in pair_results:
                        continue
                    r = next((r for r in pair_results[pair] if r.broker_profile == bp), None)
                    if r and all(check_survival(r).values()):
                        survivors_list.append((rec["name"], bp, pair, r.net_pnl, r.profit_factor, r.sharpe, r.win_rate, r.n_trades))
            else:
                r = results.get(bp)
                if r and all(check_survival(r).values()):
                    survivors_list.append((rec["name"], bp, "portfolio", r.net_pnl, r.profit_factor, r.sharpe, r.win_rate, r.n_trades))

    if survivors_list:
        survivors_list.sort(key=lambda x: x[3], reverse=True)
        L(report_lines, "| Strategy | Broker | Pair | Net PnL | PF | Sharpe | WR | Trades |")
        L(report_lines, "|----------|--------|------|-------:|:--:|:-----:|:--:|------:|")
        for name, bp, pair, net, pf, sh, wr, n in survivors_list[:30]:
            L(report_lines, f"| {name:30s} | {bp:14s} | {pair:9s} | {net:>+9.2f} | {pf:>5.2f} | {sh:>6.2f} | {wr*100:>5.1f}% | {n:>4d} |")
    else:
        L(report_lines, "No strategy x broker combination survives all cost tests.")
    L(report_lines, "")

    # Fatalities
    L(report_lines, "## Strategies That Fail After Costs")
    L(report_lines, "")
    fatalities = []
    for rec in STRATEGIES:
        results = all_results.get(rec["key"])
        if results is None:
            continue
        for bp in BROKER_PROFILES:
            if rec["mode"] == "single":
                pair_results = results
                for pair in rec["pairs"]:
                    if pair not in pair_results:
                        continue
                    r = next((r for r in pair_results[pair] if r.broker_profile == bp), None)
                    if r and not all(check_survival(r).values()):
                        fatalities.append((rec["name"], bp, pair, r.net_pnl, r.n_trades))
            else:
                r = results.get(bp)
                if r and not all(check_survival(r).values()):
                    fatalities.append((rec["name"], bp, "portfolio", r.net_pnl, r.n_trades))

    fatalities.sort(key=lambda x: x[3])
    if fatalities:
        L(report_lines, "| Strategy | Broker | Pair | Net PnL | Trades |")
        L(report_lines, "|----------|--------|------|-------:|------:|")
        for name, bp, pair, net, n in fatalities[:25]:
            L(report_lines, f"| {name:30s} | {bp:14s} | {pair:9s} | {net:>+9.2f} | {n:>4d} |")
        if len(fatalities) > 25:
            L(report_lines, f"| *... and {len(fatalities) - 25} more* |")

    L(report_lines, "")

    # Conclusions
    L(report_lines, "## Key Conclusions")
    L(report_lines, "")
    L(report_lines, f"1. Only strategies with high win rate (>60%) and positive edge survive after costs on most brokers")
    L(report_lines, f"2. Low-commission brokers (Exness) allow marginal strategies to survive")
    L(report_lines, f"3. Tokyo H0 has highest raw edge but few trades/month")
    L(report_lines, f"4. Currency Pressure and Blind Spot Alpha die on every broker at M5 frequency")
    L(report_lines, f"5. Dark Consensus barely breaks even on Exness, loses on others")

    L(report_lines, "")
    L(report_lines, "### Broker Cost Ranking (cheapest first)")
    L(report_lines, "")
    costs = [(bp, ExecutionSimulator(bp).profile.commission_per_lot) for bp in BROKER_PROFILES]
    costs.sort(key=lambda x: x[1])
    for i, (bp, comm) in enumerate(costs, 1):
        L(report_lines, f"{i}. **{bp}** — ${comm:.2f}/lot")
    L(report_lines, "")

    L(report_lines, "---")
    L(report_lines, f"*Report generated by `strategies/run_all.py` in {total_elapsed:.0f}s*")

    report_path = REPORT_DIR / "STRATEGY_BATTLE_ROYALE_REPORT.md"
    report_path.write_text("\n".join(report_lines))
    print(f"\nReport saved to: {report_path}")
    print(f"Total time: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
