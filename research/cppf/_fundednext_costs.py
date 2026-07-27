"""FundedNext Stellar 2-Step cost model for V2+z strategy.
Commission: $3/round lot (Stellar 2-Step specific)
Spread: variable (uses BASE_COST estimates from parquet data)
"""
import numpy as np
import pandas as pd
from pathlib import Path

# Per-pair data at z>=2.5 from combined backtest
data = {
    "gbpnzd":  {"net_mp": 2.27, "tpd": 31, "wr": 0.798, "is_jpy": False},
    "eurnzd":  {"net_mp": 1.79, "tpd": 30, "wr": 0.793, "is_jpy": False},
    "gbpaud":  {"net_mp": 1.66, "tpd": 30, "wr": 0.815, "is_jpy": False},
    "euraud":  {"net_mp": 1.45, "tpd": 30, "wr": 0.803, "is_jpy": False},
    "gbpcad":  {"net_mp": 1.16, "tpd": 31, "wr": 0.796, "is_jpy": False},
    "audnzd":  {"net_mp": 0.93, "tpd": 31, "wr": 0.806, "is_jpy": False},
    "eurcad":  {"net_mp": 0.90, "tpd": 31, "wr": 0.803, "is_jpy": False},
    "gbpusd":  {"net_mp": 0.95, "tpd": 33, "wr": 0.754, "is_jpy": False},
    "audusd":  {"net_mp": 0.81, "tpd": 31, "wr": 0.781, "is_jpy": False},
    "audcad":  {"net_mp": 0.91, "tpd": 30, "wr": 0.798, "is_jpy": False},
    "eurusd":  {"net_mp": 0.71, "tpd": 32, "wr": 0.763, "is_jpy": False},
    "eurgbp":  {"net_mp": 0.36, "tpd": 32, "wr": 0.817, "is_jpy": False},
    # JPY pairs
    "chfjpy":  {"net_mp": 132.69, "tpd": 30, "wr": 0.762, "is_jpy": True},
    "gbpjpy":  {"net_mp": 132.98, "tpd": 31, "wr": 0.795, "is_jpy": True},
    "usdjpy":  {"net_mp": 127.13, "tpd": 33, "wr": 0.773, "is_jpy": True},
    "eurjpy":  {"net_mp": 96.98, "tpd": 30, "wr": 0.792, "is_jpy": True},
    "nzdjpy":  {"net_mp": 58.14, "tpd": 30, "wr": 0.774, "is_jpy": True},
}

COMMISSION_PER_LOT = 3.0  # $3/round lot Stellar 2-Step
LOT_MULT = [0.01, 0.02, 0.05, 0.10]  # lot sizes to test

def mp_to_dollar(mp, is_jpy):
    if is_jpy:
        return mp / 100 * 0.067
    else:
        return mp * 0.10

def commission_cost(lot):
    return COMMISSION_PER_LOT * lot  # per trade

print("FundedNext Stellar 2-Step — V2+z z>=2.5 per-pair profitability")
print("=" * 80)
print(f"{'Pair':>8s}  {'net_mp':>7s}  {'$/trade':>8s}  {'-comm':>8s}  {'tpd':>4s}  {'$/day(0.01)':>12s}  {'$/day(0.10)':>12s}")
print("-" * 80)

total_tpd = 0
for pair, d in sorted(data.items(), key=lambda x: -mp_to_dollar(x[1]["net_mp"], x[1]["is_jpy"])):
    net_dollar_per_trade = mp_to_dollar(d["net_mp"], d["is_jpy"])
    comm_per_trade = commission_cost(0.01)
    net_after_comm = net_dollar_per_trade - comm_per_trade
    daily_01 = net_after_comm * d["tpd"]
    daily_10 = net_after_comm * d["tpd"] * 10
    total_tpd += d["tpd"]
    
    if net_after_comm > 0:
        print(f"  {pair:>8s}  {d['net_mp']:>+7.2f}  {net_dollar_per_trade:>+7.4f}  {net_after_comm:>+7.4f}  {d['tpd']:>3d}  {daily_01:>+11.2f}  {daily_10:>+11.2f}")
    else:
        print(f"  {pair:>8s}  {d['net_mp']:>+7.2f}  {net_dollar_per_trade:>+7.4f}  {net_after_comm:>+7.4f}  {d['tpd']:>3d}  {daily_01:>+11.2f}  {daily_10:>+11.2f}  << UNPROFITABLE")

print("\n" + "=" * 80)

# Scenario analysis
scenarios = {
    "Top 6 non-JPY":   ["gbpnzd", "eurnzd", "gbpaud", "euraud", "gbpcad", "audnzd"],
    "Top 6 mixed":     ["gbpnzd", "eurnzd", "gbpaud", "chfjpy", "gbpjpy", "usdjpy"],
    "Top 8 mixed":     ["gbpnzd", "eurnzd", "gbpaud", "euraud", "chfjpy", "gbpjpy", "usdjpy", "eurjpy"],
    "Top 12 mixed":    list(data.keys())[:12],
    "All 17 profitable": [k for k in data if mp_to_dollar(data[k]["net_mp"], data[k]["is_jpy"]) > commission_cost(0.01)],
}

print("\nComprehensive scenario table for FundedNext Stellar 2-Step $25k:")
print(f"{'Scenario':<22s}  {'pairs':>4s}  {'t/d':>5s}  {'WR':>5s}  {'$/d(0.01)':>10s}  {'$/d(0.10)':>10s}  {'vs $1.25k':>10s}  {'vs max':>8s}")
print("-" * 85)

for name, pairs in scenarios.items():
    n_pairs = len(pairs)
    tpd = sum(data[p]["tpd"] for p in pairs)
    wr = np.average([data[p]["wr"] for p in pairs], weights=[data[p]["tpd"] for p in pairs])
    
    for lot in [0.01, 0.02, 0.05, 0.10]:
        daily = 0
        for p in pairs:
            d = data[p]
            gross = mp_to_dollar(d["net_mp"], d["is_jpy"])
            comm = commission_cost(lot)
            net = gross - comm
            if net > 0:
                daily += net * d["tpd"] * (lot / 0.01)
        
        vs1250 = daily / 1250 * 100  # % of daily loss limit
        vs2500 = daily / 2500 * 100  # % of max loss limit
        
        if lot == 0.01:
            print(f"{name:<22s}  {n_pairs:>3d}   {tpd:>4d}  {wr:>.0%}  ${daily:>7.2f}  ${daily*10:>8.2f}  {vs1250:>7.1f}%  {vs2500:>6.1f}%")
        elif lot == 0.10:
            print(f"{'':<22s}  {'':>3s}   {'':>4s}  {'':>5s}  ${daily:>7.0f}/d  {'':>8s}  {vs1250:>7.0f}%  {vs2500:>6.0f}%  [0.10 lot]")
