"""NY session consensus exhaustion — deeper dive with spread accounting."""
import numpy as np
import pandas as pd
from collections import defaultdict

df = pd.read_parquet("data/temp/mt5_m1_9day.parquet")
pairs = sorted(df["pair"].unique())

aligned = {}
for p in pairs:
    sub = df[df["pair"] == p].sort_values("time")
    aligned[p] = dict(zip(sub["time"], sub["close"]))
all_times = sorted(set.intersection(*[set(aligned[p].keys()) for p in pairs]))
price = np.column_stack([np.array([aligned[p][t] for t in all_times]) for p in pairs])
n, npairs = price.shape

# Approximate spreads (pips) for major pairs during liquid hours
spreads = {"AUDUSD": 1.5, "EURJPY": 2.5, "EURUSD": 1.5, "GBPJPY": 4.0, "GBPUSD": 2.0, "NZDUSD": 2.0, "USDJPY": 1.8}

hours = np.array([int(pd.Timestamp(t, unit='s').hour + pd.Timestamp(t, unit='s').minute/60) for t in all_times])
ny_mask = (hours >= 12) & (hours < 21)  # NY session
ny_idx = np.where(ny_mask)[0]

print("NY SESSION CONSENSUS EXHAUSTION — FILTER ANALYSIS")
print("=" * 60)

for lookback in [30, 60, 90, 120]:
    for min_agreement in [0.75, 0.85]:
        for fwd in [10, 20, 30]:
            pair_data = defaultdict(lambda: {"trades": 0, "wins": 0, "total_pnl": 0.0})

            for base_i in ny_idx:
                i = int(base_i)
                if i < lookback or i + fwd >= n:
                    continue

                rets = np.array([price[i, j] / price[i - lookback, j] - 1 for j in range(npairs)])
                signs = np.sign(rets)
                n_pos = np.sum(signs > 0)
                n_neg = np.sum(signs < 0)
                if n_pos + n_neg < 5:
                    continue
                agreement = max(n_pos, n_neg) / (n_pos + n_neg)
                if agreement < min_agreement:
                    continue

                consensus_dir = 1 if n_pos > n_neg else -1

                for j in range(npairs):
                    if np.sign(rets[j]) != consensus_dir:
                        continue
                    pair = pairs[j]
                    fwd_ret = price[i + fwd, j] / price[i, j] - 1
                    spread_cost = spreads[pair] * 0.0001
                    if pair in ("EURJPY", "GBPJPY", "USDJPY"):
                        spread_cost = spreads[pair] * 0.01  # JPY pairs quote differently

                    # We trade AGAINST consensus (reversal)
                    if consensus_dir == 1:
                        pnl = -fwd_ret - spread_cost  # short
                    else:
                        pnl = fwd_ret - spread_cost  # long

                    k = pair
                    pair_data[k]["trades"] += 1
                    pair_data[k]["wins"] += 1 if pnl > 0 else 0
                    pair_data[k]["total_pnl"] += pnl

            active = [(k, v) for k, v in pair_data.items() if v["trades"] >= 20]
            if active:
                best = max(active, key=lambda x: x[1]["wins"]/x[1]["trades"])
                wr = best[1]["wins"]/best[1]["trades"]*100
                avg_pnl = best[1]["total_pnl"]/best[1]["trades"]*10000  # in bps
                print(f"  L={lookback:>3d} A={min_agreement:.0%} F={fwd:>2d}: {best[0]:>7s} | {best[1]['trades']:>4d} trades | WR={wr:.1f}% | avg={avg_pnl:+.1f}bps")
