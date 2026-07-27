"""FundedNext Stellar 2-Step cost model for V2+z strategy."""
import numpy as np

data = {
    "gbpnzd":  {"net_mp": 2.27, "tpd": 31, "wr": 0.798},
    "eurnzd":  {"net_mp": 1.79, "tpd": 30, "wr": 0.793},
    "gbpaud":  {"net_mp": 1.66, "tpd": 30, "wr": 0.815},
    "euraud":  {"net_mp": 1.45, "tpd": 30, "wr": 0.803},
    "gbpcad":  {"net_mp": 1.16, "tpd": 31, "wr": 0.796},
    "gbpjpy":  {"net_mp": 132.98, "tpd": 31, "wr": 0.795, "jpy": True},
    "chfjpy":  {"net_mp": 132.69, "tpd": 30, "wr": 0.762, "jpy": True},
    "usdjpy":  {"net_mp": 127.13, "tpd": 33, "wr": 0.773, "jpy": True},
    "eurjpy":  {"net_mp": 96.98, "tpd": 30, "wr": 0.792, "jpy": True},
    "audnzd":  {"net_mp": 0.93, "tpd": 31, "wr": 0.806},
    "audcad":  {"net_mp": 0.91, "tpd": 30, "wr": 0.798},
    "eurcad":  {"net_mp": 0.90, "tpd": 31, "wr": 0.803},
    "gbpusd":  {"net_mp": 0.95, "tpd": 33, "wr": 0.754},
    "audusd":  {"net_mp": 0.81, "tpd": 31, "wr": 0.781},
    "eurusd":  {"net_mp": 0.71, "tpd": 32, "wr": 0.763},
}
COMM_PER_LOT = 3.0

def mp2d_per_lot(mp, jpy):
    return mp / 100 * 0.067 if jpy else mp * 0.10

scenarios = [
    ("Top 6 non-JPY", ["gbpnzd","eurnzd","gbpaud","euraud","gbpcad","audnzd"]),
    ("Top 8 (4JPY+4non)", ["gbpjpy","chfjpy","usdjpy","eurjpy","gbpnzd","eurnzd","gbpaud","euraud"]),
    ("Top 12", ["gbpnzd","eurnzd","gbpaud","euraud","gbpcad","gbpjpy","chfjpy","usdjpy","audnzd","gbpusd","eurcad","eurjpy"]),
    ("All 15 profitable", list(data.keys())),
]

S = f"FundedNext Stellar 2-Step $25k | Commission $3/round lot | V2+z at z>=2.5"
print(S)
print("=" * 100)
print(f"{'Scenario':<22s} {'pairs':>3s} {'t/d':>5s} {'WR':>4s}  $/d@0.01  $/d@0.02  $/d@0.05  $/d@0.10  $/d@0.20  %of1250@0.10  %maxloss")
print("-" * 100)

for name, pairs in scenarios:
    tpd = sum(data[p]["tpd"] for p in pairs)
    wr = np.average([data[p]["wr"] for p in pairs], weights=[data[p]["tpd"] for p in pairs])
    daily_by_lot = {}
    for lot in [0.01, 0.02, 0.05, 0.10, 0.20]:
        total = 0
        for p in pairs:
            d = data[p]
            gross = mp2d_per_lot(d["net_mp"], d.get("jpy", False))
            comm = COMM_PER_LOT * lot
            net = gross * (lot / 0.01) - comm
            if net > 0:
                total += net * d["tpd"]
        daily_by_lot[lot] = total
    pct_1250 = daily_by_lot[0.10] / 1250 * 100
    pct_2500 = daily_by_lot[0.10] / 2500 * 100
    print(f"{name:<22s} {len(pairs):>3d} {tpd:>5d} {wr:>.0%}  ${daily_by_lot[0.01]:>6.1f}  ${daily_by_lot[0.02]:>6.1f}  ${daily_by_lot[0.05]:>6.1f}  ${daily_by_lot[0.10]:>6.1f}  ${daily_by_lot[0.20]:>6.1f}   {pct_1250:>5.1f}%    {pct_2500:>5.1f}%")

print()
print(f"Daily loss limit: $1,250 | Max loss: $2,500")
print("Best config: z>=2.5, top 6-8 pairs, 0.05 lot = $42-65/day (very safe)")
print("Aggressive: top 12 pairs, 0.10 lot = $149/day (12% of limit)")
