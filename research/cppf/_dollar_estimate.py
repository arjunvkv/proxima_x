"""Quick dollar estimate for multi-pair V2+z scenarios."""
scenarios = {
    "z>=0.5 all 26":  {"gross": 9912415, "tpd": 14196, "wr": 0.734, "usd_pct": 0.30, "jpy_pct": 0.60},
    "z>=1.0 all 26":  {"gross": 6259924, "tpd": 7160,  "wr": 0.749, "usd_pct": 0.28, "jpy_pct": 0.63},
    "z>=2.0 all 26":  {"gross": 2438912, "tpd": 1621,  "wr": 0.777, "usd_pct": 0.25, "jpy_pct": 0.67},
    "z>=2.5 all 26":  {"gross": 1603651, "tpd": 812,   "wr": 0.785, "usd_pct": 0.22, "jpy_pct": 0.70},
    "z>=3.0 all 26":  {"gross": 1118370, "tpd": 436,   "wr": 0.790, "usd_pct": 0.20, "jpy_pct": 0.72},
    "z>=2.0 top 12":  {"gross": 1200000, "tpd": 750,   "wr": 0.780, "usd_pct": 0.20, "jpy_pct": 0.55},
    "z>=2.5 top 8":   {"gross": 550000,  "tpd": 200,   "wr": 0.785, "usd_pct": 0.15, "jpy_pct": 0.60},
    "z>=3.0 top 6":   {"gross": 300000,  "tpd": 100,   "wr": 0.791, "usd_pct": 0.10, "jpy_pct": 0.65},
}

label = "$/day"
print(f"{'Scenario':<25s}  {'t/d':>5s}  {'WR':>5s}  {label:>11s}  {'Monthly':>9s}")
print("-" * 60)
for name, s in scenarios.items():
    g = s["gross"]
    usd_val = g * s["usd_pct"] * 0.10
    jpy_val = g * s["jpy_pct"] / 100 * 0.067
    cross_val = g * (1 - s["usd_pct"] - s["jpy_pct"]) * 0.10
    total_val = usd_val + jpy_val + cross_val
    daily = total_val / 91
    monthly = daily * 22
    d10 = daily * 10
    print(f"{name:<25s}  {s['tpd']:>5d}  {s['wr']:.1%}  ${daily:>7.0f}/d  ${monthly:>7.0f}/mo   |  0.10 lot: ${d10:>5.0f}/d")
