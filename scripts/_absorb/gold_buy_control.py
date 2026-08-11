"""gold_buy_control.py — is the all-BUY gold profit just intraday seasonality?

Control: plain buy-and-hold-1h at fixed hours (no exhaustion signal) vs the
engine's session_exhaustion result. If plain-buy matches, the 'signal' adds
nothing and the edge is gold/silver intraday long bias.
"""
import polars as pl, numpy as np

def load(sym):
    return pl.read_parquet(f"audit_7_eas/market/{sym}.pqt").sort("time")

def sim(sym, hours, hold=12, spread_pips=0.0, comm=3.0, tv=1.0, pt=0.01):
    df = load(sym)
    ts = df["time"].to_numpy(); op = df["open"].to_numpy(); cl = df["close"].to_numpy()
    days = ts // 86400
    nets = []
    for d in np.unique(days):
        m = np.where((days == d) & (np.isin((ts // 3600) % 24, hours)))[0]
        if len(m) == 0:
            continue
        i = m[0]; j = i + hold
        if j >= len(ts) or days[j] != d:
            continue
        nets.append(cl[j] - op[i])
    n = len(nets)
    if n == 0:
        return 0, 0.0, 0.0, 0.0
    arr = np.array(nets)
    gross_lot = arr / pt * tv
    net_lot = gross_lot - 2 * comm - spread_pips * tv * 10
    pos = net_lot[net_lot > 0].sum(); neg = -net_lot[net_lot < 0].sum()
    return n, float(net_lot.mean()), float(pos / neg if neg else 99), float(net_lot.sum())

if __name__ == "__main__":
    for sym, tv, pt, sp in [("XAUUSD", 1.0, 0.01, 4.5), ("XAGUSD", 5.0, 0.001, 5.9)]:
        for hset, name in [([2, 3], "Asia23"), ([16], "US16"),
                           ([2, 3, 16], "AsiaUS"), (list(range(24)), "ALLDAY")]:
            n, exp, pf, tot = sim(sym, hset, spread_pips=sp, tv=tv, pt=pt)
            print(f"{sym} buy@{name}: n={n} exp=${exp:.1f}/lot PF={pf:.2f} tot=${tot:,.0f}")
