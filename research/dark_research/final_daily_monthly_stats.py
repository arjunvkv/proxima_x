#!/usr/bin/env python3
"""Final realistic daily + monthly stats after ALL stress effects."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, os, calendar

DATA = r"C:\Trading\Agentic_Trading\proxima_x\research\dark_research\dukascopy_data"
HALF_SPREAD_PIPS = np.array([0.5, 0.3, 0.7])
ECN_COMM, LOT, MAG95 = 7, 100000, 0.00018741
PIPS = np.array([0.01, 0.0001, 0.01])

def pip_val(p, usdjpy):
    return 10.0 if p == 1 else 1000.0 / usdjpy

def load_all():
    frames = {}
    for p, pn in [("eurjpy","EURJPY"),("eurusd","EURUSD"),("gbpjpy","GBPJPY")]:
        dfs = []
        for y in [2024, 2026]:
            for m in range(1, 13):
                if (y==2024 and m<10) or (y==2026 and m>6): continue
                ld = calendar.monthrange(y, m)[1]
                f = os.path.join(DATA, f"{p}-m1-bid-{y}-{m:02d}-01-{y}-{m:02d}-{ld}.csv")
                if not os.path.exists(f): continue
                df = pd.read_csv(f)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                dfs.append(df)
        frames[pn] = pd.concat(dfs).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    common = sorted(set(frames["EURJPY"]["timestamp"]) & set(frames["EURUSD"]["timestamp"]) & set(frames["GBPJPY"]["timestamp"]))
    tmap = {p: {t: i for i, t in enumerate(frames[p]["timestamp"])} for p in frames}
    close = np.column_stack([frames[p]["close"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    opens = np.column_stack([frames[p]["open"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    high = np.column_stack([frames[p]["high"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    low = np.column_stack([frames[p]["low"].values[[tmap[p][t] for t in common]] for p in ["EURJPY","EURUSD","GBPJPY"]])
    times = np.array([int(t.timestamp()) for t in common], dtype=np.int64)
    return close, opens, high, low, times

close, opens, high, low, times = load_all()
T = close.shape[0]
rets = np.diff(np.log(close), axis=0)
up = rets > 0; consensus = up.all(axis=1) | (~up).all(axis=1)
direction = np.where(up.all(axis=1), 1.0, -1.0)
avg_mag = np.mean(np.abs(rets), axis=1)
pair_mags = np.abs(rets)
dt_all = pd.to_datetime(times, unit="s")
hour_arr = dt_all.hour.values[1:]
usdjpy_proxy = close[:,0] / close[:,1]

atr_arr = np.max([(high[:,p]-low[:,p])/PIPS[p] for p in range(3)], axis=0)
atr_median = np.median(atr_arr[1:])

# Generate trades with COMBINED stress: latency + variable slippage + 1.5x spread + comm
te_idx = np.where(consensus & (hour_arr >= 7) & (hour_arr <= 21) & (avg_mag > MAG95))[0]
te_idx = te_idx[te_idx + 3 < T - 1]
bi = np.argmax(pair_mags[te_idx], axis=1)
avg_usdjpy = np.mean(usdjpy_proxy[te_idx])

trades = []
for j,i in enumerate(te_idx):
    p = bi[j]; next_i = min(i+1, T-1)
    entry_price = opens[next_i, p]
    exit_price = close[min(i+3, T-1), p]
    gross = np.log(exit_price/entry_price)*direction[te_idx[j]]
    spread = HALF_SPREAD_PIPS[p]*2*1.5*pip_val(p, avg_usdjpy)
    slip_mult = atr_arr[i] / max(atr_median, 0.1)
    slip_var = (0.2 + 0.3 * min(slip_mult, 5.0)) * 2 * pip_val(p, avg_usdjpy)
    gusd = LOT*gross if p==1 else LOT*gross*entry_price/usdjpy_proxy[i]
    pnl = gusd - spread - slip_var - ECN_COMM
    trades.append({
        "ts": dt_all[i], "pnl": pnl, "dir": direction[te_idx[j]],
        "pair": ["EURJPY","EURUSD","GBPJPY"][p]
    })

df = pd.DataFrame(trades)
df["date"] = df["ts"].dt.date
df["month"] = df["ts"].dt.to_period("M")

print("=" * 100)
print("FINAL REALISTIC STATS — After combined stress (latency + ATRe slip + 1.5x spread + $7 comm)")
print("=" * 100)
print(f"Total trades: {len(df):,}  |  Period: {df['ts'].min().date()} — {df['ts'].max().date()}")
print()

# ====== DAILY STATS ======
daily = df.groupby("date").agg(n=("pnl","count"), pnl=("pnl","sum"), wr=("pnl", lambda x: np.mean(x>0)*100)).reset_index()
d = daily["pnl"].values
print("DAILY STATISTICS:")
print("-" * 70)
print(f"  Trading days:                    {len(daily)}")
print(f"  Avg trades/day:                  {daily['n'].mean():.1f}")
print(f"  Avg daily PnL:                   ${np.mean(d):.2f}")
print(f"  Median daily PnL:                ${np.median(d):.2f}")
print(f"  Daily Sharpe:                    {np.mean(d)/(np.std(d)+1e-10)*np.sqrt(252):.2f}")
print(f"  Daily win rate:                  {np.mean(d>0)*100:.1f}%")
print(f"  Std dev of daily PnL:            ${np.std(d):.2f}")
print(f"  Best day:                        ${np.max(d):,.0f}")
print(f"  Worst day:                       ${np.min(d):,.0f}")
print(f"  Daily VaR 95%:                   ${np.percentile(d, 5):,.0f}")
print(f"  Daily VaR 99%:                   ${np.percentile(d, 1):,.0f}")
print(f"  Profit factor (gross/gross loss): {daily[daily['pnl']>0]['pnl'].sum()/abs(daily[daily['pnl']<0]['pnl'].sum()):.2f}")
print(f"  Daily avg win / avg loss:        ${daily[daily['pnl']>0]['pnl'].mean():.0f} / ${abs(daily[daily['pnl']<0]['pnl'].mean()):.0f}")
print()

# ====== MONTHLY STATS ======
print("MONTHLY STATISTICS:")
print("-" * 100)
print(f"{'Month':>10s} {'n':>6s} {'tpd':>5s} {'WR%':>5s} {'Tot$':>10s} {'Avg$':>7s} {'Daily$':>8s} {'Sharpe':>7s} {'MDD':>7s}")
print("-" * 100)

monthly_rows = []
for m in sorted(df["month"].unique()):
    sub = df[df["month"] == m]
    nd = len(sub["date"].unique())
    tot = sub["pnl"].sum()
    avg = sub["pnl"].mean()
    wr = np.mean(sub["pnl"]>0)*100
    tpd = len(sub) / max(nd, 1)
    dly = tot / max(nd, 1)
    sh = np.mean(sub["pnl"])/(np.std(sub["pnl"])+1e-10)*np.sqrt(1440/3)
    mdd = sub["pnl"].cumsum().min()
    print(f"{str(m):>10s} {len(sub):6d} {tpd:5.1f} {wr:5.1f} {tot:10,.0f} {avg:7.2f} {dly:8,.0f} {sh:7.2f} {mdd:7,}")
    monthly_rows.append({"month": str(m), "n": len(sub), "tpd": tpd, "wr": wr, "total": tot, "avg": avg, "daily": dly, "sharpe": sh, "mdd": mdd})

print("-" * 100)
tot_all = df["pnl"].sum()
print(f"{'TOTAL':>10s} {len(df):6d} {len(df)/len(df['date'].unique()):5.1f} {np.mean(df['pnl']>0)*100:5.1f} {tot_all:10,.0f} {df['pnl'].mean():7.2f} {tot_all/len(daily):8,.0f} {np.mean(df['pnl'])/(np.std(df['pnl'])+1e-10)*np.sqrt(1440/3):7.2f} {df['pnl'].cumsum().min():7,}")
print()

# ====== TRADE-LEVEL STATS ======
print("TRADE-LEVEL STATS:")
print("-" * 70)
p = df["pnl"].values
print(f"  Avg PnL per trade:               ${np.mean(p):.2f}")
print(f"  Median PnL per trade:            ${np.median(p):.2f}")
print(f"  Std dev per trade:               ${np.std(p):.2f}")
print(f"  Trade Sharpe (annualized):       {np.mean(p)/(np.std(p)+1e-10)*np.sqrt(1440/3):.2f}")
print(f"  Win rate:                        {np.mean(p>0)*100:.1f}%")
print(f"  Avg win / Avg loss:              ${np.mean(p[p>0]):.2f} / ${abs(np.mean(p[p<0])):.2f}")
print(f"  Profit factor:                   {np.sum(p[p>0])/abs(np.sum(p[p<0])):.2f}")
print(f"  Max consecutive wins:            {max(len(list(g)) for k,g in __import__('itertools').groupby(p>0) if k)}")
print(f"  Max consecutive losses:          {max(len(list(g)) for k,g in __import__('itertools').groupby(p>0) if not k)}")
print(f"  Max single trade win:            ${np.max(p):.2f}")
print(f"  Max single trade loss:           ${np.min(p):.2f}")
print(f"  Trade VaR 95%:                   ${np.percentile(p, 5):.2f}")
print(f"  Trade VaR 99%:                   ${np.percentile(p, 1):.2f}")
print()

# ====== PAIR DISTRIBUTION ======
print("PAIR DISTRIBUTION:")
for pair in ["EURJPY","EURUSD","GBPJPY"]:
    sub = df[df["pair"] == pair]
    p_sub = sub["pnl"].values
    print(f"  {pair:>7s}: n={len(sub):5d} ({len(sub)/len(df)*100:.0f}%)  WR={np.mean(p_sub>0)*100:.1f}%  Sharpe={np.mean(p_sub)/(np.std(p_sub)+1e-10)*np.sqrt(1440/3):.2f}  Avg=${np.mean(p_sub):.2f}  Tot=${np.sum(p_sub):,.0f}")
print()

# ====== DIRECTION ======
print("DIRECTION:")
for dlab, cond in [("LONG", df["dir"] > 0), ("SHORT", df["dir"] < 0)]:
    p_sub = df.loc[cond, "pnl"].values
    print(f"  {dlab:>6s}: n={len(p_sub):5d} ({len(p_sub)/len(df)*100:.0f}%)  WR={np.mean(p_sub>0)*100:.1f}%  Sharpe={np.mean(p_sub)/(np.std(p_sub)+1e-10)*np.sqrt(1440/3):.2f}  Avg=${np.mean(p_sub):.2f}")

# ====== 1-LOT DOLLAR VALUES ======
print()
print("=" * 100)
print("DOLLAR VALUES (per 1 lot / $100k notional)")
print("=" * 100)
avg_daily = np.mean(d)
total_days = len(daily)
print(f"  Daily:      ${avg_daily:,.0f}")
print(f"  Weekly:     ${avg_daily*5:,.0f}")
print(f"  Monthly:    ${avg_daily*21:,.0f}")
print(f"  Quarterly:  ${avg_daily*63:,.0f}")
print(f"  Yearly:     ${avg_daily*252:,.0f}")
print(f"  9-month:    ${np.sum(d):,.0f}")
print(f"  Avg trade:  ${np.mean(p):.2f}")
