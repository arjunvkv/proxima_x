"""Analyze tick log hp/lp distribution vs backtest threshold."""
import csv, sys
logdir = r"C:\Trading\Agentic_Trading\proxima_x\paper_trade\live\logs_impulse_fade_v2"

bp = {"EURUSD": 0.0001, "EURJPY": 0.01}
thresh = {"EURUSD": 0.0005, "EURJPY": 0.10}  # 5 and 10 pips

by_pair = {}
with open(f"{logdir}/ticks.csv") as f:
    r = csv.DictReader(f)
    for row in r:
        pair = row["pair"]
        if pair not in by_pair:
            by_pair[pair] = {"hp": [], "lp": [], "n": 0, "spreads": []}
        s = by_pair[pair]
        s["n"] += 1
        try:
            h = float(row["hp"]); l = float(row["lp"])
            if h > 0: s["hp"].append(h)
            if l > 0: s["lp"].append(l)
            sp = float(row["spread"])
            if sp > 0: s["spreads"].append(sp)
        except: pass

for pair, s in by_pair.items():
    pip = bp[pair]; th = thresh[pair]
    print(f"\n{pair}: {s['n']} ticks ({s['n']/600:.1f}/sec)")
    for label, vals in [("hp", s["hp"]), ("lp", s["lp"]), ("spread", s["spreads"])]:
        if not vals: continue
        vals.sort()
        print(f"  {label}: n={len(vals)} max={max(vals):.6f} p99={vals[len(vals)//100*99]:.6f} "
              f"p95={vals[len(vals)//20*19]:.6f} p50={vals[len(vals)//2]:.6f} "
              f"avg={sum(vals)/len(vals):.6f}")
    # How many windows exceed threshold?
    exceed_hp = sum(1 for v in s["hp"] if v >= th)
    exceed_lp = sum(1 for v in s["lp"] if v >= th)
    print(f"  windows exceeding {int(th/pip)}p threshold: hp={exceed_hp} lp={exceed_lp} "
          f"({(exceed_hp+exceed_lp)/max(len(s['hp'])+len(s['lp']),1)*100:.2f}%)")
    # Backtest expected events per 10min
    bt_daily = {"EURUSD": 48.4, "EURJPY": 17.6}
    exp_per_10min = bt_daily[pair] / 144  # 144 ten-minute periods in a day
    print(f"  Backtest expects {exp_per_10min:.2f} events per 10min")
