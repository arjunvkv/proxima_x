"""batt_crypto.py — deep battery for CONTROL_big_move_btc (big_move_fade CRYPTO)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map
from costmaps_r3 import corrected_maps

TICK, SPR = corrected_maps()
BIG = (1e9, 1e9)
UNI = ["BTCUSD", "ETHUSD"]
BARS = build_bars_map(UNI)

def run(rule, sessions, lookback=48, hold=12, top_n=3, spread_x=1.0):
    spec = StrategySpec.from_dict({
        "name": "x", "universe": UNI,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": rule, "lookback": lookback, "pick": "n_worst", "top_n": top_n,
                   "side": "both", "fill_bar": 1},
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": False,
                 "jpy_sl_tp": BIG, "non_jpy_sl_tp": BIG},
        "sessions": sessions, "base_lot": 0.15})
    return run_strategy(BARS, spec, raw=False, volume=0.15, commission_per_lot=3.0,
                        tick_value_map=TICK, spread_pips_map={s: SPR[s] * spread_x for s in SPR})

def stats(tr, label, print_rows=True):
    if not tr:
        print(f"{label}: n=0"); return None
    nets = np.array([t["net"] for t in tr]); days = np.array([t["entry_ts"] // 86400 for t in tr])
    n = len(nets); net = nets.sum(); exp = net / n / 0.15
    daynet = np.array([nets[days == d].sum() for d in np.unique(days)])
    pos = daynet[daynet > 0].sum(); neg = -daynet[daynet <= 0].sum()
    pf = pos / neg if neg > 0 else 99.0
    lodo = int((nets.sum() - nets).__gt__(0).sum())
    h1, h2 = nets[days < np.median(np.unique(days))].sum(), nets[days >= np.median(np.unique(days))].sum()
    long = nets[[t["side"] == "BUY" for t in tr]].sum(); short = nets[[t["side"] == "SELL" for t in tr]].sum()
    if print_rows:
        print(f"{label}: n={n} net=${net:,.0f} exp=${exp:.1f}/lot PF={pf:.2f} wf=[{h1/1000:.0f}k,{h2/1000:.0f}k] "
              f"LODO={lodo}/{n} long=${long:,.0f} short=${short:,.0f}")
    return {"n": n, "net": net, "exp": exp, "pf": pf, "wf": [h1, h2], "lodo": lodo,
            "long": long, "short": short, "days": daynet, "nets": nets,
            "side": [t["side"] for t in tr], "sym": [t["symbol"] for t in tr], "ts": [t["entry_ts"] for t in tr]}

def sc_for(uni):
    return sum(SPR.get(s, 0.5) * TICK.get(s, 0.1) * 10.0 for s in uni) / len(uni)

def ladder(tr, sc):
    net = sum(t["net"] for t in tr); n = len(tr)
    return [(x, net - (x - 1.0) * sc * 0.15 * n) for x in (1.25, 1.5, 2.0)]

def ftmo_sim(r, limit=5000.0, dd_limit=10000.0):
    d = r["days"]; worst = -d.min()
    v = max(0.15, 0.15 * limit / (worst * 2.0))
    eq = np.cumsum(d * v / 0.15)
    mdd = (np.maximum.accumulate(eq) - eq).max()
    sgn = (d < 0).astype(int)
    streak = 0; max_streak = 0
    for x in sgn:
        streak = streak + 1 if x else 0
        max_streak = max(max_streak, streak)
    print(f"   FTMO: size={v:.3f} lots worst2x=${worst*2*v/0.15:,.0f} (limit ${limit:,.0f}) "
          f"maxDD=${mdd:,.0f} (${dd_limit:,.0f}) maxLossStreak={max_streak}d "
          f"monthly~${r['net']*v/0.15/ (len(np.unique(r['ts']))//30+1):,.0f}")

print("=== DEEP BATTERY: big_move_fade CRYPTO (BTC+ETH, sessions full, lb48, hold12) ===")
r = run("big_move_fade", list(range(24)))
s = stats(r, "BASE")
print("   ladder:", [f"{x}x=${v:,.0f}" for x, v in ladder(r, sc_for(UNI))])
ftmo_sim(s)
print("   PLATEAU:")
ok = True
for lb in (24, 48, 96):
    for hd in (6, 12, 18):
        rr = run("big_move_fade", list(range(24)), lookback=lb, hold=hd)
        ss = stats(rr, f"   lb{lb}/h{hd}", print_rows=False)
        if ss and ss["net"] > 0:
            print(f"   lb{lb}/h{hd}: +${ss['net']/1000:.1f}k PF={ss['pf']:.2f} ✓")
        else:
            ok = False
            print(f"   lb{lb}/h{hd}: +${ss['net']/1000:.1f}k PF={ss['pf']:.2f} ✗")
print("   PLATEAU:", "PASS" if ok else "FAIL")
from collections import defaultdict
bys = defaultdict(float)
for t in r:
    bys[t["symbol"]] += t["net"]
print("   per-symbol:", {k: round(v) for k, v in bys.items()})