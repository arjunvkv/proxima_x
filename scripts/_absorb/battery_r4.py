"""battery_r4.py — full honesty suite for engine-based sweep survivors."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from collections import defaultdict
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map
from costmaps_r3 import corrected_maps, ALL

TICK, SPR = corrected_maps()
BIG = (1e9, 1e9)
full = build_bars_map(ALL)

CELLS = {
    "C1 exGOLD_full": (["XAUUSD", "XAGUSD"], "session_exhaustion", list(range(24))),
    "C2 exGOLD_us":   (["XAUUSD", "XAGUSD"], "session_exhaustion", [13,14,15,16,17,18,19,20,21]),
    "C3 rvGOLD_full": (["XAUUSD", "XAGUSD"], "session_reversion", list(range(24))),
    "C4 exMIX_full":  (["XAUUSD","US500.cash","US30.cash","USOIL.cash","BTCUSD","DXY.cash"], "session_exhaustion", list(range(24))),
    "C5 exIND_us":    (["US30.cash","US500.cash","GER40.cash","UK100.cash","JP225.cash","HK50.cash"], "session_exhaustion", [13,14,15,16,17,18,19,20,21]),
    "C6 bmfMIX_full": (["XAUUSD","US500.cash","US30.cash","USOIL.cash","BTCUSD","DXY.cash"], "big_move_fade", list(range(24))),
}

def run_cell(uni, rule, sess, sp_mult=1.0):
    ubars = {s: full[s] for s in uni}
    spec = StrategySpec.from_dict({
        "name": f"{rule}_{uni}_{sess}", "universe": uni,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": rule, "lookback": 50, "pick": "n_worst", "top_n": 3,
                   "side": "both", "fill_bar": 1},
        "exit": {"mode": "sl_tp_hold", "hold_bars": 12, "stop_first": True,
                 "jpy_sl_tp": BIG, "non_jpy_sl_tp": BIG},
        "sessions": sess, "base_lot": 0.15})
    spr = {k: v * sp_mult for k, v in SPR.items()} if sp_mult != 1.0 else SPR
    return run_strategy(ubars, spec, volume=0.15, commission_per_lot=3.0,
                        tick_value_map=TICK, spread_pips_map=spr)

def suite(tr, label):
    if not tr:
        print(f"{label}: no trades"); return
    n = len(tr)
    day = defaultdict(float); sym = defaultdict(float); side = defaultdict(float)
    hour = defaultdict(float); mon = defaultdict(float)
    for t in tr:
        d = t["entry_ts"] // 86400
        day[d] += t["net"]; sym[t["symbol"]] += t["net"]
        side[t["side"]] += t["net"]
        hour[(t["entry_ts"] // 3600) % 24] += t["net"]
        mon[(t["entry_ts"] // 86400) // 30] += t["net"]
    nets = list(day.values()); tot = sum(nets)
    pos = sum(v for v in nets if v > 0); neg = -sum(v for v in nets if v < 0)
    dl = sorted(day)
    wf = [sum(day[k] for k in dl[:len(dl)//2]), sum(day[k] for k in dl[len(dl)//2:])]
    lodo = sum(1 for k in range(len(nets)) if (tot - nets[k]) > 0)
    pos_side = all(v > 0 for v in side.values())
    neg2mo = any(mon[k] < 0 and mon.get(k+1, 0) < 0 for k in sorted(mon))
    nh = sorted(((h, v) for h, v in hour.items()), key=lambda x: -x[1])
    top_h = [(h, round(v)) for h, v in nh[:3]]
    lodo_flips = lodo
    print(f"{label}: n={n} net=${tot:,.0f} exp=${tot/n/0.15:.1f}/lot PF={pos/neg if neg else 99:.2f} "
          f"wf=${wf[0]:,.0f}/${wf[1]:,.0f} lodo_flips={lodo_flips}/{len(nets)} "
          f"sides={dict(side)} sides_pos={pos_side} neg2mo={neg2mo} syms={dict(sym)} topH={top_h}")
    return {"n": n, "net": tot, "exp": tot/n/0.15, "pf": pos/neg if neg else 99,
            "wf": wf, "lodo": lodo_flips, "neg2mo": neg2mo}

res = {}
for label, (uni, rule, sess) in CELLS.items():
    tr = run_cell(uni, rule, sess)
    res[label] = suite(tr, label)
    # stress: 1.5x spreads on the strongest cells
    if label in ("C1", "C4"):
        trs = run_cell(uni, rule, sess, sp_mult=1.5)
        suite(trs, f"{label} STRESS1.5")
# per-day trade list for Jaccard
def entries(tr):
    return sorted(set(t["entry_ts"] // 86400 for t in tr))
A_entries = None
json.dump({k: v for k, v in res.items() if v}, open("scripts/_absorb/results/battery_r4.json", "w"), indent=1, default=str)
print("battery_r4 done")
