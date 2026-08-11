"""scripts/_absorb/new_assets_sweep.py — all engine rules x new-asset universes.

First-pass cost-aware sweep over the freshly acquired cache: indices, gold/
silver, crypto, oil, DXY, and a cross-asset mix. Existing FX legs NOT re-tested.
Costs: broker tick values (probe 2026-08-11) + measured live spreads (same
probe, market open) + $3/lot commission. Gate (USD/lot, keeps FX-leg bar):
  PF > 1.2, net > 0, exp_usd/lot > $15, wf_pos == 2, pos_syms >= 2.
Engine/book/live untouched. Output sorted survivor list.
"""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map
from costmaps_r3 import corrected_maps

TICK_VALUES, SPREADS = corrected_maps()
BIG = (1e9, 1e9)

UNIS = {
    "IND":  ["US30.cash","US500.cash","GER40.cash","UK100.cash","JP225.cash","HK50.cash"],
    "GOLD": ["XAUUSD","XAGUSD"],
    "CRYPTO": ["BTCUSD","ETHUSD"],
    "OIL":  ["USOIL.cash","UKOIL.cash"],
    "DXY":  ["DXY.cash"],
    "MIX":  ["XAUUSD","US500.cash","US30.cash","USOIL.cash","BTCUSD","DXY.cash"],
}
WINS = {
    "full": list(range(24)),
    "eu":   [7,8,9,10,11,12],
    "us":   [13,14,15,16,17,18,19,20,21],
    "asia": [0,1,2,3,4,5,6],
    "open_eu": [7,8], "open_us": [13,14], "close_us": [19,20,21],
}
# per-universe window subsets (crypto 24/7: session windows arbitrary, keep minimal)
UNI_WINS = {
    "IND":    ["full", "us", "open_us", "close_us"],
    "GOLD":   ["full", "us", "eu", "asia"],
    "CRYPTO": ["full", "us", "asia"],
    "OIL":    ["full", "us", "eu"],
    "DXY":    ["full", "us", "eu"],
    "MIX":    ["full", "us"],
}
RULES = ["session_exhaustion","session_reversion","range_reversion","range_breakout",
         "liquidity_sweep","session_open_breakout","intraday_momentum","fix_reversal",
         "round_barrier_fade","domestic_hours","weekend_gap","cross_momentum",
         "round_number_bounce","big_move_fade","intraday_momentum_london",
         "day_of_week_usd","vol_compress_fade","lead_lag","thin_market_fade"]

def stats(trades):
    if not trades:
        return None
    day = defaultdict(float)
    for t in trades:
        day[t["entry_ts"] // 86400] += t["net"]
    nets = list(day.values())
    tot = sum(nets)
    pos = sum(v for v in nets if v > 0); neg = -sum(v for v in nets if v < 0)
    pf = pos / neg if neg > 0 else float("inf")
    half = len(nets) // 2
    wf = int(sum(nets[:half]) > 0) + int(sum(nets[half:]) > 0)
    bysym = defaultdict(float)
    for t in trades:
        bysym[t["symbol"]] += t["net"]
    return {"n": len(trades), "net": tot, "pf": pf, "wr": sum(1 for t in trades if t["net"] > 0)/len(trades),
            "exp_lot": tot / len(trades) / 0.15, "wf": wf, "pos_syms": sum(1 for v in bysym.values() if v > 0),
            "worst_day": min(nets)}

bars = build_bars_map(sorted({s for u in UNIS.values() for s in u}))
print("bars:", {k: len(v) for k, v in list(bars.items())[:3]}, "... total", sum(len(v) for v in bars.values()))
print("NOTE: engine ignores spec.universe (caller-enforced); per-cell bars loaded below")
out = []
cells_total = sum(len(UNI_WINS[u]) for u in UNIS) * len(RULES)
ci = 0
for uni_name, uni in UNIS.items():
    ubars = {s: bars[s] for s in uni}   # caller-enforced universe (engine has no filter)
    for rule in RULES:
        for win_name in UNI_WINS[uni_name]:
            if rule == "vol_compress_fade" and win_name != "full":
                continue
            sess = WINS[win_name]
            ci += 1
            if ci % 10 == 0 or ci <= 5:
                print(f"[{ci}/{cells_total}] {rule} {uni_name} {win_name}", flush=True)
            spec = StrategySpec.from_dict({
                "name": f"{rule}_{uni_name}_{win_name}", "universe": uni,
                "feed": {"kind": "bar", "timeframe": "M5"},
                "signal": {"rule": rule, "lookback": 50, "pick": "n_worst", "top_n": 3,
                           "side": "both", "fill_bar": 1},
                "exit": {"mode": "sl_tp_hold", "hold_bars": 12, "stop_first": True,
                         "jpy_sl_tp": BIG, "non_jpy_sl_tp": BIG},
                "sessions": sess, "base_lot": 0.15})
            try:
                t = run_strategy(ubars, spec, volume=0.15, commission_per_lot=3.0,
                                 tick_value_map=TICK_VALUES, spread_pips_map=SPREADS)
            except ValueError as e:
                print("SKIP", rule, uni_name, win_name, ":", str(e)[:80])
                continue
            m = stats(t)
            if m is None or m["n"] < 30:
                continue
            g = m["pf"] > 1.2 and m["net"] > 0 and m["exp_lot"] > 15.0 and m["wf"] == 2 and m["pos_syms"] >= 2
            out.append({"rule": rule, "uni": uni_name, "win": win_name, **{k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}, "gate": g})
            if g:
                print(f"  <<< PASS {rule} {uni_name} {win_name} n={m['n']} net=${m['net']:,.0f} PF={m['pf']:.2f} exp=${m['exp_lot']:.1f} wf={m['wf']} syms={m['pos_syms']}")

out.sort(key=lambda c: -c["net"])
with open("scripts/_absorb/results/new_assets_sweep.json", "w") as f:
    json.dump(out, f, indent=1)
npass = sum(1 for c in out if c["gate"])
print(f"total cells: {len(out)}  PASS: {npass}")
for c in [c for c in out if c["gate"]][:20]:
    print(f"  {c['rule']:<24} {c['uni']:<7} {c['win']:<8} n={c['n']:>4} net=${c['net']:>8,.0f} PF={c['pf']:.2f} exp=${c['exp_lot']:>6.1f} wf={c['wf']} syms={c['pos_syms']}")