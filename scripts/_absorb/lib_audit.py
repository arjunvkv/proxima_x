"""scripts/_absorb/lib_audit.py — sweep the engine's never-validated journal rules.

Every rule in the engine registry gets mechanism-window cells run through the
cost-aware honest battery (measured + typical spreads, $3.5 commission), the
same gate the book legs must pass. Any cell passing escalates to the full
battery (per-side pips, LODO, neighbor robustness).

Rules: thin_market_fade, vol_compress_fade, domestic_hours, day_of_week_usd,
carry_clock (US-hours map), lead_lag, big_move_fade, intraday_momentum_london,
intraday_momentum, fix_reversal (TRUE fix windows per clock-check).
Engine/book/live untouched.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from collections import defaultdict
from proxima_ops.backtest import StrategySpec, run_strategy, metrics, gate
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
SPREAD_MEASURED = {"EURUSD":0.60,"USDJPY":3.50,"GBPUSD":1.30,"AUDUSD":0.90,"EURJPY":5.70,
                   "GBPJPY":7.10,"EURAUD":2.90,"EURNZD":5.10,"GBPAUD":4.90,"GBPNZD":6.30,
                   "GBPCAD":4.10,"AUDNZD":3.50,"USDCAD":1.60,"NZDUSD":1.20,"EURGBP":1.70,
                   "EURCHF":2.60,"USDCHF":1.30,"AUDJPY":4.20}
SPREAD_TYPICAL = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
                  "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
                  "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
                  "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
CARRY_V0 = {"EURUSD":0,"USDJPY":1,"GBPUSD":1,"AUDUSD":1,"EURJPY":1,"GBPJPY":1,
            "EURAUD":-1,"EURNZD":-1,"GBPAUD":-1,"GBPNZD":-1,"GBPCAD":1,"AUDNZD":0,
            "USDCAD":1,"NZDUSD":1,"EURGBP":-1,"EURCHF":1,"USDCHF":1,"AUDJPY":1}

# rule -> list of mechanism-window configs (sessions, lookback, top_n, hold, extra)
GRID = {
    "thin_market_fade":        [([2,3,4,5], 288, 5, 24, {}), ([2,3,4,5], 288, 5, 48, {})],
    "vol_compress_fade":       [([12,13], 96, 5, 12, {}), ([4,5,6], 96, 5, 24, {})],
    "domestic_hours":          [([7,8,9,10,11], 50, 5, 12, {}), ([13,14,15,16,17], 50, 5, 12, {})],
    "day_of_week_usd":         [([0,1], 6, 5, 48, {}), ([0,1,2], 6, 5, 72, {})],
    "carry_clock":             [([14,15,16,17,18,19,20], 6, 8, 72, {"direction_map": CARRY_V0, "weekdays": [2,3,4]})],
    "lead_lag":                [([7,8,9], 6, 3, 12, {}), ([13,14,15], 6, 3, 12, {})],
    "big_move_fade":           [([2,3,4], 1440, 5, 24, {}), ([7,8], 1440, 5, 12, {})],
    "intraday_momentum_london":[([7,8], 6, 3, 96, {}), ([7,8], 6, 3, 60, {})],
    "intraday_momentum":       [([13,14], 6, 3, 48, {}), ([0,1], 6, 3, 24, {})],
    "fix_reversal":            [([17], 6, 3, 12, {}), ([3], 6, 3, 12, {}), ([14], 6, 3, 12, {})],
}

def stats(trades):
    day = defaultdict(float)
    for t in trades:
        day[t["entry_ts"] // 86400] += t.get("net", t.get("pnl_pts", 0.0))
    nets = list(day.values())
    tot = sum(nets)
    pos = sum(v for v in nets if v > 0); neg = -sum(v for v in nets if v < 0)
    pf = pos / neg if neg > 0 else float("inf")
    wr = sum(1 for t in trades if t.get("net", t.get("pnl_pts", 0.0)) > 0) / len(trades) if trades else 0.0
    half = len(nets) // 2
    wf_pos = (sum(nets[:half]) > 0) + (sum(nets[half:]) > 0)
    bysym = defaultdict(float)
    for t in trades:
        bysym[t["symbol"]] += t.get("net", t.get("pnl_pts", 0.0))
    pos_syms = sum(1 for v in bysym.values() if v > 0)
    worst = min(nets) if nets else 0.0
    return {"trades": len(trades), "wr": wr, "pf": pf, "net": tot,
            "exp_lot": tot / max(len(trades), 1) / 0.15, "wf_pos": wf_pos,
            "pos_syms": pos_syms, "nsyms": len(bysym), "worst_day": worst}

bars = build_bars_map(UNIVERSE)
print(f"bars ready: {len(bars)} symbols")
out = []
for rule, cfgs in GRID.items():
    for (sessions, lb, top, hold, extra) in cfgs:
        win = "-".join(str(h) for h in sessions)
        sig = {"rule": rule, "lookback": lb, "pick": "n_worst", "top_n": top,
               "side": "both", "fill_bar": 1}
        sig.update(extra)
        spec = StrategySpec.from_dict({
            "name": f"{rule}_{win}", "universe": UNIVERSE,
            "feed": {"kind": "bar", "timeframe": "M5"},
            "signal": sig,
            "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
            "sessions": sessions, "base_lot": 0.15})
        raw = run_strategy(bars, spec, raw=True)
        usd_t = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5,
                             spread_pips_map=SPREAD_TYPICAL)
        usd_m = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5,
                             spread_pips_map=SPREAD_MEASURED)
        sr, st, sm = stats(raw), stats(usd_t), stats(usd_m)
        g = gate(metrics(usd_t), lot=0.15)
        cell = {"rule": rule, "window": win, "lb": lb, "top": top, "hold": hold,
                "raw": sr, "net_typ": st, "net_meas": sm,
                "pass": bool(st["pf"] > 1.2 and st["net"] > 0 and st["exp_lot"] > 15.0)}
        out.append(cell)
        flag = "  <<< PASS" if cell["pass"] else ""
        print(f"{rule:<22} {win:<14} n={sr['trades']:>4} raw=${sr['net']:>9,.0f} "
              f"typ=${st['net']:>9,.0f} meas=${sm['net']:>9,.0f} "
              f"PF={st['pf']:>5.2f} exp=${st['exp_lot']:>6.1f} wf={st['wf_pos']} "
              f"syms={st['pos_syms']}/{st['nsyms']}{flag}")

os.makedirs("scripts/_absorb/results", exist_ok=True)
with open("scripts/_absorb/results/lib_audit.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote scripts/_absorb/results/lib_audit.json")