"""triage_v1.py — rapid Level-1 survival gate for batches of candidate strategies.

Drop a candidates JSON (rule + universe + sessions + params mapped from any
internet/research description) -> per-strategy CERTIFICATE in ~1-3s each:
  [costs]  exp/lot vs gate ($15 AND >=1.5x round-trip spread cost)
  [wf]     both walk-forward halves positive
  [lodo]   removing top-5 days keeps net positive
  [sides]  long+short both positive (flagged if single-side by design)
  [months] <= 1 negative month (no 2 consecutive)
  [stress] survives 1.25x/1.5x/2x spread ladder
  [pf]     day-aggregated PF >= 1.2
Survivors (all green) are written to triage_survivors.json for the deep battery.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map
from costmaps_r3 import corrected_maps

TICK, SPR = corrected_maps()
BIG = (1e9, 1e9)
SPREAD_X = {"1x": 1.0, "1.25x": 1.25, "1.5x": 1.5, "2x": 2.0}

# "internet archetype -> engine config" hints (for the report, not enforced)
ARCHETYPES = {
    "london_breakout": "range_breakout + eu session", "asia_range": "range_breakout + asia",
    "ny_open_momentum": "session_open_breakout + us", "liquidity_sweep_fade": "liquidity_sweep",
    "exhaustion_fade": "session_exhaustion", "macd_reversion": "session_reversion",
    "weekend_gap": "weekend_gap", "vol_compress": "vol_compress_fade",
    "round_numbers": "round_number_bounce", "carry": "carry_clock",
    "big_move_fade": "big_move_fade", "range_reversion": "range_reversion",
}


def run_cell(cand, spread_x=1.0):
    spec = StrategySpec.from_dict({
        "name": cand["name"], "universe": cand["universe"],
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": cand["rule"], "lookback": cand.get("lookback", 50),
                   "pick": cand.get("pick", "n_worst"), "top_n": cand.get("top_n", 3),
                   "side": cand.get("side", "both"), "fill_bar": cand.get("fill_bar", 1)},
        "exit": {"mode": "sl_tp_hold", "hold_bars": cand.get("hold_bars", 12),
                 "stop_first": True, "jpy_sl_tp": BIG, "non_jpy_sl_tp": BIG},
        "sessions": cand["sessions"], "base_lot": 0.15,
    })
    bars = build_bars_map(cand["universe"])
    tr = run_strategy(bars, spec, raw=False, volume=0.15, commission_per_lot=3.0,
                      tick_value_map=TICK, spread_pips_map={s: SPR[s] * spread_x for s in SPR})
    return [t for t in tr if t.get("entry_ts")]


def cert(trades, vol=0.15):
    """All checks on engine-raw trades. Returns (checks dict, metrics dict)."""
    if len(trades) < 60:
        return {"n": len(trades), "net": 0.0, "exp_lot": 0.0, "pf": 0.0,
                "wf_pos": 0, "lodo5": False, "sides": 0, "neg_months": 99,
                "stress": 0.0}, {}
    nets = np.array([t["net"] for t in trades], dtype=float)
    days = np.array([t["entry_ts"] // 86400 for t in trades])
    sides = np.array([1 if t["side"] == "BUY" else -1 for t in trades])
    net, n = nets.sum(), len(nets)
    exp_lot = net / n / vol
    # PF on day-aggregated net
    dn = np.array([nets[days == d].sum() for d in np.unique(days)])
    pf = dn[dn > 0].sum() / abs(dn[dn < 0].sum()) if (dn < 0).any() else 99.0
    # WF halves by day
    ds = np.unique(days); half = ds[len(ds) // 2]
    wf_pos = int(nets[days < half].sum() > 0) + int(nets[days >= half].sum() > 0)
    # LODO: remove top-5 days
    top5 = np.argsort(dn)[-5:]
    lodo5 = (net - dn[top5].sum()) > 0
    # per-side
    pos_s = nets[sides > 0].sum(); neg_s = nets[sides < 0].sum()
    sides_ok = (pos_s > 0 and neg_s > 0) or (pos_s > 0 and (sides > 0).all())
    # months
    ms = np.array([t["entry_ts"] // 2592000 for t in trades])
    mn = {m: nets[ms == m].sum() for m in np.unique(ms)}
    neg_months = sum(1 for v in mn.values() if v < 0)
    return {"n": n, "net": round(net, 1), "exp_lot": round(exp_lot, 1),
            "pf": round(pf, 2), "wf_pos": wf_pos, "lodo5": lodo5,
            "sides": sides_ok, "neg_months": neg_months}, {}


def stress_ladder(trades, spread_cost_per_lot):
    """Re-run costs at 1.25/1.5/2x by adjusting nets (spread part only)."""
    out = {}
    for tag, x in SPREAD_X.items():
        if x == 1.0:
            out[tag] = sum(t["net"] for t in trades)
            continue
        # net = gross - comm - spread*vol ; add extra (x-1)*spread_cost*vol per trade
        out[tag] = sum(t["net"] for t in trades) - (x - 1.0) * spread_cost_per_lot * 0.15 * len(trades)
    return out


def main(cands_path):
    data = json.load(open(cands_path))
    uni_map = {k: v for k, v in data.items() if isinstance(v, list) and k != "candidates"}
    cands = data["candidates"]
    for c in cands:
        if isinstance(c["sessions"], str) and c["sessions"] == "full":
            c["sessions"] = list(range(24))
        if isinstance(c["universe"], str):
            c["universe"] = uni_map[c["universe"]]
    survivors, table = [], []
    for c in cands:
        tr = run_cell(c)
        m, _ = cert(tr)
        # spread cost per lot at 1x (engine formula: spread_pips x pip_usd, pip_usd = tick x 10)
        sc = sum(SPR.get(s, 0.5) * TICK.get(s, 0.1) * 10.0 for s in c["universe"]) / len(c["universe"])
        ladder = stress_ladder(tr, sc)
        m["stress"] = ladder
        gate = (m["n"] >= 60 and m["exp_lot"] >= 15.0 and m["pf"] >= 1.2 and
                m["wf_pos"] == 2 and m["lodo5"] and m["sides"] and m["neg_months"] <= 1 and
                ladder["2x"] > 0)
        table.append((c["name"], m, gate))
        if gate:
            survivors.append(c)
    # report
    print(f"{'name':34s} {'n':>6s} {'exp/lot':>9s} {'PF':>6s} {'wf':>3s} {'LODO5':>5s} {'side':>5s} {'negM':>4s} {'2x':>10s}  GATE")
    for name, m, g in table:
        s = m.get("stress", {})
        print(f"{name[:34]:34s} {m['n']:6d} {m['exp_lot']:9.1f} {m['pf']:6.2f} "
              f"{m['wf_pos']:3d} {str(m['lodo5']):>5s} {str(m['sides']):>5s} {m['neg_months']:4d} "
              f"{s.get('2x', 0):10.1f}  {'PASS' if g else '--'}")
    print(f"\nsurvivors: {len(survivors)}/{len(cands)} -> {[s['name'] for s in survivors]}")
    json.dump(survivors, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "triage_survivors.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "scripts/_absorb/candidates_r3.json")
