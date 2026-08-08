"""Vet the gap-fill survivors (esp. intraday_momentum_short h22) with the full
decisive battery: shuffle-with-costs, walk-forward, window-contrast, tokyo-overlap.
Gap survivors have no lb/top/hold in JSON -> default 50/5/24."""
import sys, os, json, time, random, statistics as st
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.validation import walk_forward, metrics

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
SPREAD = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
          "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
          "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
          "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
WINS = {"h0":[0],"h22":[22],"h23":[23],"0-1":[0,1]}
OTHER = {"h0":[10],"h22":[11],"h23":[12],"0-1":[10,11]}
bars = build_bars_map(UNIVERSE)
surv = json.load(open(os.path.join(os.path.dirname(__file__),
                                   "_sweep_gap_survivors.json")))

def make(rule, w, lb=50, top=5, hold=24, side="both"):
    sig = {"rule": rule, "lookback": lb, "pick": "n_worst", "top_n": top,
           "side": side, "fill_bar": 1}
    return StrategySpec.from_dict({
        "name": rule, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
        "sessions": w, "base_lot": 0.15})

def run(bars_, spec):
    return run_strategy(bars_, spec, volume=0.15, commission_per_lot=3.0,
                        spread_pips_map=SPREAD)

def ident(usd):
    return {(t["entry_ts"] // 86400, t["symbol"], t["side"]) for t in usd}

def shuffle_plot(bars_, spec, iters=12, seed=42):
    rng = random.Random(seed)
    real = run(bars_, spec)
    real_m = sum(t["net"] for t in real) / len(real) if real else 0.0
    means = []
    for _ in range(iters):
        sh = {s: b[:] for s, b in bars_.items()}
        for s in sh:
            rng.shuffle(sh[s])
        usd = run(sh, spec)
        if usd:
            means.append(sum(x["net"] for x in usd) / len(usd))
    real_net = sum(t["net"] for t in real)
    if not means:
        return real_net, 0.0, 0, 0
    sm = sum(means) / len(means)
    sd = st.stdev(means) if len(means) > 1 else 0.0
    z = (real_m - sm) / sd if sd > 0 else 0.0
    below = sum(1 for m_ in means if m_ < real_m)
    return real_net, z, below, len(means)

tokyo_spec = make("session_exhaustion", [0], 6, 3, 12)
tokyo_ids = ident(run(bars, tokyo_spec))
t0 = time.time()
out = []
for s in surv:
    win = WINS[s["window"]]
    rule = s["rule"].replace("_short", "")
    side = "short" if s["rule"].endswith("_short") else "both"
    spec = make(rule, win, side=side)
    real_net, z, below, _ = shuffle_plot(bars, spec)
    usd = run(bars, spec)
    wf = walk_forward([t for t in usd], train_size=250, test_size=90)
    wf_pos = wf["positive_share"]
    other_win = OTHER[s["window"]]
    usd_o = run(bars, make(rule, other_win, side=side))
    other_net = sum(t["net"] for t in usd_o)
    ids, ids_o = ident(usd), ident(usd_o)
    inter = len(ids & ids_o)
    union = len(ids | ids_o)
    other_ov = inter / union if union else 0.0
    contrast_ok = other_net <= 0 or (other_net < 1.5 * real_net and other_ov < 0.6)
    ids = ident(usd)
    inter = len(ids & tokyo_ids)
    union = len(ids | tokyo_ids)
    ov = inter / union if union else 0.0
    m = metrics(usd)
    pass_ = (z >= 2.0 and below >= 10 and wf_pos >= 0.55 and contrast_ok
             and ov < 0.40 and real_net > 0)
    out.append({"rule": s["rule"], "window": s["window"], "net": real_net,
                "pf": m["profit_factor"], "z": z, "below": below,
                "wf_pos": wf_pos, "other_net": other_net, "other_ov": other_ov,
                "tokyo_ov": ov, "pass": pass_})
    print(f"{s['rule']:<26}{s['window']:<6} net ${real_net:>8,.0f} "
          f"PF {m['profit_factor']:5.2f} z {z:>6.1f} {below}/12 "
          f"wf {wf_pos:.2f} other ${other_net:>8,.0f}(ov {other_ov:.2f}) "
          f"ovTok {ov:.2f}  {'PASS' if pass_ else 'fail'}")
print(f"\n--- {time.time()-t0:.0f}s  PASS: {sum(1 for o in out if o['pass'])}/{len(out)}")
for o in out:
    if o["pass"]:
        print(f"  ** {o['rule']} {o['window']} net ${o['net']:,.0f} PF {o['pf']:.2f} "
              f"z {o['z']:.1f} wf {o['wf_pos']:.2f} ovTok {o['tokyo_ov']:.2f}")