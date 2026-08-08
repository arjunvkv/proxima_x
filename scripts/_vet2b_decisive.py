"""Stage 2b: decisive vetting of 1b survivors (dead-window candidates).

Gates (all at REAL FTMO cost):
  1. purple 12-shuffle WITH costs: z >= 2, >=10/12 below
  2. walk-forward stability: >=60% positive OOS windows
  3. WINDOW CONTRAST: same rule/params at a DIFFERENT 3h window must not earn
     uniformly (net_other < 1.5x net OR net_other <= 0) AND overlap < 0.60 —
     a real session edge concentrates at its own hour
  4. overlap vs tokyo leg (existing book): < 0.4 Jaccard (additive, not mask)
  PASS = all four.
"""
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
OTHER = {"2-4":[7,8,9],"7-9":[9,10,11],"9-11":[2,3,4],"12-13":[14,15,16],
         "14-16":[17,18,19],"17-19":[2,3,4]}
WINS = {"2-4":[2,3,4],"7-9":[7,8,9],"9-11":[9,10,11],"12-13":[12,13],
        "14-16":[14,15,16],"17-19":[17,18,19]}

bars = build_bars_map(UNIVERSE)
surv = json.load(open(os.path.join(os.path.dirname(__file__),
                                   "_sweep1b_survivors.json")))

def make(rule, w, lb, top, hold, ph, side="both"):
    sig = {"rule": rule, "lookback": lb, "pick": "n_worst", "top_n": top,
           "side": side, "fill_bar": 1, "per_hour": ph}
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

t0 = time.time()
out = []
for s in surv:
    win = WINS[s["window"]]  # exact hour list; "17-19".split gives [17,19] (drop 18) — fixed
    spec = make(s["rule"], win, s["lb"], s["top"], s["hold"], s["per_hour"])
    # 1) shuffle w/ costs
    real_net, z, below, niters = shuffle_plot(bars, spec)
    # 2) walk-forward on real trades
    usd = run(bars, spec)
    wf = walk_forward([t for t in usd], train_size=250, test_size=90)
    wf_pos = wf["positive_share"]
    # 3) window contrast
    other_win = OTHER.get(s["window"])
    other_net, other_ov = 0.0, 0.0
    if other_win:
        spec_o = make(s["rule"], other_win, s["lb"], s["top"], s["hold"], s["per_hour"])
        usd_o = run(bars, spec_o)
        other_net = sum(t["net"] for t in usd_o)
        ids, ids_o = ident(usd), ident(usd_o)
        inter = len(ids & ids_o)
        union = len(ids | ids_o)
        other_ov = inter / union if union else 0.0
    contrast_ok = other_net <= 0 or (other_net < 1.5 * real_net and other_ov < 0.6)
    # 4) overlap vs tokyo (existing flagship)
    tokyo_spec = make("session_exhaustion", [0], 6, 3, 12, False)
    tokyo_ids = ident(run(bars, tokyo_spec))
    ids = ident(usd)
    inter = len(ids & tokyo_ids)
    union = len(ids | tokyo_ids)
    ov = inter / union if union else 0.0
    m = metrics(usd)
    pass_ = (z >= 2.0 and below >= 10 and wf_pos >= 0.55 and contrast_ok
             and ov < 0.40 and real_net > 0)
    out.append({"rule": s["rule"], "window": s["window"], "lb": s["lb"],
                "top": s["top"], "hold": s["hold"], "ph": s["per_hour"],
                "net": real_net, "pf": m["profit_factor"], "z": z,
                "below": below, "wf_pos": wf_pos, "other_net": other_net,
                "other_ov": other_ov, "tokyo_ov": ov, "pass": pass_})
    print(f"{s['rule']:<20}{s['window']:<6} lb{s['lb']:<4} top{s['top']} "
          f"h{s['hold']:<2} ph={int(s['per_hour'])}  net ${real_net:>8,.0f} "
          f"PF {m['profit_factor']:5.2f} z {z:>6.1f} {below}/12 "
          f"wf {wf_pos:.2f} other ${other_net:>8,.0f}(ov {other_ov:.2f}) "
          f"ovTok {ov:.2f}  {'PASS' if pass_ else 'fail'}")

print(f"\n--- {time.time()-t0:.0f}s  PASS: {sum(1 for o in out if o['pass'])}/{len(out)}")
json.dump(out, open(os.path.join(os.path.dirname(__file__),
                                 "_vet2b_results.json"), "w"), indent=2)
for o in out:
    if o["pass"]:
        print(f"  ** {o['rule']} {o['window']} lb{o['lb']} top{o['top']} "
              f"h{o['hold']} ph={o['ph']} net ${o['net']:,.0f} PF {o['pf']:.2f} "
              f"z {o['z']:.1f} ({o['below']}/12) wf {o['wf_pos']:.2f} "
              f"ovTok {o['tokyo_ov']:.2f}")