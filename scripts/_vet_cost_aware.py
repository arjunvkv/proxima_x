"""Stage 2: vet each BOTH-spread survivor with the cost-aware gate.

Checks (all WITH real spread + commission):
  1. 12-shuffle day-order test: z-score of real net vs shuffled distribution,
     require z >= 2 and >= 10/12 shuffled runs below real.
  2. walk-forward: split the real (cost-charged) trades into thirds; require
     >=60% of thirds net-positive (stable, not one regime).
  3. neighborhood: shift the hour window (-1 / +1) — at least 1 neighbor also
     nets > 0 at costs (plateau, not a lone grid point).
Verdict PASS only if all three hold. Reads stage-1 survivors JSON.
"""
import sys, os, json, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
from proxima_ops.backtest import StrategySpec, run_strategy, metrics
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
SPREAD = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
          "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
          "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
          "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
NORM = {"EURUSD":"EURUSD","USDJPY":"USDJPY","GBPUSD":"GBPUSD","AUDUSD":"AUDUSD",
        "EURJPY":"EURJPY","GBPJPY":"GBPJPY","EURAUD":"EURAUD","EURNZD":"EURNZD",
        "GBPAUD":"GBPAUD","GBPNZD":"GBPNZD","GBPCAD":"GBPCAD","AUDNZD":"AUDNZD",
        "USDCAD":"USDCAD","NZDUSD":"NZDUSD","EURGBP":"EURGBP","EURCHF":"EURCHF",
        "USDCHF":"USDCHF","AUDJPY":"AUDJPY"}
HOURS = list(range(24))

bars = build_bars_map(UNIVERSE)

def shuffle_days(bm, seed):
    rnd = random.Random(seed)
    out = {}
    for sym, bl in bm.items():
        days = defaultdict(list)
        for b in bl:
            days[b["ts"] // 86400].append(b)
        ks = list(days.keys())
        rnd.shuffle(ks)
        out[sym] = [b for k in ks for b in days[k]]
    return out

def make_spec(rule, sessions, side="both", lb=50, top=5, hold=24):
    sig = {"rule": rule, "lookback": lb, "pick": "n_worst", "top_n": top,
           "side": side, "fill_bar": 1}
    return StrategySpec.from_dict({
        "name": rule, "universe": UNIVERSE,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": sig,
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": True},
        "sessions": sessions, "base_lot": 0.15})

def net_of(usd):
    return sum(t["net"] for t in usd)

def parse_window(key):
    """'h7'->[7]; '7-9'->[7,8,9]."""
    if key.startswith("h"):
        return [int(key[1:])]
    if "-" in key:
        a, b = key.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(key)]

def key_of(sessions):
    if len(sessions) == 1:
        return f"h{sessions[0]}"
    return f"{sessions[0]}-{sessions[-1]}"

SURV = json.load(open(os.path.join(os.path.dirname(__file__),
                                   "_sweep_cost_survivors.json")))
verdicts = []
t0 = time.time()
print(f"{'rule':<24}{'win':<6}{'side':<6}{'net$':>9}{'z':>6}{'sh<':>5}{'wf%':>6}"
      f"{'nbr':>4}  VERDICT")
results = []
for s in SURV:
    rule = s["rule"].replace("_short", "")
    side = "short" if "_short" in s["rule"] else "both"
    sess = parse_window(s["window"])
    spec = make_spec(rule, sess, side)
    real = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.0,
                        spread_pips_map=SPREAD)
    if len(real) < 12:
        continue
    real_net = net_of(real)
    m = metrics(real)
    # 1. shuffle-with-costs
    shuf = []
    for i in range(12):
        usd = run_strategy(shuffle_days(bars, 3000 + i), spec,
                           volume=0.15, commission_per_lot=3.0,
                           spread_pips_map=SPREAD)
        shuf.append(net_of(usd) if usd else 0.0)
    msh = sum(shuf) / len(shuf)
    sd = (sum((x - msh) ** 2 for x in shuf) / len(shuf)) ** 0.5 or 1e-9
    z = (real_net - msh) / sd
    below = sum(1 for x in shuf if x < real_net)
    # 2. walk-forward thirds
    n = len(real)
    third = n // 3
    segs = [net_of(real[i:i + third]) for i in range(0, n - third + 1, third)]
    wf_pos = sum(1 for x in segs if x > 0) / len(segs) if segs else 0.0
    # 3. hour-window neighbors (shift -1 / +1)
    nbr_ok = 0
    for dh in (-1, 1):
        ws = [(h + dh) % 24 for h in sess]
        sp2 = make_spec(rule, ws, side)
        u = run_strategy(bars, sp2, volume=0.15, commission_per_lot=3.0,
                         spread_pips_map=SPREAD)
        if len(u) >= 12 and net_of(u) > 0:
            nbr_ok += 1
    net_ok = real_net > 0 and m["profit_factor"] > 1.2
    pass_ = net_ok and z >= 2.0 and below >= 10 and wf_pos >= 0.6 and nbr_ok >= 1
    verdicts.append({
        "rule": rule, "side": side, "win": s["window"], "net": round(real_net, 0),
        "pf": m["profit_factor"], "z": round(z, 2), "below": below,
        "wf_share": round(wf_pos, 2), "nbr": nbr_ok, "pass": pass_})
    print(f"{rule:<24}{s['window']:<6}{side:<5} {real_net:>9,.0f}{z:>6.1f}{below:>4}/12"
          f"{m['profit_factor']:>6.2f}{nbr_ok:>4}  "
          f"{'PASS!!!' if pass_ else 'fail'}")

json.dump(verdicts, open(os.path.join(os.path.dirname(__file__),
                                      "_sweep_cost_vet.json"), "w"), indent=2)
print(f"\n--- {time.time()-t0:.0f}s; PASS: "
      f"{sum(1 for v in verdicts if v['pass'])}/{len(verdicts)}")
for v in verdicts:
    if v["pass"]:
        print(f"  FINAL: {v['rule']} {v['win']} ({v['side']}) net ${v['net']:.0f} "
              f"PF {v['pf']} z {v['z']} wf {v['wf_share']} nbr {v['nbr']}")