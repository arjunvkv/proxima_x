"""battery_r5.py — deep battery for triage survivors.

Suite per candidate: full LODO, per-side/symbol/hour, monthly, stress ladder
(1.25/1.5/2x), plateau grid (lookback x hold), regime halves, FTMO $100k
daily-loss-limit + max-DD simulation, Jaccard vs the live book.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map
from costmaps_r3 import corrected_maps

TICK, SPR = corrected_maps()
BIG = (1e9, 1e9)
FX22 = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY", "GBPJPY", "EURAUD", "EURGBP",
        "USDCAD", "USDCHF", "NZDUSD", "AUDJPY", "CADJPY", "AUDCAD", "AUDCHF", "AUDNZD",
        "EURCAD", "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD", "EURCHF"]
BARS = build_bars_map(FX22)

def run(rule, sessions, lookback=48, hold=12, top_n=3, spread_x=1.0):
    spec = StrategySpec.from_dict({
        "name": "x", "universe": FX22,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": rule, "lookback": lookback, "pick": "n_worst",
                   "top_n": top_n, "side": "both", "fill_bar": 1},
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": False,
                 "jpy_sl_tp": BIG, "non_jpy_sl_tp": BIG},
        "sessions": sessions, "base_lot": 0.15})
    tr = run_strategy(BARS, spec, raw=False, volume=0.15, commission_per_lot=3.0,
                      tick_value_map=TICK, spread_pips_map={s: SPR[s] * spread_x for s in SPR})
    return tr

def stats(tr, label, print_rows=True):
    if not tr:
        print(f"{label}: n=0"); return None
    nets = np.array([t["net"] for t in tr]); days = np.array([t["entry_ts"] // 86400 for t in tr])
    n = len(nets); net = nets.sum(); exp = net / n / 0.15
    daynet = np.array([nets[days == d].sum() for d in np.unique(days)])
    pos = daynet[daynet > 0].sum(); neg = -daynet[daynet <= 0].sum()
    pf = pos / neg if neg > 0 else 99.0
    sgn = np.sign(nets); lodo = int((nets.sum() - nets).__gt__(0).sum())
    h1, h2 = nets[days < np.median(np.unique(days))].sum(), nets[days >= np.median(np.unique(days))].sum()
    long = nets[[t["side"] == "BUY" for t in tr]].sum(); short = nets[[t["side"] == "SELL" for t in tr]].sum()
    mo = {}
    for t in tr:
        m = t["entry_ts"] // 2592000; mo.setdefault(m, 0.0); mo[m] += t["net"]
    mo_neg = sorted([v for v in mo.values() if v < 0]); neg2 = any(mo_neg[i] < 0 and i + 1 < len(mo_neg) and mo_neg[i + 1] < 0 for i in range(len(mo_neg)))
    hrs = {}
    for t in tr:
        h = (t["entry_ts"] // 3600) % 24; hrs.setdefault(h, 0.0); hrs[h] += t["net"]
    top_hrs = sorted(hrs.items(), key=lambda kv: -kv[1])[:3]
    if print_rows:
        print(f"{label}: n={n} net=${net:,.0f} exp=${exp:.1f}/lot PF={pf:.2f} wf=[{h1/1000:.0f}k,{h2/1000:.0f}k] "
              f"LODO={lodo}/{n} long=${long:,.0f} short=${short:,.0f} neg2mo={neg2} topH={[(h, f'{v/1000:.0f}k') for h, v in top_hrs]}")
    return {"n": n, "net": net, "exp": exp, "pf": pf, "wf": [h1, h2], "lodo": lodo,
            "long": long, "short": short, "neg2": neg2, "days": daynet, "nets": nets,
            "side": [t["side"] for t in tr], "sym": [t["symbol"] for t in tr], "ts": [t["entry_ts"] for t in tr]}

def ladder(tr, sc):
    net = sum(t["net"] for t in tr); n = len(tr)
    out = []
    for x in (1.25, 1.5, 2.0):
        adj = (x - 1.0) * sc * 0.15 * n
        out.append((x, net - adj))
    return out

def ftmo_sim(r, limit=5000.0, dd_limit=10000.0):
    d = r["days"]; worst = -d.min()
    v = max(0.15, 0.15 * limit / (worst * 2.0))  # size so 2x-stress worst day stays within daily limit
    eq = np.cumsum(d * v / 0.15)
    peak = np.maximum.accumulate(eq)
    mdd = (peak - eq).max()
    wd2x = worst * 2 * v / 0.15
    # loss-streak timing: max consecutive losing days + MC P(3 losing days in a row)
    sgn = (d < 0).astype(int)
    streak = 0; max_streak = 0
    for x in sgn:
        streak = streak + 1 if x else 0
        max_streak = max(max_streak, streak)
    rng = np.random.default_rng(11)
    p3 = 0
    for _ in range(2000):
        s = rng.permutation(sgn)
        for i in range(len(s) - 2):
            if s[i] and s[i + 1] and s[i + 2]:
                p3 += 1; break
    p3 /= 2000.0
    print(f"   FTMO: size={v:.3f} lots worst1x=${worst*v/0.15:,.0f} worst2x=${wd2x:,.0f} "
          f"(daily limit ${limit:,.0f}) maxDD=${mdd:,.0f} (${dd_limit:,.0f}) "
          f"maxLossStreak={max_streak}d P(3L-in-row)={p3:.2f} "
          f"monthly~${r['net']*v/0.15/ (len(np.unique(r['ts']))//30+1):,.0f}")
    return v

def jaccard_book(r):
    p = "../../../run_core_book_live.py"
    jl = "../../../core_book_trades.jsonl"
    if not os.path.exists(jl):
        print("   Jaccard: book log not found, skipped"); return
    tset = set(r["ts"])
    bdays = set()
    for line in open(jl):
        try:
            o = json.loads(line)
            if "entry_ts" in o: bdays.add(o["entry_ts"] // 86400)
            elif "time" in o: bdays.add(int(o["time"]) // 86400)
        except Exception:
            pass
    dset = {x // 86400 for x in r["ts"]}
    print(f"   Jaccard: day-level overlap with live book = {len(dset & bdays)}/{len(dset)} ({len(dset & bdays) / max(1, len(dset)):.2f})")

sc_fx = sum(SPR.get(s, 0.5) * TICK.get(s, 0.1) * 10.0 for s in FX22) / len(FX22)

print("=== DEEP BATTERY: big_move_fade_fx (sessions full, lb48, hold12) ===")
base = run("big_move_fade", list(range(24)))
r = stats(base, "BASE")
print("   ladder:", [f"{x}x=${v:,.0f}" for x, v in ladder(base, sc_fx)])
ftmo_sim(r)
jaccard_book(r)
print("   PLATEAU (all cells must be net-positive):")
ok = True
for lb in (24, 48, 96):
    for hd in (6, 12, 18):
        rr = run("big_move_fade", list(range(24)), lookback=lb, hold=hd)
        s = stats(rr, f"   lb{lb}/h{hd}", print_rows=False)
        if s and s["net"] > 0: print(f"   lb{lb}/h{hd}: +${s['net']/1000:.1f}k PF={s['pf']:.2f} ✓")
        else: ok = False; print(f"   lb{lb}/h{hd}: +${s['net']/1000:.1f}k PF={s['pf']:.2f} ✗")
print("   PLATEAU:", "PASS" if ok else "FAIL")

print("=== DEEP BATTERY: carry_clock_fx (R2-rejected family re-check) ===")
rc = run("carry_clock", list(range(24)))
r2 = stats(rc, "BASE")
print("   ladder:", [f"{x}x=${v:,.0f}" for x, v in ladder(rc, sc_fx)])
for q in (0, 1):
    rr = run("carry_clock", list(range(24)))
    s = stats(rr, f"   run{q}", print_rows=False)
    print(f"   run{q}: net=${s['net']/1000:.1f}k PF={s['pf']:.2f}")

print("=== DEEP BATTERY: macd_reversion_fx + exhaustion_fade_fx (borderline nudges) ===")
for rule, tag in (("session_reversion", "macd_reversion_fx"), ("session_exhaustion", "exhaustion_fade_fx")):
    for sess, hd in (([7, 8, 9, 10, 11, 12], 12), (list(range(24)), 12), ([7, 8, 9, 10, 11, 12], 18)):
        rr = run(rule, sess, hold=hd)
        s = stats(rr, f"   {tag} sess={'eu' if sess==[7,8,9,10,11,12] else 'full'} h{hd}", print_rows=False)
        print(f"   {tag} sess={'eu' if sess==[7,8,9,10,11,12] else 'full'} h{hd}: n={s['n']} exp=${s['exp']:.1f} PF={s['pf']:.2f} neg2={s['neg2']} wf=[{s['wf'][0]/1000:.0f}k,{s['wf'][1]/1000:.0f}k]")
