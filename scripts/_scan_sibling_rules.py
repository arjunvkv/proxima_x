"""Sweep the FIVE signed-score sibling rules that were NEVER tested (0 cells in any
validation JSON): session_reversion, session_momentum, range_reversion,
range_breakout, liquidity_sweep. Full all-hours scan with gate + val OOS.
Run timeout-friendly: tests each rule x each hour-window."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from proxima_ops.backtest import StrategySpec, run_strategy, metrics, gate, split_by_ts
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
bars = build_bars_map(UNIVERSE)

# window spec: (name, sessions-list). Single hours + key multi-hour blocks.
WINDOWS = [
    ("h0",[0]),("h1",[1]),("h2",[2]),("h3",[3]),("h4",[4]),("h5",[5]),("h6",[6]),
    ("h7",[7]),("h8",[8]),("h9",[9]),("h10",[10]),("h11",[11]),("h12",[12]),
    ("h13",[13]),("h14",[14]),("h15",[15]),("h16",[16]),("h17",[17]),("h18",[18]),
    ("h19",[19]),("h20",[20]),("h22",[22]),
    ("0-1",[0,1]),("2-3",[2,3]),("4-5",[4,5]),("6-8",[6,7,8]),("7-9",[7,8,9]),
    ("9-11",[9,10,11]),("12-13",[12,13]),("14-16",[14,15,16]),("17-19",[17,18,19]),
]

RULES = ["session_reversion","session_momentum","range_reversion","range_breakout",
         "liquidity_sweep"]

def mk(rule, sess, top=5, lb=48, hold=24, sltp=(0.004,0.006), jpy=(0.40,0.60),
       side="both", fill_bar=1, per_hour=False):
    ss = {"frame":"signal","rule":rule,"lookback":lb,"pick":"n_worst","top_n":top,
         "side":side,"fill_bar":fill_bar,"per_hour":per_hour}
    return StrategySpec.from_dict({"name":rule,"universe":UNIVERSE,
        "feed":{"kind":"bar","timeframe":"M5"},"signal":ss,
        "exit":{"mode":"sl_tp_hold","hold_bars":hold,"stop_first":True,
                "jpy_sl_tp":jpy,"non_jpy_sl_tp":sltp},
        "sessions":sess,"base_lot":0.15})

def scan(rule, sess):
    spec = mk(rule, sess, top=5, lb=50, hold=6, side="both")
    try:
        usd = run_strategy(bars, spec, volume=0.15, commission_per_lot=3.5)
    except Exception as e:
        return None, str(e)
    if not usd:
        return None, "no trades"
    m = metrics(usd)
    _, va = split_by_ts(usd)
    v = metrics(va) if va else {"net_pnl":0.0,"profit_factor":0.0}
    return m, v

print("=== all-hours sweep of 5 never-tested sibling rules (top5 lb50 hold 6, side=both) ===")
print(f"{'rule':<22}{'win':<8}{'pf':<6}{'net':<10}{'trad':<6}{'exp':<8}oosa")
results = []
for rule in RULES:
    for wname, sess in WINDOWS:
        m, v = scan(rule, sess)
        if m is None or m.get("trades",0)==0: continue
        g = gate(m, lot=0.15)
        pf = m["profit_factor"]
        net = m["net_pnl"]
        ov = v["net_pnl"] if isinstance(v,dict) else 0
        oos_ok = pf > 1.5 and net > 0 and ov > 0
        if oos_ok:
            results.append((rule, wname, m, v))
            print(f"{rule:<22}{m['win_rate']:<8.3f}{pf:<6.2f}{net:<10.0f}"
                  f"{m['trades']:<6}{net/m['trades']:<6.1f}{ov:>8.0f}  PASS")
print(f"\ncoarse survivors (gate+PF>1.5+val>0): {len(results)}")
for rule, wname, m, v in results:
    print(f"  -> {rule} {wname}: PF {m['profit_factor']:.2f} WR {m['win_rate']:.3f} "
          f"net {m['net_pnl']:.0f} / OOS {v['net_pnl']:.0f}")