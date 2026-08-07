"""1:1 live-vs-backtest parity: dummy live firer (run_tokyo_h0_live.live_loop
offline) vs generalized batch engine — same tape, same spec. Must align on
entry, exit, PnL and EVENT COUNT exactly."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from proxima_ops.backtest import StrategySpec, run_strategy, metrics
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.live_sim import fire_live

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]

SPEC = StrategySpec.from_dict({
    "name": "tokyo_h0", "universe": UNIVERSE,
    "feed": {"kind": "bar", "timeframe": "M5"},
    "signal": {"rule": "session_exhaustion", "lookback": 6, "pick": "n_worst",
               "top_n": 5, "side": "long", "fill_bar": 1},
    "exit": {"mode": "sl_tp_hold", "hold_bars": 12, "stop_first": True},
    "sessions": [0], "base_lot": 0.15,
})

def key(t):
    """Alignment key: symbol + entry_ts + exit_ts + side."""
    return (t["symbol"], t["entry_ts"], t["exit_ts"], t["side"])

def main():
    bars = build_bars_map(UNIVERSE)
    eng = run_strategy(bars, SPEC, volume=0.15)
    live = fire_live(bars, SPEC, volume=0.15)
    print("=" * 76)
    print("1:1 LIVE-vs-BACKTEST PARITY (generalized engine vs dummy live firer)")
    print("=" * 76)
    print(f"engine trades : {len(eng)}")
    print(f"live   trades : {len(live)}")
    print(f"EVENT COUNT   : {'MATCH' if len(eng) == len(live) else 'MISMATCH'}")
    e_by = {key(t): t for t in eng}
    l_by = {key(t): t for t in live}
    e_keys = set(e_by)
    l_keys = set(l_by)
    only_e = e_keys - l_keys
    only_l = l_keys - e_keys
    print(f"aligned (entry+exit+side) : {len(e_keys & l_keys)}")
    print(f"engine-only               : {len(only_e)}")
    print(f"live-only                 : {len(only_l)}")
    if only_e:
        for k in sorted(only_e)[:5]:
            t = e_by[k]
            print(f"   E-only {t['symbol']} entry={t['entry']} pnl={t['pnl_pts']} reason={t['reason']}")
    if only_l:
        for k in sorted(only_l)[:5]:
            t = l_by[k]
            print(f"   L-only {t['symbol']} entry={t['entry']} pnl={t['pnl_pts']} reason={t['reason']}")
    # PnL alignment on common keys
    mism = 0
    for k in e_keys & l_keys:
        if abs(e_by[k]["pnl_pts"] - l_by[k]["pnl_pts"]) > 1e-9 or abs(e_by[k]["net"] - l_by[k]["net"]) > 1e-6:
            mism += 1
            if mism <= 5:
                print(f"   PNL-DIFF {k[0]} e={e_by[k]['pnl_pts']} l={l_by[k]['pnl_pts']}")
    print(f"pnl mismatches on aligned : {mism}")
    me, ml = metrics(eng), metrics(live)
    print(f"engine net ${me['net_pnl']:,.2f}  live net ${ml['net_pnl']:,.2f}  "
          f"diff ${me['net_pnl']-ml['net_pnl']:,.2f}")
    ok = (len(eng) == len(live) and not only_e and not only_l and mism == 0
          and abs(me["net_pnl"] - ml["net_pnl"]) < 0.01)
    print("=" * 76)
    print("PARITY:", "PASS — 1:1 aligned" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())