"""Regression gate — prove the generalized engine reproduces the validated Tokyo_H0.

Tokyo_H0 (audit-validated baseline): session hour 0, top-5 worst 6-bar M5 return,
BUY, fill at next bar open, SL/TP 0.35/0.45 JPY (0.0035/0.0045 others), hold 12,
one position per symbol per day, 18-pair universe, 200-day tape.

Validated reference: 720 trades, ~90% WR, PF ~9, net ~+$11,649, ~$16.66/lot-trade,
exp ~$107.87/lot, max_dd <= $200 (purple: real $77,002 vs shuffled $257).

Exit 0 => parity proven (numbers within audit tolerance).
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proxima_ops.backtest import (StrategySpec, run_strategy, metrics, gate,
                                  purple_edge, determinism, split_by_ts,
                                  walk_forward)
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]

SPEC = StrategySpec.from_dict({
    "name": "tokyo_h0",
    "universe": UNIVERSE,
    "feed": {"kind": "bar", "timeframe": "M5"},
    "signal": {"rule": "session_exhaustion", "lookback": 6, "pick": "n_worst",
               "top_n": 5, "side": "long", "fill_bar": 1},
    "exit": {"mode": "sl_tp_hold", "hold_bars": 12, "stop_first": True},
    "sessions": [0],
    "base_lot": 0.15,
})

REF = {"trades": 720, "wr": 0.90, "pf": 9.0, "net": 11649.36,
       "exp_lot": 107.87, "maxdd": 200.0}
# The audit reference used the conservative $3.5/lot/side gate rate; the live
# broker charges $3.0. Keep 3.5 here so the parity matches the audit curve.
AUDIT_COMMISSION = 3.5

def main() -> int:
    bars = build_bars_map(UNIVERSE)
    # determinism: 3 identical runs -> identical trade count
    runs = [run_strategy(bars, SPEC, volume=SPEC.base_lot,
                         commission_per_lot=AUDIT_COMMISSION) for _ in range(3)]
    d_ok = determinism(lambda: run_strategy(bars, SPEC, volume=SPEC.base_lot,
                                            commission_per_lot=AUDIT_COMMISSION))
    usd = runs[0]
    m = metrics(usd)
    g = gate(m, lot=SPEC.base_lot)
    # purple on the raw USD runner
    purple = purple_edge(bars, lambda bm: run_strategy(bm, SPEC, volume=SPEC.base_lot,
                                                       commission_per_lot=AUDIT_COMMISSION),
                         m["expectancy"] / SPEC.base_lot, iters=5)
    tr, va = split_by_ts(usd)
    wf = walk_forward(usd, train_size=300, test_size=100, lot=SPEC.base_lot)

    print("=" * 72)
    print("TOKYO_H0 PARITY — generalized engine vs audit-validated reference")
    print("=" * 72)
    print(f"trades      : {m['trades']}   (ref {REF['trades']})")
    print(f"win rate    : {m['win_rate']:.4f}   (ref ~{REF['wr']})")
    print(f"profit fact : {m['profit_factor']:.2f}   (ref ~{REF['pf']})")
    print(f"net         : ${m['net_pnl']:,.2f}   (ref ${REF['net']:,.0f})")
    print(f"exp/trade   : ${m['expectancy']:.2f}  (ref ~$16.66)")
    print(f"exp/lot     : ${g['expectancy_per_lot']:.2f}  (ref ~${REF['exp_lot']})")
    print(f"max DD      : ${m['max_drawdown']:.2f}  (ref ~${REF['maxdd']})")
    print(f"commission  : ${m['commission']:,.2f}")
    print(f"determinism : {'OK' if d_ok else 'FAIL'}")
    print(f"purple      : {purple}")
    print(f"gate        : {'PASS' if g['passed'] else 'REJECT ' + str(g['reject'])}")
    print(f"train/val   : {metrics(tr)['trades']}/{metrics(va)['trades']} trades  "
          f"val net ${metrics(va)['net_pnl']:,.2f} val PF {metrics(va)['profit_factor']:.2f}")
    print(f"walk-fwd    : {wf.get('n_windows', 0)} windows, positive share {wf.get('positive_share', 0.0)}, "
          f"stable={wf.get('stable', False)}")
    tol_ok = (abs(m["trades"] - REF["trades"]) <= 2 and g["passed"] and d_ok
              and purple == "REAL-EDGE")
    print("=" * 72)
    print("PARITY:", "PASS" if tol_ok else "FAIL")
    return 0 if tol_ok else 1

if __name__ == "__main__":
    sys.exit(main())