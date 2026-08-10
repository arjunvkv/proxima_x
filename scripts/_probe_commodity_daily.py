"""PROBE: commodity-currency daily signal (gold -> AUDUSD / NZDUSD).

Mechanism: Chen-Rogoff (2003) commodity currencies / terms-of-trade; Apergis &
Papoulakos (2013) daily bidirectional causality gold<->AUDUSD (~0.6% AUD per 1%
gold, long-run); Apergis (2014) out-of-sample predictability of AUDUSD from gold.
Literature gives DIRECTION and DAILY-cadence evidence but NO intraday entry hour —
the entry hour is our hypothesis, hence the purple-shuffle must protect it.

Signal (daily, NO lookahead): for entries on FX day D, the signal is the return of
the LAST COMPLETED gold day L < D: gold_ret[L] = close(L)/close(prev_known(L)) - 1.
Gold UP on L -> LONG AUDUSD/NZDUSD on D; gold DOWN -> SHORT. Measured both sides.

Costs: honest — full spread (typical busy-session map from _sweep_cost_aware) +
commission $3.5/lot/side (RT), volume 0.15. SL/TP wide (daily horizon): nonjpy
(0.0040, 0.0060). Exit clamped to the same server day (no rollover cross-day holds).
Same gate as the vault battery: PF>1.2, net>0, exp>$15/lot, DD<20%, 20<=trades<=20k;
purple = shuffle gold signal values across days (5x); walk-forward 120/60.

Standalone scratch: does NOT touch engine.py / parity gates. Output:
validation_probe_commodity_daily.json
"""
from __future__ import annotations
import sys, os, json, random, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from proxima_ops.backtest.feed import build_bars_map
from proxima_ops.backtest.validation import metrics, gate, walk_forward

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(BASE, "validation_probe_commodity_daily.json")

TRADE = ["AUDUSD", "NZDUSD"]
SIGNAL = "XAUUSD"
HOURS = [1, 3, 7]           # Sydney / Tokyo / London entries (server)
HOLDS = [24, 96, 288]       # 2h / 8h / full day
VOL = 0.15
COMM_SIDE = 3.5             # per lot per side
SPREAD = {"AUDUSD": 1.1, "NZDUSD": 1.4}   # typical busy-session
SL_TP = (0.0040, 0.0060)    # nonjpy
SEED = 12345

_G = {}

def _init_worker():
    global _G
    bars = build_bars_map(TRADE + [SIGNAL])
    _G["fx"] = {s: bars[s] for s in TRADE}
    closes: dict[int, float] = {}
    for b in bars[SIGNAL]:
        closes[b["ts"] // 86400] = b["close"]
    known = sorted(closes)
    rets = {}
    for a, b in zip(known, known[1:]):
        rets[b] = closes[b] / closes[a] - 1.0
    # gold_signal[D] = return of the LAST COMPLETED gold day strictly before D
    sig: dict[int, float] = {}
    last = None
    for d in range(known[0], known[-1] + 1):
        if last is not None and last in rets:
            sig[d] = rets[last]
        if d in closes:
            last = d
    _G["gold_signal"] = sig

def _sim_pair(sym: str, hour: int, hold: int, signal: dict, _rng) -> list[dict]:
    bars = _G["fx"][sym]
    pip = 0.0001
    pip_usd = 10.0 * VOL          # USD-quote majors: $10/pip per 1.0 lot
    trades: list[dict] = []
    n = len(bars)
    i = 0
    while i < n:
        b = bars[i]
        day = b["ts"] // 86400
        h = (b["ts"] // 3600) % 24
        if h == hour and day in signal:
            dirn = 1 if signal[day] >= 0 else -1
            # entry at ask (long) / bid (short): half-spread adverse
            entry = b["open"] + dirn * SPREAD[sym] * pip / 2.0
            sl = entry - dirn * SL_TP[0]
            tp = entry + dirn * SL_TP[1]
            exit_i, exit_px, reason = None, None, "HOLD"
            last_same = min(i + hold - 1, n - 1)
            for j in range(i, last_same + 1):
                x = bars[j]
                if x["ts"] // 86400 != day:
                    last_same = j - 1
                    break
                if (dirn > 0 and x["low"] <= sl) or (dirn < 0 and x["high"] >= sl):
                    exit_i, exit_px, reason = j, sl, "SL"
                    break
                if (dirn > 0 and x["high"] >= tp) or (dirn < 0 and x["low"] <= tp):
                    exit_i, exit_px, reason = j, tp, "TP"
                    break
            if exit_i is None:
                exit_i = last_same
                exit_px = bars[exit_i]["close"]
            pnl_pts = (exit_px - entry) * dirn     # price units (0.0001 = 1 pip)
            gross = (pnl_pts / pip) * pip_usd      # pips x $/pip  (units fix: /pip)
            comm = COMM_SIDE * 2 * VOL
            # exit half-spread in dollars: (spread pips/2) x $/pip
            exit_cost = SPREAD[sym] * pip_usd / 2.0
            net = gross - comm - exit_cost
            trades.append({"sym": sym, "entry_ts": b["ts"],
                           "exit_ts": bars[exit_i]["ts"], "reason": reason,
                           "pnl_pts": round(pnl_pts / pip, 2),  # pips
                           "gross_usd": round(gross, 2),
                           "commission": round(comm, 2), "net": round(net, 2)})
            i = exit_i
        i += 1
    return trades

def process_cell(d: dict) -> dict:
    sym, hour, hold = d["sym"], d["hour"], d["hold"]
    rng = random.Random(SEED)
    base = _sim_pair(sym, hour, hold, _G["gold_signal"], rng)
    m = metrics(base)
    g = gate(m, lot=VOL)
    exp_base = m["expectancy"] / VOL if m["trades"] else 0.0
    keys = list(_G["gold_signal"])
    vals = list(_G["gold_signal"].values())
    purples = []
    for _ in range(5):
        rng.shuffle(vals)
        shuf = dict(zip(keys, vals))
        mm = metrics(_sim_pair(sym, hour, hold, shuf, rng))
        purples.append(mm["expectancy"] / VOL if mm["trades"] else 0.0)
    purple = "REAL-EDGE" if exp_base > max(purples) else "no-edge"
    wf = walk_forward(base, train_size=120, test_size=60, lot=VOL)
    passed = (g["passed"] and purple == "REAL-EDGE"
              and wf.get("stable", False) and wf.get("positive_share", 0) > 0.5)
    return {"rule": "commodity_daily", "sym": sym, "signal": SIGNAL, "hour": hour,
            "hold": hold, "trades": m["trades"], "win_rate": round(m["win_rate"], 4),
            "profit_factor": round(m["profit_factor"], 2), "net": round(m["net_pnl"], 2),
            "exp_lot": g["expectancy_per_lot"], "max_dd": round(m["max_drawdown"], 2),
            "gate": g["passed"], "reject": g["reject"], "purple": purple,
            "purple_floor_exp": round(max(purples), 2),
            "wf_share": wf.get("positive_share", 0.0), "wf_stable": wf.get("stable", False),
            "PASSES_BATTERY": passed}

def main() -> int:
    cells = [{"sym": s, "hour": h, "hold": hd}
             for s, h, hd in itertools.product(TRADE, HOURS, HOLDS)]
    print(f"commodity_daily probe cells: {len(cells)}")
    _init_worker()
    res = []
    for i, c in enumerate(cells):
        r = process_cell(c)
        res.append(r)
        print(f"[{i+1}/{len(cells)}] {r['sym']:<7} h={r['hour']} hold={r['hold']:<3} "
              f"n={r['trades']:4d} WR={r['win_rate']:.2f} PF={r['profit_factor']:.2f} "
              f"net=${r['net']:>9,.0f} exp=${r['exp_lot']:>6,.0f} purple={r['purple']:<9}"
              f" wf={r['wf_share']:.2f} PASS={r['PASSES_BATTERY']}", flush=True)
    json.dump(res, open(OUT, "w"), indent=2)
    ships = [r for r in res if r["PASSES_BATTERY"]]
    print(f"\n=== PASSES_BATTERY: {len(ships)}/{len(cells)} ===")
    for r in ships:
        print(f"  {r['sym']} h={r['hour']} hold={r['hold']} PF={r['profit_factor']} "
              f"net=${r['net']:,.0f} exp=${r['exp_lot']:,.0f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())