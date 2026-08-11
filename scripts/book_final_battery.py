"""book_final_battery.py — combined-book risk check via NOVA (fast path).

Tier-NOW legs (current worker, config-only): tokyo / cascade / london /
usfade at LIVE lots (run_core_book_live.py STRATS) + S3 GOLD exhaustion at
proposed 0.15. Checks combined daily net vs $1,250 daily / $2,500 maxDD.

Tier-NEXT legs (worker extension / NOVA live P3) are listed separately with
their battery-validated standalone sizes — NOT run here (signed/daily rules
the current worker cannot fire).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "_absorb"))
sys.path.insert(0, os.getcwd())
import numpy as np
from costmaps_r3 import corrected_maps
from proxima_ops.backtest.feed import load_bars_cached, build_bars_map
from proxima_ops.backtest.spec import StrategySpec
from proxima_ops.backtest.pnl import FTMO_TICK_VALUES
from proxima_ops.nova import engine as NV

TICK, SPR = corrected_maps()
BIG = (1e9, 1e9)
FX18 = [s for s in FTMO_TICK_VALUES if os.path.exists(
    os.path.join("audit_7_eas", "market", f"{s}.pqt"))]

from proxima_ops.nova import feed as NF

def load(syms):
    return {s: NF.bars_list_to_arrays(load_bars_cached(s)) for s in syms}

def leg(rule, sessions, lb, tn, hold, univ, side="long"):
    return StrategySpec.from_dict({
        "name": "leg", "universe": univ,
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": rule, "lookback": lb, "pick": "n_worst",
                   "top_n": tn, "side": side, "fill_bar": 1},
        "exit": {"mode": "sl_tp_hold", "hold_bars": hold, "stop_first": False,
                 "jpy_sl_tp": BIG, "non_jpy_sl_tp": BIG},
        "sessions": sessions, "base_lot": 0.15})

# ---- Tier NOW: live legs (STRATS truth, tokyo EXCLUDED) + S3 gold ----
# tokyo removed per hour-shift probe (2026-08-10, live daemon spreads):
# hour-0 rollover spread tax $39,514 > edge $22,883 -> net -$16,630 @0.52 lots;
# hours 1-3 have no edge at all. Not viable in any hour slot.
LIVE = {
    "cascade": {"rule": "session_exhaustion", "sessions": [2, 3, 4], "lb": 1440, "tn": 8, "hold": 24, "lot": 0.14, "univ": FX18},
    "london":  {"rule": "session_exhaustion", "sessions": [7, 8, 9], "lb": 1440, "tn": 5, "hold": 12, "lot": 0.23, "univ": FX18},
    "usfade":  {"rule": "session_exhaustion", "sessions": [14, 15, 16, 17, 18, 19], "lb": 50, "tn": 5, "hold": 24, "lot": 0.45, "univ": FX18},
    "gold_s3": {"rule": "session_exhaustion", "sessions": None,      "lb": 50,   "tn": 3, "hold": 12, "lot": 0.15, "univ": ["XAUUSD", "XAGUSD"]},
}
BARS_FX = load(FX18)
BARS_GOLD = load(["XAUUSD", "XAGUSD"])

def run_leg(cfg):
    bm = BARS_FX if cfg["univ"] == FX18 else BARS_GOLD
    spec = leg(cfg["rule"], cfg["sessions"], cfg["lb"], cfg["tn"], cfg["hold"], cfg["univ"])
    return NV.run_strategy(bm, spec, volume=0.15, commission_per_lot=3.0,
                           tick_value_map=TICK, spread_pips_map={s: SPR[s] for s in SPR})

# ---- run all tier-now legs ----
import time
t0 = time.time()
book = {}
for name, cfg in LIVE.items():
    tr = run_leg(cfg)
    nets = np.array([t["net"] for t in tr])
    days = np.array([t["entry_ts"] // 86400 for t in tr])
    dnet = np.array([nets[days == d].sum() * cfg["lot"] / 0.15 for d in np.unique(days)])
    book[name] = {"n": len(tr), "net": float(nets.sum() * cfg["lot"] / 0.15),
                  "days": dnet, "worst": float(-dnet.min()), "lot": cfg["lot"]}
    print(f"  {name:9s} n={book[name]['n']:5d} net=${book[name]['net']:9,.0f} "
          f"worstDay=${book[name]['worst']:7,.0f} @ {cfg['lot']} lots")
print(f"  (NOVA runtime {time.time()-t0:.2f}s)")

# ---- combined daily series over the union of days ----
daykey = {}
for name, cfg in LIVE.items():
    tr = run_leg(cfg)
    nets = np.array([t["net"] for t in tr])
    for t, n in zip(tr, nets):
        d = t["entry_ts"] // 86400
        daykey.setdefault(d, {}).setdefault(name, 0.0)
        daykey[d][name] += n * cfg["lot"] / 0.15
days = sorted(daykey)
comb = np.array([sum(daykey[d].values()) for d in days])
cum = np.cumsum(comb)
peak = np.maximum.accumulate(cum)
mdd = float((peak - cum).max())
worst = float(-comb.min())
green = float((comb > 0).mean())
streak = 0; mx = 0
for x in comb < 0:
    streak = streak + 1 if x else 0
    mx = max(mx, streak)
print(f"\nCOMBINED Tier-NOW ({len(LIVE)} legs): net=${comb.sum():,.0f} "
      f"worstDay=${worst:,.0f} (limit $1,250) maxDD=${mdd:,.0f} (limit $2,500) "
      f"greenDays={green*100:.1f}% maxLossStreak={mx}d")
print(f"  budget used: daily {worst/1250*100:.0f}%  maxDD {mdd/2500*100:.0f}%")

# ---- per-leg daily loss contribution at worst day ----
wd = days[int(np.argmin(comb))]
print(f"  worst combined day {wd}: " + ", ".join(
    f"{k}=${v:,.0f}" for k, v in sorted(daykey[wd].items(), key=lambda kv: -abs(kv[1]))))

json.dump({"tier_now": {k: {"lot": v["lot"], "n": v["n"], "net": v["net"], "worst_day": v["worst"]}
                        for k, v in book.items()},
           "combined": {"net": float(comb.sum()), "worst_day": worst, "max_dd": mdd,
                        "green_days": green, "max_loss_streak": mx}},
          open("scripts/book_final_combined.json", "w"), indent=1)
print("saved scripts/book_final_combined.json")
