"""Session-exhaustion mean reversion — multi-session generalization sweep.

Hypothesis (from web research; strongest fit to our validated machinery):
  The Tokyo_H0 edge — fade the day's first extreme move at a scheduled session
  open, fill next M5 bar, hold 12 bars, JPY 0.35/0.45 or 0.0035/0.0045 — should
  hold at OTHER major FX session opens too, each open being a fresh event:
  Tokyo 00:00, London 07:00, New York 12:00 UTC.

This harness reuses EXACTLY the audited output layer (run_audit.load_bars /
trade_to_usd / metrics / gate / split_by_ts) and the tokyo_h0 port
(ea_ports.tokyo_h0), parameterized ONLY by session_hour — the same machine
shipped in scripts/run_tokyo_h0_live.py. Nothing is rewritten; the "new idea"
is CHOOSING which session hours to run it at, so a live port is a one-constant
change and stays apples-to-apples with the validated 200-day cache.

Anti-lookahead / cost contract preserved by the existing port:
  * closed-bar rank (lookback=6), entry at NEXT bar open (no forming-bar fill)
  * real broker commission $3.5/lot/side, pip-value-true USD conversion
  * SL/TP absolute levels, hold 12 M5 bars, one position per symbol per session
Reoptimization caution: the full 24-h sweep is diagnostic transparency, NOT the
live pick. The live candidate is the *structurally motivated* session set
{0,7,12} (major FX session opens), pre-specified from theory, so we are not
chasing the top of the sweep. Every hour is run against shuffled returns
(purple discriminator) from the same tape.
"""
from __future__ import annotations
import os, sys, random, statistics as st
import polars as pl

ROOT = r"C:\Trading\Proxima_X"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "audit_7_eas"))
from run_audit import load_bars, trade_to_usd, metrics, gate, split_by_ts
import ea_ports as EP

CACHE = os.path.join(ROOT, "audit_7_eas", "market")
TOKYO_UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
                  "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
                  "USDCHF","AUDJPY"]
LOT = 0.15          # matches the live BASE_LOT
DAYS = 200
STRUCTURAL_HOURS = [0, 7, 12]           # Tokyo, London, New York opens (UTC)
SWEEP_HOURS = list(range(24))           # full diagnostic sweep
PURPLE_ITERS = 10


def load_all() -> dict[str, list[dict]]:
    bars = {}
    for s in TOKYO_UNIVERSE:
        p = os.path.join(CACHE, f"{s}.pqt")
        df = pl.read_parquet(p).sort("time")
        bars[s] = [{"ts": int(r["time"]), "open": r["open"], "high": r["high"],
                    "low": r["low"], "close": r["close"]}
                   for r in df.iter_rows(named=True)]
    return bars


def run_hour_usd(bmap: dict, hour: int) -> list[dict]:
    """Full list of net-of-commission trades at this session hour."""
    tr = EP.tokyo_h0({s: b for s, b in bmap.items()}, session_hour=hour)
    return [trade_to_usd(t, LOT) for t in tr if t is not None]


def purple_edge(bars, hour, exp_lot) -> str:
    rng = random.Random(42)
    means, cnt = [], 0
    for _ in range(PURPLE_ITERS):
        sh = {s: b[:] for s, b in bars.items()}
        for s in sh:
            rng.shuffle(sh[s])
        t2 = EP.tokyo_h0({s: b for s, b in sh.items()}, session_hour=hour)
        u2 = [trade_to_usd(t, LOT) for t in t2 if t is not None]
        if u2:
            means.append(sum(x["net"] for x in u2) / len(u2)); cnt += len(u2)
    if not means:
        return "no-trades"
    sm = sum(means) / len(means)
    sd = st.stdev(means) if len(means) > 1 else 0.0
    return "REAL-EDGE" if exp_lot > sm + 2 * sd else "no-edge"


def main() -> None:
    bars = load_all()
    print(f"universe={len(bars)} pairs  cache={DAYS} days  lot={LOT}\n")
    print(f"{'hr':>3} {'#trades':>8}{'per/day':>8}{'WR%':>6}{'net$':>10}"
          f"{'PF':>6}{'exp$lot':>8}{'maxdd$':>8}  edge")
    sweep = {}
    for h in SWEEP_HOURS:
        usd = run_hour_usd(bars, h)
        if not usd:
            print(f"{h:>3}{'-':>7}  (no trades)")
            sweep[h] = {"hour": h, "usd": usd}
            continue
        m = metrics(usd)
        exp_lot = m["expectancy"] / LOT if m["trades"] else 0.0
        tag = purple_edge(bars, h, exp_lot)
        sweep[h] = {"hour": h, "usd": usd, "m": m, "exp_lot": round(exp_lot, 2), "edge": tag}
        print(f"{h:>3}{m['trades']:>8}{m['trades']/DAYS:>8.1f}{m['win_rate']*100:>6.1f}"
              f"{m['net_pnl']:>10,.0f}{m['profit_factor']:>6.2f}{exp_lot:>8.2f}"
              f"{m['max_drawdown']:>8,.0f}  {tag}")

    # ---- structural composite {0,7,12} = the live candidate ----
    emit_composite(bars, STRUCTURAL_HOURS)

    with open(os.path.join(ROOT, "audit_7_eas", "session_reversion_sweep.json"), "w") as f:
        import json
        ser = {str(h): {"trades": (v.get("m") or {}).get("trades", 0),
                        "exp_lot": v.get("exp_lot"),
                        "edge": v.get("edge")} for h, v in sweep.items()}
        json.dump(ser, f, indent=2)
    print("\nwrote audit_7_eas/session_reversion_sweep.json")


def emit_composite(bars, hours) -> None:
    print(f"\n=== COMPOSITE session set {hours} (live candidate) ===")
    all_usd = []
    for h in hours:
        all_usd += run_hour_usd(bars, h)
    if not all_usd:
        print("no trades in composite")
        return
    m = metrics(all_usd)
    tr, va = split_by_ts(all_usd)
    print(f"  total={m['trades']} ({m['trades']/DAYS:.1f}/day across {len(hours)} sessions)")
    print(f"  overall: n={m['trades']} wr={m['win_rate']*100:.1f}% PF={m['profit_factor']:.2f} "
          f"net=${m['net_pnl']:,.0f} exp=${m['expectancy']:.2f} "
          f"({m['expectancy']/LOT:.2f}/lot) dd=${m['max_drawdown']:,.0f}")
    for w, subset in (("train", tr), ("val", va)):
        if not subset:
            continue
        mm = metrics(subset)
        g = gate(mm, lot=LOT)
        print(f"  {w}: n={mm['trades']} wr={mm['win_rate']*100:.1f}% PF={mm['profit_factor']:.2f} "
              f"net=${mm['net_pnl']:,.0f} exp=${mm['expectancy']:.2f} "
              f"({mm['expectancy']/LOT:.2f}/lot) dd=${mm['max_drawdown']:,.0f} "
              f"-> {'PASS' if g['passed'] else 'REJECT ' + str(g['reject'][:2])}")


if __name__ == "__main__":
    main()