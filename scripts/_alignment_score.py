"""_alignment_score.py — measured live-vs-model alignment for the core book.

Reads the journal written by run_core_book_live.py --execute
(logs/core_book_trades.jsonl; hourly spread snapshots in core_book_spreads.csv)
and answers, per entry and book-level:

  * ENTRY ALIGNMENT — live fill price vs the model's validated fill bar OPEN
    (recorded in journal as model_open). Median/p95 abs distance in pips:
    0.00 = pixel-perfect parity with the backtest fill convention.
  * SLIP — requested (quote used) vs actual deal fill price, in pips.
  * SPREAD AT FILL vs the research TYPICAL envelope (excess pips).
  * REALIZED PnL — summed from closed tickets.

ALIGNMENT SCORE (0-100, 100 = perfect):
  100 - 30*median_entryΔ - 20*median_slip - 10*median_spread_excess, floored.

Usage (offline, no broker needed):
    python scripts/_alignment_score.py
"""
from __future__ import annotations
import json, os, csv
from statistics import median

ROOT = os.environ.get("PROXIMA_ROOT", r"C:\Trading\Proxima_X")
JOURNAL = os.path.join(ROOT, "logs", "core_book_trades.jsonl")
SPREAD_LOG = os.path.join(ROOT, "logs", "core_book_spreads.csv")

# {symbol: (TYPICAL busy pips, MEASURED worst pips)} research envelope
ENVELOPES = {
    "EURUSD": (0.8, 3.5), "USDJPY": (0.8, 3.5), "GBPUSD": (1.6, 4.0),
    "AUDUSD": (1.0, 4.0), "EURJPY": (1.2, 4.5), "GBPJPY": (2.3, 5.5),
    "AUDJPY": (0.9, 4.0), "EURAUD": (2.0, 6.0), "EURNZD": (3.0, 8.0),
    "GBPAUD": (3.0, 8.0), "GBPNZD": (3.5, 9.0), "GBPCAD": (2.5, 7.0),
    "USDCAD": (1.0, 4.0), "NZDUSD": (1.4, 5.0), "AUDNZD": (2.5, 7.0),
    "EURGBP": (1.0, 4.0), "EURCHF": (1.5, 5.0), "USDCHF": (1.8, 5.5),
}
DEFAULT_ENV = (1.5, 5.0)


def pip_size(sym: str) -> float:
    return 0.01 if "JPY" in sym else 0.0001


def load_journal() -> tuple[list[dict], list[dict]]:
    entries, closes = [], []
    if not os.path.exists(JOURNAL):
        return entries, closes
    with open(JOURNAL) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("kind") == "entry":
                entries.append(rec)
            elif rec.get("kind") == "close":
                closes.append(rec)
    return entries, closes


def load_spreads() -> list[list[float]]:
    rows = []
    if not os.path.exists(SPREAD_LOG):
        return rows
    with open(SPREAD_LOG) as f:
        for row in csv.reader(f):
            if not row or not row[0].isdigit():
                continue
            rows.append([float(x) if x else float("nan") for x in row[1:]])
    return rows


def main() -> None:
    entries, closes = load_journal()
    spreads = load_spreads()
    print(f"journal: {len(entries)} entries, {len(closes)} closes | "
          f"spread snapshots: {len(spreads)}")
    if not entries:
        print("ALIGNMENT: n/a (no fills journaled yet)")
        return

    close_by_ticket: dict = {}
    for c in closes:
        close_by_ticket.setdefault(c.get("ticket"), []).append(c)

    by_strat: dict[str, dict] = {}
    entry_deltas: list[float] = []
    slips: list[float] = []
    spread_excess: list[float] = []
    realized = 0.0
    for e in entries:
        strat = e.get("strategy", "?")
        s = by_strat.setdefault(strat, {"n": 0, "deltas": [], "slips": []})
        s["n"] += 1
        sym = e["symbol"]
        model_open = e.get("model_open")
        delta = (abs(e["actual"] - model_open) / pip_size(sym)) if model_open else 0.0
        slip = abs(e["actual"] - e["requested"]) / pip_size(sym)
        s["deltas"].append(delta)
        s["slips"].append(slip)
        entry_deltas.append(delta)
        slips.append(slip)
        sp = e.get("spread_pips_at_fill")
        if sp is not None:
            env = ENVELOPES.get(sym, DEFAULT_ENV)
            excess = float(sp) - env[0]
            if excess > 0:
                spread_excess.append(excess)
        for c in close_by_ticket.get(e.get("ticket"), []):
            realized += float(c.get("pnl", 0.0))

    names = sorted(by_strat)
    print("\n== per-strategy ENTRY alignment (live fill vs model bar) ==")
    print(f"{'strategy':<10}{'n':>5}{'medΔpips':>10}{'p95Δpips':>10}")
    for n in names:
        d = sorted(by_strat[n]["deltas"])
        p95 = d[int(len(d) * 0.95) - 1] if len(d) >= 20 else (d[-1] if d else 0.0)
        print(f"{n:<10}{by_strat[n]['n']:>5}{median(d):>10.2f}{p95:>10.2f}")

    dl = sorted(entry_deltas) if entry_deltas else [0.0]
    entry_med = median(dl)
    entry_p95 = dl[int(len(dl) * 0.95) - 1] if len(dl) >= 20 else dl[-1]
    slip_med = median(slips) if slips else 0.0
    excess_med = median(spread_excess) if spread_excess else 0.0

    score = max(0.0, 100.0 - 30.0 * entry_med - 20.0 * slip_med
                - 10.0 * excess_med)

    print("\n== book ENTRY alignment ==")
    print(f"  median entryΔ   : {entry_med:.2f} pips (model=next-bar open)")
    print(f"  p95 entryΔ      : {entry_p95:.2f} pips")
    print(f"  median slip     : {slip_med:.2f} pips (requested vs deal fill)")
    print(f"  median spread + : {excess_med:.2f} pips over TYPICAL envelope")
    print(f"  realized pnl    : ${realized:.2f} on {len(closes)} closed tickets")
    print(f"\nALIGNMENT SCORE: {score:.0f}/100")

    if spreads:
        env = list(ENVELOPES)
        print("\n== LIVE spread snapshot envelope (hourly, all symbols) ==")
        print(f"{'symbol':<10}{'n':>5}{'median':>8}{'p95':>8}{'worst':>8}")
        for i, sym in enumerate(env):
            vals = [r[i] for r in spreads if r[i] == r[i]]  # skip NaN
            if not vals:
                continue
            vals.sort()
            p95 = vals[int(len(vals) * 0.95) - 1] if len(vals) >= 20 else vals[-1]
            print(f"{sym:<10}{len(vals):>5}{median(vals):>8.2f}{p95:>8.2f}"
                  f"{vals[-1]:>8.2f}")


if __name__ == "__main__":
    main()