"""Reconcile SessionStats trade log vs MT5 deals for a single run."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

TRADE_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "trade_logs")


def _load_trade_log(run_id):
    path = os.path.join(TRADE_LOG_DIR, f"run_{run_id}.jsonl")
    if not os.path.exists(path):
        print(f"Trade log not found: {path}")
        return []
    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def reconcile(run_id):
    events = _load_trade_log(run_id)
    if not events:
        return

    fills = [e for e in events if e["event"] == "fill"]
    closes = [e for e in events if e["event"] == "close"]

    tracked_pnl = sum(c.get("gross_pnl", 0) for c in closes)
    fill_tickets = {f.get("ticket") for f in fills if f.get("ticket")}
    close_tickets = {c.get("ticket") for c in closes if c.get("ticket")}

    print(f"=== Reconcile Run {run_id} ===")
    print(f"  Fills : {len(fills)}  (tickets: {len(fill_tickets)})")
    print(f"  Closes: {len(closes)}  (tickets: {len(close_tickets)})")
    print(f"  Tracked PnL: ${tracked_pnl:.2f}")
    print()

    unmatched_fills = fill_tickets - close_tickets
    if unmatched_fills:
        print(f"  ! {len(unmatched_fills)} fills without close (still open or missing):")
        for t in sorted(unmatched_fills):
            f = next((e for e in fills if e.get("ticket") == t), None)
            if f:
                print(f"    ticket={t} {f['pair']} dir={f['direction']} entry={f['entry_price']}")

    unmatched_closes = close_tickets - fill_tickets
    if unmatched_closes:
        print(f"  ! {len(unmatched_closes)} closes without matching fill (tracking mismatch):")
        for t in sorted(unmatched_closes):
            c = next((e for e in closes if e.get("ticket") == t), None)
            if c:
                print(f"    ticket={t} {c['pair']} pnl=${c['gross_pnl']:.2f}")

    if not unmatched_fills and not unmatched_closes:
        print("  All fills matched to closes. Tracking is consistent.")
    print()

    print("  Per-trade detail:")
    print(f"  {'ticket':>12} {'pair':>8} {'pnl':>8} {'dir':>3} {'entry':>10} {'exit':>10}")
    print("  " + "-" * 55)
    for c in sorted(closes, key=lambda x: x.get("entry_time", 0)):
        t = c.get("ticket", "?")
        p = c.get("pair", "?")
        g = c.get("gross_pnl", 0)
        d = c.get("direction", "?")
        ep = c.get("entry_price", 0)
        xp = c.get("exit_price", 0)
        print(f"  {t:>12} {p:>8} {g:>8.2f} {d:>3} {ep:>10.5f} {xp:>10.5f}")

    print()
    print(f"  FINAL: Tracked PnL = ${tracked_pnl:.2f}")
    print("  (Compare against MT5 deals with comment 'pap_{run_id}')")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python paper_trade/scripts/reconcile_run.py <run_id>")
        sys.exit(1)
    reconcile(int(sys.argv[1]))
