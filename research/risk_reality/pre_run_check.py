"""V2.4A Phase 3 — pre-run DB state dump."""
import sys
sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
from proxima_ops.ledger.trade_ledger import TradeLedger
from proxima_ops.ledger.signal_ledger import SignalLedger

tl = TradeLedger()
print("=== Pre-run DB State ===")
print(f"Trades: {tl.total_trades}")
print(f"Profit: {tl.total_profit}")
print(f"Open: {len(tl.get_open())}")
recent = tl.get_recent(5)
for t in recent:
    print(f"  Trade #{t['trade_id']}: {t['symbol']} {t['signal_type']} {t['status']}")

sl = SignalLedger()
sm = sl.summary()
print(f"Signals: {sm}")
sig_recent = sl.get_recent(10)
for s in sig_recent:
    print(f"  Signal: {s['symbol']} state={s['signal_state']} exec={s['executed']}")
