"""Capture 15 minutes of live ticks for parity benchmark."""
import sys; sys.path.insert(0, '.')
import time, json
from datetime import datetime
sys.path.insert(0, '..')
from research.live_capture.recorder import LiveCapturer
from proxima_ops.config.settings import SETTINGS
from proxima_ops.execution.mt5_connector import MT5Connector
from signals.outcome_surface_signal import OutcomeSurfaceSignal
from research.replay_cache import ReplayCache
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine

capturer = LiveCapturer()
mt5 = MT5Connector()
if not mt5.connect():
    print("FAILED to connect to MT5")
    sys.exit(1)

print("Capturing 15 minutes of live ticks for", SETTINGS.symbols)
start_wall = time.time()
tick_count = 0
cycle_count = 0
last_signal_check = 0

while time.time() - start_wall < 900:
    now = int(time.time())
    for sym in SETTINGS.symbols:
        tick = mt5.get_tick(sym)
        if tick:
            capturer.record_tick({"symbol": sym, **tick})
            tick_count += 1
    if now - last_signal_check >= 60:
        last_signal_check = now
        cycle_count += 1
        for sym in SETTINGS.symbols:
            tick = mt5.get_tick(sym)
            if tick:
                capturer.record_signal(f"live_{now}_{sym}", {
                    "price": tick.get("ask", 0),
                    "spread": tick.get("spread", 0),
                    "ts": now,
                })
                print(f"  Cycle {cycle_count}: {sym} price={tick.get('ask', 0):.3f}")
    time.sleep(0.1)

mt5.disconnect()
path = capturer.save(f"live_15min_{datetime.now().strftime('%H%M%S')}")
print(f"\nDone: {tick_count} ticks, {cycle_count} cycles. Saved to {path}")
