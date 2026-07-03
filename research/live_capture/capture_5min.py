"""Live capture for 5 minutes — records ticks + OSS signals."""
import sys; sys.path.insert(0, '.')
import time
import json
from datetime import datetime
sys.path.insert(0, '..')
from research.live_capture.recorder import LiveCapturer
from proxima_ops.config.settings import SETTINGS
from proxima_ops.execution.mt5_connector import MT5Connector
from signals.outcome_surface_signal import OutcomeSurfaceSignal
from research.replay_cache import ReplayCache

capturer = LiveCapturer()
mt5 = MT5Connector()
if not mt5.connect():
    print("FAILED to connect to MT5")
    sys.exit(1)

# Load OSS for signal capture
oss = OutcomeSurfaceSignal()
try:
    cache = ReplayCache(["EURJPY"], "2026-04-01", "2026-04-20", tick_limit=50000, seed=42)
    from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
    ticks = cache.compute()
    recs = []
    ed, doa = {}, DelayedOutcomeEngine(horizon_ticks=20)
    for t in ticks[:10000]:
        s = t["sym"]
        d = t.get("ecdf", 0.5) - 0.5
        sig = 1 if d > 0.05 else (-1 if d < -0.05 else 0)
        ed[s] = {"price": t["price"], "ecdf_rank": t["ecdf"], "entropy": t["entropy"], "signal": sig}
        doa.record_snapshot(ed)
        if doa.ready:
            for s2, outcome in doa.evaluate({s2: ed[s2]["price"] for s2 in ed}).items():
                recs.append({"sym": s2, "ecdf": ed[s2]["ecdf_rank"], "outcome": outcome})
    if recs:
        oss = OutcomeSurfaceSignal.from_pipeline_records(recs, ev_threshold=0.05)
        print(f"OSS trained: {oss.bucket_count()} buckets, density={oss.signal_density():.2f}")
except Exception as e:
    print(f"OSS load failed (non-fatal): {e}")

print(f"Capturing live ticks for {SETTINGS.symbols} for 5 minutes...")
start_wall = time.time()
tick_count = 0
cycle_count = 0
last_signal_check = 0

while time.time() - start_wall < 300:
    now = int(time.time())
    
    # Collect ticks every 100ms
    for sym in SETTINGS.symbols:
        tick = mt5.get_tick(sym)
        if tick:
            capturer.record_tick({"symbol": sym, **tick})
            tick_count += 1
    
    # 60-second evaluation cycle
    if now - last_signal_check >= 60:
        last_signal_check = now
        cycle_count += 1
        for sym in SETTINGS.symbols:
            tick = mt5.get_tick(sym)
            if tick:
                ecdf_rank = 0.5  # simplified for capture
                sig = oss.predict(ecdf_rank)
                capturer.record_signal(f"live_{now}_{sym}", {
                    "oss_signal": sig,
                    "ecdf_rank": ecdf_rank,
                    "price": tick.get("ask", 0),
                    "spread": tick.get("spread", 0),
                    "ts": now,
                })
                print(f"  Cycle {cycle_count}: {sym} signal={sig}")
    
    time.sleep(0.1)

mt5.disconnect()
path = capturer.save(f"live_5min_{datetime.now().strftime('%H%M%S')}")
print(f"Done: {tick_count} ticks, {cycle_count} cycles. Saved to {path}")
