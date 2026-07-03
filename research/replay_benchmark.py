"""Replay-vs-live benchmark — compute composite realism score."""
import sys; sys.path.insert(0, '.')
import json
import os
import logging
logging.basicConfig(level=logging.WARNING)

from research.parity_logger import ParityLogger
from research.live_capture.recorder import LiveCapturer
from replay.environment import build_replay_environment, ReplayConfig
from replay.clock_patcher import patch_clock


def benchmark(capture_path: str, replay_start: str = "2026-04-01",
              replay_end: str = "2026-04-02", symbols: list[str] = None,
              warmup: int = 5000, seed: int = 42) -> dict:
    symbols = symbols or ["EURJPY", "USDJPY"]

    # Load live capture
    capturer = LiveCapturer()
    capturer.load(capture_path)
    live_events = capturer._events

    # Build replay environment
    config = ReplayConfig(
        symbols=symbols,
        start=replay_start,
        end=replay_end,
        speed=500000,
        burst=True,
        latency=True,
        slippage=True,
        seed=seed,
    )
    env = build_replay_environment(config)
    patch_clock(env.clock)

    # Drain warmup ticks
    for _ in range(warmup):
        if env.tick_source.next_tick() is None:
            break

    # Align live events to replay tick sequence
    logger = ParityLogger()
    replay_idx = 0
    for ev in live_events:
        if ev["type"] != "signal":
            continue
        eid = ev.get("event_id")
        if not eid:
            continue

        # Consume replay ticks up to this event
        rd = {}
        while replay_idx < env.replay_feed.total:
            tick = env.tick_source.next_tick()
            if tick is None:
                break
            replay_idx += 1
            tick_eid = tick.get("_event_id")
            if tick_eid == eid:
                rd = {
                    "oss_signal": tick.get("signal", 0),
                    "latency_ms": 0,
                    "decision_ts": tick.get("time_sec", 0),
                    "realized_pnl": tick.get("pnl"),
                }
                break
            if replay_idx >= warmup + len(live_events):
                break

        logger.record_live(eid, ev["data"])
        logger.record_replay(eid, rd)

    score = logger.composite_score()
    print(json.dumps(score, indent=2))
    return score


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Replay-vs-live realism benchmark")
    parser.add_argument("capture", help="Path to live capture JSON file")
    parser.add_argument("--start", default="2026-04-01")
    parser.add_argument("--end", default="2026-04-02")
    parser.add_argument("--symbols", default="EURJPY,USDJPY")
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    benchmark(args.capture, args.start, args.end,
              [s.strip() for s in args.symbols.split(",")],
              args.warmup, args.seed)
