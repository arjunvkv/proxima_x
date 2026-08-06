"""Deterministic island runner — proof of the whole validator offline.

Structure (per ChatGPT project restoration, two LiveRunner legs):

  aligned = _align_bars(data)                      # align ONCE
  SIM  leg: LiveRunner + LiveExecutor(paper)  -> sim.jsonl
  BROK leg: LiveRunner + FakeBroker           -> broker.jsonl

Both legs share the SAME strategy, run_id and aligned bars, so decision_ids
match; each has its own emitter stream and executor state. Then an 8-section
sign-off is produced from a single load of both streams.

Performance optimizations (no architecture change):
  - alignment runs once and is reused by both legs
  - each stream is read from disk exactly once (lists passed around)
  - emitters run in ISLAND mode (buffered, flush every 256 events / on close)
  - default months=[1] for fast PR feedback; --months 1,2,3 for release
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

for _root in (Path(__file__).resolve().parents[3], Path(__file__).resolve().parents[2]):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from proxima_honest_backtest.live.events.emitter import EmitterMode, EventEmitter, replay_stream
from proxima_honest_backtest.live.executor import LiveExecutor
from proxima_honest_backtest.live.feed import ReplayFeed
from proxima_honest_backtest.live.runner import LiveRunner
from proxima_honest_backtest.live.reconciliation.monitor import ReconMonitor
from proxima_honest_backtest.live.reconciliation.report import generate_signoff
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator
from proxima_honest_backtest.strategies.multi_pair_engine import MultiPairBacktestEngine
from proxima_honest_backtest.validation.island.config import IslandConfig
from proxima_honest_backtest.validation.island.diff import compare_level2
from proxima_honest_backtest.validation.island.fake_broker import FakeBroker


def load_pairs(pairs: List[str], months: List[int], year: int = 2026) -> Dict[str, pd.DataFrame]:
    try:
        from data.providers.mt5_provider import MT5Provider
        provider = MT5Provider()
    except Exception:
        return {}
    out: Dict[str, pd.DataFrame] = {}
    for sym in pairs:
        frames = []
        for m in months:
            try:
                df = provider.load_rates(sym, year, m, "m5")
            except Exception:
                df = pd.DataFrame()
            if not df.empty:
                frames.append(df)
        if frames:
            d = pd.concat(frames, ignore_index=True)
            d.sort_values("time", inplace=True)
            d.reset_index(drop=True, inplace=True)
            out[sym] = d
    return out


def _aligned_records(data: Dict[str, pd.DataFrame], ship) -> List[Dict[str, Any]]:
    engine = MultiPairBacktestEngine(ship.factory(), ExecutionSimulator("exness"))
    return engine._align_bars(data)


def _run_leg(aligned, ship, executor, run: str) -> int:
    runner = LiveRunner(
        strategy=ship.factory(),
        feed=ReplayFeed(aligned),
        executor=executor,
        pairs=ship.pairs,
        persist=False,
    )
    runner.run_replay(ReplayFeed(aligned))
    return runner.n_enter_decisions


def run_island(cfg: IslandConfig) -> Dict[str, Any]:
    ship = cfg.ship()
    # run_id is a property that recomputes a fresh UTC timestamp on every access —
    # capture ONCE so both legs + output dir share a single stable identity.
    run_id = cfg.run_id
    data = load_pairs(ship.pairs, cfg.months, cfg.year)
    if not data:
        return {"error": "no data", "run_id": run_id}

    out_dir = Path(cfg.out_dir) if cfg.out_dir else (
        Path(__file__).resolve().parents[2] / "validation" / "island" / "runs"
    )
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- align ONCE ----
    aligned = _aligned_records(data, ship)

    # ---- SIM leg ----
    sim_path = run_dir / "sim.jsonl"
    sim_emit = EventEmitter(str(sim_path), strategy=ship.key, run_id=run_id,
                            validate=True, mode=EmitterMode.ISLAND)
    sim_ex = LiveExecutor(ship.pairs, magic_base=cfg.magic_base, base_lot=cfg.base_lot,
                          mode="paper", spread_model_half=0.0, emitter=sim_emit)
    sim_decisions = _run_leg(aligned, ship, sim_ex, run_id)
    sim_emit.close()

    # ---- BROKER leg ----
    broker_path = run_dir / "broker.jsonl"
    bro_emit = EventEmitter(str(broker_path), strategy=ship.key, run_id=run_id,
                            validate=True, mode=EmitterMode.ISLAND)
    fake = FakeBroker(emitter=bro_emit, seed=cfg.fake_seed,
                      scenarios={p: cfg.scenario for p in ship.pairs})
    broker_decisions = _run_leg(aligned, ship, fake, run_id)
    bro_emit.close()

    # ---- single load each, then evaluate ----
    sim_events = replay_stream(str(sim_path))
    broker_events = replay_stream(str(broker_path))
    sim_verdict = ReconMonitor().evaluate(sim_events)
    broker_verdict = ReconMonitor().evaluate(broker_events)
    level2 = compare_level2(sim_events, broker_events)

    signoff = generate_signoff(
        run_id=run_id,
        events=broker_events,
        env=cfg.env,
        strategy=ship.key,
        level2=level2,
        out_path=str(run_dir / "signoff_report.json"),
    )

    metadata = {
        "run_id": run_id,
        "strategy": ship.key,
        "env": cfg.env,
        "months": cfg.months,
        "scenario": cfg.scenario,
        "seed": cfg.fake_seed,
        "n_aligned": len(aligned),
        "sim_decisions": sim_decisions,
        "broker_decisions": broker_decisions,
        "n_sim_fills": level2.get("n_sim_fills", 0),
        "n_broker_fills": level2.get("n_broker_fills", 0),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    return {
        "run_id": run_id,
        "dir": str(run_dir),
        "n_aligned": len(aligned),
        "sim_verdict": sim_verdict["verdict"],
        "broker_verdict": broker_verdict["verdict"],
        "level2": level2,
        "signoff_verdict": signoff["verdict"],
        "n_sim_fills": level2.get("n_sim_fills", 0),
        "n_broker_fills": level2.get("n_broker_fills", 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Island validator (two-leg offline signoff)")
    ap.add_argument("--months", default="1", help="comma-separated month numbers, e.g. 1,2,3")
    ap.add_argument("--scenario", default="instant", help="FakeBroker scenario")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = IslandConfig(
        months=[int(m) for m in args.months.split(",") if m.strip()],
        scenario=args.scenario,
        fake_seed=args.seed,
    )
    res = run_island(cfg)
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()