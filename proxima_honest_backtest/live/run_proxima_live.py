#!/usr/bin/env python3
"""Proxima LIVE entrypoint — deploy a validated strategy on a real FTMO demo.

Production sequence (ship-killer contract):
  1. connect MT5, verify account
  2. warmup bootstrap  (copy_rates_from_pos -> MarketStateBuilder history)
  3. history-hash gate (live hash vs expected backtest hash; HALT on mismatch)
  4. restart ledger    (replay JSONL + positions_get reconcile; block dup entry)
  5. LIVE_READY        (only then does LiveRunner emit/execute ENTERs)
  6. run forever on LiveM5Feed

RUN:  python live/run_proxima_live.py --dry
      python live/run_proxima_live.py --seed-data ../data/m5_live --hash <expected-sha>
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "proxima_honest_backtest"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from proxima_honest_backtest.live.bootstrap import BootstrapHistory, history_hash
from proxima_honest_backtest.live.events.emitter import EventEmitter, EmitterMode
from proxima_honest_backtest.live.executor import LiveExecutor
from proxima_honest_backtest.live.feed import LiveM5Feed
from proxima_honest_backtest.live.ledger import Ledger, RecoveryChecker
from proxima_honest_backtest.live.runner import LiveRunner
from proxima_honest_backtest.strategies.tokyo_h0.strategy import ALL_PAIRS, TokyoH0Strategy

LOGS_DIR = Path(__file__).resolve().parents[1] / "live_logs"
STREAM_PATH = LOGS_DIR / "stream.jsonl"
STATE_PATH = LOGS_DIR / "state.jsonl"


def load_strategy(pairs: List[str]) -> TokyoH0Strategy:
    return TokyoH0Strategy({"pairs": pairs})


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="paper execution, no real orders")
    ap.add_argument("--bars", type=int, default=100, help="warmup M5 bars per pair")
    ap.add_argument("--magic", type=int, default=400000)
    ap.add_argument("--lot", type=float, default=0.10, help="base lot for --dry sizing")
    ap.add_argument("--hash", default=None, help="expected backtest history hash (HALT on mismatch)")
    ap.add_argument("--seed-data", default=None,
                    help="dir of parquet (same export the backtest consumed) to seed live history; "
                         "when set, the --hash must be computed over that same data")
    ap.add_argument("--pairs", nargs="*", default=None, help="restrict pairs (default all 18)")
    args = ap.parse_args(argv)

    pairs = args.pairs or ALL_PAIRS
    mode = "paper" if args.dry else "live"

    import MetaTrader5 as mt5
    if not mt5.initialize():
        print("MT5 initialize FAILED")
        return 1
    try:
        acct = mt5.account_info()
        if acct is None:
            print("no account_info")
            return 1
        print(f"[live] account={acct.login} server={acct.server} "
              f"balance={acct.balance} trade_allowed={acct.trade_allowed}")

        LOGS_DIR.mkdir(exist_ok=True)

        # 1. warmup bootstrap
        bs = BootstrapHistory(pairs, warmup_bars=args.bars)
        if args.seed_data:
            seed_dir = Path(args.seed_data)
            import pandas as pd
            data = {}
            for p in pairs:
                path = seed_dir / f"{p}.parquet"
                if path.exists():
                    data[p] = pd.read_parquet(path)
            bs_res = bs.seed_from_data(data)
            print(f"[live] warmup seeded from {seed_dir} "
                  f"pairs={ {p: c for p, c in bs_res['seeded'].items()} }")
        else:
            bs_res = bs.bootstrap(mt5)
            print(f"[live] warmup pulled from broker "
                  f"complete={bs.warmup_complete}")
        now_hash = history_hash(bs.history())
        print(f"[live] warmup hash={now_hash[:16]}")

        # 2. history-hash gate
        if args.hash and now_hash != args.hash:
            print("HALT: bootstrap history hash != expected backtest hash")
            return 1

        # 3. restart ledger
        ledger = Ledger()
        n_events = ledger.load(str(STREAM_PATH))
        print(f"[live] ledger loaded {n_events} events, last_seq={ledger.last_seq}")

        # 4. executor + emitter + READY barrier
        emitter = EventEmitter(str(STREAM_PATH), strategy="tokyo_h0",
                               run_id=f"live_{time.strftime('%Y%m%dT%H%M%S')}",
                               mode=EmitterMode.LIVE)
        ex = LiveExecutor(pairs, magic_base=args.magic, base_lot=args.lot,
                          mode=mode, mt5=None if args.dry else mt5,
                          hard_sl_pips=50.0, hard_tp_pips=0.0, emitter=emitter)
        checker = RecoveryChecker(mt5, args.magic, pairs)
        runner = LiveRunner(load_strategy(pairs), LiveM5Feed(pairs, mt5),
                            ex, pairs, state_path=str(STATE_PATH),
                            persist=not args.dry)

        if args.dry:
            runner.set_ready(True)
            print("[live] DRY paper mode, LIVE_READY=true")
        else:
            ex.recover_positions()
            rec = checker.reconcile(ledger)
            print(f"[live] recovery: ok={rec['ok']} matched={rec['matched']} "
                  f"orphan={rec['orphan_broker_positions']}")
            if not rec["ok"]:
                print(f"[live] HALT: reconciliation not clean: {rec}")
                emitter.close()
                return 1
            runner.set_ready(True)
            print("[live] LIVE_READY=true (live)")

        print(f"[live] running mode={mode} on {len(pairs)} pairs ...")
        try:
            while True:
                bar = runner.feed.wait_for_new_bar(timeout=5.0)
                if bar is None:
                    continue
                runner.process_bar(bar)
        except KeyboardInterrupt:
            print("[live] shutdown")
        finally:
            emitter.close()
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())