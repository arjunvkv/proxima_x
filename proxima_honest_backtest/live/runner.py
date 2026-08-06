"""LiveRunner — drives a validated strategy through the shared DecisionKernel
with LiveExecutor, fed by LiveM5Feed (live) or ReplayFeed (replay/parity).

Mirrors MultiPairBacktestEngine.run()'s decision semantics:
  - build causal history via MarketStateBuilder (closes strictly before ts)
  - ENTER fills at metadata['entry_price'] (bar open)
  - EXIT fills at bar OPEN (same A3 open-of-bar semantics as the backtest)
  - position tracking + state persistence for restart safety
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from proxima_honest_backtest.core.decision_kernel import generate_decisions
from proxima_honest_backtest.live.events.schema import decision_id as make_did
from proxima_honest_backtest.live.executor import LiveExecutor
from proxima_honest_backtest.live.feed import BaseFeed
from proxima_honest_backtest.live.market_state import MarketStateBuilder
from proxima_honest_backtest.live.parity import decision_id as parity_did


class LiveRunner:
    def __init__(
        self,
        strategy: Any,
        feed: BaseFeed,
        executor: LiveExecutor,
        pairs: List[str],
        state_path: Optional[str] = None,
        persist: bool = True,
    ) -> None:
        self.strategy = strategy
        self.feed = feed
        self.executor = executor
        self.pairs = pairs
        self.state_path = state_path
        self.persist = persist
        self.state = MarketStateBuilder()
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.decisions: List[Dict[str, Any]] = []
        self._ready = True  # paper/replay default ready; live supervisor sets False until bootstrap

    def set_ready(self, ready: bool = True) -> None:
        """LIVE_READY barrier: no ENTER orders before warmup+reconcile complete."""
        self._ready = ready

    # ------------------------------------------------------------------
    def _decide(self, bars: Dict[str, Dict], ts) -> None:
        history = self.state.history()
        strategy_bars = bars  # causal contract: open + history only; masked probe handled by caller
        signals = generate_decisions(self.strategy, strategy_bars, history)
        strat_name = getattr(self.strategy, "name", "unknown")

        for signal in signals:
            pair = signal.metadata.get("pair", "")
            action = signal.metadata.get("action", "")
            if not pair:
                continue
            run_id = getattr(self.executor, "run_id", "local")
            if "ENTER" in action and pair not in self.positions:
                direction = "LONG" if signal.signal > 0 else "SHORT"
                if not self._ready:
                    self._emit_event("DECISION", pair, make_did(pair, ts, run_id, "ENTER"),
                                     action="ENTER", side=direction, quantity=0.0,
                                     requested_price=float(signal.metadata.get("entry_price", bars.get(pair, {}).get("open")) or 0.0),
                                     bar_ts_utc=str(ts), skipped="warmup_not_ready")
                    continue
                qty = self.executor.position_size(pair)  # broker lots from backtest units
                bar_open = bars.get(pair, {}).get("open")
                requested_price = float(signal.metadata.get("entry_price", bar_open))
                did = make_did(pair, ts, run_id, "ENTER")
                self._emit_event("DECISION", pair, did, action="ENTER", side=direction,
                                 quantity=qty, requested_price=requested_price,
                                 bar_ts_utc=str(ts))
                # Level-1 parity: record the emitted decision regardless of fill.
                self.decisions.append({
                    "ts": str(ts), "strategy": strat_name, "symbol": pair,
                    "side": direction, "type": "ENTER",
                    "requested_price": round(requested_price, 8),
                    "execution_status": "pending",
                })
                report = self.executor.execute_order(
                    side=direction, quantity=qty, symbol=pair,
                    price=requested_price, volatility=0.001,
                    hour_utc=int(ts.hour) if hasattr(ts, "hour") else 0,
                    timestamp=ts, decision_id=did,
                )
                self.decisions[-1]["decision_id"] = parity_did(strat_name, self.decisions[-1])
                if report.filled:
                    self.positions[pair] = {
                        "side": direction,
                        "entry_price": report.fill_price or requested_price,
                        "quantity": getattr(report.trade, "quantity", qty)
                        if report.trade else qty,
                    }
                    self.decisions[-1]["execution_status"] = "filled"
                    self.decisions[-1]["fill_price"] = round(float(report.fill_price or requested_price), 8)
            elif "EXIT" in action and pair in self.positions:
                bar_open = bars.get(pair, {}).get("open")
                if bar_open is None:
                    continue
                exit_event = {
                    "ts": str(ts), "strategy": strat_name, "symbol": pair,
                    "side": "SELL" if self.positions[pair]["side"] == "LONG" else "BUY",
                    "type": "EXIT", "exit_open": bar_open,
                }
                did = make_did(pair, ts, run_id, "EXIT")
                self._emit_event("DECISION", pair, did, action="EXIT", side="S",
                                 quantity=self.positions[pair].get("quantity", 0.0),
                                 requested_price=bar_open, bar_ts_utc=str(ts))
                report = self.executor.close_position(pair, price=bar_open, timestamp=ts,
                                                      decision_id=did)
                exit_event["execution_status"] = report.reject_reason or "filled"
                exit_event["decision_id"] = parity_did(strat_name, exit_event)
                self.decisions.append(exit_event)
                if report.filled:
                    self.positions.pop(pair, None)

    # ------------------------------------------------------------------
    def _emit_event(self, event_type: str, symbol, decision_id, **payload) -> None:
        if self.executor.emitter is not None:
            self.executor.emitter.emit(event_type, symbol=symbol, decision_id=decision_id, **payload)

    # ------------------------------------------------------------------
    def process_bar(self, bars: Dict[str, Any]) -> None:
        if not bars:
            return
        # candidate ts = union of bar times; use the first available
        ts = next((b["time"] for _, b in bars.items() if b and b.get("time")), None)
        if ts is None:
            return
        # Mirror the backtest engine: strategy only sees snapshots with >=2 pairs
        # (sparse weekend/holiday rows still contribute closes, but never decide).
        if len(bars) >= 2:
            self._decide(bars, ts)
        self.state.append_bar(bars)  # append closes AFTER decision (causal)
        if self.persist and self.state_path:
            self._save_state(ts)

    # ------------------------------------------------------------------
    def run_replay(self, bars) -> None:  # accept iterable/engine-style records
        provider = bars if hasattr(bars, "wait_for_new_bar") else None
        while True:
            bar = provider.wait_for_new_bar() if provider else None
            if bar is None:
                break
            self.process_bar(bar)

    # ------------------------------------------------------------------
    def run_forever(self) -> None:
        self.executor.recover_positions()
        while True:
            bar = self.feed.wait_for_new_bar()
            if bar is None:
                continue
            self.process_bar(bar)
            self.decisions.clear()  # decisions are bounded; state persists

    # ------------------------------------------------------------------
    def _save_state(self, ts) -> None:
        if not self.state_path:
            return
        payload = {
            "strategy": getattr(self.strategy, "name", "unknown"),
            "last_processed_bar": str(ts),
            "positions": self.positions,
        }
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, self.state_path)

    # ------------------------------------------------------------------
    @property
    def n_enter_decisions(self) -> int:
        return sum(1 for d in self.decisions if d["type"] == "ENTER")