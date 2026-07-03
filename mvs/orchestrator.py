from __future__ import annotations

from typing import Dict, List
import time
import numpy as np
import polars as pl

from mvs.models.market_plane import MarketRealityPlane
from mvs.models.perception_plane import PerceptionStatePlane
from mvs.models.action_plane import ActionStatePlane
from mvs.models.outcome_plane import OutcomeStatePlane
from mvs.storage.duckdb_store import TruthStore
from mvs.reconstruction.tick_loader import TickLoader
from mvs.reconstruction.state_rebuilder import StateRebuilder
from mvs.conflict.truth_conflict import TruthConflictEngine
from mvs.honesty.layer_honesty import LayerHonestyEngine
from mvs.honesty.honesty_ranker import HonestyRanker
from mvs.honesty.layer_attribution import LayerAttribution
from mvs.geometry.mfe_mae import MfeMaeCalculator
from mvs.geometry.path_signature import PathSignatureEngine
from mvs.geometry.excursion_topology import ExcursionTopologyEngine
from mvs.geometry.opportunity_geometry import OpportunityGeometryEngine
from mvs.continuation.shadow_continuation import ShadowContinuationEngine
from mvs.utils import TickIndexer


class MVSEngine:
    __slots__ = (
        "symbol", "market", "perception", "action", "outcome",
        "tick_loader", "state_rebuilder", "store",
        "conflict_engine", "honesty_engine", "honesty_ranker", "layer_attribution",
        "mfe_mae", "path_signature", "excursion_topology", "opportunity_geometry",
        "shadow_continuation", "tick_indexer",
        "tick_count", "open_trades", "conflict_interval", "honesty_interval", "flush_interval",
        "_running", "_open_trades_map", "_trade_tick_paths",
    )

    def __init__(self, symbol: str, db_path: str = "mvs.duckdb") -> None:
        self.symbol = symbol
        self.market = MarketRealityPlane()
        self.perception = PerceptionStatePlane()
        self.action = ActionStatePlane()
        self.outcome = OutcomeStatePlane()
        self.tick_loader = TickLoader(symbol)
        self.state_rebuilder = StateRebuilder(symbol)
        self.store = TruthStore(db_path)
        self.conflict_engine = TruthConflictEngine()
        self.honesty_engine = LayerHonestyEngine()
        self.honesty_ranker = HonestyRanker()
        self.layer_attribution = LayerAttribution()
        self.mfe_mae = MfeMaeCalculator()
        self.path_signature = PathSignatureEngine()
        self.excursion_topology = ExcursionTopologyEngine()
        self.opportunity_geometry = OpportunityGeometryEngine()
        self.shadow_continuation = ShadowContinuationEngine()
        self.tick_indexer = TickIndexer()
        self.tick_count = 0
        self.open_trades: List[dict] = []
        self.conflict_interval = 50
        self.honesty_interval = 100
        self.flush_interval = 256
        self._running = True
        self._open_trades_map: Dict[int, dict] = {}
        self._trade_tick_paths: Dict[int, list] = {}

    def run_tick(self) -> dict:
        start = time.time()
        tick_data = self.tick_loader.next()
        if (time.time() - start) > 30:
            raise TimeoutError("MT5 tick timeout exceeded")
        self.market.append_tick(tick_data)
        self.tick_indexer.add(tick_data["ts_ns"], tick_data["tick_id"])
        self.tick_count += 1
        perception_state = self.state_rebuilder.on_tick(
            tick_data["tick_id"], self.symbol, tick_data["ts_ns"],
            tick_data, list(self._open_trades_map.values())
        )
        self.perception.append_state(perception_state)
        for trade_id in self._trade_tick_paths:
            self._trade_tick_paths[trade_id].append(tick_data["mid"])
        if perception_state["observer_state"] == "EXECUTE" and abs(perception_state["tpi"]) >= perception_state["calibration_threshold"]:
            trade_id = int(tick_data["tick_id"] * 1000 + self.tick_count)
            action_data = {
                "action_id": self.tick_count, "trade_id": trade_id, "ticket": trade_id,
                "symbol": self.symbol, "ts_ns": tick_data["ts_ns"], "action_type": "ENTRY",
                "direction": int(np.sign(perception_state["tpi"])), "entry_price": tick_data["mid"],
                "sl": 0.0, "tp": 0.0, "size": 1.0, "regime": perception_state["regime"],
                "signal_strength": abs(perception_state["tpi"]), "rf_prob": 0.5,
                "reason_code": "MVS_EXECUTE", "forced_close": False, "manual_intervention": False,
            }
            self.action.append_action(action_data)
        if self.tick_count % self.flush_interval == 0:
            self.store.write_market(self.market.flush())
            self.store.write_perception(self.perception.flush())
        if self.tick_count % self.conflict_interval == 0:
            self.detect_conflicts()
        if self.tick_count % self.honesty_interval == 0:
            self.update_honesty()
        return tick_data

    def on_trade_event(self, trade_data: dict) -> None:
        trade_id = trade_data["trade_id"]
        if trade_data["action_type"] == "OPEN":
            self._open_trades_map[trade_id] = trade_data
            self._trade_tick_paths[trade_id] = []
        elif trade_data["action_type"] == "CLOSE":
            self.build_outcome(trade_data)
            if trade_data.get("forced_close", False):
                tick_path = np.array(self._trade_tick_paths.get(trade_id, []))
                self.shadow_continuation.continue_trade(trade_id, trade_data["entry_price"], trade_data["direction"], tick_path, 0.0001, trade_data.get("pnl", 0.0))
            self._open_trades_map.pop(trade_id, None)

    def detect_conflicts(self):
        return self.conflict_engine.detect(self.market, self.perception, self.action, self.outcome)

    def update_honesty(self):
        conflicts = self.detect_conflicts()
        layers = ["tpi", "entropy", "regime", "vpl", "observer", "calibration", "drift", "action", "market"]
        scores = [self.honesty_engine.score_layer(layer, self.market, self.perception, self.action, self.outcome, conflicts) for layer in layers]
        ranking = self.honesty_ranker.rank_layers(scores)
        self.layer_attribution.from_conflicts(conflicts)
        return ranking

    def build_outcome(self, trade: dict) -> None:
        trade_id = trade["trade_id"]
        tick_path = np.array(self._trade_tick_paths.get(trade_id, []))
        mfe_mae_result = self.mfe_mae.compute(trade["entry_price"], trade["exit_price"], tick_path, trade["direction"])
        path_sig = self.path_signature.classify(tick_path, trade["entry_price"], trade["direction"])
        topology = self.excursion_topology.compute(tick_path, trade["entry_price"], trade["direction"])
        geometry = self.opportunity_geometry.build(mfe_mae_result, path_sig, topology)
        avg_spread = float(self.market.window(10)["spread"].mean()) if self.market.count > 0 else 0.0001
        shadow = self.shadow_continuation.continue_trade(trade_id, trade["entry_price"], trade["direction"], tick_path, avg_spread, trade.get("pnl", 0.0))
        outcome_data = {
            "trade_id": trade_id, "symbol": self.symbol,
            "entry_ts_ns": trade["entry_ts_ns"], "entry_price": trade["entry_price"],
            "exit_ts_ns": trade["exit_ts_ns"], "actual_exit_price": trade["exit_price"],
            "model_exit_price": trade.get("model_exit_price", trade["exit_price"]),
            "shadow_exit_price": shadow["shadow_exit_price"],
            **mfe_mae_result, "path_signature": path_sig,
            "continuation_alpha": shadow["continuation_alpha"],
            "optimal_exit_price": geometry.optimal_exit,
            "optimal_hold_ticks": geometry.optimal_hold_ticks,
        }
        self.outcome.append_outcome(outcome_data)
        self._trade_tick_paths.pop(trade_id, None)

    def close(self) -> None:
        self.store.write_market(self.market.flush())
        self.store.write_perception(self.perception.flush())
        self.store.write_action(self.action.flush())
        self.store.write_outcome(self.outcome.flush())
        self.store.close()
