"""
cross_layer_causal_auditor.py — Track which layer caused what decision.

Produces a full causality trace DAG per trade.

Example flow:
  OSS triggered signal → ALT disagreed → SAAL resolved conflict → execution executed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CausalNode:
    """A single decision node in the causality DAG."""

    cycle_id: int
    layer: str
    module: str
    action: str
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    parent_nodes: list[int] = field(default_factory=list)
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CausalNode:
        return cls(**d)


# ---------------------------------------------------------------------------
# Internal implementation (singleton-wrapped below)
# ---------------------------------------------------------------------------

class _CrossLayerCausalAuditor:
    """Internal causal auditor — use the ``CrossLayerCausalAuditor()`` factory."""

    def __init__(self, instance_id: str = "default") -> None:
        self._instance_id = instance_id
        # cycle_id -> node
        self._nodes: dict[int, CausalNode] = {}
        # trade_id -> last cycle_id that produced the trade
        self._trade_to_cycle: dict[str, int] = {}
        logger.info("CrossLayerCausalAuditor(%r) initialised", instance_id)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_node(
        self,
        cycle_id: int,
        layer: str,
        module: str,
        action: str,
        input: dict[str, Any] | None = None,
        output: dict[str, Any] | None = None,
        parent_nodes: list[int] | None = None,
        *,
        trade_id: str | None = None,
    ) -> CausalNode:
        """Record a causal decision node.

        Parameters
        ----------
        cycle_id : int
            Unique (within the instance) identifier for this decision cycle.
        layer : str
            One of the recognised layer names (e.g. ``"oss_surface"``).
        module : str
            Specific module / sub-component that produced the decision.
        action : str
            What happened (e.g. ``"signal_generated"``).
        input, output : dict
            The data that entered and left the layer.
        parent_nodes : list[int] or None
            ``cycle_id``\s of the nodes that caused *this* node.
        trade_id : str or None
            If provided, associates the node with a trade for reverse-lookup.
        """
        node = CausalNode(
            cycle_id=cycle_id,
            layer=layer,
            module=module,
            action=action,
            input=input or {},
            output=output or {},
            parent_nodes=parent_nodes or [],
        )
        self._nodes[cycle_id] = node
        if trade_id is not None:
            self._trade_to_cycle[trade_id] = cycle_id
        logger.debug(
            "Recorded node cycle_id=%s layer=%s action=%s",
            cycle_id, layer, action,
        )
        return node

    # ------------------------------------------------------------------
    # Query — by cycle
    # ------------------------------------------------------------------

    def _layer_order_key(self, node: CausalNode) -> int:
        """Return a sort key that respects the expected layer pipeline order."""
        order = [
            "tick_ingestion",
            "oss_surface",
            "alt_signal",
            "sdil",
            "csfr",
            "saal",
            "execution",
        ]
        # Match the *first* keyword in the layer name
        for idx, prefix in enumerate(order):
            if node.layer.startswith(prefix):
                return idx
        # Unknown layers go last
        return len(order)

    def get_trace(self, cycle_id: int) -> list[CausalNode]:
        """Return all nodes belonging to *cycle_id*, ordered by layer sequence."""
        node = self._nodes.get(cycle_id)
        if node is None:
            logger.warning("No node found for cycle_id=%s", cycle_id)
            return []
        # Flatten: collect the node + all parent nodes recursively
        seen: set[int] = set()
        collected: list[CausalNode] = []

        def _collect(cid: int) -> None:
            if cid in seen:
                return
            seen.add(cid)
            n = self._nodes.get(cid)
            if n is None:
                return
            collected.append(n)
            for pid in n.parent_nodes:
                _collect(pid)

        _collect(cycle_id)
        collected.sort(key=self._layer_order_key)
        return collected

    # ------------------------------------------------------------------
    # Query — by trade
    # ------------------------------------------------------------------

    def get_trade_causality(self, trade_id: str) -> list[CausalNode]:
        """Trace from a specific trade back through all causal parents.

        Returns nodes ordered by layer sequence (earliest → latest).
        """
        cycle_id = self._trade_to_cycle.get(trade_id)
        if cycle_id is None:
            logger.warning("No causality trail found for trade_id=%r", trade_id)
            return []
        return self.get_trace(cycle_id)

    # ------------------------------------------------------------------
    # Decision chain (human-readable)
    # ------------------------------------------------------------------

    def get_decision_chain(self, cycle_id: int) -> list[str]:
        """Return a simple string representation of the decision path.

        Example::

            [
                "tick_ingestion → bid=1.1000, ask=1.1002",
                "oss_surface → p_cont=0.50, signal=0",
                ...
                "execution → EXECUTE(signal=+1)",
            ]
        """
        nodes = self.get_trace(cycle_id)
        chain: list[str] = []
        for n in nodes:
            parts = ", ".join(f"{k}={v}" for k, v in n.output.items())
            chain.append(f"{n.layer} → {parts}")
        return chain

    # ------------------------------------------------------------------
    # DAG construction
    # ------------------------------------------------------------------

    def build_causality_dag(self, cycle_ids: list[int]) -> dict[str, list]:
        """Build a serialisable DAG structure for the given cycle IDs.

        Returns
        -------
        dict
            ``{"nodes": [...], "edges": [{"from": ..., "to": ..., "layer_from": ..., "layer_to": ...}]}``
        """
        seen: set[int] = set()
        nodes_list: list[dict[str, Any]] = []
        edges_list: list[dict[str, Any]] = []

        def _visit(cid: int) -> None:
            if cid in seen:
                return
            seen.add(cid)
            n = self._nodes.get(cid)
            if n is None:
                return
            nodes_list.append(n.to_dict())
            for pid in n.parent_nodes:
                parent = self._nodes.get(pid)
                parent_layer = parent.layer if parent else "unknown"
                edges_list.append({
                    "from": pid,
                    "to": cid,
                    "layer_from": parent_layer,
                    "layer_to": n.layer,
                })
                _visit(pid)

        for cid in cycle_ids:
            _visit(cid)

        return {"nodes": nodes_list, "edges": edges_list}

    # ------------------------------------------------------------------
    # Summary / reset
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return statistics about causality tracking."""
        total_nodes = len(self._nodes)
        total_trades = len(self._trade_to_cycle)
        layers: dict[str, int] = {}
        for n in self._nodes.values():
            layers[n.layer] = layers.get(n.layer, 0) + 1
        return {
            "instance_id": self._instance_id,
            "total_nodes": total_nodes,
            "total_trades": total_trades,
            "layers": layers,
        }

    def reset(self) -> None:
        """Clear all recorded nodes and trade associations."""
        self._nodes.clear()
        self._trade_to_cycle.clear()
        logger.info("CrossLayerCausalAuditor(%r) reset", self._instance_id)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_instances: dict[str, _CrossLayerCausalAuditor] = {}


def CrossLayerCausalAuditor(instance_id: str = "default") -> _CrossLayerCausalAuditor:
    """Return the singleton ``_CrossLayerCausalAuditor`` for *instance_id*."""
    if instance_id not in _instances:
        _instances[instance_id] = _CrossLayerCausalAuditor(instance_id)
    return _instances[instance_id]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """Record a full decision chain and verify trace, decision chain, and DAG."""
    logger.info("=== CrossLayerCausalAuditor self-test ===")
    auditor = CrossLayerCausalAuditor("self_test")
    auditor.reset()

    # 1. tick_ingestion
    auditor.record_node(
        cycle_id=1,
        layer="tick_ingestion",
        module="tick_receiver",
        action="tick_received",
        input={"raw": "EURUSD 1.1000/1.1002"},
        output={"bid": 1.1000, "ask": 1.1002},
    )

    # 2. oss_surface
    auditor.record_node(
        cycle_id=2,
        layer="oss_surface",
        module="oss_surface",
        action="signal_generated",
        input={"bid": 1.1000, "ask": 1.1002},
        output={"p_cont": 0.50, "signal": 0},
        parent_nodes=[1],
    )

    # 3. alt_signal
    auditor.record_node(
        cycle_id=3,
        layer="alt_signal",
        module="alt_signal",
        action="signal_generated",
        input={"bid": 1.1000, "ask": 1.1002},
        output={"signal": 1, "conf": 0.72},
        parent_nodes=[1],
    )

    # 4. sdil
    auditor.record_node(
        cycle_id=4,
        layer="sdil",
        module="sdil_duality",
        action="duality_resolved",
        input={"oss_signal": 0, "alt_signal": 1},
        output={"duality": "OSS_UNIQUELY_FLAT"},
        parent_nodes=[2, 3],
    )

    # 5. csfr
    auditor.record_node(
        cycle_id=5,
        layer="csfr",
        module="csfr_truth",
        action="truth_labeled",
        input={"oss_signal": 0, "alt_signal": 1},
        output={"truth_label": "ALT"},
        parent_nodes=[2, 3],
    )

    # 6. saal
    auditor.record_node(
        cycle_id=6,
        layer="saal",
        module="saal_authority",
        action="conflict_resolved",
        input={"sdil_duality": "OSS_UNIQUELY_FLAT", "csfr_truth": "ALT"},
        output={"authority": "ALT", "policy": "ALT"},
        parent_nodes=[4, 5],
    )

    # 7. execution — associate with a trade
    auditor.record_node(
        cycle_id=7,
        layer="execution",
        module="execution_engine",
        action="decision_made",
        input={"authority": "ALT", "policy": "ALT"},
        output={"decision": "EXECUTE(signal=+1)"},
        parent_nodes=[6],
        trade_id="TRADE-001",
    )

    # ------------------------------------------------------------------
    # Verify get_trace
    # ------------------------------------------------------------------
    trace = auditor.get_trace(cycle_id=7)
    assert len(trace) == 7, f"Expected 7 nodes in trace, got {len(trace)}"
    expected_layers = [
        "tick_ingestion",
        "oss_surface",
        "alt_signal",
        "sdil",
        "csfr",
        "saal",
        "execution",
    ]
    for i, node in enumerate(trace):
        assert node.layer == expected_layers[i], (
            f"Trace[{i}] expected {expected_layers[i]!r}, got {node.layer!r}"
        )
    print(f"[PASS] get_trace: {len(trace)} nodes in correct order")

    # ------------------------------------------------------------------
    # Verify get_decision_chain
    # ------------------------------------------------------------------
    chain = auditor.get_decision_chain(cycle_id=7)
    assert len(chain) == 7
    # Note: Python formats 1.1000 as "1.1" and 0.50 as "0.5", 0.72 as "0.72"
    assert chain[0] == "tick_ingestion → bid=1.1, ask=1.1002"
    assert chain[1] == "oss_surface → p_cont=0.5, signal=0"
    assert chain[2] == "alt_signal → signal=1, conf=0.72"
    assert chain[3] == "sdil → duality=OSS_UNIQUELY_FLAT"
    assert chain[4] == "csfr → truth_label=ALT"
    assert chain[5] == "saal → authority=ALT, policy=ALT"
    assert chain[6] == "execution → decision=EXECUTE(signal=+1)"
    print("[PASS] get_decision_chain:")
    for line in chain:
        print(f"        {line}")

    # ------------------------------------------------------------------
    # Verify get_trade_causality
    # ------------------------------------------------------------------
    trade_trace = auditor.get_trade_causality("TRADE-001")
    assert len(trade_trace) == 7
    assert trade_trace[-1].layer == "execution"
    print(f"[PASS] get_trade_causality('TRADE-001'): {len(trade_trace)} nodes")

    # ------------------------------------------------------------------
    # Verify build_causality_dag
    # ------------------------------------------------------------------
    dag = auditor.build_causality_dag([7])
    assert len(dag["nodes"]) == 7
    # Edges: 1->2, 1->3, 2->4, 3->4, 2->5, 3->5, 4->6, 5->6, 6->7 = 9 edges
    assert len(dag["edges"]) == 9, f"Expected 9 edges, got {len(dag['edges'])}"
    print(f"[PASS] build_causality_dag: {len(dag['nodes'])} nodes, {len(dag['edges'])} edges")

    # ------------------------------------------------------------------
    # Verify get_summary
    # ------------------------------------------------------------------
    summary = auditor.get_summary()
    assert summary["total_nodes"] == 7
    assert summary["total_trades"] == 1
    assert summary["instance_id"] == "self_test"
    print(f"[PASS] get_summary: {summary}")

    auditor.reset()
    assert len(auditor._nodes) == 0
    print("[PASS] reset")

    print("=== CrossLayerCausalAuditor self-test ALL PASSED ===")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    _run_self_test()
