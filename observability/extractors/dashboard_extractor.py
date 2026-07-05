"""
dashboard_extractor.py — Extracts ALL dashboard-level state from the ProximaDemo
runtime and converts it into the structured TelemetrySchema dataclasses.

This is a PURE extraction layer:
  - NO string building / NO formatting / NO printing
  - Every method returns typed dataclass instances
  - Handles missing attributes gracefully via ``getattr`` with defaults
  - Handles NaN / None values properly
"""

from __future__ import annotations

import time
from typing import Optional

from ..schema.telemetry_schema import (
    AccountSnapshot,
    PerformanceSnapshot,
    SymbolEvalData,
    SystemHealthSnapshot,
    TelemetrySnapshot,
)


class DashboardExtractor:
    """
    Extracts dashboard-level state from a ProximaDemo instance.

    Usage::

        demo = ProximaDemo(...)          # from run_proxima_demo.py
        extractor = DashboardExtractor(demo)
        account = extractor.extract_account()
        perf    = extractor.extract_performance()
        symbols = extractor.extract_symbols(eval_data)
        health  = extractor.extract_system_health(eval_data)
    """

    def __init__(self, demo) -> None:
        """
        Args:
            demo: The ``ProximaDemo`` instance (from ``run_proxima_demo.py``).
        """
        self._demo = demo

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def extract_account(self) -> AccountSnapshot:
        """Extract current account state from the MT5 broker bridge."""
        info = getattr(self._demo, 'mt5', None)
        if info is not None:
            info = info.get_account() or {}
        else:
            info = {}

        return AccountSnapshot(
            login=str(info.get("login", "N/A")),
            balance=float(info.get("balance", 0.0)),
            equity=float(info.get("equity", 0.0)),
            margin=float(info.get("margin", 0.0)),
            pnl=float(info.get("profit", 0.0)),
        )

    # ------------------------------------------------------------------
    # Performance
    # ------------------------------------------------------------------

    def extract_performance(self) -> PerformanceSnapshot:
        """Extract aggregated performance metrics from the
        ``OpsPerformanceMonitor``."""
        if hasattr(self._demo, 'perf'):
            perf = self._demo.perf.summary() if callable(
                getattr(self._demo.perf, 'summary', None)) else {}
        else:
            perf = {}

        return PerformanceSnapshot(
            n_trades=int(perf.get("n_trades", 0)),
            today_pnl=float(perf.get("today_pnl", 0.0)),
            sharpe=perf.get("sharpe"),
            pp=perf.get("pp"),
            max_dd=perf.get("max_dd"),
            avg_hold_bars=perf.get("avg_hold_bars"),
            win_rate=perf.get("win_rate"),
        )

    # ------------------------------------------------------------------
    # Symbol-level evaluation data
    # ------------------------------------------------------------------

    def extract_symbols(self, eval_data: Optional[dict] = None) -> list[SymbolEvalData]:
        """Convert the per-symbol ``eval_data`` dict into a list of
        structured ``SymbolEvalData`` snapshots.

        Args:
            eval_data: The ``eval_data`` dict from the demo's main loop.
                       Each key is a symbol string; each value is a dict
                       with keys such as ``price``, ``spread``,
                       ``ecdf_rank``, ``es_val``, ``es_rank``,
                       ``at_rank``, ``sizing_mult``, ``regime``,
                       ``status``, ``entropy``, ``prod_signal``,
                       ``p_cont``, ``oss_ev``, ``oss_conf``,
                       ``expected_move``, ``research_drift``,
                       ``exec_drift``, etc.
        """
        symbols: list[SymbolEvalData] = []
        data = eval_data if eval_data is not None else {}

        for sym, entry in data.items():
            if not isinstance(entry, dict):
                continue

            symbols.append(SymbolEvalData(
                symbol=str(sym),
                price=float(entry.get("price", float('nan'))),
                spread=_as_optional_float(entry.get("spread")),
                ecdf_rank=float(entry.get("ecdf_rank", 0.5)),
                es_val=float(entry.get("es_val", float('nan'))),
                es_rank=float(entry.get("es_rank", float('nan'))),
                at_rank=float(entry.get("at_rank", float('nan'))),
                sizing_mult=float(entry.get("sizing_mult", 0.0)),
                regime=str(entry.get("regime", "N/A")),
                status=str(entry.get("status", "WATCH")),
                entropy=_as_optional_float(entry.get("entropy")),
                prod_signal=_as_optional_int(entry.get("prod_signal")),
                p_cont=_as_optional_float(entry.get("p_cont")),
                oss_ev=_as_optional_float(entry.get("oss_ev")),
                oss_conf=_as_optional_float(entry.get("oss_conf")),
                expected_move=_as_optional_float(entry.get("expected_move")),
                research_drift=_as_optional_int(entry.get("research_drift")),
                exec_drift=_as_optional_int(entry.get("exec_drift")),
            ))

        return symbols

    # ------------------------------------------------------------------
    # System health
    # ------------------------------------------------------------------

    def extract_system_health(self) -> SystemHealthSnapshot:
        """Extract overall system health, deployment metadata, and runtime
        phase classification.

        The phase is derived from the number of closed trades:
            <10   → COLLECTING_EVIDENCE
            <25   → EARLY_VALIDATION
            <50   → INTERMEDIATE_VALIDATION
            >=50  → FULL_VALIDATION
        """
        # Deployment score
        score = {}
        if hasattr(self._demo, 'score'):
            score = self._demo.score.summary() if callable(
                getattr(self._demo.score, 'summary', None)) else {}

        # Runtime duration
        _start = getattr(self._demo, '_start_time', None)
        if _start is not None:
            runtime_sec = time.time() - _start
        else:
            runtime_sec = 0.0

        hours = int(runtime_sec // 3600)
        minutes = int((runtime_sec % 3600) // 60)

        # Closed trades
        closed_trades = getattr(self._demo, '_closed_trades', 0)
        if hasattr(self._demo, 'perf') and hasattr(self._demo.perf, 'n_trades'):
            closed_trades = self._demo.perf.n_trades

        # Deployment id
        deployment_id = getattr(self._demo, 'deployment_id', 'dev')

        # Derive phase
        if closed_trades < 10:
            phase = "COLLECTING_EVIDENCE"
        elif closed_trades < 25:
            phase = "EARLY_VALIDATION"
        elif closed_trades < 50:
            phase = "INTERMEDIATE_VALIDATION"
        else:
            phase = "FULL_VALIDATION"

        # Stability score (heuristic from DeploymentScore if available)
        stability_score = getattr(self._demo, '_stability_score', 0.0)

        return SystemHealthSnapshot(
            stability_score=stability_score,
            kill_switch_pressure=getattr(self._demo, '_kill_switch_pressure', 0.0),
            rollout_progress=getattr(self._demo, '_rollout_progress', 0.0),
            system_integrity=getattr(self._demo, '_system_integrity', 0.0),
            deployment_score=float(score.get("current_score", 0.0)),
            deployment_classification=str(score.get("classification", "UNKNOWN")),
            deployment_id=deployment_id,
            runtime_hours=hours,
            runtime_minutes=minutes,
            phase=phase,
        )

    # ------------------------------------------------------------------
    # Full dashboard (placeholder — override in subclass)
    # ------------------------------------------------------------------

    def extract_full_dashboard(self, eval_data: Optional[dict] = None) -> TelemetrySnapshot:
        """Extract every observable dimension into a single
        ``TelemetrySnapshot``.

        The default implementation raises ``NotImplementedError``.
        Subclasses or users who need the full snapshot should override
        this method, composing the individual extractor methods together
        and supplying the remaining snapshot fields from other sources.

        Args:
            eval_data: The per-symbol evaluation dict from the demo loop.

        Raises:
            NotImplementedError: Always — override in a concrete subclass.
        """
        raise NotImplementedError(
            "extract_full_dashboard is not implemented. "
            "Override in a subclass to compose a complete TelemetrySnapshot."
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _as_optional_float(value) -> Optional[float]:
    """Convert *value* to ``float`` or ``None``.

    ``None``, NaN, and objects that cannot be coerced to a finite float
    are returned as ``None``.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (ValueError, TypeError):
        return None
    if v != v or v == float('inf') or v == float('-inf'):
        return None
    return v


def _as_optional_int(value) -> Optional[int]:
    """Convert *value* to ``int`` or ``None``."""
    if value is None:
        return None
    try:
        v = int(value)
    except (ValueError, TypeError):
        return None
    return v
