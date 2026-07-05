"""
telemetry_schema.py — Canonical structured data types for the observability system.

Schema Version: 1.0.0

This module defines the typed dataclasses that convert raw runtime state into
JSON-serialisable structured data. It is a pure data normalisation layer with
zero logic, no formatting, and no string rendering beyond what the types require.

Every dataclass here mirrors a section of the terminal dashboard produced by
`run_proxima_demo.py`. Together they enable 1:1 reconstruction of that dashboard
on any consumer (CLI, WebSocket, REST, file export).

Usage:
    snap = TelemetrySnapshot(...)
    raw_dict = dataclass_to_dict(snap)
    json_str = snapshot_to_json(snap)
"""

from __future__ import annotations

import dataclasses
import json
import time
from typing import Optional


# ── Account / Performance ──────────────────────────────────────────────────────


@dataclasses.dataclass
class AccountSnapshot:
    """Current account state as reported by the broker bridge."""
    login: str
    balance: float
    equity: float
    margin: float
    pnl: float


@dataclasses.dataclass
class PerformanceSnapshot:
    """Aggregate performance metrics over the current session/day."""
    n_trades: int
    today_pnl: float
    sharpe: Optional[float]
    pp: Optional[float]
    max_dd: Optional[float]
    avg_hold_bars: Optional[float]
    win_rate: Optional[float]


# ── Symbol Evaluation ──────────────────────────────────────────────────────────


@dataclasses.dataclass
class SymbolEvalData:
    """Per-symbol evaluation result from the research-execution pipeline."""
    symbol: str
    price: float
    spread: Optional[float]
    ecdf_rank: float
    es_val: float
    es_rank: float
    at_rank: float
    sizing_mult: float
    regime: str
    status: str
    entropy: Optional[float]
    prod_signal: Optional[int]
    p_cont: Optional[float]
    oss_ev: Optional[float]
    oss_conf: Optional[float]
    expected_move: Optional[float]
    research_drift: Optional[int]
    exec_drift: Optional[int]


# ── Position & Funnel ──────────────────────────────────────────────────────────


@dataclasses.dataclass
class PositionSnapshot:
    """Live open position state."""
    ticket: int
    symbol: str
    side: str
    volume: float
    entry_price: float
    current_price: float
    profit: float
    bars_elapsed: int
    entry_es_rank: Optional[float]
    entry_at_rank: Optional[float]
    econ_ratio: Optional[float]
    expected_move: Optional[float]
    trigger_count_while_open: int


@dataclasses.dataclass
class FunnelSnapshot:
    """Order life-cycle funnel — how many signals survive each gate."""
    generated: int
    threshold_passed: int
    triggered: int
    submitted: int
    accepted: int
    opened: int
    closed: int
    blocked: int
    rejected: int
    timeout: int
    pass_rate: float
    submit_rate: float
    accept_rate: float
    open_rate: float
    leakage_pct: float
    survival_rate: float
    block_breakdown: dict  # key -> count


# ── Trade Policy Indicator (TPI) ────────────────────────────────────────────────


@dataclasses.dataclass
class TpiSnapshot:
    """Trade Policy Indicator — alignment / conflict statistics between research and execution."""
    per_symbol: dict  # symbol -> {tpi, direction, confidence, percentile, session, eligible, alignment, persistence, curvature}
    base_wins: int
    base_losses: int
    aligned_wins: int
    aligned_losses: int
    conflict_wins: int
    conflict_losses: int
    veto_avoided_losses: int
    total_shadow_observations: int
    alignments: int
    conflicts: int
    weak_alignments: int
    gates_passed: int  # 0-3
    live_decay: dict  # H1/H3 hit rates, half-life, etc.


# ── Regime / Thermodynamics / Topology / Shadow ────────────────────────────────


@dataclasses.dataclass
class RegimeSnapshot:
    """Market regime state and transition dynamics."""
    regime_state: float  # encoded regime
    regime_transition_pressure: float
    regime_entropy_gradient: float
    regime_stability_velocity: float
    per_symbol_regime: dict  # symbol -> regime string


@dataclasses.dataclass
class InformationThermodynamicsSnapshot:
    """Information-theoretic measures of the signal environment."""
    entropy_level: float
    entropy_derivative: float
    compression_ratio: float
    signal_entropy: float
    noise_floor_estimate: float
    predictability_index: float


@dataclasses.dataclass
class ExecutionTopologySnapshot:
    """Execution topology metrics — density, fill quality, rotation/lock events."""
    signal_density: float
    execution_rate: float
    fill_ratio: float
    slippage_proxy: float
    win_rate_proxy: float
    risk_exposure: float
    rotation_events: int
    lock_events: int
    migration_events: int


@dataclasses.dataclass
class ShadowSnapshot:
    """Mirror / shadow system alignment and edge preservation."""
    shadow_alignment: float
    sof_score: float
    edge_decay: float
    mirror_divergence: float
    alpha_transfer_rate: float
    false_signal_rate: float
    gt_corr: float
    sy_corr: float
    stas: float
    winner: str
    edge_preservation: float
    execution_efficiency: float
    phase2_enabled: bool
    samples: int


# ── Reality / Convergence ──────────────────────────────────────────────────────


@dataclasses.dataclass
class DeploymentRealitySnapshot:
    """Reality-check metrics for the deployment leg."""
    asr: float
    execution_quality: str
    mean_slippage_pts: float
    score_trend: str
    classification: str


@dataclasses.dataclass
class FrequencyRealitySnapshot:
    """Reality-check metrics for the frequency leg."""
    blocked_total: int
    blocked_profitable: int
    leakage_rate: str  # e.g. "12.5%"
    adr: float
    classification: str


@dataclasses.dataclass
class RealityConvergenceSnapshot:
    """Convergence metrics between deployment and frequency reality checks."""
    ate: float
    frequency_match: str  # e.g. "75%"
    friction_index: float
    health_index: float
    classification: str


@dataclasses.dataclass
class DplValidationSnapshot:
    """DPL validation — how many snapshots resolved with actual outcomes."""
    total_snapshots: int
    resolved: int
    pct_resolved: float
    symbols: list
    regime_distribution: dict
    has_short_outcomes: bool


# ── Director / RCL / Session / System ──────────────────────────────────────────


@dataclasses.dataclass
class DirectorPipelineSnapshot:
    """Director-level assessment of pipeline health."""
    evidence_strength: float
    research_confidence: float
    deployment_confidence: float
    alpha_transfer: float
    biggest_risk: str
    biggest_strength: str
    recommendation: str
    classification: str


@dataclasses.dataclass
class RclDashboardSnapshot:
    """RCL (Research Confidence Log) dashboard — win-rate by horizon."""
    h5_resolved: int
    h20_resolved: int
    h5_wins: int
    h20_wins: int
    h5_win_rate: float
    h20_win_rate: float
    divergence: float


@dataclasses.dataclass
class SessionBalanceSnapshot:
    """Trade distribution across sessions and balance classification."""
    asia: int
    london: int
    overlap: int
    ny: int
    dead: int
    total: int
    imbalance: float
    status: str  # BALANCED, SKEWED, or BUILDING


@dataclasses.dataclass
class SystemHealthSnapshot:
    """Overall system health, stability, and rollout progress."""
    stability_score: float
    kill_switch_pressure: float
    rollout_progress: float
    system_integrity: float
    deployment_score: float
    deployment_classification: str
    deployment_id: str
    runtime_hours: int
    runtime_minutes: int
    phase: str  # COLLECTING_EVIDENCE, EARLY_VALIDATION, etc.


# ── Master Telemetry Snapshot ──────────────────────────────────────────────────


@dataclasses.dataclass
class TelemetrySnapshot:
    """Master telemetry snapshot — every observable dimension in one object.

    This is the top-level contract for the WebSocket streaming protocol and
    the canonical input to any dashboard renderer.
    """
    cycle_id: int
    timestamp: float

    account: AccountSnapshot
    performance: PerformanceSnapshot
    symbols: list[SymbolEvalData]
    positions: list[PositionSnapshot]
    funnel: FunnelSnapshot
    tpi: TpiSnapshot

    regime: RegimeSnapshot
    thermodynamics: InformationThermodynamicsSnapshot
    execution_topology: ExecutionTopologySnapshot
    shadow: ShadowSnapshot

    deployment_reality: DeploymentRealitySnapshot
    frequency_reality: FrequencyRealitySnapshot
    reality_convergence: RealityConvergenceSnapshot
    dpl_validation: DplValidationSnapshot
    director_pipeline: DirectorPipelineSnapshot
    rcl_dashboard: RclDashboardSnapshot
    session_balance: SessionBalanceSnapshot
    system_health: SystemHealthSnapshot

    engine_vector: list[float]  # 32 floats


# ── Utility Functions ──────────────────────────────────────────────────────────


def dataclass_to_dict(obj) -> dict:
    """Convert any dataclass to a JSON-serialisable dict.

    Recursively converts all nested dataclass fields into plain dicts
    via ``dataclasses.asdict``.
    """
    return dataclasses.asdict(obj)


def snapshot_to_json(snapshot: TelemetrySnapshot) -> str:
    """Convert a full ``TelemetrySnapshot`` to a JSON string.

    Uses ``str`` as a fallback for any non-serialisable values (e.g. ``None``
    inside ``Optional`` fields that have been set to a non-standard type, or
    NumPy scalars).
    """
    return json.dumps(dataclass_to_dict(snapshot), default=str)


# ── Verification (runs when module is executed directly) ────────────────────────


if __name__ == "__main__":
    # Construct a minimal valid snapshot with dummy values and prove
    # it round-trips through JSON without error.
    snap = TelemetrySnapshot(
        cycle_id=1,
        timestamp=time.time(),
        account=AccountSnapshot(login="123", balance=10000.0, equity=10000.0, margin=0.0, pnl=0.0),
        performance=PerformanceSnapshot(
            n_trades=0, today_pnl=0.0, sharpe=None, pp=None,
            max_dd=None, avg_hold_bars=None, win_rate=None,
        ),
        symbols=[],
        positions=[],
        funnel=FunnelSnapshot(
            generated=0, threshold_passed=0, triggered=0, submitted=0,
            accepted=0, opened=0, closed=0, blocked=0, rejected=0, timeout=0,
            pass_rate=0.0, submit_rate=0.0, accept_rate=0.0, open_rate=0.0,
            leakage_pct=0.0, survival_rate=0.0, block_breakdown={},
        ),
        tpi=TpiSnapshot(
            per_symbol={},
            base_wins=0, base_losses=0,
            aligned_wins=0, aligned_losses=0,
            conflict_wins=0, conflict_losses=0,
            veto_avoided_losses=0,
            total_shadow_observations=0,
            alignments=0, conflicts=0, weak_alignments=0,
            gates_passed=0,
            live_decay={},
        ),
        regime=RegimeSnapshot(
            regime_state=0.0, regime_transition_pressure=0.0,
            regime_entropy_gradient=0.0, regime_stability_velocity=0.0,
            per_symbol_regime={},
        ),
        thermodynamics=InformationThermodynamicsSnapshot(
            entropy_level=0.0, entropy_derivative=0.0, compression_ratio=0.0,
            signal_entropy=0.0, noise_floor_estimate=0.0, predictability_index=0.0,
        ),
        execution_topology=ExecutionTopologySnapshot(
            signal_density=0.0, execution_rate=0.0, fill_ratio=0.0,
            slippage_proxy=0.0, win_rate_proxy=0.0, risk_exposure=0.0,
            rotation_events=0, lock_events=0, migration_events=0,
        ),
        shadow=ShadowSnapshot(
            shadow_alignment=0.0, sof_score=0.0, edge_decay=0.0,
            mirror_divergence=0.0, alpha_transfer_rate=0.0, false_signal_rate=0.0,
            gt_corr=0.0, sy_corr=0.0, stas=0.0,
            winner="NONE", edge_preservation=0.0, execution_efficiency=0.0,
            phase2_enabled=False, samples=0,
        ),
        deployment_reality=DeploymentRealitySnapshot(
            asr=0.0, execution_quality="unknown", mean_slippage_pts=0.0,
            score_trend="flat", classification="unknown",
        ),
        frequency_reality=FrequencyRealitySnapshot(
            blocked_total=0, blocked_profitable=0, leakage_rate="0.0%",
            adr=0.0, classification="unknown",
        ),
        reality_convergence=RealityConvergenceSnapshot(
            ate=0.0, frequency_match="0%", friction_index=0.0,
            health_index=0.0, classification="unknown",
        ),
        dpl_validation=DplValidationSnapshot(
            total_snapshots=0, resolved=0, pct_resolved=0.0,
            symbols=[], regime_distribution={}, has_short_outcomes=False,
        ),
        director_pipeline=DirectorPipelineSnapshot(
            evidence_strength=0.0, research_confidence=0.0,
            deployment_confidence=0.0, alpha_transfer=0.0,
            biggest_risk="none", biggest_strength="none",
            recommendation="hold", classification="unknown",
        ),
        rcl_dashboard=RclDashboardSnapshot(
            h5_resolved=0, h20_resolved=0, h5_wins=0, h20_wins=0,
            h5_win_rate=0.0, h20_win_rate=0.0, divergence=0.0,
        ),
        session_balance=SessionBalanceSnapshot(
            asia=0, london=0, overlap=0, ny=0, dead=0, total=0,
            imbalance=0.0, status="BUILDING",
        ),
        system_health=SystemHealthSnapshot(
            stability_score=0.0, kill_switch_pressure=0.0,
            rollout_progress=0.0, system_integrity=0.0,
            deployment_score=0.0, deployment_classification="unknown",
            deployment_id="dev",
            runtime_hours=0, runtime_minutes=0,
            phase="COLLECTING_EVIDENCE",
        ),
        engine_vector=[0.0] * 32,
    )

    # Round-trip test
    json_str = snapshot_to_json(snap)
    parsed = json.loads(json_str)

    assert parsed["cycle_id"] == 1, f"Expected 1, got {parsed['cycle_id']}"
    assert parsed["account"]["login"] == "123"
    assert len(parsed["engine_vector"]) == 32
    assert parsed["system_health"]["phase"] == "COLLECTING_EVIDENCE"
    assert parsed["rcl_dashboard"]["h5_win_rate"] == 0.0
    assert parsed["shadow"]["winner"] == "NONE"

    print(f"✅ TelemetrySnapshot round-trip OK ({len(json_str)} bytes)")
    print(f"   JSON keys: {list(parsed.keys())}")
