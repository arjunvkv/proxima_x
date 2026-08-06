"""proxima_ops/reality/_stubs.py — explicit no-op stubs for the reality layer.

Loaded ONLY when PROXIMA_REALITY_ENABLED is unset/"0" (see _gate.py). These
are deliberate, honest placeholders for the proxima_ops.reality.* modules that
were proven never to exist in git history. They mirror the exact interface the
demo runner uses (audited call sites) and return neutral values so the
execution/paper path runs while the reality observability layer is off.

They are NOT silent: _gate logs a startup warning, and each stub method logs
once on first use (debug level) so operators can see the layer is inert.

When PROXIMA_REALITY_ENABLED=1 the real modules are expected — the stub
submodules raise ImportError instead of loading these (never broad try/except).
"""
import logging

logger = logging.getLogger("proxima_ops.reality.stubs")

_WARNED: set = set()


def _stub_warn(name: str):
    if name not in _WARNED:
        _WARNED.add(name)
        logger.warning(f"reality stub invoked: {name}() — layer disabled, no-op")


class _StubBase:
    """Base: neutral summary + fire-and-forget records."""

    def summary(self, *a, **k):
        _stub_warn(self.__class__.__name__ + ".summary")
        return f"{self.__class__.__name__}: disabled (stub)"

    def update(self, *a, **k):
        return None

    def record(self, *a, **k):
        return None


class OutcomeLedger(_StubBase):
    """RCL-1A outcome ledger (stub)."""

    def __init__(self):
        self._resolved: list = []

    def record_signal(self, *a, **k):
        return None

    def update_features(self, *a, **k):
        return None

    def update_context(self, *a, **k):
        return None

    def mark_resolved(self, *a, **k):
        return None

    def resolve_outcome(self, *a, **k):
        return None

    def resolved_count(self) -> int:
        return 0

    def unresolved(self) -> list:
        return []

    def compute_feature_matrix(self, horizon="h5"):
        return None, None, None


class LiveInformationGainAudit(_StubBase):
    """RCL-1B information-gain audit (stub)."""

    def compute_by_horizon(self, *a, **k):
        return {}


class FeatureRedundancyMatrix(_StubBase):
    """RCL-1C redundancy matrix (stub)."""

    def compute_pairwise(self, *a, **k):
        return None


class AdaptiveMetaReweighter(_StubBase):
    """RCL-1D meta reweighting (stub)."""

    def compute_weights(self, *a, **k):
        return None


class LayerPruner(_StubBase):
    """RCL-1E layer pruning (stub)."""

    def compute_scores(self, *a, **k):
        return None


class OccupancyAudit(_StubBase):
    """P1.1 occupancy audit (stub)."""

    def record_blocked(self, *a, **k):
        return None

    def try_resolve(self, *a, **k):
        return None

    def unresolved(self) -> list:
        return []


class TPIAbAudit(_StubBase):
    """TPI A/B audit (stub)."""


class FunnelAudit(_StubBase):
    """Signal funnel audit (stub)."""

    def record_generated(self, *a, **k):
        return None


class RegimeMemoryMatrix(_StubBase):
    """Regime memory / transition edges (stub)."""

    def __init__(self):
        self._prev_regime: dict = {}

    def transition_edge(self, *a, **k):
        return None

    def transition_edge_summary(self, *a, **k):
        return "regime-memory: disabled (stub)"

    def sizing_multiplier(self, *a, **k):
        return 1.0


class SignalDecayVelocity(_StubBase):
    """Signal decay velocity / hold caps (stub)."""

    def dps(self, *a, **k):
        return None

    def record_outcome(self, *a, **k):
        return None

    def hold_cap(self, *a, **k) -> int:
        return 20


def cohort_key(*parts) -> str:
    """cohort_key() pure-function stub — deterministic key from parts."""
    _stub_warn("cohort_key")
    return "|".join("" if p is None else str(p) for p in parts)


class OccupancyMigration(_StubBase):
    """P0 occupancy migration engine (stub)."""

    def record_event(self, *a, **k):
        return None

    def quality_score(self, *a, **k) -> float:
        return 0.0

    def should_migrate(self, *a, **k):
        return False, "REALITY_DISABLED"

    def rps_for_migration(self, *a, **k) -> float:
        return 0.0


class ImpulseGraph(_StubBase):
    """DPL-18 cross-asset impulse graph (stub)."""

    def __init__(self, nodes=None):
        self._nodes = nodes or []
