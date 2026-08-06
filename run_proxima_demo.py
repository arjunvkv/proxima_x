"""
PROXIMA OPS — MT5 Demo Deployment Platform
python run_proxima_demo.py

Connects Proxima V2 to MT5 Demo for live deployment validation.

Modes:
  demo       — Full demo deployment (default)
  monitor    — Monitoring only, no trading
  backfill   — Backfill signal ledger from historical data
  report     — Generate daily report

No optimization. No recalibration. No alpha modification.
"""
import sys
import os

# Ensure project root is on path for core_runtime imports
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import math
import random
import logging
from collections import defaultdict
from proxima_ops.decision.gate.mra_signal import MarketRealityAnchor
from proxima_ops.decision.gate.emd_signal import ExecutionMicrostructureDrift
from proxima_ops.decision.gate.recovery_policy import RecoveryPolicy, check_regime_failure
from proxima_ops.decision.gate.phase6_rollout_controller import Phase6RolloutController
from proxima_ops.decision.gate.phase6_kill_switch import Phase6KillSwitch
from proxima_ops.decision.gate.phase6_scaling_engine import Phase6ScalingEngine
from proxima_ops.decision.gate.phase6_recovery_protocol import Phase6RecoveryProtocol
from proxima_ops.decision.gate.phase6_audit_logger import Phase6AuditLogger

# Reconfigure stdout/stderr to UTF-8 to prevent encoding errors on Windows console
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Initialize logging immediately with force=True to override sub-module defaults
import os as _log_os
_log_path = _log_os.environ.get("PROXIMA_LOG_FILE", "proxima_demo.log")
_log_dir = _log_os.path.dirname(_log_path)
if _log_dir:
    _log_os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(_log_path, encoding="utf-8")
    ],
    force=True
)
logger = logging.getLogger("proxima_demo")

# Core runtime integration
from core_runtime.gate_audit_logger import GateAuditLogger, get_gate_audit
from core_runtime.spread_model import SpreadModel, get_spread_model
from core_runtime.position_state_sync import PositionStateSynchronizer, get_position_sync
from core_runtime.execution_lifecycle_manager import ExecutionLifecycleManager, get_lifecycle_manager
from core_runtime.microstructure_calibrator import MicrostructureCalibrator, get_micro_calibrator
from core_runtime.reality_integrity_dashboard import RealityIntegrityDashboard, get_reality_dashboard
from proxima_ops.governance.system_mode_contract import (
    SystemMode, Plane, ExecutionMode, UIMode, MOFPolicy,
    ModeValidator, RuntimeInvariantChecker, ModeSnapshotLogger,
)

# Acceptance mode setup
import tempfile as _tf
ACCEPTANCE_MODE = "--acceptance" in sys.argv
ACCEPTANCE_LOG_PATH = os.path.join(_tf.gettempdir(), "proxima_acceptance_only.log")
if ACCEPTANCE_MODE:
    _ah = logging.FileHandler(ACCEPTANCE_LOG_PATH, encoding="utf-8", mode="w")
    _ah.setLevel(logging.INFO)
    _ah.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    _af = lambda r: any(t in r.getMessage() for t in ("[CYCLE]", "[ENTROPY_RANK]", "[TOPOLOGY_DECISION]", "[TOPOLOGY_TRANSITION]", "[EXHAUST_DETAIL]", "[EXHAUST_AUDIT]", "[EXHAUST_VS_SHADOW]", "[EXHAUST_DBG]", "[EXHAUST_NEAR_MISS]", "[EXHAUST_NEAREST]", "[SHADOW_EXHAUST_OVERRIDE]", "[SHADOW_EXHAUST_RATE]", "[THESIS_BUFFER]", "[THESIS_LABELS]", "[B4_RECORD]", "[B4_RESOLVE]", "[B4_ACCEPT]", "[THESIS_RF_STATUS]", "[THESIS_GRAPH]", "[AUTO_STOP]", "[SHUTDOWN]", "[TOPOLOGY_INIT]", "[TOPOLOGY_DBG]"))
    _ah.addFilter(type('AcceptFilter', (), {'filter': lambda self, r: _af(r)})())
    logger.addHandler(_ah)
    logger.info("[ACCEPTANCE_MODE] enabled — logging CYCLE/ENTROPY_RANK/TOPOLOGY_DECISION/TOPOLOGY_TRANSITION to %s", ACCEPTANCE_LOG_PATH)

SYSTEM_MODE = SystemMode()
_invariant_checker = RuntimeInvariantChecker()
_mode_snapshot_logger = ModeSnapshotLogger(interval=60)

_SHUTDOWN = False
PROXIMA_MAX_CYCLES = int(os.environ.get("PROXIMA_MAX_CYCLES", "0"))


def compute_micro_volatility(symbol: str, rates: list, lookback: int = None) -> float:
    if lookback is None:
        lookback = 20
    if not rates or len(rates) < lookback:
        return 0.0001
    _slice = rates[-lookback:]
    _deltas = []
    for i in range(1, len(_slice)):
        _prev = _slice[i-1].get("close", _slice[i-1].get("open", 0))
        _cur = _slice[i].get("close", _slice[i].get("open", 0))
        _deltas.append(abs(_cur - _prev))
    if not _deltas:
        return 0.0001
    _deltas.sort()
    _med = _deltas[len(_deltas) // 2]
    point_val = 0.01 if "JPY" in symbol else (0.1 if "XAU" in symbol or "XAG" in symbol else 0.0001)
    return max(_med / max(point_val, 1e-9), 0.0001)

import time
import signal
import argparse

# Capture real wall clock before any monkey-patching
_wall_perf_counter = time.perf_counter
from datetime import datetime, date
from typing import Optional, Dict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_VALIDATION_ENABLED = os.environ.get("VALIDATION_ENABLED", "1") == "1"
if _VALIDATION_ENABLED:
    from validation.validation_integration import ValidationIntegration as _ValidationIntegration, OrganismState, DecisionContext, TradeOutcome
else:
    from validation.validation_integration import NullValidationIntegration as _ValidationIntegration, OrganismState, DecisionContext, TradeOutcome

from proxima_ops.config.settings import SETTINGS
from proxima_ops.execution.mt5_connector import MT5Connector
from proxima_ops.execution.order_manager import OrderManager
from proxima_ops.execution.position_manager import PositionManager
from execution.execution_router import ExecutionRouter
from proxima_ops.ledger.trade_ledger import TradeLedger
from proxima_ops.ledger.signal_ledger import SignalLedger
from proxima_ops.ledger.deployment_ledger import DeploymentLedger
from proxima_ops.control.permissions import Permissions
from proxima_ops.control.command_router import CommandRouter
from proxima_ops.control.telegram_bot import TelegramBot
from proxima_ops.monitoring.deployment_score import DeploymentScore
from proxima_ops.monitoring.mt5_monitor import MT5Monitor
from proxima_ops.monitoring.performance_monitor import OpsPerformanceMonitor
from proxima_ops.reporting.daily_report import DailyReport
from proxima_ops.reporting.weekly_report import WeeklyReport
from research.mechanism_discovery.energy_dynamics import EnergyDynamics
from research.mechanism_discovery.temporal_topology import TemporalTopology
from research.ucf.integration.ucf_compute_entry import compute_ucf_field
from research.ucf.integration.ucf_propagation_schema import UCFPropagationField

# TPI Flow Overlay (Layer 7) — Shadow Mode
from layer7.get_tpi_signal import get_tpi_signal, TPI_ELIGIBLE
from layer7.execution_risk_manager import KillSwitch, DrawdownManager, LockManager, risk_gate, PortfolioGraph
from layer7.market_graph import MarketGraph
from layer7.memory_compressor import MemoryCompressor
from layer7.rejection_engine import RejectionEngine, RejectionType
from layer7.metric_normalization import MetricNormalizer
from layer7.observer_collapse import ObserverCollapseModel
from layer7.non_markovian_field import NonMarkovianBeliefField
from layer7.causal_fusion_layer import CausalFusionLayer
from proxima_ops.monitoring.execution_statistics import survivorship_bias_correction
from data.tick_buffer import TickBuffer
from layer7.tpi_outcomes import TPIOutcomeTracker, TPIPersistenceTracker, TPICurvatureTracker
from layer7.tpi_calibration import TPICalibrationLayer
from layer7.propagation import CrossAssetPropagationEngine
from layer7.tick_thermodynamics import TickThermodynamicsEngine
from layer7.rf_gate import LiveRfGate
from layer7.meta_state import MetaStateFusionEngine
from layer7.session_conditional import SessionConditionalEngine, get_session
from layer7.entropy_compression import EntropyCompressionEngine
from mvs.adaptation.observer_decay import ObserverDecayEngine
from mvs.adaptation.weak_day_detector import WeakDayDetector
from mvs.observer.observer_features import (
    normalize_tpi, persistence_ratio_from_streak,
    curvature_strength_from_state, compute_entropy_alignment,
    compute_confidence, state_from_confidence,
)
from proxima_ops.reality.outcome_ledger import OutcomeLedger
from mvs.analysis.battle_decay_exit import BattleDecayExitEngine, get_tpi_threshold
from mvs.analysis.counterfactual_engine import CounterfactualEngine, DecisionBoundaryLog
from proxima_ops.reality.live_ig_audit import LiveInformationGainAudit
from proxima_ops.reality.redundancy_matrix import FeatureRedundancyMatrix
from proxima_ops.reality.meta_reweighter import AdaptiveMetaReweighter
from proxima_ops.reality.layer_pruner import LayerPruner
from proxima_ops.reality.occupancy_audit import OccupancyAudit
from proxima_ops.reality.tpi_ab_audit import TPIAbAudit
from proxima_ops.reality.funnel_audit import FunnelAudit
from proxima_ops.reality.regime_memory import RegimeMemoryMatrix
from proxima_ops.reality.signal_decay import SignalDecayVelocity, cohort_key
from proxima_ops.risk.spread_normalizer import SpreadNormalizer
from proxima_ops.execution.restricted_bridge import (
    RestrictedExecutionBridge,
    format_bridge_dashboard,
)
from proxima_ops.execution.symbol_direction_lock import SymbolDirectionLock
from layer7.types import TPIObservation
from dashboard.tpi_dashboard import generate as tpi_dashboard_generate, record_tpi_shadow as tpi_record_shadow, cache_tpi_signal as tpi_cache_signal
from signals.thesis_buffer import ThesisBuffer
from signals.thesis_rf_trainer import ThesisRfTrainer
from signals.thesis_graph import ThesisGraph

# Observability Imports
from proxima_ops.monitoring.signal_statistics import SignalStatistics
from proxima_ops.monitoring.trigger_statistics import TriggerStatistics
from proxima_ops.monitoring.execution_statistics import ExecutionStatistics, RealityVector, classify_execution_mode
from proxima_ops.monitoring.opportunity_tracker import OpportunityTracker
from proxima_ops.monitoring.global_rank_engine import GlobalRankEngine
from proxima_ops.monitoring.sample_integrity_guard import SampleIntegrityGuard, SUPPRESSED_CLASSIFICATIONS
from proxima_ops.monitoring.research_alignment_monitor import ResearchAlignmentMonitor
from proxima_ops.monitoring.exception_dashboard import ExceptionDashboard
from proxima_ops.monitoring.deployment_dashboard import DeploymentDashboard
from proxima_ops.monitoring.signal_funnel import SignalFunnel
from proxima_ops.monitoring.lifecycle_audit import LifecycleAudit
from proxima_ops.monitoring.deployment_context import DeploymentContext
from proxima_ops.risk.market_observability_filter import (
    MarketObservabilityFilter,
    ObservabilityState,
)
from proxima_ops.monitoring.order_tracker import OrderTracker
from proxima_ops.monitoring.trade_reconciler import TradeReconciler
from proxima_ops.monitoring.funnel_dashboard import FunnelDashboard
from research.frequency_reality.blocked_signal_tracker import BlockedSignalTracker
from research.frequency_reality.executed_signal_tracker import ExecutedSignalTracker
from research.frequency_reality.delayed_outcome_engine import DelayedOutcomeEngine
from research.frequency_reality.frequency_cost_analysis import FrequencyCostAnalysis
from research.frequency_reality.frequency_classifier import FrequencyClassifier
from research.frequency_reality.frequency_pipeline import FrequencyRealityPipeline
from research.frequency_reality.dpl_live_validation import DPLLiveValidation
from research.deployment_reality.latency_analysis import LatencyAnalysis
from research.deployment_reality.spread_reality import SpreadReality
from research.deployment_reality.signal_decay import SignalDecay
from research.deployment_reality.execution_quality import ExecutionQuality
from research.deployment_reality.blocked_vs_executed import BlockedVsExecuted
from research.deployment_reality.drawdown_forensics import DrawdownForensics
from research.deployment_reality.regime_reality import RegimeReality
from research.deployment_reality.deployment_classifier import DeploymentClassifier
from research.deployment_reality.deployment_pipeline import DeploymentRealityPipeline
from research.reality_convergence.expectation_engine import ExpectationEngine
from research.reality_convergence.reality_engine import RealityEngine
from research.reality_convergence.divergence_detector import DivergenceDetector
from research.reality_convergence.convergence_tracker import ConvergenceTracker
from research.reality_convergence.alpha_transfer import AlphaTransfer
from research.reality_convergence.operational_friction import OperationalFriction
from research.reality_convergence.deployment_health import DeploymentHealth
from research.reality_convergence.anomaly_detector import AnomalyDetector
from research.reality_convergence.reality_classifier import RealityClassifier
from proxima_ops.reality.occupancy_migration import OccupancyMigration
from proxima_ops.reality.impulse_graph import ImpulseGraph
from research.reality_convergence.reality_pipeline import RealityConvergencePipeline
from research.autonomous_director.director_pipeline import DirectorPipeline
from proxima_ops.risk.risk_manager import RiskManager
from proxima_ops.risk.catastrophic_stop import catastrophic_sl, catastrophic_tp, pip_distance
from features.ecdf_transform import PerSymbolECDF
from engine.ranking_engine import RankingEngine
from signals.outcome_surface_signal import OutcomeSurfaceSignal
from signals.transition_oss import TransitionOSS
from signals.sal_mapper import SignalAggregationLayer
from engine.topk_rotation_engine import TopKRotationEngine
from risk.h20_cap_engine import H20CapEngine
from execution.execution_mapper import ExecutionMapper
from fusion_kernel.fusion_kernel import SignalFusionKernel
from evaluation.delayed_outcome_engine import DelayedOutcomeEngine as V4DelayedOutcomeEngine
from learning.afl_engine import AFLFeedbackEngine
from learning.fwo_engine import FeatureWeightOptimizer
from learning.rsl_engine import RegimeSegmentedLearning
from regime.rtd_engine import RegimeTransitionDetector
from learning.tca_engine import TemporalCreditAssignment
from learning.cwf_engine import CausalWeightFusion
from monitoring.cdm_engine import ConsensusDriftMonitor
from learning.drl_engine import DriftResolutionLayer
from stability.mso_engine import MetaStabilityOptimizer
from stability.lct_engine import LongHorizonConvergenceTracker
from meta.ssol_engine import SystemSelfOptimizationLoop
from causal.cal_engine import CALEngine
from validation.wfv_engine import WalkForwardValidator, StatisticalEdgeTest
from validation.svr_engine import SystemValidationReduction
from analysis.edge_trace import EdgeTraceAnalyzer
from research.layer_config import LayerConfig
from proxima_ops.decision.shadow_mirror import ShadowDecisionMirror
from proxima_ops.decision.shadow_execution_engine import ShadowExecutionOrchestrator
from proxima_ops.decision.shadow_engine.shadow_core import ShadowCore
from proxima_ops.decision.shadow_engine.shadow_worker import ShadowWorker

RUNNING = True
MIN_HOLD_TICKS_FLIP = 12
MIN_HOLD_TICKS_MIGRATION = 20
MAX_HOLD_TICKS = 200  # thesis expiry: force-close if held longer (increased from 48 to allow BattleDecay warmup)
EXPLORATION_TTL = 24  # fixed observation window for exploration impulse response (ticks)
MICRO_VOL_LOOKBACK = 20  # ticks for micro-volatility baseline computation


def signal_handler(sig, frame):
    global RUNNING, _SHUTDOWN
    logger.info("[SHUTDOWN] Signal %s received", sig)
    RUNNING = False
    _SHUTDOWN = True


class DashboardLogHandler(logging.Handler):
    def __init__(self, demo_instance):
        super().__init__()
        self.demo = demo_instance

    def emit(self, record):
        try:
            msg = record.getMessage()
            self.demo.add_activity(msg)
        except Exception:
            self.handleError(record)


class SymbolTrustModel:
    """Online adaptive trust per symbol. Replaces static priors with outcome-driven Bayesian-style EMA."""

    def __init__(self, alpha: float = 0.08, prior: float = 1.0):
        self.alpha = alpha
        self.prior = prior
        self.trust = defaultdict(lambda: prior)
        self.observations = defaultdict(int)

    def get(self, symbol: str) -> float:
        return self.trust[symbol]

    def update(self, symbol: str, pnl: float):
        reward = 1.0 if pnl > 0 else 0.0
        current = self.trust[symbol]
        updated = (1 - self.alpha) * current + self.alpha * reward
        self.trust[symbol] = min(1.2, max(0.05, updated))
        self.observations[symbol] += 1


import threading as _threading


class TickCache:
    def __init__(self, mt5, symbols, poll_interval=0.2):
        self._mt5 = mt5
        self._symbols = list(symbols)
        self._poll_interval = poll_interval
        self._cache = {}
        self._lock = _threading.Lock()
        self._running = True
        self._thread = _threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self):
        while self._running:
            for sym in self._symbols:
                try:
                    tick = self._mt5.get_tick(sym)
                    if tick:
                        with self._lock:
                            self._cache[sym] = tick
                except Exception:
                    pass
            _threading.Event().wait(self._poll_interval)

    def get_tick(self, sym):
        with self._lock:
            return self._cache.get(sym)

    def get_all(self):
        with self._lock:
            return dict(self._cache)

    def stop(self):
        self._running = False


class ProximaDemo:
    def __init__(self, env=None, layer_config: LayerConfig = None,
                 tick_source=None, broker=None, replay_mode=False):
        self._env = env
        self.config = layer_config or LayerConfig()
        self._clock = getattr(env, 'clock', None) if env else None
        self._metrics = getattr(env, 'metrics', None) if env else None
        self._runtime_limit = 0
        self._tick_limit = 0
        self._warmup_ticks = 0
        self._replay_feed = getattr(env, 'replay_feed', None) if env else None
        self._warmup_mode = False
        self._replay_mode = replay_mode or (env is not None)
        self.tick_source = tick_source or (getattr(env, 'tick_source', None) if env else None)
        self._broker = broker or (getattr(env, 'broker', None) if env else None)

        self._activity_log = []
        logging.getLogger().addHandler(DashboardLogHandler(self))

        # Per-symbol execution gate — collapse all computation to this set
        self._execution_symbols = ["EURUSD"]  # Single symbol for validation mode
        # In acceptance mode, keep 4 symbols
        if os.environ.get("VALIDATION_MODE"):
            self._execution_symbols = ["EURUSD"]
        elif ACCEPTANCE_MODE:
            self._execution_symbols = ["EURUSD", "USDJPY", "EURJPY", "AUDJPY"]
        else:
            self._execution_symbols = list(SETTINGS.symbols)
        self._execution_universe = list(self._execution_symbols)
        self._observation_universe = list(set(SETTINGS.symbols + SETTINGS.shadow_symbols))
        self._shadow_set = set(SETTINGS.shadow_symbols)
        logger.info(f"EXECUTION_MODE: single_symbol={len(self._execution_symbols) == 1} cycle_est={len(self._execution_symbols)}s")

        # P3.2: Boot-time integrity audit — abort if M5 data is corrupt
        from proxima_ops.bootstrap.integrity_audit import run_integrity_audit, audit_summary
        # Skip VPL check (heavy, ~20s per symbol); VPL is evaluated live anyway
        audit = run_integrity_audit(skip_vpl=True)
        print(audit_summary(audit))
        if audit["status"] == "ABORT":
            raise RuntimeError(f"Boot integrity audit FAILED — M5 data corrupt. Cannot start.")
        self._boot_audit = audit
        print("[CONSTRUCTOR DEBUG] Boot audit done, creating MT5 connector...", flush=True)

        if env and hasattr(env, 'broker') and env.broker is not None:
            from core.adapters.replay_mt5_connector import ReplayMT5Connector
            self.mt5 = ReplayMT5Connector(
                tick_source=env.tick_source,
                clock=env.clock,
                broker=env.broker,
            )
        else:
            self.mt5 = MT5Connector()
        print("[CONSTRUCTOR DEBUG] MT5Connector created", flush=True)
        self._tick_cache = None
        if ACCEPTANCE_MODE:
            self._tick_cache = TickCache(self.mt5, self._observation_universe, poll_interval=0.5)
            self.mt5._tick_cache = self._tick_cache
            print("[CONSTRUCTOR DEBUG] TickCache created for acceptance mode", flush=True)
        self._bootstrap = None
        # Bootstrap market seed data from MT5 parquet at engine start
        print("[CONSTRUCTOR DEBUG] Starting MarketSeedLoader.seed_all...", flush=True)
        try:
            from bootstrap.market_seed import MarketSeedLoader
            loader = MarketSeedLoader(self.mt5)
            self._bootstrap = loader.seed_all(self._observation_universe, bars=1440)
        except Exception as e:
            logger.warning(f"[BOOTSTRAP] failed: {e}")
            self._bootstrap = {}
        print(f"[CONSTRUCTOR DEBUG] MarketSeedLoader done, bootstrap has {len(self._bootstrap)} symbols", flush=True)
        print("[CONSTRUCTOR DEBUG] Creating OrderManager...", flush=True)
        self._sdl = SymbolDirectionLock()
        self.orders = OrderManager(self.mt5, symbol_direction_lock=self._sdl)
        # Restricted Execution Bridge — gated execution firewall
        self._execution_bridge = RestrictedExecutionBridge(order_manager=self.orders)
        print("[CONSTRUCTOR DEBUG] Creating PositionManager...", flush=True)
        self.positions = PositionManager(self.mt5)
        # Shared TickBuffer for replay mode — fed from replay feed during tick dispatch
        self._replay_tpi_buf = TickBuffer() if self._replay_mode else None
        print("[CONSTRUCTOR DEBUG] Creating ledgers...", flush=True)
        self.trade_ledger = TradeLedger()
        self.signal_ledger = SignalLedger()
        self.deployment_ledger = DeploymentLedger()
        print("[C1]", flush=True)
        self._cycle_id: int = 0
        self._cycle_clock = None
        print("[C2]", flush=True)
        self.score = DeploymentScore()
        self.mt5_monitor = MT5Monitor(self.mt5)
        self.perf = OpsPerformanceMonitor()
        self.ed = EnergyDynamics()
        self.tt = TemporalTopology()
        print("[C3]", flush=True)
        self._price_history = {}  # sym -> list of H1 rates
        self._m5_history = {}     # sym -> list of M5 rates (for bar elapsed computation)
        self._active_positions_metadata = {}  # ticket -> {"entry_bar_idx": int, "symbol": str}
        self._load_active_positions_metadata()
        print("[C4]", flush=True)
        self._current_bar_idx = {}  # sym -> int (tracks current bar index)
        self._position_lock_until = {}  # broker_symbol -> int (remaining lock bars)
        self._position_tick_age = {}  # ticket -> int (ticks held for flip prevention)
        self._thesis_buffer = ThesisBuffer()
        self._thesis_graph = ThesisGraph(self._thesis_buffer)
        self._thesis_trainer = ThesisRfTrainer()
        self._flip_cooldown = {}  # broker_symbol -> int (ticks before next flip allowed)
        self._funnel_failures = {}  # reason -> int counter for funnel attribution
        self._reinforcement_blocks = 0  # P8: same-direction position exists blocks
        self._flip_blocks = 0  # P8: sign flip blocks (not yet flipped)
        self._rotation_event_count = 0
        self._lock_event_count = 0
        self._migration_event_count = 0  # observation only — no trading decisions
        self._top3_history = []  # list of (timestamp, top3_list) for rotation tracking
        self._last_top3_qualified = []
        # Gate audit logger — tracks every rejection with full context
        self._gate_audit_logger = get_gate_audit()
        # Spread model — probabilistic filter replacing hard cutoff
        self._spread_model = get_spread_model(mode="sigmoid")
        # Position state synchronizer — syncs position state with MT5 every cycle
        self._position_sync = get_position_sync(self.mt5)
        # Execution lifecycle manager — tracks full signal lifecycle
        self._lifecycle_mgr = get_lifecycle_manager(h20_bars=20)
        # Microstructure calibrator — measures model vs broker reality
        self._micro_calibrator = get_micro_calibrator()
        # Reality integrity dashboard — tracks SPR, GFR, ECS, MAI
        self._reality_dashboard = get_reality_dashboard()
        # Wire components together
        self._reality_dashboard.register_micro_calibrator(self._micro_calibrator)
        self._budget_block_ttl: dict[str, int] = {}  # sym -> cycle_id when block expires
        self._pending_close: dict[str, tuple] = {}  # broker_sym -> (ticket, cycle_id) for atomic close before re-entry
        self._symbol_trust = SymbolTrustModel()
        self._spinners = ["/", "-", "\\", "|"]
        self._spinner_idx = 0

        # P0.24: Reconciliation event log for causal tracking
        # P0.26: Restart quarantine — reduced risk after restart for N cycles
        self._quarantine_cycles_remaining = 20  # ~20 minutes at 60s/cycle
        self._has_run_full_cycle = False  # becomes True after first complete 60s cycle

        # P0.25: Idempotency guard — tracks which tickets have been fed back to SymbolTrust,
        # preventing duplicate BD feedback injection after restart recovery
        self._applied_feedback_tickets: set[int] = set()

        # Exploration mode state
        self._exploration_cooldown: dict[str, int] = {}  # sym -> cycles until next exploration
        self._exploration_dispatch: set = set()

        # P0.14: Canonical per-symbol TPI buffer — single source of truth for all TPI consumers
        self._canonical_tpi_buffer = TickBuffer()  # per-symbol ring buffer, fed from tick dispatch

        self._reconciliation_events: list[dict] = []

        # Causal Execution Gate — SHADOW mode (observability only, no blocking)
        self._gate_mra = MarketRealityAnchor()
        self._gate_emd = ExecutionMicrostructureDrift()
        self._gate_recovery = RecoveryPolicy()
        self._gate_decisions: list[dict] = []

        # Phase 6 — Deployment Control System (SHADOW mode initially)
        self._phase6_rollout = Phase6RolloutController()
        self._phase6_killswitch = Phase6KillSwitch()
        self._phase6_scaling = Phase6ScalingEngine()
        self._phase6_recovery = Phase6RecoveryProtocol()
        self._phase6_audit = Phase6AuditLogger()
        self._phase6_log: list[dict] = []
        self._phase6_current_mult: float = 1.0

        # BattleDecay v7.2 exit engine (all symbols)
        self._battle_decay = self._create_battle_decay_engine()
        self._bd_feed_buffer: dict[int, list] = {}  # ticket -> [(bid, ask, ts)]

        # TPI Layer 7 — Tick→Bar outcome tracker
        self._tpi_tracker = TPIOutcomeTracker()
        self._tpi_persistence = TPIPersistenceTracker()
        # TPI inversion exit state: ticket -> {direction, entry_tpi, inversion_count, symbol}
        self._exit_state = {}
        self._max_hold_ticks = 5000
        self._tpi_curvature = TPICurvatureTracker()
        self._pyramid_log: list[dict] = []
        self._pyramid_count: dict[str, int] = {}
        self._tpi_propagation = CrossAssetPropagationEngine()
        self._tick_thermo = TickThermodynamicsEngine()
        _rf_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research", "models", "edge_state_rf_oos.joblib")
        self._rf_gate = LiveRfGate(model_path=_rf_model_path if os.path.exists(_rf_model_path) else None)
        self._meta_fusion = MetaStateFusionEngine()
        self._session_cond = SessionConditionalEngine()
        self._entropy_compression = EntropyCompressionEngine()
        # Market Observability Filter (MOF) — pre-perception gate
        self._mof = MarketObservabilityFilter()
        self._mof_blocked = False
        self._last_mof_state = ObservabilityState.INFORMATION_RICH.value
        self._topology_prev: Dict[str, str] = {}
        self._ecdf = PerSymbolECDF(window_size=5000)

        # Pre-seed ECDF from bootstrap data for all symbols (bulk hydration)
        if self._bootstrap:
            for sym, seed in self._bootstrap.items():
                closes = seed.get("closes", [])
                n_seeded = 0
                self._ecdf.hydrate(sym, closes)
                n_seeded = len([p for p in closes if p > 0])
                if n_seeded > 0:
                    logger.info(f"[BOOTSTRAP] {sym}: ECDF bulk-seeded={n_seeded} bars ecdf_rank={seed.get('ecdf_seed',0.5)}")

        # Production signal path (research-validated, replaces V3/V4 adaptive stack)
        from bootstrap.oss_bootstrap_trainer import OSSBootstrapTrainer
        self._oss_bootstrap = OSSBootstrapTrainer()
        self._sal = SignalAggregationLayer(
            entry_threshold=0.50, accum_window=100, accum_decay=0.977, consensus_min=0.60)
        self._trained_oss = False

        # Bootstrap-train OSS from MT5 parquet data for immediate paper trading
        if self._bootstrap:
            try:
                result = self._oss_bootstrap.train_all(self._bootstrap)
                if result["trained"]:
                    self._trained_oss = True
                    logger.info(f"[BOOTSTRAP OSS] trained=TRUE — paper trading enabled immediately")
                else:
                    logger.warning(f"[BOOTSTRAP OSS] training failed: {result}")
            except Exception as e:
                logger.error(f"[BOOTSTRAP OSS] training error: {e}")
        else:
            logger.warning("No bootstrap data available — OSS training deferred")

        # OSS expected-value ranking (replaces ECDF+entropy ranking)
        self._ranking = RankingEngine(oss=self._oss_bootstrap.get_oss())
        self._ranked_symbols = []
        self._top_k = min(8, max(3, len(SETTINGS.symbols)//4))
        self._active_symbols = set()
        self._cycle_execution_set = set()
        self._dir_stats = {"BUY_TRIGGER": 0, "SELL_TRIGGER": 0, "BUY_SUBMIT": 0, "SELL_SUBMIT": 0, "BUY_ACCEPT": 0, "SELL_ACCEPT": 0}
        self._top3_sticky_cycles = {}  # sym -> consecutive cycles in TOP3
        self._top3_cooldown = {}       # sym -> remaining cooldown cycles
        self._rotation = TopKRotationEngine(
            top_k=self._top_k,
            min_margin=0.03,
            persistence=3,
        )
        self._h20 = H20CapEngine(max_cap_per_symbol=0.60, min_cap_per_symbol=0.10)
        self._allocations = {}
        self._exec_mapper = ExecutionMapper(
            base_lot=1.0,
            max_lot=5.0,
            min_lot=0.01,
            risk_per_unit=1000.0,
        )
        self._execution_plan = {}
        self._fusion = SignalFusionKernel(
            entropy_flip_threshold=0.65,
            coherence_penalty=0.15,
        )
        self._doa = V4DelayedOutcomeEngine(horizon_ticks=20)
        self._afl = AFLFeedbackEngine(
            learning_rate=0.05,
            entropy_sensitivity=1.0,
            rotation_sensitivity=1.0,
        )
        self._cal = CALEngine()
        self._fwo = FeatureWeightOptimizer(lr=0.05)
        self._rsl = RegimeSegmentedLearning(
            base_weights={"ecdf": 0.40, "entropy": 0.35, "spread": 0.15, "signal": 0.10},
        )
        self._rtd = RegimeTransitionDetector(
            enter_threshold=0.65,
            exit_threshold=0.55,
            min_persistence=3,
        )
        self._tca = TemporalCreditAssignment(decay=0.85, max_history=50)
        self._cwf = CausalWeightFusion(cal_weight=0.5, tca_weight=0.5)
        self._cdm = ConsensusDriftMonitor(drift_threshold=0.75)
        self._drl = DriftResolutionLayer(base_cal_weight=0.5, base_tca_weight=0.5)
        self._mso = MetaStabilityOptimizer(window=10, oscillation_threshold=0.25)
        self._lct = LongHorizonConvergenceTracker(window=50)
        self._ssol = SystemSelfOptimizationLoop()
        self._wfv = WalkForwardValidator(train_size=200, test_size=50)
        self._svr = SystemValidationReduction()
        self._wfv_records: list[dict] = []
        self._edge_trace = EdgeTraceAnalyzer()
        # P1.2: Persistent deployment_id and deterministic A/B arm
        import uuid
        depl_dir = os.path.join(os.getcwd(), "state")
        os.makedirs(depl_dir, exist_ok=True)
        depl_path = os.path.join(depl_dir, "deployment_id.txt")
        if os.path.exists(depl_path):
            with open(depl_path, "r") as f:
                self.deployment_id = f.read().strip()
        else:
            self.deployment_id = uuid.uuid4().hex[:8]
            with open(depl_path, "w") as f:
                f.write(self.deployment_id)
        self._tpi_arm = "SOFT_SCORE"
        logger.info(f"[TPI_AB] Deployment arm = {self._tpi_arm} (deployment_id={self.deployment_id})")
        self._tpi_calibration = TPICalibrationLayer(mode=self._tpi_arm)
        self._tpi_ab_audit = TPIAbAudit()
        self._funnel_audit = FunnelAudit()
        self._regime_memory = RegimeMemoryMatrix()
        self._signal_decay = SignalDecayVelocity()
        self._spread_normalizer = SpreadNormalizer()
        self._cycle_tpi_snapshot: dict[str, dict] = {}
        self._cycle_context: dict = {}
        self._outcome_ledger = OutcomeLedger()
        self._occupancy_audit = OccupancyAudit()

        # Phase E — Observer Validation & Promotion Framework
        self._validation = _ValidationIntegration()
        _vmods = len(self._validation._module_names) if hasattr(self._validation, '_module_names') else 0
        logger.info(f"[VALIDATION] {'enabled' if _VALIDATION_ENABLED else 'disabled'} — {_vmods} observer modules")
        self._ig_audit = LiveInformationGainAudit()
        self._redundancy_matrix = FeatureRedundancyMatrix()
        self._meta_reweighter = AdaptiveMetaReweighter()
        self._layer_pruner = LayerPruner()
        self._impulse_graph = ImpulseGraph(nodes=TPI_ELIGIBLE)
        self._session_balance: dict[str, int] = {s: 0 for s in ["ASIA", "LONDON", "OVERLAP", "NY", "DEAD"]}
        self._last_meta_scores: dict[str, dict] = {}
        # Preload tick data for thermodynamics engine — ALL symbols
        print(f"[CONSTRUCTOR DEBUG] Subscribing to ticks for {len(self._observation_universe)} symbols...", flush=True)
        # Access raw MetaTrader5 module for symbol_select subscription
        try:
            import MetaTrader5 as _raw_mt5
        except ImportError:
            _raw_mt5 = None
        _sub_count = 0
        for sym in self._observation_universe:
            try:
                if _raw_mt5 is not None:
                    _raw_mt5.symbol_select(sym, True)
                    _ = _raw_mt5.symbol_info_tick(sym)
                    _sub_count += 1
            except Exception:
                pass
        print(f"[CONSTRUCTOR DEBUG] Subscribed {_sub_count}/{len(self._observation_universe)} symbols", flush=True)

        print(f"[CONSTRUCTOR DEBUG] Loading tick thermo for {len(self._observation_universe)} symbols...", flush=True)
        _thermo_loaded = 0
        for sym in self._observation_universe:
            try:
                self._tick_thermo.load_offline(sym)
                _thermo_loaded += 1
            except Exception:
                pass
        print(f"[CONSTRUCTOR DEBUG] Loaded thermo data for {_thermo_loaded}/{len(self._observation_universe)} symbols", flush=True)

        # Pre-seed RF gate rolling buffer from MT5 historical tick data
        # This fills the 2000-tick window immediately so the RF gate is ready from cycle 1
        print(f"[CONSTRUCTOR DEBUG] Pre-seeding RF gate for {len(self._execution_symbols)} symbols...", flush=True)
        self._preseed_rf_gate()
        print(f"[CONSTRUCTOR DEBUG] RF gate pre-seed complete", flush=True)

        # Observability Initialization
        self._start_time = time.time()
        self._ctx = DeploymentContext()
        self._global_rank_engine = GlobalRankEngine()
        self._sample_guard = SampleIntegrityGuard()
        self._alignment_monitor = ResearchAlignmentMonitor()
        self._exception_dashboard = ExceptionDashboard()
        self.sig_stats = SignalStatistics()
        stats_file = os.path.join(os.path.dirname(SETTINGS.db_path), "observability_stats.json")
        self.trig_stats = TriggerStatistics(save_path=stats_file)
        self.exec_stats = ExecutionStatistics()
        # Phase 2: Counterfactual engine for lifecycle reversal learning
        self._cf_engine = CounterfactualEngine()
        # Phase 3.2: CF hard gating state
        self._cf_gate_block_counter = 0
        self.opp_tracker = OpportunityTracker()
        self.dashboard = DeploymentDashboard(self._start_time, self.trig_stats, self.opp_tracker,
                                             deployment_context=self._ctx)
        funnel_file = os.path.join(os.path.dirname(SETTINGS.db_path), "funnel_stats.json")
        self.funnel = SignalFunnel(save_path=funnel_file, deployment_context=self._ctx)
        self.audit = LifecycleAudit(self.funnel)
        self.order_tracker = OrderTracker()
        self.reconciler = TradeReconciler(self.trade_ledger, self.positions)
        self._ard = DirectorPipeline()
        self._risk = RiskManager()
        self._risk.set_position_manager(self.positions)
        self.execution_router = ExecutionRouter(self._risk, self.positions, self.orders,
                                                 symbol_direction_lock=self._sdl)
        self._migration = OccupancyMigration()
        self.observer_decay = ObserverDecayEngine(half_life_ticks=300, execute_threshold=0.55)
        self.weak_day = WeakDayDetector()
        self._observer_seq: dict[str, int] = {}
        self._active_signal: dict[str, str] = {}

        # Frequency Reality Audit — Delayed Outcome Engine (true forward returns)
        def _price_for_outcome(sym: str) -> float:
            tick = self.tick_source.next_tick(sym) if self.tick_source else (self._tick_cache.get_tick(sym) if self._tick_cache else self.mt5.get_tick(sym))
            return tick["ask"] if tick else 0.0

        self._freq_blocked = BlockedSignalTracker(deployment_context=self._ctx)
        self._freq_executed = ExecutedSignalTracker(deployment_context=self._ctx)
        self._freq_future = DelayedOutcomeEngine(
            price_provider=_price_for_outcome,
            horizons=[20, 50, 100])
        self._freq_analysis = FrequencyCostAnalysis(self._freq_blocked, self._freq_executed)
        self._freq_classifier = FrequencyClassifier(self._freq_analysis)
        self._dpl_live = DPLLiveValidation()
        self._freq_pipeline = FrequencyRealityPipeline(
            self._freq_blocked, self._freq_executed,
            self._freq_future, self._freq_analysis, self._freq_classifier,
            dpl_live=self._dpl_live)

        # Deployment Reality Lab
        self._drl_latency = LatencyAnalysis()
        self._drl_spread = SpreadReality()
        self._drl_decay = SignalDecay()
        self._drl_exec_q = ExecutionQuality()
        self._drl_bve = BlockedVsExecuted(self._freq_blocked, self._freq_executed)
        self._drl_dd = DrawdownForensics()
        self._drl_regime = RegimeReality()
        self._drl_classifier = DeploymentClassifier(
            self.perf, self.score, self._drl_latency, self._drl_spread,
            self._drl_decay, self._drl_exec_q, self._drl_bve,
            self._drl_dd, self._drl_regime)
        self._drl_pipeline = DeploymentRealityPipeline(
            self._drl_latency, self._drl_spread, self._drl_decay,
            self._drl_exec_q, self._drl_bve, self._drl_dd,
            self._drl_regime, self._drl_classifier)

        # Reality Convergence Engine
        self._rce_exp = ExpectationEngine()
        self._rce_real = RealityEngine(
            perf_monitor=self.perf, deployment_score=self.score,
            freq_reality=self._freq_blocked,
            executed_reality=self._freq_executed,
            drl_module=self._drl_exec_q, mt5_monitor=self.mt5_monitor)
        self._rce_real.set_start_time(self._start_time)
        self._rce_div = DivergenceDetector(self._rce_exp, self._rce_real)
        self._rce_conv = ConvergenceTracker(self._rce_exp, self._rce_real)
        self._rce_ate = AlphaTransfer(self._rce_exp, self._rce_real)
        self._rce_friction = OperationalFriction(
            self._drl_exec_q.summary() if hasattr(self._drl_exec_q, 'summary') else {},
            self._freq_blocked, self._rce_real)
        self._rce_health = DeploymentHealth(
            alpha_transfer=self._rce_ate, friction=self._rce_friction,
            convergence=self._rce_conv, divergence=self._rce_div)
        self._rce_anomaly = AnomalyDetector(
            perf_monitor=self.perf, mt5_monitor=self.mt5_monitor,
            freq_reality=self._freq_blocked)
        self._rce_classifier = RealityClassifier(
            alpha_transfer=self._rce_ate, divergence=self._rce_div,
            friction=self._rce_friction, health=self._rce_health,
            anomaly=self._rce_anomaly)
        self._rce_pipeline = RealityConvergencePipeline(
            self._rce_exp, self._rce_real, self._rce_div,
            self._rce_conv, self._rce_ate, self._rce_friction,
            self._rce_health, self._rce_anomaly, self._rce_classifier)

        self.funnel_dash = FunnelDashboard(self.funnel, self.reconciler, self.score, self.perf,
                                           freq_classifier=self._freq_classifier,
                                           drl_classifier=self._drl_classifier,
                                           rce_pipeline=self._rce_pipeline,
                                           director=self._ard,
                                           dpl_live=self._dpl_live)

        self._permissions = Permissions()
        self._router = CommandRouter(self._permissions)
        self._setup_commands()
        self.telegram = TelegramBot(self._router)

        self._daily_report = DailyReport(
            self.trade_ledger, self.signal_ledger, self.deployment_ledger,
            self.score, self.perf, self.mt5_monitor)
        self._weekly_report = WeeklyReport(
            self.trade_ledger, self.signal_ledger, self.deployment_ledger,
            self.score, self.perf)

        self._last_daily_report = None
        self._last_weekly_report = None
        self._last_deployment_snapshot = None
        self._paused = False
        # Wave 1: Thread-safe position registration
        self._position_registry_lock = None
        import threading
        self._position_registry_lock = threading.Lock()
        # Wave 3: Execution + Portfolio Safety
        self.kill_switch = KillSwitch()
        self.drawdown_manager = DrawdownManager()
        self.lock_manager = LockManager()
        # Wave 4: Cross-symbol + memory + portfolio coupling
        self.market_graph = MarketGraph()
        self.memory_compressor = MemoryCompressor()
        self.portfolio_graph = PortfolioGraph()
        # Wave 5: Rejection semantics
        self.rejection_engine = RejectionEngine()

        # Shadow Decision Mirror — non-intrusive parallel DecisionGate
        self._shadow_mirror = ShadowDecisionMirror(
            max_positions=SETTINGS.max_positions_active,
            min_hold_ticks_flip=5,
            min_hold_ticks_migration=10,
        )
        self._shadow_orchestrator = ShadowExecutionOrchestrator()

        # Ground-truth shadow — parallel observer capturing real per-layer state
        self._shadow_gt = ShadowCore()
        self._shadow_gt_worker = ShadowWorker(self._shadow_gt, "state/shadow_gt_trace.jsonl")

        # STR-E — Shadow Truth Reconciliation Engine
        from proxima_ops.decision.shadow_engine.stre.stre_engine import STREngine, STRECoordinator
        from proxima_ops.decision.shadow_engine.phase2.pipeline_gt import GTSuppressionTracker, CounterfactualConvictionGT
        from proxima_ops.decision.shadow_engine.sof.objective import evaluate_system
        self._sof_evaluate = evaluate_system
        self._stre_engine = STREngine.load(window=100)
        self._stre_coordinator = STRECoordinator(self._stre_engine)
        if len(self._stre_engine.pnl_buffer) > 0:
            logger.info(f"[STR-E] Loaded {len(self._stre_engine.pnl_buffer)} persisted samples")
        self._gt_suppression = GTSuppressionTracker(window=100)
        self._gt_cf = CounterfactualConvictionGT()

    def _create_battle_decay_engine(self):
        return BattleDecayExitEngine()

    def _compute_observer_state(self, tpi_confidence: float, persistence_streak: int,
                                 curvature_state: str, normalized_entropy: float) -> dict:
        _ntpi = normalize_tpi(tpi_confidence)
        _pers = persistence_ratio_from_streak(persistence_streak)
        _curv = curvature_strength_from_state(curvature_state)
        _ent = compute_entropy_alignment(normalized_entropy, max_entropy=1.0)
        confidence = compute_confidence(_ntpi, _pers, _curv, _ent)
        state = state_from_confidence(confidence)
        return {"observer_state": state, "observer_confidence": float(confidence),
                "reality_score": min(1.0, max(0.0, confidence + 0.1))}

    def _freq_rates_provider(self, symbol: str) -> list:
        return self.mt5.get_rates(symbol, count=150, timeframe="H1")

    def _bars_elapsed(self, entry_bar_time, symbol) -> int:
        if not entry_bar_time:
            return -1
        elapsed_secs = self._now_ts() - entry_bar_time
        # entry_bar_time may be hour-floored (error up to ±30 min ~6 M5 bars);
        # time-based monotonic increment is correct for H5/H20 exit logic
        return max(0, int(elapsed_secs // 300))

    def _atomic_write_json(self, path: str, data) -> None:
        """P0.27: Atomic JSON write — temp file + fsync + atomic rename."""
        import json, os
        try:
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception as e:
            logger.error(f"Atomic write failed for {path}: {e}")

    def _save_active_positions_metadata(self):
        meta_file = os.path.join(os.path.dirname(SETTINGS.db_path), "active_positions_metadata.json")
        self._atomic_write_json(meta_file, self._active_positions_metadata)
        # P0.25: Persist idempotency guard
        guard_file = os.path.join(os.path.dirname(SETTINGS.db_path), "applied_feedback_tickets.json")
        self._atomic_write_json(guard_file, list(self._applied_feedback_tickets))

    def _update_symbol_trust_from_bd(self, ticket: int, symbol: str) -> None:
        """P0.10: Feed BattleDecay exit evidence back into SymbolTrust entry priors."""
        # P0.25: Idempotency guard — each ticket applied exactly once
        if ticket in self._applied_feedback_tickets:
            return
        self._applied_feedback_tickets.add(ticket)
        if not hasattr(self, '_battle_decay') or not self._battle_decay:
            return
        summary = self._battle_decay.get_exit_quality(ticket)
        if summary is None:
            return
        # BD quality: positive when FF dominates (favorable path), negative when EF dominates (adverse)
        ff = summary.get("ff_total", 0.0)
        ef = summary.get("ef_total", 0.0)
        total = ff + ef
        if total > 1e-9:
            ratio = (ff - ef) / total  # +1 = perfectly favorable, -1 = perfectly adverse
            # Scale to [-1, 1] and dampen by continuity
            cs = summary.get("mean_continuity", 1.0)
            quality = ratio * cs * 0.5  # max adjustment of ±0.5 to trust
            current_trust = self._symbol_trust.get(symbol)
            adjusted = current_trust + quality * 0.3  # 30% blend
            # Feed as pseudo-PnL: positive quality → positive PnL signal
            pseudo_pnl = quality * 100.0
            self._symbol_trust.update(symbol, pseudo_pnl)
            logger.info(f"[BD_TRUST] {symbol} ticket={ticket} ff={ff:.2f} ef={ef:.2f} "
                        f"ratio={ratio:.3f} cs={cs:.2f} quality={quality:.3f} "
                        f"trust={current_trust:.3f}→{self._symbol_trust.get(symbol):.3f}")

    def _reconcile_broker_positions(self):
        """Reconcile local position state against broker truth (source-of-truth).
        Emits structured reconciliation events for DFAD/RealityScore causal tracking (P0.24).
        """
        try:
            broker_positions = self.positions.positions or []
        except Exception as e:
            logger.error(f"[RECON] broker fetch failed: {e}")
            return
        broker_tickets = {}
        for p in broker_positions:
            ticket = p.get("ticket")
            if ticket:
                broker_tickets[ticket] = p
        local_tickets = set(self._active_positions_metadata.keys())
        broker_ticket_set = set(broker_tickets.keys())
        stale_local = local_tickets - broker_ticket_set
        for ticket in stale_local:
            meta = self._active_positions_metadata.get(ticket, {})
            sym_for_trust = meta.get("symbol", None) or next(
                (s for s, m in self._active_positions_metadata.items()
                 if m.get("ticket") == ticket or m.get("symbol")),
                None
            )
            last_profit = meta.get("last_profit", 0.0)
            if sym_for_trust and last_profit != 0.0:
                self._symbol_trust.update(sym_for_trust, last_profit)
            self._emit_reconciliation_event("ORPHAN_RECONCILE_CLOSE", ticket, sym_for_trust,
                                            {"last_profit": last_profit, "meta": meta})
            logger.warning(f"[RECON] removing stale local ticket={ticket}")
            self._active_positions_metadata.pop(ticket, None)
            if ticket in self._exit_state:
                self._exit_state.pop(ticket, None)
        missing_local = broker_ticket_set - local_tickets
        for ticket in missing_local:
            p = broker_tickets[ticket]
            sym = p.get("symbol", "?")
            self._emit_reconciliation_event("ORPHAN_RECONCILE_OPEN", ticket, sym,
                                            {"broker_type": p.get("type"), "price": p.get("price_open")})
            logger.warning(f"[RECON] hydrating orphan broker ticket={ticket}")
            self._active_positions_metadata[ticket] = {
                "entry_bar_time": p.get("time"),
                "entry_warmup_ticks": self._warmup_ticks,
                "entry_ev": 0.0,
                "entry_es_rank": 0.0,
                "entry_at_rank": 0.0,
                "entry_price": p.get("price_open", 0.0),
                "min_price": p.get("price_open", 0.0),
                "max_price": p.get("price_open", 0.0),
                "entry_time": time.time(),
                "direction": 1 if p.get("type") == "BUY" else -1,
                "volume": p.get("volume", 0.0),
                "hydrated": True,
            }
        for ticket in local_tickets & broker_ticket_set:
            local_vol = self._active_positions_metadata.get(ticket, {}).get("volume", 0.0)
            broker_vol = broker_tickets[ticket].get("volume", 0.0)
            if abs(local_vol - broker_vol) > 1e-6:
                self._emit_reconciliation_event("POSITION_SPLIT_MERGE", ticket,
                                                self._active_positions_metadata[ticket].get("symbol"),
                                                {"local_vol": local_vol, "broker_vol": broker_vol})
                logger.warning(f"[RECON] volume drift ticket={ticket} local={local_vol} broker={broker_vol}")
                self._active_positions_metadata[ticket]["volume"] = broker_vol
        # Wave 1: Detect ghost and missing positions
        local_ids = set(self.positions.positions.keys()) if hasattr(self.positions, 'positions') and isinstance(self.positions.positions, dict) else set()
        broker_ids = set()
        try:
            broker_positions = self.positions.get_all() if hasattr(self.positions, 'get_all') else []
            broker_ids = {p.get("ticket", 0) for p in broker_positions}
        except Exception:
            pass
        missing = broker_ids - local_ids
        ghost = local_ids - broker_ids
        if missing:
            logger.warning(f"[RECONCILE] {len(missing)} broker positions missing locally: {list(missing)[:5]}")
        if ghost:
            logger.warning(f"[RECONCILE] {len(ghost)} ghost positions (local but not broker): {list(ghost)[:5]}")

    # ─────────────────────────────────────────────────────────────────────
    # Bridge Gate — Centralized Exit Authority
    # ─────────────────────────────────────────────────────────────────────

    def _bridge_allows_exit(self, symbol: str, ticket: int,
                            exit_reason: str = "unknown") -> bool:
        """Check if the RestrictedExecutionBridge permits closing *symbol*.

        Returns ``True`` if the bridge allows the exit, ``False`` if it
        blocks it.  Logs ``EXIT_ALLOWED`` / ``EXIT_BLOCKED`` so we can
        verify bridge participation in the main loop.

        Parameters
        ----------
        symbol : str
            Broker symbol being closed.
        ticket : int
            MT5 ticket number (used for logging only).
        exit_reason : str
            Human-readable reason for the attempted exit (BD, TPI, TIMEOUT,
            H20, MIGRATION, etc.).

        Returns
        -------
        bool
            ``True`` if exit may proceed, ``False`` if bridge blocks it.
        """
        if not hasattr(self, '_execution_bridge') or self._execution_bridge is None:
            return True  # bridge not available — allow

        bridge = self._execution_bridge

        # Fast-path: shadow instability blocks everything
        if bridge.shadow_instability_flag:
            logger.warning(
                f"[EXIT_BLOCKED] {symbol} ticket={ticket} "
                f"exit={exit_reason} — shadow_instability"
            )
            return False

        # Check the last bridge evaluation result (computed once per cycle)
        if not hasattr(self, '_last_bridge_result') or not self._last_bridge_result:
            logger.info(
                f"[EXIT_ALLOWED] {symbol} ticket={ticket} "
                f"exit={exit_reason} — no bridge result, default allow"
            )
            return True

        br_decisions = self._last_bridge_result.get("decisions", {})
        br_dec = br_decisions.get(symbol, {})

        if br_dec.get("bridge_state") == "BLOCKED":
            reasons = "; ".join(br_dec.get("blocking_reasons", ["unknown"]))
            logger.warning(
                f"[EXIT_BLOCKED] {symbol} ticket={ticket} "
                f"exit={exit_reason}: {reasons}"
            )
            return False

        logger.info(
            f"[EXIT_ALLOWED] {symbol} ticket={ticket} exit={exit_reason}"
        )
        return True

    def _emit_reconciliation_event(self, event_type: str, ticket: int, symbol: str,
                                    details: dict = None) -> None:
        """Emit a structured reconciliation event for causal tracking (P0.24)."""
        event = {
            "type": event_type,
            "ticket": ticket,
            "symbol": symbol or "?",
            "cycle_id": self._cycle_id,
            "timestamp": time.time(),
            "details": details or {},
        }
        self._reconciliation_events.append(event)
        # Pipe to exec_stats for DFAD/RealityScore visibility
        if hasattr(self, 'exec_stats') and self.exec_stats:
            self.exec_stats.record_reconciliation(event_type, symbol or "?", ticket, details)
        # Limit event log size
        if len(self._reconciliation_events) > 1000:
            self._reconciliation_events = self._reconciliation_events[-500:]

    def _select_balanced_top3(self, candidates, execution_plan, n=3):
        selected = []
        for sym in candidates:
            if len(selected) >= n:
                break
            selected.append(sym)
        return set(selected[:n])

    def _load_active_positions_metadata(self):
        import json
        meta_file = os.path.join(os.path.dirname(SETTINGS.db_path), "active_positions_metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._active_positions_metadata = {int(k): v for k, v in data.items()}
            except Exception as e:
                logger.error(f"Error loading active positions metadata: {e}")
        # P0.25: Restore idempotency guard
        guard_file = os.path.join(os.path.dirname(SETTINGS.db_path), "applied_feedback_tickets.json")
        if os.path.exists(guard_file):
            try:
                with open(guard_file, "r", encoding="utf-8") as f:
                    self._applied_feedback_tickets = set(json.load(f))
            except Exception as e:
                logger.error(f"Error loading idempotency guard: {e}")

    def _train_oss_from_cache(self) -> bool:
        """Load pre-computed OSS training data from ReplayCache with drift."""
        try:
            from research.replay_cache import ReplayCache
            from evaluation.delayed_outcome_engine import DelayedOutcomeEngine
            from bootstrap.oss_bootstrap_trainer import OSSBootstrapTrainer
            train_ticks = ReplayCache(["EURJPY"], "2026-04-01", "2026-04-20",
                                       tick_limit=50000, seed=42).compute()
            recs, ed, doa = [], {}, DelayedOutcomeEngine(horizon_ticks=20)
            _ema = {}
            _diff_hist = {}
            for t in train_ticks:
                s = t["sym"]
                price = t["price"]
                if s not in _ema:
                    _ema[s] = price
                else:
                    alpha = 2.0 / 21.0
                    _ema[s] = alpha * price + (1 - alpha) * _ema[s]
                _diff = price - _ema[s]
                _diff_hist.setdefault(s, []).append(_diff)
                if len(_diff_hist[s]) > 20:
                    _diff_hist[s] = _diff_hist[s][-20:]
                _local_std = float(np.std(_diff_hist[s])) if len(_diff_hist[s]) > 5 else 0.0
                drift = 0
                if _local_std > 1e-10:
                    _z = _diff / _local_std
                    drift = 1 if _z > 0.5 else (-1 if _z < -0.5 else 0)
                d = t.get("ecdf", 0.5) - 0.5
                sig = 1 if d > 0.05 else (-1 if d < -0.05 else 0)
                ed[s] = {"price": price, "ecdf_rank": t["ecdf"], "entropy": t["entropy"],
                         "signal": sig, "drift": drift}
                doa.record_snapshot(ed)
                if doa.ready:
                    for s2, outcome in doa.evaluate({s2: ed[s2]["price"] for s2 in ed}).items():
                        recs.append({"sym": s2, "ecdf": ed[s2]["ecdf_rank"],
                                    "drift": ed[s2].get("drift", 0), "outcome": outcome})
            self._oss_bootstrap = OSSBootstrapTrainer()
            for s2, outcome in doa.evaluate({s2: ed[s2]["price"] for s2 in ed}).items():
                recs.append({"sym": s2, "ecdf": ed[s2]["ecdf_rank"],
                            "drift": ed[s2].get("drift", 0), "outcome": outcome})
            if recs:
                from signals.outcome_surface_signal import OutcomeSurfaceSignal
                oss_h10 = OutcomeSurfaceSignal.from_pipeline_records(recs, ev_threshold=0.05)
                for sym in SETTINGS.symbols:
                    self._oss_bootstrap._models[sym] = {10: oss_h10}
            self._ranking.set_oss(self._oss_bootstrap.get_oss())
            self._trained_oss = True
            logger.info(f"[OSS TRAIN] Trained from cache: {len(recs)} records")
            return True
        except Exception as e:
            logger.error(f"[OSS TRAIN] Failed: {e}")
            return False

    def _now_ts(self) -> float:
        if self._clock:
            return self._clock.time()
        return time.time()

    def _now_dt(self):
        if self._clock:
            return self._clock.now()
        return datetime.now()

    def reset(self):
        self._sal.reset()
        self._spread_normalizer.reset()
        self._tpi_calibration.reset_stats()
        self._wfv_records.clear()
        self._activity_log.clear()
        self._active_positions_metadata.clear()
        self._top3_history.clear()
        self._last_top3_qualified.clear()
        self._ranked_symbols.clear()
        self._active_symbols.clear()
        self._cycle_id = 0
        self._pyramid_log.clear()
        self._pyramid_count.clear()
        self._session_balance = {s: 0 for s in ["ASIA", "LONDON", "OVERLAP", "NY", "DEAD"]}
        self._last_meta_scores.clear()
        self._position_lock_until.clear()
        self._rotation_event_count = 0
        self._lock_event_count = 0
        self._migration_event_count = 0
        self._session_balance = {s: 0 for s in ["ASIA", "LONDON", "OVERLAP", "NY", "DEAD"]}
        self._budget_block_ttl.clear()
        for v in self.__dict__.values():
            if hasattr(v, "reset") and v is not self._sal:
                try:
                    v.reset()
                except Exception:
                    pass
        self._ecdf = type(self._ecdf)(window_size=5000)

    def warmup(self, n_ticks: int):
        """Warmup: advance replay feed and pre-seed RF gate with historical ticks."""
        self._warmup_mode = True
        feed = self._replay_feed
        for i in range(n_ticks):
            if feed and not feed.done:
                tick = feed.next()
                if tick:
                    sym = tick.get("symbol", "")
                    if sym and hasattr(self, '_rf_gate'):
                        _ts = tick.get("time_sec", tick.get("timestamp", 0))
                        self._rf_gate.feed_tick(sym, tick.get("bid", 0), tick.get("ask", 0), _ts, i)
                    if sym:
                        self._warmup_ticks += 1
            elif self.tick_source:
                tick = self.tick_source.get_tick("EURJPY")
                if tick is None:
                    break
                if hasattr(self, '_rf_gate'):
                    self._rf_gate.feed_tick("EURJPY", tick.get("bid", 0), tick.get("ask", 0), tick.get("time", 0), i)
                self._warmup_ticks += 1
            else:
                break
        self._warmup_mode = False

    def _preseed_rf_gate(self):
        """Pre-seed the RF gate rolling buffer from MT5 historical ticks.
        This fills the 2000-tick window immediately so the RF gate is ready from cycle 1.
        Skips if in replay mode or if MT5 connector doesn't support historical ticks."""
        if not hasattr(self, '_rf_gate') or self._rf_gate is None:
            return
        if not hasattr(self, 'mt5'):
            return
        if not hasattr(self.mt5, 'get_historical_ticks'):
            logger.info("[RF_PRESEED] MT5 connector does not support historical ticks — skipping")
            return
        if self._replay_mode:
            logger.info("[RF_PRESEED] Replay mode — RF gate will be fed during warmup instead")
            return
        total_seeded = 0
        for sym in self._execution_symbols:
            try:
                ticks = self.mt5.get_historical_ticks(sym, count=self._rf_gate.window)
                if ticks and len(ticks) > 0:
                    n = self._rf_gate.pre_seed(sym, ticks)
                    total_seeded += n
                    logger.info(f"[RF_PRESEED] {sym}: seeded {n}/{self._rf_gate.window} ticks — ready={self._rf_gate.ready(sym)} prob={self._rf_gate.prob(sym):.4f}")
                else:
                    logger.info(f"[RF_PRESEED] {sym}: no historical ticks available — will warm up live")
            except Exception as e:
                logger.warning(f"[RF_PRESEED] {sym} failed: {e}")
        if total_seeded > 0:
            logger.info(f"[RF_PRESEED] Total: seeded {total_seeded} ticks across {len(self._execution_symbols)} symbols")

    def _shadow_regime(self, sym: str) -> str:
        """Lightweight regime estimate for shadow symbols using spread + entropy."""
        ed = getattr(self, '_shadow_state', {})
        state = ed.get(sym, {})
        spread_p95 = state.get("spread_p95", 0)
        spread_p50 = state.get("spread_p50", 0)
        entropy = state.get("entropy", 0.5)

        if spread_p95 > 0 and spread_p50 > 0 and spread_p95 > spread_p50 * 2:
            return "WIDE"
        elif entropy > 0.85:
            return "CHAOTIC"
        elif entropy < 0.45:
            return "COMPRESSED"
        else:
            return "NORMAL"

    def _process_shadow_tick(self, sym: str, tick: dict) -> None:
        """Lightweight shadow symbol processing — no signal generation."""
        price = tick.get("ask", 0)
        spread = tick.get("spread", 0)
        self._ecdf.update(sym, price)
        if not hasattr(self, '_shadow_state'):
            self._shadow_state = {}
        if sym not in self._shadow_state:
            self._shadow_state[sym] = {"spreads": [], "entropy": 0.5}
        ss = self._shadow_state[sym]
        ss["spreads"].append(spread)
        if len(ss["spreads"]) > 5000:
            ss["spreads"] = ss["spreads"][-5000:]
        if len(ss["spreads"]) >= 100:
            arr = np.array(ss["spreads"])
            ss["spread_p50"] = float(np.median(arr))
            ss["spread_p95"] = float(np.percentile(arr, 95))

    def _spread_points(self, tick: dict) -> float:
        """Spread in POINTS regardless of tick shape.

        Demo spread consumers (spread_normalizer, spread model, gates) all
        expect points — the live connector historically fed points. Canonical
        ticks carry spread (price units) + spread_pts (points); raw live ticks
        carry only spread-as-points; replay archive rows carry price units.
        This is the single unit-normalization point so replay == live.
        """
        pts = tick.get("spread_pts") or tick.get("spread_points")
        if not pts:
            raw = tick.get("spread", 0)
            if raw > 1:
                pts = raw
            else:
                point = tick.get("point") or 1e-5
                if raw:
                    pts = int(raw / max(point, 1e-12))
                else:
                    # No explicit spread: derive from bid/ask (old default),
                    # then convert to points so units stay consistent.
                    bid = tick.get("bid", 0.0)
                    ask = tick.get("ask", 0.0)
                    if bid or ask:
                        pts = int(max(ask - bid, 0.0) / max(point, 1e-12))
                    else:
                        pts = 0
        return float(pts or 0)

    def _dispatch_tick(self, sym: str, tick: dict, eval_data: dict) -> None:
        """Shared tick-dispatch path for BOTH live and replay modes.

        Phase 1 (apples-to-apples): live and replay previously fed the same
        engines through two divergent code paths with different field names
        (``time`` vs ``time_sec`` vs ``timestamp``). This single method is the
        one place a tick is ingested: tolerant field access accepts the
        canonical contract (data.canonical_tick) OR any raw producer shape, so
        a strategy tested on replay consumes byte-identical fields live.
        """
        bid = tick.get("bid", 0.0)
        ask = tick.get("ask", bid)
        price = ask if ask else bid
        spread = self._spread_points(tick)
        _ts = tick.get("time_sec") or tick.get("time") or tick.get("timestamp") or 0

        if sym in eval_data:
            eval_data[sym]["price"] = price
            eval_data[sym]["spread"] = spread
            eval_data[sym]["ecdf_rank"] = self._ecdf.compute_and_update(sym, price)

            if sym in self._shadow_set:
                self._process_shadow_tick(sym, tick)
                if "regime" not in eval_data.get(sym, {}):
                    eval_data[sym]["regime"] = self._shadow_regime(sym)

            self._tick_thermo.feed_ticks(sym, bid, ask, _ts)
            self._rf_gate.feed_tick(sym, bid, ask, _ts, self._warmup_ticks)
            self._thesis_graph.tick(sym, price)
            self._validation.on_tick(sym, price, _ts)
            self._canonical_tpi_buffer.append(sym, bid, ask, _ts)
            if self._replay_tpi_buf is not None:
                self._replay_tpi_buf.append(sym, bid, ask, _ts)
            for _bd_t in list(self._battle_decay._states):
                _bd_s = self._battle_decay._states[_bd_t]
                if _bd_s.symbol == sym:
                    _bd_s.feed_tick(bid, ask, _ts)
        if sym:
            self._warmup_ticks += 1

    def add_activity(self, msg: str):
        # Ignore verbose startup, mapping, and redundant syncing logs
        ignore_terms = [
            "DAILY REPORT", "WEEKLY REPORT", "====", "----", "DEPLOYMENT", 
            "PERFORMANCE", "SIGNALS", "OPEN POSITIONS", "Markets closed",
            "Mapping symbol", "Warming up price buffers", "Initialized buffer",
            "Initializing metadata", "Syncing trade ledger", "Sync: Position"
        ]
        if any(term in msg for term in ignore_terms):
            return
            
        # Format known messages beautifully
        formatted_msg = msg
        if "Order failed for" in msg:
            try:
                parts = msg.split("Order failed for ")[1].split(": ")
                symbol = parts[0]
                details = parts[1]
                formatted_msg = f"❌ FAILED: {symbol} | {details}"
            except Exception:
                pass
        elif "Sync: Closed trade" in msg:
            try:
                trade_id = msg.split("Closed trade ")[1].split(" ")[0]
                ticket = msg.split("(ticket ")[1].split(")")[0]
                exit_p = msg.split("Exit price: ")[1].split(",")[0]
                profit = msg.split("Profit: ")[1]
                formatted_msg = f"✔ CLOSED: Trade #{trade_id} (Ticket {ticket}) | Exit: {exit_p} | PnL: {profit}"
            except Exception:
                pass
        elif "Spread too high for" in msg:
            try:
                symbol = msg.split("Spread too high for ")[1].split(",")[0]
                formatted_msg = f"⌛ BLOCKED: {symbol} | Spread too high"
            except Exception:
                pass
        elif "H20 EXIT: Closing position" in msg:
            try:
                ticket = msg.split("Closing position ")[1].split(" ")[0]
                symbol = msg.split("for ")[1].split(" ")[0]
                formatted_msg = f"⌛ H20 EXIT: Closing position {ticket} for {symbol}"
            except Exception:
                pass
        elif "Starting Proxima Ops" in msg:
            formatted_msg = "ℹ Engine initialized and active"
        elif "Connected to MT5" in msg:
            try:
                acc = msg.split("Account: ")[1].split(",")[0]
                formatted_msg = f"ℹ Connected to MT5 | Account: {acc}"
            except Exception:
                pass
        elif "BUY " in msg and " - ticket=" in msg:
            try:
                parts = msg.split("BUY ")[1].split(" ")
                symbol = parts[0]
                volume = parts[1]
                price = msg.split("@ ")[1].split(" - ")[0]
                ticket = msg.split("ticket=")[1]
                formatted_msg = f"✈ EXECUTED: Buy {volume} {symbol} @ {price} | Ticket: {ticket}"
            except Exception:
                pass
        elif "Closed ticket " in msg:
            try:
                ticket = msg.split("ticket ")[1]
                formatted_msg = f"✔ CLOSED: Ticket {ticket} closed on MT5"
            except Exception:
                pass
        elif "Failed to close ticket " in msg:
            try:
                ticket = msg.split("ticket ")[1].split(":")[0]
                err = msg.split(": ")[1]
                formatted_msg = f"❌ CLOSE FAILED: Ticket {ticket} | {err}"
            except Exception:
                pass

        t = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{t}] {formatted_msg}"
        
        # Avoid duplicate consecutive logs
        if self._activity_log and self._activity_log[-1][11:] == formatted_msg:
            return
            
        self._activity_log.append(formatted)
        if len(self._activity_log) > 6:
            self._activity_log.pop(0)

    def _print_dashboard(self, eval_data: dict, open_positions: list, account: dict, score_data: dict, seconds_to_next_eval: int, rotation_events=0, lock_events=0, migration_events=0, avg_hold_bars=0, top3_ranked=None):
        if not hasattr(self, "_ansi_initialized"):
            if sys.platform == "win32":
                os.system('')
            self._ansi_initialized = True

        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinners)
        spinner = self._spinners[self._spinner_idx]

        # Build paper metrics (shared between deployment dashboard and funnel dashboard)
        perf_summary_inner = self.perf.summary()
        paper_metrics = {
            'pp': perf_summary_inner.get('pp') if isinstance(perf_summary_inner.get('pp'), (int, float)) else 0,
            'avg_hold': perf_summary_inner.get('avg_hold_bars') if isinstance(perf_summary_inner.get('avg_hold_bars'), (int, float)) else 0,
            'sharpe': perf_summary_inner.get('sharpe') if isinstance(perf_summary_inner.get('sharpe'), (int, float)) else 0,
            'active_assets': len([s for s, d in eval_data.items() if d.get('status') != 'WATCH']),
        }

        # Build the full dashboard output
        out = []
        out.append(self.dashboard.render(
            eval_data=eval_data, account_info=account, score_data=score_data,
            seconds_to_next_eval=seconds_to_next_eval, spinner=spinner,
            closed_trades=self.perf.n_trades,
            rotation_events=rotation_events,
            lock_events=lock_events,
            migration_events=migration_events,
            avg_hold_bars=avg_hold_bars,
            top3_ranked=top3_ranked,
            paper_metrics=paper_metrics))

        out.append("OPEN POSITIONS:")
        out.append(f"{'Ticket':<12s} {'Symbol':<10s} {'Side':<6s} {'Volume':<8s} {'Entry Price':<12s} {'Current Price':<14s} {'PnL':<10s} {'Bars':<6s}")
        for pos in open_positions:
            ticket = pos["ticket"]
            meta = self._active_positions_metadata.get(ticket, {})
            entry_bar = meta.get("entry_bar_time")
            broker_sym = self.mt5._get_broker_symbol(pos["symbol"])
            elapsed_bars = self._bars_elapsed(entry_bar, broker_sym)
            elapsed_str = f"{'20+' if elapsed_bars >= 20 else elapsed_bars}/20" if elapsed_bars >= 0 else "N/A"
            out.append(f"{ticket:<12d} {pos['symbol']:<10s} {pos['type']:<6s} {pos['volume']:<8.2f} {pos['price_open']:<12.3f} {pos['price_current']:<14.3f} ${pos['profit']:<8.2f} {elapsed_str:<6s}")
        if not open_positions:
            out.append(" No open positions")
        out.append("")

        # Open trade context
        if open_positions:
            out.append("OPEN TRADE CONTEXT")
            out.append("=" * 52)
            for pos in open_positions:
                ticket = pos["ticket"]
                meta = self._active_positions_metadata.get(ticket, {})
                entry_bar = meta.get("entry_bar_time")
                broker_sym = self.mt5._get_broker_symbol(pos["symbol"])
                elapsed_bars = self._bars_elapsed(entry_bar, broker_sym)
                elapsed_str = f"{'20+' if elapsed_bars >= 20 else elapsed_bars} bars" if elapsed_bars >= 0 else "N/A"
                es_str = f"{meta.get('entry_es_rank', 0) * 100:.1f}%" if isinstance(meta.get('entry_es_rank'), (int, float)) else "N/A"
                at_str = f"{meta.get('entry_at_rank', 0) * 100:.1f}%" if isinstance(meta.get('entry_at_rank'), (int, float)) else "N/A"
                sym_data = eval_data.get(pos["symbol"], eval_data.get(broker_sym, {}))
                econ_r = sym_data.get("econ_ratio", 0.0)
                exp_m = sym_data.get("expected_move", 0)
                out.append(f" ECON: ratio={econ_r:.4f}x move={exp_m:.6f}")
                sig = meta.get("trigger_count_while_open", 0)
                out.append(f" Ticket {ticket} | Age {elapsed_str} | ES/AT {es_str}/{at_str} | SigOpen {sig} | PnL ${pos['profit']:.2f} | THESIS_ACTIVE")
            out.append("=" * 52)
            out.append("")

        # Recent activity
        out.append("RECENT ACTIVITY:")
        for log_line in self._activity_log:
            out.append(f" {log_line}")
        if not self._activity_log:
            out.append(" No recent activity")
        out.append("")

        # Full funnel dashboard — all research layers in one block
        full = self.funnel_dash.generate(
            order_attempts=self.order_tracker.get_recent(1),
            paper_metrics=paper_metrics)
        out.append(full)

        if SYSTEM_MODE.ui != UIMode.TRADER_VIEW:
            # TPI Flow Overlay (Layer 7) — Shadow Dashboard
            tpi_panel = tpi_dashboard_generate(
                tracker=self._tpi_tracker,
                persistence=self._tpi_persistence,
                curvature=self._tpi_curvature,
                eligible_symbols=[s for s in self._observation_universe if s in TPI_ELIGIBLE],
            )
            out.append(tpi_panel)

            # Deployment Integrity Guards — V2.2
            n_trades = self.perf.n_trades
            sig_counts = {}
            for sym in self._observation_universe:
                sig_counts[sym] = len([x for x in self.funnel._records if x.get("symbol") == sym]) if hasattr(self.funnel, "_records") else 0
            guard = self._sample_guard.guard("DEPLOYMENT_CLASSIFICATION")
            alignment_line = self._alignment_monitor.dashboard_line(sig_counts)
            out.append(f"V2.2 GUARDS: SampleInspector={guard} | {alignment_line}")
            # Universe integrity status
            canonical = set(SETTINGS.execution_symbols)
            deployed = set(self._execution_universe)
            core_present = canonical.issubset(deployed)
            extra = deployed - canonical
            uni_ok = "CORE_OK" if core_present else f"CORE_MISSING diff={canonical - deployed}"
            out.append(f"UNIVERSE: {len(deployed)}-asset ({len(extra)} extra) | {sorted(deployed)} | {uni_ok}")
            # Pyramid event log (compact)
            if self._pyramid_log:
                out.append(f"PYRAMID EVENTS ({len(self._pyramid_log)} total):")
                for pe in self._pyramid_log[-5:]:
                    out.append(f"  {pe['time'][:19]} {pe['symbol']} #{pe['pyramid_number']} ES={pe['es_percentile']:.3f}")
            else:
                out.append("PYRAMID EVENTS: none")
            # P8: Leakage accounting split
            total_blocked = self._reinforcement_blocks + self._flip_blocks
            if total_blocked > 0:
                pct = self._flip_blocks / total_blocked * 100
                out.append(f"POSITION EXISTS: {total_blocked} total (reinforcement={self._reinforcement_blocks}, flip_blocked={self._flip_blocks}, flip_pct={pct:.1f}%)")
            if self._exception_dashboard.has_active():
                out.append(f"EXCEPTIONS: {self._exception_dashboard.summary().splitlines()[-1]}")
            # DPL-18: Cross-asset propagation
            prop_syms = [s for s in self._observation_universe if s in TPI_ELIGIBLE]
            if prop_syms:
                out.append(self._tpi_propagation.summary(prop_syms))
                dpl_matrix = self._tpi_propagation.compute(prop_syms)
                self._impulse_graph.update(dpl_matrix)
                out.append(self._impulse_graph.summary())
            # DPL-19: Tick thermodynamics
            thermo_syms = [s for s in self._observation_universe]
            if thermo_syms:
                out.append(self._tick_thermo.summary(thermo_syms))
            # DPL-21: Meta-State Fusion
            meta_syms = [s for s in self._observation_universe if s in self._last_meta_scores]
            if meta_syms:
                out.append(self._meta_fusion.summary(meta_syms, self._last_meta_scores))
            # DPL-20: Session Conditional
            out.append(self._session_cond.summary())
            # DPL-22: Entropy Compression
            ent_syms = [s for s in self._observation_universe]
            if ent_syms:
                out.append(self._entropy_compression.summary(ent_syms))
            # RCL-1A: Outcome Ledger
            out.append(self._outcome_ledger.summary())
            # RCL-1B: Information Gain Audit
            out.append(self._ig_audit.summary(self._outcome_ledger))
            # RCL-1C: Redundancy Matrix (uses H5 for faster sample accumulation)
            try:
                fnames, X, Y = self._outcome_ledger.compute_feature_matrix(horizon="h5")
                if fnames and X:
                    self._redundancy_matrix.compute_pairwise(fnames, X)
            except Exception:
                pass
            out.append(self._redundancy_matrix.summary(self._outcome_ledger))
            # RCL-1D: Meta Reweighting
            ig_by_h = self._ig_audit.compute_by_horizon(self._outcome_ledger)
            h20_resolved = [r for r in self._outcome_ledger._resolved if "h20" in r.get("outcomes", {})]
            self._meta_reweighter.compute_weights(ig_by_h, self._redundancy_matrix, h20_count=len(h20_resolved))
            out.append(self._meta_reweighter.summary())
            # RCL-1E: Layer Pruning
            resolved_n = self._outcome_ledger.resolved_count()
            self._layer_pruner.compute_scores(ig_by_h, self._redundancy_matrix, self._meta_reweighter, resolved_samples=resolved_n)
            out.append(self._layer_pruner.summary(self._outcome_ledger))
            # P2.3: RCL Dual Horizon Dashboard
            h5_records = [r for r in self._outcome_ledger._resolved if "h5" in r.get("outcomes", {})]
            h20_records = h20_resolved
            h5_wins = sum(1 for r in h5_records if r["outcomes"]["h5"].get("win"))
            h20_wins = sum(1 for r in h20_records if r["outcomes"]["h20"].get("win"))
            out.append("  RCL DUAL HORIZON")
            out.append("-" * 52)
            out.append(f"  H5 Resolved:        {len(h5_records)}")
            out.append(f"  H20 Resolved:       {len(h20_records)}")
            if h5_records:
                out.append(f"  H5 WR:              {h5_wins}/{len(h5_records)} = {h5_wins/max(len(h5_records),1):.1%}")
            if h20_records:
                out.append(f"  H20 WR:             {h20_wins}/{len(h20_records)} = {h20_wins/max(len(h20_records),1):.1%}")
            if h5_records and h20_records:
                h5_wr = h5_wins / max(len(h5_records), 1)
                h20_wr = h20_wins / max(len(h20_records), 1)
                out.append(f"  Divergence:         {h5_wr - h20_wr:+.1%}")
            out.append("")
            # P2.5: Session Balance
            known = {k: v for k, v in self._session_balance.items() if k != "UNKNOWN"}
            total_signals = sum(known.values())
            out.append("  SESSION BALANCE")
            out.append("-" * 52)
            for sess in ["ASIA", "LONDON", "OVERLAP", "NY", "DEAD"]:
                cnt = known.get(sess, 0)
                bar = "█" * min(cnt, 50) + (" " * max(0, 50 - min(cnt, 50)))
                out.append(f"  {sess:<10s} {cnt:<5d} {bar}")
            if self._session_balance.get("UNKNOWN", 0):
                out.append(f"  UNKNOWN:            {self._session_balance['UNKNOWN']}")
            if total_signals > 0:
                max_c = max(known.values())
                min_c = max(min(v for v in known.values() if v > 0), 1)
                imbalance = max_c / min_c
                if max_c >= 20:
                    out.append(f"  Imbalance:          {imbalance:.1f}x")
                    out.append(f"  Status:             {'BALANCED' if imbalance < 5 else 'SKEWED'}")
                else:
                    out.append(f"  Status:             BUILDING (need >=20 signals)")
            out.append("")
            # P1.1: Occupancy Leakage Audit
            out.append(self._occupancy_audit.summary())
            # P1.2: TPI A/B Validation
            out.append(self._tpi_ab_audit.summary())
            # P1.3: Spread normalization — session baselines tracked
            sess_info = self._spread_normalizer.session_baseline_summary()
            if sess_info:
                out.append(sess_info)
            # P1.4: Funnel Starvation Audit
            out.append(self._funnel_audit.summary())
            out.append("")
            # P3.4: Regime Memory Matrix
            out.append(self._regime_memory.summary())
            out.append("")
            # P4.1: Live Transition Edge Dashboard
            lines_edge = []
            lines_edge.append("  LIVE TRANSITION EDGE")
            lines_edge.append("-" * 60)
            has_edge = False
            for sym in self._observation_universe:
                prev_r = self._regime_snapshot.get(sym) if hasattr(self, '_regime_snapshot') else None
                curr_r = self._regime_memory._prev_regime.get(sym)
                if prev_r is not None and curr_r is not None and prev_r != curr_r:
                    edge_line = self._regime_memory.transition_edge_summary(sym, prev_r, curr_r)
                    if edge_line:
                        lines_edge.append(edge_line)
                        has_edge = True
            if not has_edge:
                lines_edge.append("  No active transitions to evaluate")
            lines_edge.append("")
            out.extend(lines_edge)

            # P4.2: Signal Decay Velocity
            out.append(self._signal_decay.summary())
            out.append("")
            # P4.3: Occupancy Migration
            out.append(self._migration.summary())

        # RHL Risk Hardening Layer section
        ai = self.mt5.get_account() or {}
        out.append(self._risk.dashboard_section(ai.get("balance", 0.0), open_positions))

        # Write to file as well for reference
        try:
            report_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "live_observability_report.md")
            with open(report_file, "w", encoding="utf-8") as f:
                f.write("# PROXIMA OPS \u2014 LIVE OBSERVABILITY STATS BREAKDOWN\n\n")
                f.write(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                f.write("```text\n")
                f.write(full)
                f.write("\n```\n")
        except Exception as e:
            logger.error(f"Error writing live_observability_report.md: {e}")

        # Shadow System Status — SOF / STR-E / Phase 2
        _sr = getattr(self, '_last_stre_result', None)
        if _sr:
            out.append("")
            out.append("=" * 52)
            out.append("  SHADOW SYSTEM — TRUTH RECONCILIATION")
            out.append("=" * 52)
            out.append(f"  GT_corr={_sr.get('gt_corr',0):.4f} SY_corr={_sr.get('sy_corr',0):.4f} STAS={_sr.get('stas',0):.4f} Winner={_sr.get('winner','N/A')}")
            sof = _sr.get("SOF")
            if sof is not None:
                out.append(f"  SOF={sof:.6f} EdgePres={_sr.get('edge_preservation',0):.6f} ExecEff={_sr.get('execution_efficiency',0):.6f}")
            p2 = "ENABLED" if getattr(self, '_stre_coordinator', None) and self._stre_coordinator.phase2_enabled else "BLOCKED"
            out.append(f"  Phase 2: {p2} | Samples: {_sr.get('samples',0)}")
            out.append("")

        # Funnel failure attribution
        if self._funnel_failures:
            out.append(f"\n  FUNNEL FAILURE BREAKDOWN ({sum(self._funnel_failures.values())} total)")
            for reason, count in sorted(self._funnel_failures.items(), key=lambda x: -x[1]):
                out.append(f"    {reason}: {count}")
        # Clear screen and write everything in one shot
        if hasattr(self, "_clear_seq"):
            sys.stdout.write(self._clear_seq)
        else:
            self._clear_seq = '\033c'
            sys.stdout.write(self._clear_seq)
        sys.stdout.write("\n".join(out) + "\n")
        sys.stdout.flush()

    def _setup_commands(self):
        self._router.register("status", self._cmd_status)
        self._router.register("portfolio", self._cmd_portfolio)
        self._router.register("trades", self._cmd_trades)
        self._router.register("signal", self._cmd_signal)
        self._router.register("pause", self._cmd_pause)
        self._router.register("resume", self._cmd_resume)
        self._router.register("closeall", self._cmd_closeall)
        self._router.register("health", self._cmd_health)
        self._router.register("report", self._cmd_report)
        self._router.register("alpha", self._cmd_alpha)
        self._router.register("tpi_mode", self._cmd_tpi_mode)

    def _cmd_alpha(self, args, update) -> str:
        return self.dashboard.generate_alpha_snapshot()

    def _cmd_tpi_mode(self, args, update) -> str:
        if args:
            mode = args[0].upper()
            try:
                self._tpi_calibration.set_mode(mode)
                return f"TPI_MODE set to {mode}"
            except ValueError as e:
                return str(e)
        # Report current mode and gate stats
        stats = self._tpi_calibration.gate_stats()
        return (f"TPI_MODE: {stats['mode']}\n"
                f"Gate blocks: {stats['total_triggers_blocked']}\n"
                f"By gate: {stats['by_gate']}\n"
                f"Shadow opps: {stats['shadow_opportunities']}\n"
                f"Usage: /tpi_mode [HARD_GATE|SOFT_SCORE]")

    def print_alpha_snapshot(self):
        print(self.dashboard.generate_alpha_snapshot())

    def _cmd_status(self, args, update) -> str:
        ds = self.score.summary()
        perf = self.perf.summary()
        mt5_h = self.mt5_monitor.health_summary
        positions = self.positions.positions
        return (f"Proxima Ops — Status\n"
                f"Score: {ds['current_score']:.3f} ({ds['classification']})\n"
                f"Positions: {len(positions)}\n"
                f"Today PnL: ${perf['today_pnl']:.2f}\n"
                f"Sharpe: {perf['sharpe']:.3f}\n"
                f"PP: {perf['pp']:.3f}\n"
                f"MT5: {mt5_h['mt5_status']}\n"
                f"Paused: {self._paused}")

    def _cmd_portfolio(self, args, update) -> str:
        positions = self.positions.positions
        if not positions:
            return "No open positions"
        lines = ["Portfolio:"]
        for p in positions:
            lines.append(f"{p['symbol']} {p['type']} | {p['volume']} | "
                         f"Entry: {p['price_open']:.3f} | PnL: ${p['profit']:.2f}")
        return "\n".join(lines)

    def _cmd_trades(self, args, update) -> str:
        trades = self.trade_ledger.get_recent(10)
        if not trades:
            return "No trades recorded"
        lines = ["Last 10 Trades:"]
        for t in trades:
            lines.append(f"{t['symbol']} {t['signal_type']} | "
                         f"Entry: {t['entry_price']} | "
                         f"PnL: ${t['profit_money']:.2f} | {t['status']}")
        return "\n".join(lines)

    def _cmd_signal(self, args, update) -> str:
        if not args:
            return "Usage: /signal EURJPY"
        symbol = args[0].upper()
        tick = self.tick_source.next_tick(symbol) if self.tick_source else (self._tick_cache.get_tick(symbol) if self._tick_cache else self.mt5.get_tick(symbol))
        if tick is None:
            return f"Could not get tick for {symbol}"
        return (f"{symbol} — Live Tick\n"
                f"Bid: {tick['bid']:.5f}\n"
                f"Ask: {tick['ask']:.5f}\n"
                f"Spread: {tick['spread']}")

    def _cmd_pause(self, args, update) -> str:
        self._paused = True
        return "Trading PAUSED. No new entries. Monitoring continues."

    def _cmd_resume(self, args, update) -> str:
        self._paused = False
        return "Trading RESUMED."

    def _cmd_closeall(self, args, update) -> str:
        self.positions.refresh()
        for pos_ca in self.positions.positions:
            m_ca = self._active_positions_metadata.get(pos_ca["ticket"])
            if m_ca:
                m_ca["expected_exit_reason"] = "MANUAL"
        self._save_active_positions_metadata()
        self._sdl.reset()
        results = self.orders.close_all()
        closed = sum(1 for r in results if r["closed"])
        failed = sum(1 for r in results if not r["closed"])
        return f"Close All: {closed} closed, {failed} failed"

    def _cmd_health(self, args, update) -> str:
        mt5_h = self.mt5_monitor.health_summary
        ds = self.score.summary()
        return (f"Health Check:\n"
                f"MT5: {mt5_h['mt5_status']}\n"
                f"Uptime: {mt5_h['uptime_minutes']}m\n"
                f"Deployment Score: {ds['current_score']:.3f} ({ds['classification']})")

    def _cmd_report(self, args, update) -> str:
        return self._daily_report.generate()

    def sync_ledger_with_mt5(self):
        logger.info("Syncing trade ledger with MT5 active positions...")
        self.positions.refresh()
        open_positions = self.positions.positions
        active_tickets = {p["ticket"] for p in open_positions}
        
        open_ledger_trades = self.trade_ledger.get_open()
        for lt in open_ledger_trades:
            ticket = lt.get("mt5_ticket")
            trade_id = lt.get("trade_id")
            if not ticket or not trade_id:
                continue
            
            if ticket not in active_tickets:
                logger.info(f"Sync: Position {ticket} is no longer active on MT5. Closing in ledger...")
                
                # Read metadata for exit reason and excursion
                pos_meta = self._active_positions_metadata.get(ticket, {})
                exit_reason = pos_meta.get("expected_exit_reason", "UNKNOWN")
                
                # Fetch deals from history (replay-safe)
                deals = self.mt5.get_deal_history(ticket) if hasattr(self.mt5, "get_deal_history") else None
                if not deals and not hasattr(self.mt5, 'get_deal_history'):
                    try:
                        import MetaTrader5 as mt5
                        deals = mt5.history_deals_get(position=ticket)
                    except Exception:
                        deals = None
                
                # MT5 deal reason mapping (authoritative over expected_exit_reason)
                MT5_REASON_MAP = {0: "BROKER_UNKNOWN", 1: "SL", 2: "TP", 3: "BROKER_STOP", 4: "MANUAL", 6: "SL", 8: "MANUAL"}
                exit_price = lt.get("entry_price", 0.0)
                profit_money = 0.0
                exit_time_sec = int(self._now_ts())
                
                if deals:
                    out_deals = [d for d in deals if getattr(d, "entry", None) == 1]
                    if out_deals:
                        out_deal = out_deals[0]
                        exit_price = getattr(out_deal, "price", exit_price)
                        profit_money = getattr(out_deal, "profit", 0.0)
                        exit_time_sec = getattr(out_deal, "time", exit_time_sec)
                        mt5_reason = getattr(out_deal, "reason", -1)
                        exit_reason = MT5_REASON_MAP.get(mt5_reason, exit_reason)
                    else:
                        last_deal = deals[-1]
                        exit_price = getattr(last_deal, "price", exit_price)
                        profit_money = getattr(last_deal, "profit", 0.0)
                        exit_time_sec = getattr(last_deal, "time", exit_time_sec)
                        mt5_reason = getattr(last_deal, "reason", -1)
                        exit_reason = MT5_REASON_MAP.get(mt5_reason, exit_reason)
                else:
                    profit_money = pos_meta.get("last_profit", 0.0)
                
                duration = max(0, exit_time_sec - lt.get("timestamp", exit_time_sec))
                duration_bars = max(1, duration // 3600)
                
                entry_price = lt.get("entry_price", 0.0)
                symbol = lt.get("symbol", "")
                point = 0.01 if "JPY" in symbol else 0.0001
                profit_points = (exit_price - entry_price) / point if point > 0 else 0.0
                if lt.get("signal_type") == "SHORT":
                    profit_points = -profit_points
                    
                # Record signal lifecycle closure
                close_signal_id = pos_meta.get("signal_id")
                if close_signal_id:
                    self.audit.record_closed(close_signal_id, profit_points, profit_money)
                    self._funnel_audit.update(close_signal_id, closed=True, final_pnl=profit_money)
                    self._freq_pipeline.record_executed_result(ticket, profit_money)
                else:
                    sig = self.funnel.make_signal_id(symbol)
                    self.funnel.generate(sig, symbol, 0, 0, 0, "UNKNOWN")
                    self.audit.record_closed(sig, profit_points, profit_money)
                    self._funnel_audit.update(sig, closed=True, final_pnl=profit_money)
                    self._freq_pipeline.record_executed_result(ticket, profit_money)

                # RHL: record trade result in risk governor
                ai = self.mt5.get_account() or {}
                self._risk.post_trade_result(profit_money, ai.get("equity", 0.0))

                # Store excursion data before deleting metadata
                min_px = pos_meta.get("min_price", entry_price)
                max_px = pos_meta.get("max_price", entry_price)

                # V6: Record position lock on strategy-exit close only (H20/FLIP/MIGRATION set this before close)
                # DO NOT set lock here — this is the reconciliation path (ghost locks from killed processes)
                broker_sym_lock = self.mt5._get_broker_symbol(symbol) if hasattr(self.mt5, '_get_broker_symbol') else symbol
                self._lock_event_count += 1

                # B4: Resolve thesis at position close
                _label = self._thesis_buffer.resolve(ticket, profit_money, exit_reason)
                # Phase E: resolve observer evidence at trade close
                try:
                    _outcome = TradeOutcome(
                        thesis_id=str(ticket),
                        symbol=symbol,
                        exit_price=exit_price,
                        pnl=profit_money,
                        success=_label == 1 if _label is not None else False,
                        exit_reason=exit_reason,
                        duration_bars=duration_bars,
                    )
                    self._validation.on_trade_close(str(ticket), _outcome)
                except Exception as ex:
                    logger.warning(f"[VALIDATION] on_trade_close error: {ex}")
                if _label is not None:
                    logger.info(f"[B4_RESOLVE] ticket={ticket} pnl={profit_money:+.2f} "
                                f"label={_label} reason={exit_reason}")
                    self._thesis_trainer.retrain(self._thesis_buffer.get_resolved())
                    logger.info(f"[THESIS_RF_STATUS] ready={self._thesis_trainer.ready()} "
                                f"samples={len(self._thesis_buffer.get_resolved())}")
                    global RUNNING
                    if getattr(self, '_stop_on_first_thesis', False):
                        logger.info("[B4_ACCEPT] first thesis resolved, stopping")
                        RUNNING = False

                # Delete from metadata and save
                if ticket in self._active_positions_metadata:
                    del self._active_positions_metadata[ticket]
                    self._save_active_positions_metadata()

                # DRL: drawdown/regime recording on trade close
                es_val = lt.get("signal_score", 0.0)
                at_val = lt.get("residual_energy", 0.0)
                regime_val = lt.get("regime", "UNKNOWN")
                forecast_val = lt.get("persistence_forecast", "N/A")
                self._drl_pipeline.record_trade_close(
                    close_signal_id or sig, symbol, regime_val,
                    es_val, at_val,
                    entry_price, exit_price,
                    profit_points, profit_money,
                    max(1, duration // 3600), forecast_val)

                self.trade_ledger.close_trade(
                    trade_id=trade_id,
                    exit_price=exit_price,
                    profit_points=profit_points,
                    profit_money=profit_money,
                    duration=duration,
                    exit_reason=exit_reason,
                    min_price=min_px,
                    max_price=max_px
                )

                self.perf.record_trade(profit_money, profit_points, date.today().isoformat(), hold_bars=duration_bars)
                # P1.2: Record trade outcome in TPI A/B audit
                entry_price_ab = lt.get("entry_price", 0.0)
                ret_ab = profit_money / max(abs(entry_price_ab), 1e-10) if entry_price_ab != 0 else 0.0
                self._tpi_ab_audit.record(self._tpi_arm, profit_money > 0, ret_ab, duration_bars)
                logger.info(f"Sync: Closed trade {trade_id} (ticket {ticket}) in ledger. Exit price: {exit_price}, Profit: ${profit_money:.2f}")
                
                # Trade Closed Observability Printout
                pnl_pts = int(profit_points)
                pnl_sign = "+" if pnl_pts >= 0 else ""
                
                es_val = lt.get("signal_score", 0.0)
                at_val = lt.get("residual_energy", 0.0)
                regime = lt.get("regime", "UNKNOWN")
                threshold = lt.get("threshold", 0.80)
                forecast = lt.get("persistence_forecast", "N/A")
                
                trade_closed_msg = (
                    "=========================================================\n"
                    "TRADE CLOSED\n"
                    "=========================================================\n"
                    f"Symbol:\n{symbol}\n\n"
                    f"Entry:\n{entry_price:.5f}\n\n"
                    f"Exit:\n{exit_price:.5f}\n\n"
                    f"PnL:\n{pnl_sign}{pnl_pts} points\n\n"
                    f"Duration:\n{duration_bars} bars\n\n"
                    f"Entry State:\n\n"
                    f"ES:\n{es_val:.2f}\n\n"
                    f"AT:\n{at_val:.2f}\n\n"
                    f"Regime:\n{regime}\n\n"
                    f"Threshold:\n{threshold:.2f}\n\n"
                    f"Forecast Persistence:\n{forecast}\n\n"
                    f"Actual Persistence:\n{duration_bars}\n"
                    "========================================================="
                )
                print(trade_closed_msg)
                logger.info(f"Trade closed printout:\n{trade_closed_msg}")

    def run_demo(self):
        print("Starting Proxima Ops Demo Deployment...")
        print(f"  Threshold: {SETTINGS.threshold}")
        print(f"  Risk: {SETTINGS.risk_per_trade:.2%}")
        print(f"  Assets: {SETTINGS.symbols}")
        print(f"  Shadow: {len(SETTINGS.shadow_symbols)} symbols")
        print("  Connecting to MT5...", end=" ", flush=True)

        # Force MT5 shutdown first in case a previous process left it hanging (skip in replay)
        if not self._env or not hasattr(self._env, 'broker') or self._env.broker is None:
            try:
                import MetaTrader5 as _mt5_cleanup
                _mt5_cleanup.shutdown()
            except Exception:
                pass

        logger.info("Starting Proxima Ops Demo Deployment...")
        logger.info(f"Threshold: {SETTINGS.threshold}")
        logger.info(f"Risk: {SETTINGS.risk_per_trade:.2%}")
        logger.info(f"Execution universe: {SETTINGS.symbols} | Shadow: {len(SETTINGS.shadow_symbols)} symbols")

        if not self.mt5.connect():
            print("FAILED")
            logger.error("Failed to connect to MT5. Aborting.")
            self.telegram.send_sync("⚠ PROXIMA OPS: MT5 connection FAILED. Aborting deployment.")
            return
        print("OK")

        # Start ground-truth shadow observer
        try:
            self._shadow_gt_worker.start()
            logger.info("[SHADOW_GT] Ground-truth observer started")
        except Exception as _sgt_e:
            logger.warning(f"[SHADOW_GT] Failed to start observer: {_sgt_e}")

        account = self.mt5.get_account()
        if account:
            msg = (f"PROXIMA OPS DEPLOYMENT STARTED\n"
                   f"Account: {account['login']} | Balance: ${account['balance']:.2f}\n"
                   f"Threshold: {SETTINGS.threshold} | Risk: {SETTINGS.risk_per_trade:.2%}\n"
                   f"Mode: DEMO VALIDATION")
            self.telegram.send_sync(msg)

        # Sync database with MT5 positions on startup
        self.sync_ledger_with_mt5()

        logger.info("[CALIBRATION] Microstructure calibrator initialized — collecting baseline data")

        # Train OSS from replay cache for production signal path
        if ACCEPTANCE_MODE and self._trained_oss:
            logger.info("[ACCEPTANCE] OSS already trained from bootstrap — skipping cache training")
        else:
            logger.info("Training OSS from replay cache...")
            self._train_oss_from_cache()

        # Warm up price history buffers (550 H1 bars per asset) and initialize eval_data
        logger.info("Warming up price buffers (fetching 550 H1 bars per asset)...")
        print("  Warming up price buffers...", end=" ", flush=True)
        eval_data = {}
        for sym in self._observation_universe:
            eval_data[sym] = {
                "price": np.nan, "spread": None, "es_val": np.nan,
                "es_rank": np.nan, "at_rank": np.nan, "thermo_sizing_mult": np.nan,
                "status": "WATCH"
            }
            rates = self.mt5.get_rates(sym, count=550, timeframe="H1")
            if rates is not None and len(rates) >= 524:
                broker_sym = self.mt5._get_broker_symbol(sym)
                self._price_history[broker_sym] = list(rates)
                logger.info(f"Initialized buffer for {sym} (mapped to {broker_sym}) with {len(rates)} bars.")
                
                # Precompute initial features
                prices = np.array([r["close"] for r in rates], dtype=np.float64)
                highs = np.array([r["high"] for r in rates], dtype=np.float64)
                lows = np.array([r["low"] for r in rates], dtype=np.float64)
                volumes = np.array([r["volume"] for r in rates], dtype=np.float64)
                returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
                data_dict = {"price": prices, "returns": returns, "volume": volumes, "high": highs, "low": lows}
                
                res_ed = self.ed.compute(data_dict)
                es_history = np.nan_to_num(res_ed.get("energy_storage", np.zeros(len(prices))), nan=0.0)
                current_es = es_history[-1]
                es_window = es_history[-504:]
                es_percentile = float(np.sum(es_window <= current_es)) / len(es_window)
                self._global_rank_engine.record_evaluation(sym, es_percentile, current_es)

                res_tt = self.tt.compute(data_dict)
                time_density = np.nan_to_num(res_tt.get("time_density", np.zeros(len(prices))), nan=0.0)
                event_density = np.nan_to_num(res_tt.get("event_density", np.zeros(len(prices))), nan=0.0)
                info_density = np.nan_to_num(res_tt.get("information_density", np.zeros(len(prices))), nan=0.0)
                behavior_density = np.nan_to_num(res_tt.get("behavior_density", np.zeros(len(prices))), nan=0.0)
                combined_density = (time_density + event_density + info_density + behavior_density) / 4.0
                current_density = combined_density[-1]
                density_window = combined_density[-504:]
                at_percentile = float(np.sum(density_window <= current_density)) / len(density_window)

                # Freeze sizing at neutral during topology cold-start
                sizing_mult = 0.80
                if self._warmup_ticks < 504:
                    pass  # keep neutral sizing until topology converges
                elif at_percentile <= 0.20:
                    sizing_mult = 0.50
                elif at_percentile <= 0.40:
                    sizing_mult = 0.65
                elif at_percentile <= 0.60:
                    sizing_mult = 0.80
                elif at_percentile <= 0.80:
                    sizing_mult = 0.90
                else:
                    sizing_mult = 1.0
                    
                eval_data[sym]["price"] = prices[-1]
                eval_data[sym]["es_val"] = current_es
                eval_data[sym]["es_rank"] = es_percentile
                eval_data[sym]["at_rank"] = at_percentile
                eval_data[sym]["thermo_sizing_mult"] = sizing_mult

                # P3.5: Entropy unsupervised warmup from bar data
                try:
                    m5_rates = self.mt5.get_rates(sym, count=500, timeframe="M1")
                    if m5_rates is not None and len(m5_rates) >= 100:
                        self._entropy_compression.warmup_from_bars(
                            sym,
                            [r.get("open", 0) for r in m5_rates],
                            [r.get("high", 0) for r in m5_rates],
                            [r.get("low", 0) for r in m5_rates],
                            [r.get("close", 0) for r in m5_rates],
                        )
                        # P2: Seed drift EMA from warmup bar closes
                        closes = [r.get("close", 0) for r in m5_rates if r.get("close", 0) > 0]
                        if len(closes) >= 21:
                            if not hasattr(self, '_drift_ema'):
                                self._drift_ema = {}
                            ema = closes[0]
                            alpha = 2.0 / 21.0
                            for c in closes[1:]:
                                ema = alpha * c + (1 - alpha) * ema
                            self._drift_ema[sym] = ema
                except Exception as e:
                    logger.warning(f"[ENTROPY_WARMUP] {sym} failed: {e}")
            else:
                loaded = len(rates) if rates else 0
                logger.warning(f"Could not fetch enough H1 bars for {sym}. Loaded: {loaded}")
        print("OK")

        # Compute initial global ranks after all symbols processed
        self._global_rank_engine.compute()
        logger.debug(f"V2.2 Warmup GlobalRanks:\n{self._global_rank_engine.summary()}")

        # Close any orphan positions from previous sessions
        logger.info("Closing orphan MT5 positions from previous sessions...")
        try:
            self.positions.refresh()
            orphan_tickets = [p["ticket"] for p in self.positions.positions]
            for ticket in orphan_tickets:
                self.orders.close(ticket)
                pos_sym = next((p["symbol"] for p in self.positions.positions if p["ticket"] == ticket), None)
                if pos_sym:
                    self._sdl.release(pos_sym)
                logger.info(f"Closed orphan position ticket={ticket}")
            self._reconcile_broker_positions()
        except Exception as e:
            logger.warning(f"Orphan close error: {e}")

        # Initialize metadata for any pre-existing positions
        logger.info("Initializing metadata for existing positions...")
        print("  Loading existing positions...", flush=True)
        self._load_active_positions_metadata()
        self.positions.refresh()
        existing_positions = self.positions.positions
        for pos in existing_positions:
            ticket = pos["ticket"]
            pos_time = pos.get("time", 0)
            entry_bar_time = int(pos_time // 3600) * 3600
            if ticket not in self._active_positions_metadata:
                entry_p = pos.get("price_open", 0.0)
                self._active_positions_metadata[ticket] = {
                    "entry_bar_time": entry_bar_time,
                    "entry_warmup_ticks": self._warmup_ticks,
                    "symbol": pos["symbol"],
                    "entry_es_rank": None,
                    "entry_at_rank": None,
                    "trigger_count_while_open": 0,
                    "max_es_rank": 0.0,
                    "max_at_rank": 0.0,
                    "entry_price": entry_p,
                    "min_price": entry_p,
                    "max_price": entry_p,
                    "entry_time": datetime.fromtimestamp(pos_time).strftime("%Y-%m-%d %H:%M:%S") if pos_time else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            logger.info(f"Position {ticket} for {pos['symbol']} mapped to entry bar time {entry_bar_time}.")
        self._save_active_positions_metadata()

        check_interval = 0.1 if ACCEPTANCE_MODE else 10  # minimal sleep in acceptance mode for speed
        last_signal_check = 0
        last_hourly_check = 0
        last_daily_snapshot = None

        # Position state sync — refresh at cycle start
        if hasattr(self, '_position_sync'):
            self._position_sync.sync()

        print("  Starting main loop...", flush=True)
        print()

        print("[WHILE_DEBUG] Pre-loop init complete", flush=True)
        global RUNNING
        _wall_start = _wall_perf_counter()
        if self._runtime_limit > 0:
            print(f"  [DBUG] runtime_limit={self._runtime_limit}s, _wall_start={_wall_start}", flush=True)
        print("[WHILE_DEBUG] Entering while RUNNING loop", flush=True)
        while RUNNING:
            print("[WHILE_DEBUG] Top of loop", flush=True)
            if self._runtime_limit > 0 and (_wall_perf_counter() - _wall_start) >= self._runtime_limit:
                logger.info(f"Runtime limit ({self._runtime_limit}s) reached — shutting down")
                RUNNING = False
                break
            if self._tick_limit > 0 and self._replay_feed and self._replay_feed.cursor >= self._tick_limit:
                logger.info(f"Tick limit ({self._tick_limit}) reached — shutting down")
                RUNNING = False
                break
            try:
                if not self.mt5.ensure_connection():
                    logger.warning("MT5 reconnection failed, retrying...")
                    time.sleep(30)
                    continue

                now = self._now_ts()
                timestamp = int(now)
                today_str = self._now_dt().date().isoformat()

                print("[WHILE_DEBUG] Before mt5_monitor.check()", flush=True)
                mt5_status = self.mt5_monitor.check()
                print("[WHILE_DEBUG] After mt5_monitor.check()", flush=True)
                print(f"[WHILE_DEBUG] _replay_mode={getattr(self, '_replay_mode', None)} _replay_feed={getattr(self, '_replay_feed', None)}", flush=True)
                if not mt5_status["connected"]:
                    self.telegram.send_sync("MT5 DISCONNECTED")
                    time.sleep(10)
                    continue

                print("[WHILE_DEBUG] Before tick dispatch", flush=True)
                # Tick dispatch: live polls per-symbol, replay consumes merged events
                if self._replay_mode and self._replay_feed:
                    tick = self._replay_feed.next()
                    if tick is None:
                        logger.info("Replay feed exhausted — shutting down")
                        break
                    sym = tick.get("symbol")
                    self._dispatch_tick(sym, tick, eval_data)
                else:
                    _tick_start = _wall_perf_counter()
                    for sym in self._execution_symbols:
                        tick = self.tick_source.next_tick(sym) if self.tick_source else (self._tick_cache.get_tick(sym) if self._tick_cache else self.mt5.get_tick(sym))
                        if tick:
                            self._dispatch_tick(sym, tick, eval_data)
                        # Time cap: don't spend more than 15s per loop on tick dispatch
                        if _wall_perf_counter() - _tick_start > 15:
                            break
                    if not ACCEPTANCE_MODE and not getattr(self, '_replay_mode', False):
                        for tpi_sym in TPI_ELIGIBLE:
                            if tpi_sym not in (getattr(self, '_active_symbols', set()) or set()):
                                tpi_tick = self.tick_source.next_tick(tpi_sym) if self.tick_source else (self._tick_cache.get_tick(tpi_sym) if self._tick_cache else self.mt5.get_tick(tpi_sym))
                                if tpi_tick:
                                    _tpi_ts = tpi_tick.get("time_sec") or tpi_tick.get("time") or tpi_tick.get("timestamp") or 0
                                    self._tick_thermo.feed_ticks(tpi_sym, tpi_tick.get("bid", 0), tpi_tick.get("ask", 0), _tpi_ts)
                                    self._rf_gate.feed_tick(tpi_sym, tpi_tick.get("bid", 0), tpi_tick.get("ask", 0), _tpi_ts, self._warmup_ticks)
                                    self._canonical_tpi_buffer.append(tpi_sym, tpi_tick.get("bid", 0), tpi_tick.get("ask", 0), _tpi_ts)
                    print(f"[WHILE_DEBUG] After tick dispatch, _warmup_ticks={self._warmup_ticks}, eval_data keys={len(eval_data)}", flush=True)

                # V3/V4 STEP 2+3+4: Ranking Engine + TOP-K + Rotation Stability (execution symbols only)
                if self.config.ranking:
                    self._ranked_symbols = self._ranking.rank_all({s: eval_data[s] for s in self._execution_symbols if s in eval_data})
                else:
                    self._ranked_symbols = [(sym, 0.5) for sym in self._execution_symbols]
                # P14: Apply OSS quality discount to ranking scores
                self._ranked_symbols = [
                    (sym, score * 0.65) if not self._oss_bootstrap.has_surface(sym) else (sym, score)
                    for sym, score in self._ranked_symbols
                ]
                if self.config.rotation:
                    selected = self._rotation.select(self._ranked_symbols)
                else:
                    selected = [sym for sym, _ in self._ranked_symbols]
                self._active_symbols = set(selected)
                logger.info(
                    f"[RANKING TOP-{self._top_k}] "
                    f"ranked={self._ranked_symbols[:self._top_k]} "
                    f"active={selected}"
                )

                # Production signal path (multi-horizon OSS — regime selects horizon)
                # Signal hierarchy: regime-aware OSS → ECDF floor → flat
                prod_signals = {}
                for sym in self._execution_symbols:
                    if sym not in eval_data:
                        continue
                    ecdf = eval_data[sym].get("ecdf_rank", 0.5)
                    price = eval_data[sym].get("price", 0.0)
                    # Hydrate VPL regime for all symbols before OSS horizon selection
                    try:
                        from deployment.get_vpl_signal import get_current_signal
                        vpl_sig = get_current_signal(sym)
                        if vpl_sig and vpl_sig.get("regime"):
                            eval_data[sym]["regime"] = vpl_sig["regime"]
                    except Exception:
                        pass
                    # P5: Populate dynamic entropy from entropy engine
                    entropy_state = self._entropy_compression.compute_state(sym)
                    if entropy_state.get("state") == "ACTIVE" or entropy_state.get("warmup_prior"):
                        eval_data[sym]["entropy"] = entropy_state.get("normalized_entropy", 0.5)
                    if self._trained_oss and self._oss_bootstrap.has_surface(sym):
                        regime = eval_data[sym].get("regime", "")
                        # OSS horizon blending: triangular weights across 3/10/20 using entropy
                        entropy = eval_data[sym].get("entropy", 0.5)
                        w3 = max(0.0, 1.0 - 2.0 * entropy)
                        w20 = max(0.0, 2.0 * entropy - 1.0)
                        w10 = 1.0 - w3 - w20
                        horizon = 10  # nominal for logging
                        # Compute z-score drift: (price - EMA20) / rolling_std(diff)
                        ema_val = getattr(self, '_drift_ema', {}).get(sym, price)
                        live_drift = 0
                        if ema_val > 0 and price > 0:
                            _diff = price - ema_val
                            if not hasattr(self, '_diff_hist'):
                                self._diff_hist = {}
                            self._diff_hist.setdefault(sym, []).append(_diff)
                            if len(self._diff_hist[sym]) > 20:
                                self._diff_hist[sym] = self._diff_hist[sym][-20:]
                            _local_std = float(np.std(self._diff_hist[sym])) if len(self._diff_hist[sym]) > 5 else 0.0
                            if _local_std > 1e-10:
                                _z = _diff / _local_std
                                live_drift = 1 if _z > 0.5 else (-1 if _z < -0.5 else 0)
                        exec_drift = 0
                        _warmup_min = 5000 if not os.environ.get("VALIDATION_MODE") else 100
                        if getattr(self, '_warmup_ticks', 0) < _warmup_min:
                            exec_drift = 0
                        else:
                            exec_drift = live_drift
                        # Research drift: always live (for persistence telemetry, not execution)
                        research_drift = live_drift
                        # Compute OSS for all 3 horizons and blend by entropy weights
                        oss_3 = self._oss_bootstrap.predict_with_info(sym, ecdf, horizon=3, drift=exec_drift)
                        oss_10 = self._oss_bootstrap.predict_with_info(sym, ecdf, horizon=10, drift=exec_drift)
                        oss_20 = self._oss_bootstrap.predict_with_info(sym, ecdf, horizon=20, drift=exec_drift)
                        oss_list = [(w3, oss_3), (w10, oss_10), (w20, oss_20)]
                        def blend_float(key):
                            return sum(w * float(o.get(key, 0.0)) for w, o in oss_list)
                        def blend_bool(key):
                            return sum(w * float(o.get(key, 0.0)) for w, o in oss_list) / max(1e-10, w3 + w10 + w20)
                        p_cont = blend_float("p_cont")
                        p_cont = max(0.0, min(1.0, p_cont))
                        if exec_drift == 0 and not os.environ.get("VALIDATION_MODE"):
                            prod_sig = 0
                        elif p_cont >= 0.60:
                            prod_sig = exec_drift
                        elif p_cont <= 0.40:
                            prod_sig = -exec_drift
                        else:
                            prod_sig = 0
                        eval_data[sym]["expected_move"] = blend_float("mean_abs_move")
                        persist_hits = int(round(blend_float("persist_hits")))
                        persist_total = int(round(blend_float("persist_total")))
                        # Research-only lookup: blended persistence from live drift buckets
                        r_oss_3 = self._oss_bootstrap.predict_with_info(sym, ecdf, horizon=3, drift=research_drift)
                        r_oss_10 = self._oss_bootstrap.predict_with_info(sym, ecdf, horizon=10, drift=research_drift)
                        r_oss_20 = self._oss_bootstrap.predict_with_info(sym, ecdf, horizon=20, drift=research_drift)
                        r_oss_list = [(w3, r_oss_3), (w10, r_oss_10), (w20, r_oss_20)]
                        research_p_cont = max(0.0, min(1.0, sum(w * float(o.get("p_cont", 0.5)) for w, o in r_oss_list)))
                        research_ph = int(round(sum(w * float(o.get("persist_hits", 0)) for w, o in r_oss_list)))
                        research_pt = int(round(sum(w * float(o.get("persist_total", 0)) for w, o in r_oss_list)))
                        research_bucket = oss_10.get("diagnostics", {}).get("found_bucket", "?")
                        research_fr = oss_10.get("diagnostics", {}).get("fallback_reason", "?")
                        oss_info = oss_10  # nominal for log fields
                        logger.info(
                            f"[OSS SURFACE] {sym} "
                            f"ecdf={ecdf:.4f} exec_drift={exec_drift} live_drift={live_drift} "
                            f"horizon=blended(w3={w3:.2f},w10={w10:.2f},w20={w20:.2f}) regime={regime} "
                            f"p_cont={p_cont:.2f} ph={persist_hits} pt={persist_total} "
                            f"r_pc={research_p_cont:.2f} r_ph={research_ph} r_pt={research_pt} "
                            f"r_bucket={research_bucket} r_fb={research_fr} "
                            f"signal={prod_sig} "
                            f"up={oss_info.get('up_pct', 0):.1f}% dn={oss_info.get('dn_pct', 0):.1f}%"
                        )
                        prod_signals[sym] = prod_sig
                        # Store OSS metadata for Phase A arbitration
                        eval_data[sym]["oss_ev"] = oss_info.get("ev", 0.0)
                        eval_data[sym]["oss_conf"] = oss_info.get("confidence", 0.5)
                        eval_data[sym]["p_cont"] = p_cont
                        eval_data[sym]["oss_count"] = oss_info.get("count", 0)
                        eval_data[sym]["research_p_cont"] = research_p_cont
                        eval_data[sym]["research_drift"] = research_drift
                        eval_data[sym]["exec_drift"] = exec_drift
                        eval_data[sym]["oss_ev_signal"] = oss_info.get("ev_signal", 0)
                    else:
                        # Mean-reversion fallback: cheap → buy, expensive → sell
                        if ecdf <= 0.20:
                            prod_signals[sym] = 1
                        elif ecdf >= 0.80:
                            prod_signals[sym] = -1
                        else:
                            prod_signals[sym] = 0
                for sym, sig in prod_signals.items():
                    eval_data[sym]["prod_signal"] = sig
                # Update running EMA20 for drift
                if not hasattr(self, '_drift_ema'):
                    self._drift_ema = {}
                alpha = 2.0 / 21.0
                for sym in self._execution_symbols:
                    price = eval_data.get(sym, {}).get("price", 0.0)
                    if price > 0:
                        prev = self._drift_ema.get(sym, price)
                        self._drift_ema[sym] = alpha * price + (1 - alpha) * prev
                # DRIFT_AUDIT: compare exec vs research drift
                _exec_drifts = [eval_data.get(s, {}).get("exec_drift", 0) for s in self._execution_symbols if s in eval_data]
                _research_drifts = [eval_data.get(s, {}).get("research_drift", 0) for s in self._execution_symbols if s in eval_data]
                _exec_neg = sum(1 for d in _exec_drifts if d < 0)
                _exec_zero = sum(1 for d in _exec_drifts if d == 0)
                _exec_pos = sum(1 for d in _exec_drifts if d > 0)
                _research_neg = sum(1 for d in _research_drifts if d < 0)
                _research_zero = sum(1 for d in _research_drifts if d == 0)
                _research_pos = sum(1 for d in _research_drifts if d > 0)
                logger.info(f"[DRIFT_AUDIT] exec: neg={_exec_neg} zero={_exec_zero} pos={_exec_pos} | "
                            f"research: neg={_research_neg} zero={_research_zero} pos={_research_pos}")
                # OSS_SEMANTIC_AUDIT: compare EV-based signal vs persistence-based signal (research drift)
                _ev_sigs = []
                _persist_sigs = []
                for _s in self._execution_symbols:
                    if _s not in eval_data:
                        continue
                    _ed = eval_data[_s]
                    _rd = _ed.get("research_drift", 0)
                    _ecdf = _ed.get("ecdf_rank", 0.5)
                    if self._trained_oss and self._oss_bootstrap.has_surface(_s):
                        _rinfo = self._oss_bootstrap.predict_with_info(_s, _ecdf, horizon=10, drift=_rd)
                        _ev_sigs.append(_rinfo.get("ev_signal", 0))
                        _persist_sigs.append(_rinfo.get("signal", 0))
                _ev_buy = sum(1 for s in _ev_sigs if s == 1)
                _ev_sell = sum(1 for s in _ev_sigs if s == -1)
                _ev_flat = sum(1 for s in _ev_sigs if s == 0)
                _pst_buy = sum(1 for s in _persist_sigs if s == 1)
                _pst_sell = sum(1 for s in _persist_sigs if s == -1)
                _pst_flat = sum(1 for s in _persist_sigs if s == 0)
                _same = sum(1 for i in range(len(_ev_sigs)) if _ev_sigs[i] == _persist_sigs[i])
                _diff = sum(1 for i in range(len(_ev_sigs)) if _ev_sigs[i] != _persist_sigs[i] and _ev_sigs[i] != 0 and _persist_sigs[i] != 0)
                logger.info(f"[OSS_SEMANTIC_AUDIT] EV: buy={_ev_buy} sell={_ev_sell} flat={_ev_flat} | "
                            f"PERSIST: buy={_pst_buy} sell={_pst_sell} flat={_pst_flat} | "
                            f"same={_same} diff={_diff}")
                # OSS_STATE_USAGE: which drift buckets are exercised with research drift
                _db_d0 = sum(1 for s in self._execution_symbols if eval_data.get(s, {}).get("research_drift", 0) == 0)
                _db_dp = sum(1 for s in self._execution_symbols if eval_data.get(s, {}).get("research_drift", 0) == 1)
                _db_dn = sum(1 for s in self._execution_symbols if eval_data.get(s, {}).get("research_drift", 0) == -1)
                logger.info(f"[OSS_STATE_USAGE] drift=0:{_db_d0} drift=+1:{_db_dp} drift=-1:{_db_dn}")
                # DRIFT_DIST: compare training vs runtime drift distribution
                _rd_list = [eval_data.get(s, {}).get("research_drift", 0) for s in self._execution_symbols]
                _rd0 = sum(1 for d in _rd_list if d == 0)
                _rdp = sum(1 for d in _rd_list if d == 1)
                _rdn = sum(1 for d in _rd_list if d == -1)
                logger.info(f"[DRIFT_DIST] runtime: 0={_rd0} +1={_rdp} -1={_rdn}")

                # UCF computation — immutable field, computed once per cycle
                ucf_field = compute_ucf_field(
                    symbols=self._execution_symbols,
                    technical_states=self._current_technical_states if hasattr(self, '_current_technical_states') else {},
                    fsv_states=self._current_fsv_states if hasattr(self, '_current_fsv_states') else {},
                    regime_state={"regime": "neutral", "regime_stability": 0.5, "fsv_entropy": 0.0, "technical_volatility": 0.5, "recent_prediction_error": 0.0, "exposure_concentration": 0.0},
                )
                if ucf_field is not None:
                    if hasattr(self, '_fusion'):
                        self._fusion.ucf_field = ucf_field
                    self._cycle_context["ucf_field"] = ucf_field

                # GATE — SHADOW OBSERVABILITY LAYER (no execution control, log-only)
                _gate_log: dict[str, dict] = {}
                _high_vol_syms: list[str] = []
                _regime_degraded: bool = False
                _gate_regime_failures: int = 0
                for _gsym in self._execution_symbols:
                    _u = eval_data.get(_gsym, {})
                    _price = _u.get("price", 0.0)
                    _spread = _u.get("spread", 0.0002)
                    _latency = _u.get("fill_latency", 0.05)
                    _exp_slip = _u.get("expected_slippage", 0.0001)
                    _act_slip = _u.get("actual_slippage", 0.0001)
                    _rv = _u.get("recovery_velocity", random.uniform(0.2, 0.8))
                    _rc = _u.get("recovery_confidence", random.uniform(0.3, 0.9))
                    self._gate_mra.update(_gsym, _price, _spread)
                    self._gate_emd.record_fill(_gsym, _latency, _exp_slip, _act_slip)
                    self._gate_recovery.update_rv(_gsym, _rv)
                    self._gate_recovery.update_rc(_gsym, _rc)
                    _regime_vol = self._gate_mra.get_regime_volatility(_gsym)
                    self._gate_recovery.set_regime_volatility(_gsym, _regime_vol)
                    _dampen = _regime_vol > 0.7
                    if _dampen:
                        _high_vol_syms.append(_gsym)
                    _mra = self._gate_mra.get_mra(_gsym, dampen=_dampen)
                    _emd = self._gate_emd.get_emd(_gsym, dampen=_dampen)
                    _gdec = self._gate_recovery.resolve(_gsym)
                    _gate_log[_gsym] = {
                        "mra": _mra["mra_score"],
                        "emd": _emd["emd_score"],
                        "rv": _gdec["rv_score"],
                        "rc": _gdec["rc_score"],
                        "classification": _gdec["classification"],
                        "veto": _gdec["veto_applied"],
                        "regime_vol": round(_regime_vol, 4),
                    }
                    if _gdec["classification"] == "STRUCTURAL":
                        _gate_regime_failures += 1
                self._gate_decisions.append({
                    "cycle": self._cycle_id,
                    "symbols": _gate_log,
                    "high_vol_syms": len(_high_vol_syms),
                    "regime_degraded": _regime_degraded,
                    "structural_count": _gate_regime_failures,
                })
                if _high_vol_syms:
                    _hv_info = " ".join(f"{s}(vol={_gate_log[s]['regime_vol']:.2f})" for s in _high_vol_syms)
                    logger.info(f"[GATE_HIGH_VOL] dampened={len(_high_vol_syms)} syms={_hv_info}")

                # UCF alignment from field coherence (fallback 0.5 if unavailable)
                _ucf_field = self._cycle_context.get("ucf_field")
                _ucf_coherence = _ucf_field.field_coherence if _ucf_field is not None else 0.0
                if _ucf_coherence > 0.01:
                    _ucf_align = _ucf_coherence
                else:
                    _stre = getattr(self, '_last_stre_result', None)
                    if _stre is not None and _stre.get("samples", 0) >= 5:
                        _stas = abs(float(_stre.get("gt_corr", 0.0)) - float(_stre.get("sy_corr", 0.0)))
                        _sample_confidence = min(1.0, _stre.get("samples", 0) / 200.0)
                        _stas_score = max(0.0, 1.0 - _stas)
                        _ucf_align = 0.3 + 0.7 * (_stas_score * _sample_confidence)
                    else:
                        _ucf_align = 0.5
                _rf = check_regime_failure(_ucf_align)
                if _rf == "DEGRADED":
                    _regime_degraded = True
                if _regime_degraded or _gate_regime_failures > 0:
                    logger.info(f"[GATE_DEGRADED] regime={_regime_degraded} structural={_gate_regime_failures}")

                # PHASE 6 — Deployment Control (SHADOW mode initially, log-only)
                _p6_metrics = {
                    "alignment": min(1.0, max(0.0, _ucf_align)),
                    "rc_veto_rate": _gate_regime_failures / max(1, len(self._execution_symbols)),
                    "mra_score": sum(v["mra"] for v in _gate_log.values()) / max(1, len(_gate_log)),
                    "emd_score": sum(v["emd"] for v in _gate_log.values()) / max(1, len(_gate_log)),
                }
                _ks = self._phase6_killswitch.evaluate(_p6_metrics)
                if _ks["triggered"]:
                    logger.warning(f"[PHASE6_KILL_SWITCH] triggered reasons={_ks.get('failures', [])}")
                    self._phase6_rollout.force_state("SHADOW")
                    self._phase6_recovery.trigger(self._cycle_id)
                    self._phase6_current_mult = 0.0
                    self._phase6_audit.log_kill_switch(_p6_metrics, "; ".join(_ks.get("failures", [])))
                _roll = self._phase6_rollout.evaluate(_p6_metrics)
                if _roll.get("transition"):
                    self._phase6_audit.log_transition(
                        _roll.get("from_state", "SHADOW"), _roll["state"], _p6_metrics,
                        _roll.get("reason", "state_change"),
                    )
                    logger.info(f"[PHASE6_ROLLOUT] {_roll['direction']} -> {_roll['state']}")
                _scaling = self._phase6_scaling.evaluate(
                    _p6_metrics["alignment"],
                    _p6_metrics["rc_veto_rate"],
                    _p6_metrics["emd_score"],
                )
                self._phase6_current_mult = _scaling["position_size_multiplier"]
                if _roll["state"] == "SHADOW":
                    self._phase6_current_mult = 0.0
                _pv6 = self._phase6_recovery.evaluate(self._cycle_id, _p6_metrics["alignment"], _p6_metrics["rc_veto_rate"])
                if _pv6.get("active"):
                    self._phase6_current_mult = min(self._phase6_current_mult, _pv6.get("max_exposure", 1.0))
                    logger.info(f"[PHASE6_RECOVERY] phase={_pv6['phase']} max_exposure={_pv6['max_exposure']}")
                self._phase6_log.append({
                    "cycle": self._cycle_id,
                    "rollout_state": _roll["state"],
                    "kill_switch_triggered": _ks["triggered"],
                    "multiplier": self._phase6_current_mult,
                    "stability_score": _scaling["stability_score"],
                    "stability_tier": _scaling["stability_tier"],
                    "recovery_active": _pv6.get("active", False),
                    "recovery_phase": _pv6.get("phase", "NORMAL"),
                })

                # SHADOW PATH: V3/V4 Fusion Kernel (observability only, no execution authority)
                if self.config.fusion_kernel:
                    shadow_signals = self._fusion.generate(eval_data)
                else:
                    shadow_signals = {}
                for sym, sig in shadow_signals.items():
                    eval_data[sym]["shadow_signal"] = sig
                # Fallback: if no production signal, use shadow signal
                for sym in eval_data:
                    if "signal" not in eval_data[sym]:
                        eval_data[sym]["signal"] = eval_data[sym].get("prod_signal",
                                                                     eval_data[sym].get("shadow_signal", 0))
                # SHADOW_RAW telemetry: show shadow computation internals per symbol
                for _s in self._execution_symbols:
                    _sd = eval_data.get(_s, {})
                    _secdf = _sd.get("ecdf_rank", 0.5)
                    _sentropy = _sd.get("entropy", 0.5)
                    _sscore = _secdf - _sentropy
                    _ssig_raw = 1 if _sscore > 0.05 else (-1 if _sscore < -0.05 else 0)
                    _ssig_final = shadow_signals.get(_s, 0)
                    if _ssig_raw != _ssig_final or abs(_ssig_final) > 0:
                        logger.info(f"[SHADOW_RAW] {_s} ecdf={_secdf:.4f} entropy={_sentropy:.4f} "
                                    f"score={_sscore:+.4f} raw={int(_ssig_raw):+d} final={_ssig_final} "
                                    f"flip_suppress={_sentropy > 0.65}")
                logger.info(f"[SHADOW FUSION] { {k:v for k,v in shadow_signals.items() if k in self._execution_symbols} }")

                # SHADOW EXHAUST OVERRIDE: B3.2 — replace base shadow with exhaustion direction
                _shadow_base = dict(shadow_signals)
                _exhaust_overrides = {}
                for _es in self._execution_symbols:
                    _eed = eval_data.get(_es, {})
                    _eecdf = _eed.get("ecdf_rank", 0.5)
                    _etopo = self._entropy_compression.topology(_es, all_symbols=self._execution_symbols)
                    if _etopo.get("status") != "ACTIVE":
                        continue
                    _ex_result = self._fusion._detect_exhaustion(_es, _eecdf, _etopo)
                    if _ex_result["exhausted"]:
                        _old_sig = shadow_signals.get(_es, 0)
                        shadow_signals[_es] = _ex_result["direction"]
                        _exhaust_overrides[_es] = (_old_sig, _ex_result["direction"], _ex_result["reason"], _ex_result["score"])
                if _exhaust_overrides:
                    for _es, (_old, _new, _reason, _score) in _exhaust_overrides.items():
                        logger.info(f"[SHADOW_EXHAUST_OVERRIDE] {_es} base={int(_old):+d} exhaust={int(_new):+d} "
                                    f"reason={_reason} score={_score:.3f}")
                self._exhaust_override_count = getattr(self, '_exhaust_override_count', 0) + len(_exhaust_overrides)
                _exhaust_rate = self._exhaust_override_count / max(self._cycle_id, 1)
                logger.info(f"[SHADOW_EXHAUST_RATE] cycle={self._cycle_id} "
                            f"overrides_this={len(_exhaust_overrides)} total={self._exhaust_override_count} "
                            f"rate={_exhaust_rate:.3f}/cycle")

                # PHASE A: Confidence-gated arbitration (replaces weighted ensemble)
                import math
                # Compute system-level avg_entropy for regime classification
                _entropies = [v.get("entropy", 0.5) for v in eval_data.values()]
                _avg_entropy = sum(_entropies) / max(len(_entropies), 1)
                _regime = "CHAOTIC" if _avg_entropy > 0.65 else ("STRUCTURED" if _avg_entropy < 0.40 else "TRANSITION")
                _reg_req = {"STRUCTURED": 0.60, "TRANSITION": 0.55, "CHAOTIC": 0.50}
                for sym in self._execution_symbols:
                    oss_sig = prod_signals.get(sym, 0)
                    shadow_sig = shadow_signals.get(sym, 0)
                    # OSS confidence
                    oss_ev = eval_data.get(sym, {}).get("oss_ev", 0.0)
                    oss_mag = 1.0 / (1.0 + math.exp(-8.0 * (abs(oss_ev) - 0.05))) if oss_ev != 0 else 0.0
                    oss_conf_raw = eval_data.get(sym, {}).get("oss_conf", 0.5)
                    oss_support = min(1.0, oss_conf_raw * 5.0)  # proxy count/20
                    oss_conf = oss_mag * oss_support
                    # Shadow confidence (no entropy multiplier per Phase A)
                    _entropy = eval_data.get(sym, {}).get("entropy", 0.5)
                    _score = eval_data.get(sym, {}).get("ecdf_rank", 0.5) - _entropy
                    shadow_conf = min(1.0, abs(_score) / 0.30)
                    # Arbitration
                    if oss_sig == 0 and shadow_sig == 0:
                        prod_signal = 0
                        _reason = "BOTH_ZERO"
                    elif oss_sig == shadow_sig:
                        prod_signal = oss_sig
                        _reason = "AGREE"
                    elif oss_sig == 0:
                        if shadow_conf > 0.25:
                            prod_signal = shadow_sig
                            _reason = f"OSS_NEUTRAL_{'BUY' if shadow_sig>0 else 'SELL'}"
                            logger.info(f"[PURE_SHADOW_TRADE] {sym} oss=0 shadow={int(shadow_sig):+d} "
                                        f"conf={shadow_conf:.2f} regime={_regime}")
                        else:
                            prod_signal = 0
                            _reason = "OSS_NEUTRAL_FLAT"
                    elif shadow_sig == 0:
                        prod_signal = oss_sig
                        _reason = "SHADOW_NEUTRAL"
                    else:
                        # Disagreement
                        if shadow_conf > _reg_req[_regime]:
                            prod_signal = shadow_sig
                            _reason = f"SHADOW_OVERRIDE({_regime})"
                        elif oss_conf > 0.60:
                            prod_signal = oss_sig
                            _reason = f"OSS_OVERRIDE({_regime})"
                        else:
                            prod_signal = 0
                            _reason = "CONFLICT_FLAT"
                    eval_data[sym]["prod_signal"] = prod_signal
                    prod_signals[sym] = prod_signal
                    eval_data[sym]["arb_reason"] = _reason
                    _pc = eval_data.get(sym, {}).get("research_p_cont", 0.5)
                    _ev_sig_show = eval_data.get(sym, {}).get("oss_ev_signal", 0)
                    logger.info(f"[PROD_SIGNAL_BREAKDOWN] {sym} "
                                f"oss={int(oss_sig):+d}(ev={oss_ev:.4f},conf={oss_conf:.2f}) "
                                f"ev_sig={_ev_sig_show:+d} "
                                f"shadow={int(shadow_sig):+d}(conf={shadow_conf:.2f}) "
                                f"regime={_regime} reason={_reason} final={int(prod_signal):+d} "
                                f"pc={_pc:.2f}")
                    # Track branch usage distribution
                    if not hasattr(self, '_branch_counts'):
                        self._branch_counts = {}
                    self._branch_counts[_reason] = self._branch_counts.get(_reason, 0) + 1
                    _orig = prod_signal
                    _gated = self._rf_gate.gate(sym, prod_signal)
                    if _gated != _orig:
                        _p = self._rf_gate.prob(sym)
                        logger.info(f"[RF GATE] {sym}: blocked signal {_orig}->0 prob={_p:.3f} < {self._rf_gate.prob_thresh}")
                        eval_data[sym]["prod_signal"] = _gated
                        prod_signals[sym] = _gated
                    # Track RF gate survival
                    if not hasattr(self, '_pre_rf_count'):
                        self._pre_rf_count = 0
                    if not hasattr(self, '_post_rf_count'):
                        self._post_rf_count = 0
                    if _orig != 0:
                        self._pre_rf_count += 1
                    if _gated != 0:
                        self._post_rf_count += 1
                logger.info(f"[PROD SIGNAL] { {k: v for k, v in prod_signals.items() if k in self._execution_symbols} } (phase A arbitration)")
                # Shadow Mirror — capture final signal outputs (SZ3)
                for _sym_sz3, _sig_sz3 in prod_signals.items():
                    if _sym_sz3 not in eval_data:
                        continue
                    _ed_sz3 = eval_data[_sym_sz3]
                    self._shadow_mirror.observe_signal(
                        symbol=_sym_sz3,
                        direction=_sig_sz3,
                        strength=_ed_sz3.get("research_p_cont", 0.5),
                        ecdf_rank=_ed_sz3.get("ecdf_rank", 0.5),
                        confidence=_ed_sz3.get("oss_conf", _ed_sz3.get("research_p_cont", 0.5)),
                        source="arbitration",
                        horizon=10,
                    )
                    _sz3_conv = _ed_sz3.get("research_p_cont", 0.5)
                    self._shadow_orchestrator.registry.intercept(
                        "L0_Raw", _sym_sz3, {"conviction": _sz3_conv, "direction": _sig_sz3}
                    )
                    self._shadow_orchestrator.registry.intercept(
                        "L1_DecisionGate", _sym_sz3, {"conviction": _sz3_conv * 0.95, "direction": _sig_sz3}
                    )
                    # Ground-truth capture
                    self._shadow_gt.capture("L0_GT", _sym_sz3, _ed_sz3)
                    self._shadow_gt.capture("L1_GT", _sym_sz3, _ed_sz3)
                # TOPOLOGY_DECISION: topology vs arbitration cross-reference
                _topo_dec = defaultdict(lambda: {"symbols": 0, "AGREE": 0, "SHADOW_OVERRIDE": 0, "OSS_OVERRIDE": 0, "CONFLICT": 0, "NEUTRAL": 0})
                for _s in self._execution_symbols:
                    if _s not in eval_data:
                        continue
                    _t = self._entropy_compression.topology(_s, all_symbols=list(eval_data.keys()))
                    if _t.get("status") != "ACTIVE":
                        continue
                    _tp = _t["topology"]
                    _topo_dec[_tp]["symbols"] += 1
                    _reason = eval_data.get(_s, {}).get("arb_reason", "?")
                    if "AGREE" in _reason:
                        _topo_dec[_tp]["AGREE"] += 1
                    elif "SHADOW_OVERRIDE" in _reason or "PURE_SHADOW" in _reason:
                        _topo_dec[_tp]["SHADOW_OVERRIDE"] += 1
                    elif "OSS_OVERRIDE" in _reason:
                        _topo_dec[_tp]["OSS_OVERRIDE"] += 1
                    elif "CONFLICT" in _reason:
                        _topo_dec[_tp]["CONFLICT"] += 1
                    else:
                        _topo_dec[_tp]["NEUTRAL"] += 1
                for _tp in sorted(_topo_dec.keys()):
                    _d = _topo_dec[_tp]
                    logger.info(f"[TOPOLOGY_DECISION] {_tp}: syms={_d['symbols']} "
                                f"agree={_d['AGREE']} shadow={_d['SHADOW_OVERRIDE']} "
                                f"oss={_d['OSS_OVERRIDE']} conflict={_d['CONFLICT']} neutral={_d['NEUTRAL']}")
                # PERSIST_COMPARE: compare exec OSS vs research persistence + Shadow arbitration
                _same = 0
                _diff = 0
                _exec_final_by_dir = {}
                _research_final_by_dir = {}
                for _s in self._execution_symbols:
                    if _s not in eval_data:
                        continue
                    _ed = eval_data[_s]
                    _shadow_sig = shadow_signals.get(_s, 0)
                    _ecdf = _ed.get("ecdf_rank", 0.5)
                    _entropy = _ed.get("entropy", 0.5)
                    _shadow_conf = min(1.0, abs(_ed.get("ecdf_rank", 0.5) - _entropy) / 0.30)
                    _exec_oss_sig = prod_signals.get(_s, 0)
                    _research_oss_sig = 0
                    if self._trained_oss and self._oss_bootstrap.has_surface(_s):
                        _rinfo = self._oss_bootstrap.predict_with_info(_s, _ecdf, horizon=10,
                                    drift=_ed.get("research_drift", 0))
                        _research_oss_sig = _rinfo.get("signal", 0)
                    _exec_final = _exec_oss_sig  # simplified: actual final from arbitration already in prod_signals
                    # Research arbitration (same Phase A logic)
                    if _research_oss_sig == 0 and _shadow_sig == 0:
                        _research_final = 0
                    elif _research_oss_sig == _shadow_sig:
                        _research_final = _research_oss_sig
                    elif _research_oss_sig == 0:
                        _research_final = _shadow_sig if _shadow_conf > 0.25 else 0
                    elif _shadow_sig == 0:
                        _research_final = _research_oss_sig
                    else:
                        _shadow_conf_v = min(1.0, abs(_ed.get("ecdf_rank", 0.5) - _entropy) / 0.30)
                        _oss_ev = _ed.get("oss_ev", 0.0)
                        _oss_mag = 1.0 / (1.0 + math.exp(-8.0 * (abs(_oss_ev) - 0.05))) if _oss_ev != 0 else 0.0
                        _oss_conf = _oss_mag * min(1.0, _ed.get("oss_conf", 0.5) * 5.0)
                        if _shadow_conf_v > _reg_req[_regime]:
                            _research_final = _shadow_sig
                        elif _oss_conf > 0.60:
                            _research_final = _research_oss_sig
                        else:
                            _research_final = 0
                    if _exec_final == _research_final:
                        _same += 1
                    else:
                        _diff += 1
                    # Count direction changes
                    _dir_key = f"exec_{_exec_final:+d}_res_{_research_final:+d}"
                    _exec_final_by_dir[_dir_key] = _exec_final_by_dir.get(_dir_key, 0) + 1
                logger.info(f"[PERSIST_COMPARE] same={_same} diff={_diff} details={_exec_final_by_dir}")
                oss_buy = sum(1 for s in self._execution_symbols if prod_signals.get(s, 0) == 1)
                oss_sell = sum(1 for s in self._execution_symbols if prod_signals.get(s, 0) == -1)
                shadow_buy = sum(1 for s in self._execution_symbols if shadow_signals.get(s, 0) == 1)
                shadow_sell = sum(1 for s in self._execution_symbols if shadow_signals.get(s, 0) == -1)
                prod_buy = sum(1 for s in self._execution_symbols if prod_signals.get(s, 0) == 1)
                prod_sell = sum(1 for s in self._execution_symbols if prod_signals.get(s, 0) == -1)
                logger.info(f"[POLARITY_AUDIT] OSS: buy={oss_buy} sell={oss_sell} | SHADOW: buy={shadow_buy} sell={shadow_sell} | PROD: buy={prod_buy} sell={prod_sell}")
                # THESIS_BUFFER telemetry: B4 thesis validity tracking
                _ts = self._thesis_buffer.stats()
                logger.info(f"[THESIS_BUFFER] total={_ts['total']} pending={_ts['pending']} "
                            f"resolved={_ts['resolved']} pos_rate={_ts['positive_rate']:.3f}")
                if _ts['resolved'] > 0:
                    logger.info(f"[THESIS_LABELS] positive={sum(1 for r in self._thesis_buffer.get_resolved() if r.label==1)} "
                                f"negative={sum(1 for r in self._thesis_buffer.get_resolved() if r.label==0)} "
                                f"pos_rate={_ts['positive_rate']:.3f}")
                _gs = self._thesis_graph.graph_stats()
                if _gs["registered"] > 0:
                    logger.info(f"[THESIS_GRAPH] reg={_gs['registered']} "
                                f"resolved={_gs['probes_resolved']} "
                                f"fracture={_gs['fracture_rate']:.2f} "
                                f"vectors={_gs['distinct_vectors']} "
                                f"top={_gs['vectors'][:3] if _gs['vectors'] else '[]'}")
                # Branch usage report
                if hasattr(self, '_branch_counts') and self._branch_counts:
                    _total = sum(self._branch_counts.values())
                    _pcts = {k: f"{v/_total*100:.1f}%" for k, v in sorted(self._branch_counts.items(), key=lambda x: -x[1])}
                    logger.info(f"[BRANCH_USAGE] total={_total} dist={_pcts}")
                # RF Gate audit: signal survival before/after gate
                _pre = getattr(self, '_pre_rf_count', 0)
                _post = getattr(self, '_post_rf_count', 0)
                logger.info(f"[RF_AUDIT] pre_rf_nonzero={_pre} post_rf_nonzero={_post} lost={_pre - _post}")
                self._pre_rf_count = 0
                self._post_rf_count = 0
                # Entropy distribution audit
                _e_all = [v.get("entropy", 0.5) for v in eval_data.values()]
                _e_arr = sorted(_e_all)
                _e_n = len(_e_arr)
                _e_mean = sum(_e_all) / max(_e_n, 1)
                _e_p10 = _e_arr[max(0, int(_e_n * 0.1) - 1)] if _e_n > 0 else 0
                _e_p50 = _e_arr[_e_n // 2] if _e_n > 0 else 0
                _e_p90 = _e_arr[min(_e_n - 1, int(_e_n * 0.9))] if _e_n > 0 else 0
                _e_struct = sum(1 for e in _e_all if e < 0.40)
                _e_trans = sum(1 for e in _e_all if 0.40 <= e <= 0.65)
                _e_chaos = sum(1 for e in _e_all if e > 0.65)
                logger.info(f"[ENTROPY_AUDIT] mean={_e_mean:.4f} p10={_e_p10:.4f} p50={_e_p50:.4f} p90={_e_p90:.4f} "
                            f"STRUCTURED={_e_struct} TRANSITION={_e_trans} CHAOTIC={_e_chaos}")
                # ENTROPY_COMPONENTS: decompose for most and least chaotic symbols
                _e_syms = [(v.get("entropy", 0.5), s) for s, v in eval_data.items()]
                if _e_syms:
                    _e_syms.sort(key=lambda x: -x[0])
                    _most_chaotic = _e_syms[0][1]
                    _least_chaotic = _e_syms[-1][1]
                    for _ec_label, _ec_sym in [("MOST_CHAOTIC", _most_chaotic), ("LEAST_CHAOTIC", _least_chaotic)]:
                        _ec = self._entropy_compression.decompose(_ec_sym)
                        if _ec.get("status") == "ACTIVE":
                            logger.info(f"[ENTROPY_COMPONENTS] {_ec_label}={_ec_sym} "
                                        f"occupied={_ec['occupied_bins']}/{_ec['total_bins']} "
                                        f"dominant_p={_ec['dominant_prob']:.3f} "
                                        f"raw_H={_ec['raw_entropy']:.3f} norm_H={_ec['normalized_entropy']:.3f} "
                                        f"max_H={_ec['max_entropy']:.3f} window={_ec['window']}")
                # ENTROPY_WINDOW_AUDIT: multi-window entropy for most chaotic symbol
                if _e_syms:
                    _wa = self._entropy_compression.window_audit(_most_chaotic)
                    if _wa.get(f"H{_wa.get('symbol') and '32' or '32'}") is not None or True:
                        def _fmt_h(v): return f"{v:.4f}" if v is not None else "None"
                        def _fmt_n(v): return f"{v:.3f}" if v is not None else "None"
                        _wa_str = " ".join(f"H{w}={_fmt_h(_wa.get(f'H{w}'))}(n={_fmt_n(_wa.get(f'nH{w}'))})"
                                           for w in [32, 64, 128, 256])
                        logger.info(f"[ENTROPY_WINDOW_AUDIT] {_most_chaotic} {_wa_str}")
                # ENTROPY_RANK: cross-sectional entropy ranking for all symbols
                _e_sym_list = list(eval_data.keys())
                _topology_states = defaultdict(int)
                _topology_transitions = defaultdict(int)
                for _es in _e_sym_list:
                    try:
                        _t = self._entropy_compression.topology(_es, all_symbols=_e_sym_list)
                        if _t.get("status") == "ACTIVE":
                            _topology_states[_t["topology"]] += 1
                            _prev = self._topology_prev.get(_es)
                            if _prev and _prev != _t["topology"]:
                                _topology_transitions[f"{_prev}->{_t['topology']}"] += 1
                            self._topology_prev[_es] = _t["topology"]
                        else:
                            logger.info(f"[ENTROPY_RANK_DEBUG] {_es} topology status={_t.get('status')}")
                    except Exception as _e:
                        logger.error(f"[ENTROPY_RANK_ERR] {_es}: {_e}")
                _top_str = " ".join(f"{k}={v}" for k, v in sorted(_topology_states.items()))
                logger.info(f"[ENTROPY_RANK] {_top_str}")
                if _topology_transitions:
                    _trans_str = " ".join(f"{k}={v}" for k, v in sorted(_topology_transitions.items(), key=lambda x: -x[1])[:5])
                    logger.info(f"[TOPOLOGY_TRANSITION] {_trans_str}")
                # EXHAUST_DETAIL: per-symbol exhaustion detection
                _exhaust_buy = 0
                _exhaust_sell = 0
                _exhaust_none = 0
                _exhaust_events = []
                _exhaust_ecdf_vals = sorted([eval_data.get(s, {}).get("ecdf_rank", 0.5) for s in _e_sym_list])
                if _exhaust_ecdf_vals:
                    logger.info(f"[EXHAUST_DBG] ecdf_range: min={_exhaust_ecdf_vals[0]:.4f} max={_exhaust_ecdf_vals[-1]:.4f} n_extreme_ge85={sum(1 for v in _exhaust_ecdf_vals if v>=0.85)} n_extreme_le15={sum(1 for v in _exhaust_ecdf_vals if v<=0.15)}")
                for _es in _e_sym_list:
                    _eed = eval_data.get(_es, {})
                    _eecdf = _eed.get("ecdf_rank", 0.5)
                    _etopo = self._entropy_compression.topology(_es, all_symbols=_e_sym_list)
                    if _etopo.get("status") != "ACTIVE":
                        continue
                    _ex_result = self._fusion._detect_exhaustion(_es, _eecdf, _etopo)
                    if _ex_result["exhausted"]:
                        _exhaust_events.append((_es, _ex_result, _etopo))
                        if _ex_result["direction"] == 1:
                            _exhaust_buy += 1
                        elif _ex_result["direction"] == -1:
                            _exhaust_sell += 1
                        logger.info(f"[EXHAUST_DETAIL] {_es} ecdf={_eecdf:.4f} "
                                    f"topo={_etopo.get('topology','?')} "
                                    f"H={_etopo.get('entropy',0):.4f} "
                                    f"dH={_etopo.get('d_entropy',0):+.4f} "
                                    f"pmax={_etopo.get('dominant_prob',0):.3f} "
                                    f"dp={_etopo.get('d_pmax',0):+.4f} "
                                    f"-> {_ex_result['reason']} "
                                    f"score={_ex_result['score']:.3f}")
                    else:
                        _exhaust_none += 1
                logger.info(f"[EXHAUST_AUDIT] buy={_exhaust_buy} sell={_exhaust_sell} none={_exhaust_none}")
                # EXHAUST_NEAR_MISS: count failures by gate
                _nm_ecdf = 0
                _nm_entropy = 0
                _nm_dH = 0
                _nm_dp = 0
                for _es in _e_sym_list:
                    _exh = self._fusion._exhaustion_hist.get(_es, {})
                    _nm = _exh.get("near_miss", {})
                    if _nm.get("ecdf_fail"):
                        _nm_ecdf += 1
                    if _nm.get("entropy_fail"):
                        _nm_entropy += 1
                    if _nm.get("dH_fail"):
                        _nm_dH += 1
                    if _nm.get("dp_fail"):
                        _nm_dp += 1
                logger.info(f"[EXHAUST_NEAR_MISS] ecdf_fail={_nm_ecdf} entropy_fail={_nm_entropy} dH_fail={_nm_dH} dp_fail={_nm_dp}")
                # EXHAUST_NEAREST: symbol closest to triggering exhaustion
                _best_margin = float('inf')
                _best_sym = None
                _best_detail = {}
                for _es in _e_sym_list:
                    _eed = eval_data.get(_es, {})
                    _eecdf = _eed.get("ecdf_rank", 0.5)
                    _etopo = self._entropy_compression.topology(_es, all_symbols=_e_sym_list)
                    if _etopo.get("status") != "ACTIVE":
                        continue
                    _exh = self._fusion._exhaustion_hist.get(_es, {})
                    if _exh.get("exhausted"):
                        continue
                    _H = _etopo.get("entropy", 0.5)
                    _dH = _etopo.get("d_entropy", 0)
                    _dp = _etopo.get("d_pmax", 0)
                    _m_sell = max(0.80 - _eecdf, 0.88 - _H, -_dH, -0.010 - _dp)
                    _m_buy = max(_eecdf - 0.20, 0.88 - _H, -_dH, -0.010 - _dp)
                    _m = min(_m_sell, _m_buy)
                    if _m < _best_margin:
                        _best_margin = _m
                        _best_sym = _es
                        _failed = []
                        if _eecdf > 0.20 and _eecdf < 0.80:
                            _failed.append("ecdf")
                        if _H < 0.88:
                            _failed.append("H")
                        if _dH < 0:
                            _failed.append("dH")
                        if _dp > -0.010:
                            _failed.append("dp")
                        _best_detail = {"ecdf": _eecdf, "H": _H, "dH": _dH, "dp": _dp, "failed": _failed, "margin": round(_m, 4)}
                if _best_sym:
                    logger.info(f"[EXHAUST_NEAREST] {_best_sym} "
                                f"ecdf={_best_detail['ecdf']:.4f} H={_best_detail['H']:.4f} "
                                f"dH={_best_detail['dH']:+.4f} dp={_best_detail['dp']:+.4f} "
                                f"failed={'/'.join(_best_detail['failed'])} "
                                f"margin={_best_detail['margin']:.4f}")
                # EXHAUST_VS_SHADOW: compare exhaustion direction vs shadow base signal
                _ex_base_buy = 0
                _ex_base_sell = 0
                _ex_agree = 0
                _ex_conflict = 0
                _ex_shadow_sigs = _shadow_base if isinstance(_shadow_base, dict) else (shadow_signals if isinstance(shadow_signals, dict) else {})
                for _es, _ex_res, _etopo in _exhaust_events:
                    _shadow_sig = _ex_shadow_sigs.get(_es, 0)
                    if _shadow_sig == _ex_res["direction"]:
                        _ex_agree += 1
                    else:
                        _ex_conflict += 1
                    if _shadow_sig == 1:
                        _ex_base_buy += 1
                    elif _shadow_sig == -1:
                        _ex_base_sell += 1
                if _exhaust_events:
                    logger.info(f"[EXHAUST_VS_SHADOW] base=BUY:{_ex_base_buy} SELL:{_ex_base_sell} "
                                f"agree={_ex_agree} conflict={_ex_conflict}")
                # ENTROPY_DH: per-symbol entropy + pmax momentum (most chaotic + a sample)
                for _ed_sym in [_most_chaotic, _least_chaotic]:
                    if _ed_sym in _e_sym_list:
                        _t = self._entropy_compression.topology(_ed_sym, all_symbols=_e_sym_list)
                        if _t.get("status") == "ACTIVE":
                            logger.info(f"[ENTROPY_DH] {_ed_sym} "
                                        f"H={_t['entropy']:.4f} dH={_t['d_entropy']:+.4f} "
                                        f"pmax={_t['dominant_prob']:.3f} dpmax={_t['d_pmax']:+.4f} "
                                        f"emaH={_t['ema_h']:.4f} emaP={_t['ema_pmax']:.3f}")
                # Persistence audit: distribution of research_p_cont across symbols
                _p_all = [v.get("research_p_cont", 0.5) for v in eval_data.values() if v.get("research_p_cont", 0.5) != 0.5 or v.get("oss_count", 0) > 0]
                if _p_all:
                    _p_arr = sorted(_p_all)
                    _p_n = len(_p_arr)
                    _p_mean = sum(_p_all) / max(_p_n, 1)
                    _p_p10 = _p_arr[max(0, int(_p_n * 0.1) - 1)] if _p_n > 0 else 0
                    _p_p50 = _p_arr[_p_n // 2] if _p_n > 0 else 0
                    _p_p90 = _p_arr[min(_p_n - 1, int(_p_n * 0.9))] if _p_n > 0 else 0
                    _p_low = sum(1 for p in _p_all if p < 0.40)
                    _p_mid = sum(1 for p in _p_all if 0.40 <= p <= 0.60)
                    _p_high = sum(1 for p in _p_all if p > 0.60)
                    logger.info(f"[PERSIST_AUDIT] mean={_p_mean:.4f} p10={_p_p10:.4f} p50={_p_p50:.4f} p90={_p_p90:.4f} "
                                f"low={_p_low} mid={_p_mid} high={_p_high}")
                # Persistence vs EV divergence matrix
                _div_ev_pos_cont_high = 0
                _div_ev_pos_cont_low = 0
                _div_ev_neg_cont_high = 0
                _div_ev_neg_cont_low = 0
                _div_ev_zero_cont_high = 0
                _div_ev_zero_cont_low = 0
                _div_total = 0
                for _sym in self._execution_symbols:
                    _ed = eval_data.get(_sym, {})
                    _ev = _ed.get("oss_ev", 0.0)
                    _pc = _ed.get("research_p_cont", 0.5)
                    if _pc == 0.5 and not _ed.get("oss_count", 0):
                        continue
                    _div_total += 1
                    if _ev > 0.01:
                        if _pc > 0.60:
                            _div_ev_pos_cont_high += 1
                        elif _pc < 0.40:
                            _div_ev_pos_cont_low += 1
                    elif _ev < -0.01:
                        if _pc > 0.60:
                            _div_ev_neg_cont_high += 1
                        elif _pc < 0.40:
                            _div_ev_neg_cont_low += 1
                    else:
                        if _pc > 0.60:
                            _div_ev_zero_cont_high += 1
                        elif _pc < 0.40:
                            _div_ev_zero_cont_low += 1
                logger.info(f"[PERSIST_DIVERGENCE] total={_div_total} "
                            f"EV+_P_CONT>0.60={_div_ev_pos_cont_high} "
                            f"EV+_P_CONT<0.40={_div_ev_pos_cont_low} "
                            f"EV-_P_CONT>0.60={_div_ev_neg_cont_high} "
                            f"EV-_P_CONT<0.40={_div_ev_neg_cont_low} "
                            f"EV≈0_P_CONT>0.60={_div_ev_zero_cont_high} "
                            f"EV≈0_P_CONT<0.40={_div_ev_zero_cont_low}")
                # Bucket coverage histogram
                _bc_all = [eval_data.get(_sym, {}).get("oss_count", 0) for _sym in self._execution_symbols]
                if _bc_all:
                    _bc_0_5 = sum(1 for c in _bc_all if 0 < c <= 5)
                    _bc_6_10 = sum(1 for c in _bc_all if 6 <= c <= 10)
                    _bc_11_20 = sum(1 for c in _bc_all if 11 <= c <= 20)
                    _bc_20p = sum(1 for c in _bc_all if c > 20)
                    _bc_total = sum(1 for c in _bc_all if c > 0)
                    logger.info(f"[PERSIST_COVERAGE] total={_bc_total} 1-5={_bc_0_5} 6-10={_bc_6_10} 11-20={_bc_11_20} 20+={_bc_20p}")
                # SURFACE_DIAGNOSTICS: bucket occupancy, lookup hit rate, fallback reasons
                _diag_exact = 0
                _diag_drift_fb = 0
                _diag_default = 0
                _diag_buckets = {}
                _diag_total = 0
                for _s in self._execution_symbols:
                    if _s not in eval_data:
                        continue
                    _ed = eval_data[_s]
                    _rd = _ed.get("research_drift", 0)
                    _ecdf = _ed.get("ecdf_rank", 0.5)
                    if self._trained_oss and self._oss_bootstrap.has_surface(_s):
                        _rinfo = self._oss_bootstrap.predict_with_info(_s, _ecdf, horizon=10, drift=_rd)
                        _diag = _rinfo.get("diagnostics", {})
                        _diag_total += 1
                        _fr = _diag.get("fallback_reason", "unknown")
                        if _fr == "exact":
                            _diag_exact += 1
                        elif _fr == "drift_fallback":
                            _diag_drift_fb += 1
                        else:
                            _diag_default += 1
                        _req = _diag.get("requested_bucket", "?")
                        _found = _diag.get("found_bucket", "?")
                        _avail = _diag.get("available_drifts", [])
                        _key = f"{_req}->{_found}"
                        _diag_buckets[_key] = _diag_buckets.get(_key, 0) + 1
                logger.info(f"[SURFACE_DIAGNOSTICS] total={_diag_total} "
                            f"exact={_diag_exact} drift_fallback={_diag_drift_fb} default={_diag_default}")
                if _diag_buckets:
                    _top = sorted(_diag_buckets.items(), key=lambda x: -x[1])[:10]
                    logger.info(f"[SURFACE_DIAGNOSTICS] top_routes: {dict(_top)}")
                # Cache occupancy dump (first surface symbol)
                if self._trained_oss:
                    _any_sym = next((s for s in self._execution_symbols if self._oss_bootstrap.has_surface(s)), None)
                    if _any_sym:
                        _occ = self._oss_bootstrap.get_model(_any_sym, 10).get_cache_occupancy()
                        _tot_buckets = len(_occ)
                        _pop_buckets = sum(1 for v in _occ.values() if v["count"] >= 10)
                        _persist_buckets = sum(1 for v in _occ.values() if v["persist_total"] > 0)
                        logger.info(f"[SURFACE_DIAGNOSTICS] {_any_sym}: buckets={_tot_buckets} "
                                    f"populated(>=10)={_pop_buckets} persist_nonzero={_persist_buckets}")
                        # Log sparse vs populated ECDF-drift combos
                        _by_ecdf = {}
                        for _k, _v in _occ.items():
                            _ec, _dr = _k.split("|")
                            _by_ecdf.setdefault(_ec, {})[int(_dr)] = _v["count"]
                        _sparse = sum(1 for ec, drs in _by_ecdf.items() if len(drs) < 3)
                        logger.info(f"[SURFACE_DIAGNOSTICS] ecdf_with_all_3_drifts={len(_by_ecdf)-_sparse}/{len(_by_ecdf)} "
                                    f"sparse(lt_3)={_sparse}")
                        # Persist_total distribution across buckets
                        _pt_vals = sorted([v["persist_total"] for v in _occ.values() if v["persist_total"] > 0])
                        if _pt_vals:
                            _n = len(_pt_vals)
                            _pt_p10 = _pt_vals[max(0, int(_n * 0.1) - 1)]
                            _pt_p50 = _pt_vals[_n // 2]
                            _pt_p90 = _pt_vals[min(_n - 1, int(_n * 0.9))]
                            _pt_max = _pt_vals[-1]
                            _pt_mean = sum(_pt_vals) / _n
                        else:
                            _pt_p10 = _pt_p50 = _pt_p90 = _pt_max = _pt_mean = 0
                        logger.info(f"[PERSIST_STATS] {_any_sym}: non_zero_buckets={len(_pt_vals)} "
                                    f"mean={_pt_mean:.1f} p10={_pt_p10} p50={_pt_p50} p90={_pt_p90} max={_pt_max}")
                # RF probability distribution
                _rf_probs = [self._rf_gate.prob(s) for s in self._execution_symbols]
                if _rf_probs:
                    _rf_arr = sorted(_rf_probs)
                    _rf_p10 = _rf_arr[max(0, int(len(_rf_arr)*0.1)-1)]
                    _rf_p50 = _rf_arr[len(_rf_arr)//2]
                    _rf_p90 = _rf_arr[min(len(_rf_arr)-1, int(len(_rf_arr)*0.9))]
                    logger.info(f"[RF_PROB_DIST] mean={sum(_rf_probs)/len(_rf_probs):.4f} "
                                f"p10={_rf_p10:.4f} p50={_rf_p50:.4f} p90={_rf_p90:.4f}")
                # RF state telemetry: model loaded, warmup progress per symbol
                _rf_loaded = self._rf_gate._initd and self._rf_gate.model is not None
                _rf_ready_ct = sum(1 for s in self._execution_symbols if self._rf_gate.ready(s))
                _rf_warmup_progress = {
                    s: len(self._rf_gate._tpi_vals.get(s, []))
                    for s in self._execution_symbols
                }
                _rf_min_ticks = min(_rf_warmup_progress.values()) if _rf_warmup_progress else 0
                _rf_max_ticks = max(_rf_warmup_progress.values()) if _rf_warmup_progress else 0
                logger.info(f"[RF_STATE] model_loaded={_rf_loaded} ready_ct={_rf_ready_ct}/{len(self._execution_symbols)} "
                            f"tick_range=[{_rf_min_ticks},{_rf_max_ticks}] window={self._rf_gate.window}")
                # TOP3 churn tracking
                _cur_top3 = getattr(self, '_top3_submit_set', set())
                _prev_top3 = getattr(self, '_prev_top3_set', set())
                if _prev_top3:
                    _entered = _cur_top3 - _prev_top3
                    _exited = _prev_top3 - _cur_top3
                    _held = _prev_top3 & _cur_top3
                    _churn = len(_entered | _exited) / max(len(_prev_top3 | _cur_top3), 1)
                    if _entered or _exited:
                        logger.info(f"[TOP3_CHURN] entered={_entered} exited={_exited} held={_held} churn={_churn:.2f}")
                self._prev_top3_set = _cur_top3.copy()



                # V3/V4 STEP 5: H20 CAP ALLOCATION
                if self.config.h20:
                    self._allocations = self._h20.allocate(selected, eval_data)
                else:
                    self._allocations = {sym: 1.0 / max(1, len(selected)) for sym in selected}
                logger.info(f"[H20 ALLOCATION] {self._allocations}")

                # V3/V4 STEP 6: EXECUTION MAPPER
                if self.config.execution:
                    self._execution_plan = self._exec_mapper.map(self._allocations, eval_data)
                else:
                    self._execution_plan = {}
                logger.info(f"[EXECUTION PLAN] {self._execution_plan}")

                # V3/V4 STEP 8: DOA — record snapshot + evaluate
                if self.config.doa:
                    self._doa.record_snapshot(eval_data)
                if self.config.tca:
                    self._tca.record(eval_data)
                if self.config.doa and self._doa.ready:
                    current_prices = {
                        sym: data.get("price")
                        for sym, data in eval_data.items()
                        if "price" in data
                    }
                    doa_results = self._doa.evaluate(current_prices)
                    logger.info(f"[DOA RESULTS] {doa_results}")
                    # V3/V4 STEP 19: LCT — long-horizon convergence
                    if self.config.lct:
                        self._lct.record(doa_results)
                        lct_score = self._lct.convergence_score()
                    else:
                        lct_score = 0.5
                    logger.info(f"[LCT CONVERGENCE] {lct_score}")
                    if lct_score < -0.3:
                        logger.warning(f"[LCT WARNING] SYSTEM DEGRADING ({lct_score})")
                    elif lct_score > 0.3:
                        logger.info(f"[LCT POSITIVE TREND] SYSTEM IMPROVING ({lct_score})")
                    # WFV: collect prediction-outcome pair for validation
                    for sym, outcome in doa_results.items():
                        ed = eval_data.get(sym, {})
                        ecdf = float(ed.get("ecdf_rank", 0.5))
                        entropy = float(ed.get("entropy", 0.5))
                        score = ecdf - entropy
                        self._wfv_records.append({
                            "signal": prod_signals.get(sym, 0),
                            "outcome": outcome,
                            "regime": regime if 'regime' in dir() else "UNKNOWN",
                            "sym": sym,
                            "drift": drift_scores.get(sym, 0) if 'drift_scores' in dir() else 0,
                            "ecdf": ecdf,
                            "entropy": entropy,
                            "rotation_stability": 1.0 if sym in selected else 0.0,
                            "allocation_weight": self._allocations.get(sym, 0.0),
                            "signal_strength": abs(score),
                            "regime_confidence": abs(score),
                        })
                    # V3/V4 STEP 14: TCA — temporal credit assignment
                    if self.config.tca:
                        tca_report = self._tca.assign_credit(current_prices, doa_results)
                    else:
                        tca_report = {}
                    logger.info(f"[TCA REPORT] {tca_report}")
                    # V3/V4 STEP 9: AFL — update + feedback loop
                    if self.config.afl:
                        afl_state = self._afl.update(doa_results)
                        self._fusion.entropy_flip_threshold = 0.65 * afl_state["entropy_sensitivity"]
                        self._rotation.persistence = max(1, int(3 * afl_state["rotation_sensitivity"]))
                        self._h20.max_cap = min(0.8, 0.6 * afl_state["rotation_sensitivity"])
                        logger.info(f"[AFL STATE] {afl_state}")
                    else:
                        afl_state = {"entropy_sensitivity": 1.0, "rotation_sensitivity": 1.0}
                    # V3/V4 STEP 10: CAL — causal attribution
                    if self.config.cal:
                        cal_report = self._cal.attribute(eval_data, doa_results)
                    else:
                        cal_report = {}
                    logger.info(f"[CAL REPORT] {cal_report}")
                    self._last_cal_report = cal_report
                    # V3/V4 STEP 11: FWO — feature weight optimization
                    if self.config.fwo:
                        fwo_weights = self._fwo.update(cal_report)
                    else:
                        fwo_weights = {}
                    logger.info(f"[FWO WEIGHTS] {fwo_weights}")
                    # V3/V4 STEP 17+18: CDM + DRL + MSO
                    if self.config.cdm:
                        drift_scores = self._cdm.compute_drift(cal_report, tca_report)
                    else:
                        drift_scores = {sym: 0.0 for sym in doa_results} if doa_results else {}
                    if self.config.drl:
                        drift_state = self._drl.adapt(drift_scores)
                    else:
                        drift_state = {"cal_weight": 0.5, "tca_weight": 0.5, "regularization": 1.0}
                    if self.config.mso:
                        self._mso.record(drift_state)
                        stable_state = self._mso.stabilize(drift_state)
                        if stable_state != drift_state:
                            logger.info(f"[MSO APPLIED DAMPING] {stable_state}")
                            drift_state = stable_state
                    logger.info(f"[DRL STATE] {drift_state} [DRIFT] {drift_scores}")
                    if self.config.cwf:
                        cwf_report = self._cwf.fuse_with_weights(
                            cal_report, tca_report,
                            drift_state["cal_weight"], drift_state["tca_weight"],
                        )
                    else:
                        cwf_report = cal_report if cal_report else tca_report if tca_report else {}
                    logger.info(f"[CWF REPORT] {cwf_report}")
                    # V3/V4 STEP 12+13+17: RSL — regime-segmented + RTD gate + DRL adapted
                    if self.config.rtd:
                        regime = self._rtd.detect(eval_data)
                    else:
                        regime = "STABLE_DEFAULT"
                    logger.info(f"[RTD REGIME] {regime}")
                    if self.config.rsl and regime != "TRANSITION":
                        clean_regime = regime.replace("STABLE_", "")
                        rsl_weights = self._rsl.update(clean_regime, cwf_report)
                        logger.info(f"[RSL UPDATED] {clean_regime} [WEIGHTS] {rsl_weights}")
                    else:
                        logger.info("[RSL SKIPPED - TRANSITION STATE]")

                    # V3/V4 STEP 20: SSOL — system self-optimization loop
                    if self.config.ssol:
                        ssol_state = self._ssol.update(
                            lct_score=lct_score,
                            drift_scores=drift_scores,
                            mso_state=drift_state,
                            drl_state=drift_state,
                        )
                        logger.info(f"[SSOL STATE] {ssol_state}")
                        self._fusion.entropy_flip_threshold = 0.65 * ssol_state["stability"]
                        self._rotation.persistence = max(1, int(3 * ssol_state["stability"]))
                        self._h20.max_cap = 0.6 * ssol_state["stability"]
                        self._afl.lr = ssol_state["learning_rate"]
                    else:
                        ssol_state = {"stability": 0.5, "exploration": 0.5, "learning_rate": 0.05}

                # Run heavy signal calculations, exits, and DB syncs every 60 seconds
                if timestamp - last_signal_check >= 60:
                    last_signal_check = timestamp

                    # P3.3: Build the single CycleClock for this 60s cycle
                    self._cycle_id += 1
                    logger.info("[CYCLE] cycle=%d", self._cycle_id)
                    # Wave 3: Global kill switch check — hard system halt
                    if hasattr(self, 'kill_switch') and self.kill_switch.is_active():
                        logger.critical(f"[KILL_SWITCH] {self.kill_switch._reason} — halting all execution")
                        RUNNING = False
                        break
                    # Wave 4: Portfolio overexposure → kill switch
                    if hasattr(self, 'portfolio_graph') and self.portfolio_graph.is_overexposed():
                        self.kill_switch.trigger(f"PORTFOLIO_OVEREXPOSURE risk={self.portfolio_graph.portfolio_risk():.2f}")
                        self.rejection_engine.reject("PORTFOLIO", RejectionType.PORTFOLIO_OVEREXPOSURE,
                            self.portfolio_graph.portfolio_risk(), time.time())
                        logger.critical(f"[OVEREXPOSURE] total_risk={self.portfolio_graph.portfolio_risk():.2f} threshold={self.portfolio_graph.kill_threshold}")
                        RUNNING = False
                        break
                    # P0.26: Decay quarantine counter — reduced risk after restart
                    if self._quarantine_cycles_remaining > 0:
                        self._quarantine_cycles_remaining -= 1
                        if self._quarantine_cycles_remaining == 0:
                            logger.info("[QUARANTINE] completed — full risk resumes")
                    # Broker truth reconciliation before any state mutation
                    self._reconcile_broker_positions()
                    # Auto-stop if ACCEPTANCE_MODE and cycle limit reached
                    if ACCEPTANCE_MODE and PROXIMA_MAX_CYCLES > 0 and self._cycle_id >= PROXIMA_MAX_CYCLES:
                        logger.info("[AUTO_STOP] acceptance mode: max_cycles=%d reached", PROXIMA_MAX_CYCLES)
                        RUNNING = False
                        break
                    self._sticky_removed_this_cycle = set()
                    from proxima_ops.core.cycle_clock import CycleClock
                    bar_time = next(
                        (r["timestamp"] if "timestamp" in r else r.get("time", now)
                         for rates in self._price_history.values() if rates
                         for r in [rates[-1]]),
                        now
                    )
                    self._cycle_clock = CycleClock(
                        cycle_id=self._cycle_id,
                        bar_time=bar_time,
                        wall_time=now,
                    )
                    logger.info(self._cycle_clock.log_line())

                    # Update H1 and M5 price history buffers using broker-mapped symbols (active only)
                    for sym in list(getattr(self, '_active_symbols', self._execution_symbols[:3])):
                        rates = self.mt5.get_rates(sym, count=550, timeframe="H1")
                        if rates is not None and len(rates) >= 524:
                            broker_sym = self.mt5._get_broker_symbol(sym)
                            self._price_history[broker_sym] = list(rates)
                            # P2.2: Feed fine-grain bars to synthetic thermodynamics
                            if sym in self._tick_thermo._synthetic_symbols:
                                m1_rates = self.mt5.get_rates(sym, count=200, timeframe="M1")
                                if m1_rates:
                                    for bar in m1_rates:
                                        self._tick_thermo.feed_bar(sym, bar.get("open", 0),
                                                                    bar.get("high", 0),
                                                                    bar.get("low", 0),
                                                                    bar.get("close", 0))
                        m5_rates = self.mt5.get_rates(sym, count=1000, timeframe="M5")
                        if m5_rates is not None and len(m5_rates) >= 10:
                            broker_sym = self.mt5._get_broker_symbol(sym)
                            self._m5_history[broker_sym] = list(m5_rates)

                    # Enforce H20 exits on active positions
                    self.positions.refresh()
                    open_positions = self.positions.positions
                    for pos in open_positions:
                        ticket = pos["ticket"]
                        symbol = pos["symbol"]
                        _tpc = int(os.environ.get("PROXIMA_TICKS_PER_CYCLE", "1"))
                        self._position_tick_age[ticket] = self._position_tick_age.get(ticket, 0) + _tpc
                        ttl_age = self._position_tick_age[ticket]
                        if ticket in self._battle_decay._states:
                            self._battle_decay._states[ticket].age_ticks = ttl_age
                        _es_check = self._exit_state.get(ticket, {})
                        if _es_check.get("entry_mode") == "EXPLORATION":
                            _mv = compute_micro_volatility(symbol, self._price_history.get(symbol, []))
                            _mae_cur = abs(pos.get("price_current", 0) - pos.get("price_open", 0))
                            _pt_ec = 0.01 if "JPY" in symbol else (0.1 if "XAU" in symbol or "XAG" in symbol else 0.0001)
                            _mae_pts_ec = _mae_cur / max(_pt_ec, 1e-9)
                            if ttl_age >= EXPLORATION_TTL:
                                exit_decision = ("EXP_EXIT", "TIME_BASED", f"ttl={ttl_age}")
                            elif _mae_pts_ec > 3.0 * _mv:
                                exit_decision = ("EXP_EXIT", "VOL_SHOCK", f"mae={_mae_pts_ec:.1f} mv={_mv:.3f}")
                            else:
                                continue
                            logger.info(f"[{exit_decision[0]}] {symbol} ticket={ticket} reason={exit_decision[1]} detail={exit_decision[2]}")
                            logger.info(f"[EXP_RESPONSE_KERNEL] {symbol} ticket={ticket} ttl={ttl_age} mae_pts={_mae_pts_ec:.1f} mv={_mv:.3f} bucket={_es_check.get('entropy_bucket','?')}")
                            # Bridge gate — centralised exit authority
                            if not self._bridge_allows_exit(symbol, ticket, exit_decision[1]):
                                logger.info(f"[BRIDGE_SKIP] {symbol} ticket={ticket} — EXP_EXIT blocked, deferring")
                                continue
                            self.orders.close(ticket)
                            self._sdl.release(symbol)
                            self.trade_ledger.close_by_ticket(ticket, exit_reason=exit_decision[1], exit_detail=exit_decision[2])
                            self._reconcile_broker_positions()
                            self._exit_state.pop(ticket, None)
                            continue
                        # P0.19: Parallel evaluation — check all exit conditions, pick best
                        exit_decision = None  # (reason, detail)
                        # 1. BattleDecay (most informed, highest priority)
                        if ticket in self._battle_decay._states:
                            state = self._battle_decay._states[ticket]
                            meta_bd = self._active_positions_metadata.get(ticket, {})
                            entry_px = meta_bd.get("entry_price", pos.get("price_open", 0))
                            min_px = meta_bd.get("min_price", entry_px)
                            max_px = meta_bd.get("max_price", entry_px)
                            direction_bd = 1 if state.direction == 1 else -1
                            pt = 0.01 if "JPY" in symbol else (0.1 if "XAU" in symbol or "XAG" in symbol else 0.0001)
                            mfe_raw = (max_px - entry_px) if direction_bd == 1 else (entry_px - min_px)
                            mfe_pts_bd = mfe_raw / pt if pt > 0 else 0.0
                            bd_decision = self._battle_decay.evaluate_cycle(
                                ticket, pos.get("bid", 0), pos.get("ask", 0),
                                pos.get("time", 0), mfe_pts_bd)
                            if bd_decision is not None:
                                exit_decision = ("BD", bd_decision[0], bd_decision[1])
                        # 2. TPI inversion (only if BD not firing)
                        if exit_decision is None and ticket in self._exit_state:
                            es = self._exit_state[ticket]
                            prox = es.get("symbol")
                            tpi_sig = self._cycle_tpi_snapshot.get(prox) if prox else None
                            if tpi_sig and ttl_age >= 1:
                                tpi_val = tpi_sig.get("tpi", 0.0)
                                tpi_thr = get_tpi_threshold(prox)
                                if (tpi_val * es["direction"] < 0) and abs(tpi_val) >= tpi_thr:
                                    es["inversion_count"] += 1
                                else:
                                    es["inversion_count"] = 0
                                if es["inversion_count"] >= 3:
                                    exit_decision = ("TPI", "TPI_INVERSION",
                                                     f"tpi={tpi_val:.4f} thr={tpi_thr} inv={es['inversion_count']}")
                        # 3. TIMEOUT (last resort)
                        if exit_decision is None:
                            bd_ready = self._battle_decay.is_ticket_ready(ticket) if hasattr(self, '_battle_decay') else False
                            if ttl_age >= MAX_HOLD_TICKS and bd_ready:
                                exit_decision = ("TIMEOUT", "TIMEOUT", f"max_hold={ttl_age}")
                        # Execute best exit decision
                        if exit_decision is not None:
                            priority, exit_reason, exit_detail = exit_decision
                            logger.info(f"[{exit_reason}] {symbol} ticket={ticket} priority={priority} detail={exit_detail}")
                            # Bridge gate — centralised exit authority
                            if not self._bridge_allows_exit(symbol, ticket, exit_reason):
                                logger.info(f"[BRIDGE_SKIP] {symbol} ticket={ticket} — {exit_reason} blocked, deferring")
                                continue
                            self.orders.close(ticket)
                            self._sdl.release(symbol)
                            self.trade_ledger.close_by_ticket(
                                ticket, exit_reason=exit_reason, exit_detail=exit_detail)
                            self._update_symbol_trust_from_bd(ticket, symbol)
                            # Phase 2: Record cycle boundary for CF engine (future ETCS evaluation)
                            _cf_dir = self._battle_decay._states[ticket].direction if ticket in self._battle_decay._states else 0
                            self._cf_engine.record_cycle_boundary(DecisionBoundaryLog(
                                position_id=str(ticket), symbol=symbol,
                                cycle_index=self._cycle_id,
                                timestamp=pos.get("time", 0),
                                cycle_close_price=pos.get("price_current", 0),
                                entry_price=pos.get("price_open", 0.0),
                                direction=_cf_dir,
                                pnl=pos.get("profit", 0), fee=0.0,
                                tpi=self._cycle_tpi_snapshot.get(symbol, {}).get("tpi", 0.0),
                                continuity_score=self._battle_decay._states[ticket]._signal_confidence if ticket in self._battle_decay._states else 0.5,
                                cycle_quality=0.5, regime_state="",
                            ))
                            # Phase 2: Compute exit realized score and pump to system_confidence
                            _cf_realized = 1.0 if pos.get("profit", 0) > 0 else 0.0
                            _cf_err = abs(0.5 - _cf_realized)  # prior = neutral 0.5
                            if hasattr(self, 'exec_stats') and self.exec_stats:
                                self.exec_stats.record_reversal_event(
                                    symbol, success=_cf_realized > 0.5)
                            # Temporal decay weighting — dampens EMA update as trades accumulate
                            _trade_n = self.exec_stats.total_orders if hasattr(self, 'exec_stats') and self.exec_stats else 0
                            _recency_decay = math.exp(-0.01 * max(0, _trade_n - 1))
                            realized_score = max(0.0, min(1.0, 1.0 - _cf_err * _recency_decay))
                            self._battle_decay.update_system_confidence(realized_score)
                            # Phase 3.2: Log CF error for gating layer
                            if hasattr(self, 'exec_stats') and self.exec_stats:
                                self.exec_stats.log_cf_record({
                                    "symbol": symbol, "ticket": ticket,
                                    "cf_error": _cf_err, "realized": _cf_realized,
                                    "exit_reason": exit_reason,
                                })
                            if priority == "BD":
                                price_bd = pos.get("price_current", 0)
                                mfe_remaining = mfe_raw - (abs(price_bd - entry_px)) if pt > 0 else 0.0
                                self._battle_decay.record_exit(
                                    ticket, symbol, exit_reason, exit_detail,
                                    mfe_pts_bd, mfe_remaining)
                                self._battle_decay.remove_ticket(ticket)
                            self._exit_state.pop(ticket, None)
                            self._reconcile_broker_positions()
                            continue
                        if ticket not in self._active_positions_metadata:
                            pos_time = pos.get("time", 0)
                            entry_bar_time = int(pos_time // 3600) * 3600
                            entry_p = pos.get("price_open", 0.0)
                            self._active_positions_metadata[ticket] = {
                                "entry_bar_time": entry_bar_time,
                                "entry_warmup_ticks": self._warmup_ticks,
                                "pos_open_time": int(pos_time),
                                "symbol": symbol,
                                "entry_es_rank": None,
                                "entry_at_rank": None,
                                "trigger_count_while_open": 0,
                                "max_es_rank": 0.0,
                                "max_at_rank": 0.0,
                                "entry_price": entry_p,
                                "min_price": entry_p,
                                "max_price": entry_p,
                                "entry_time": datetime.fromtimestamp(pos_time).strftime("%Y-%m-%d %H:%M:%S") if pos_time else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            self._save_active_positions_metadata()
                        
                        meta = self._active_positions_metadata[ticket]
                        # Update min/max prices for BattleDecay MFE tracking
                        _cur_px = pos.get("price_current", pos.get("price_open", 0))
                        meta["min_price"] = min(meta.get("min_price", _cur_px), _cur_px)
                        meta["max_price"] = max(meta.get("max_price", _cur_px), _cur_px)
                        entry_bar = meta["entry_bar_time"]
                        elapsed_bars = self._bars_elapsed(entry_bar, symbol)
                        # P4.2: Get dynamic hold cap from signal decay velocity
                        ck_hold = meta.get("cohort_key", "")
                        hold_cap = self._signal_decay.hold_cap(ck_hold) if ck_hold else 20
                        if elapsed_bars >= hold_cap:
                                    logger.info(f"H20 EXIT: Closing position {ticket} for {symbol} (open for {elapsed_bars} bars)")
                                    meta["expected_exit_reason"] = "H20"
                                    meta["last_profit"] = pos.get("profit", 0.0)
                                    self._save_active_positions_metadata()
                                    # Bridge gate — centralised exit authority
                                    if self._bridge_allows_exit(symbol, ticket, "H20"):
                                        self.orders.close(ticket)
                                        self._sdl.release(symbol)
                                        self._reconcile_broker_positions()
                                        # V6: Record position lock (normalize to broker symbol)
                                        broker_sym_lock = self.mt5._get_broker_symbol(pos["symbol"])
                                        self._position_lock_until[broker_sym_lock] = SETTINGS.position_lock_bars
                                        self._lock_event_count += 1
                                    else:
                                        logger.info(f"[BRIDGE_SKIP] {symbol} ticket={ticket} — H20 blocked, deferring")

                    # Sync the trade ledger database with active MT5 positions
                    self.sync_ledger_with_mt5()

                    # Update positions list and check signals
                    self.positions.refresh()
                    positions_now = self.positions.positions
                    open_tickets = {p["ticket"] for p in positions_now}
                    # Wave 4: Update portfolio graph exposures
                    for p in positions_now:
                        exposure = p.get("volume", 0) * p.get("price_current", 1)
                        self.portfolio_graph.update(p.get("symbol", "?"), exposure)
                    for t in list(self._exit_state):
                        if t not in open_tickets:
                            del self._exit_state[t]
                    for t in list(self._battle_decay._states):
                        if t not in open_tickets:
                            _bd_s = self._battle_decay._states[t]
                            # Compute EOL for closed trade: (MFE_peak - ExitPnL) / (MFE_peak + 1)
                            _eol = 0.0
                            if _bd_s.mfe_peak > 0:
                                _meta_eol = self._active_positions_metadata.get(t, {})
                                _entry_eol = _meta_eol.get("entry_price", _bd_s.entry_price)
                                _price_eol = _meta_eol.get("max_price" if _bd_s.direction == 1 else "min_price", _entry_eol)
                                _mfe_peak_eol = (_price_eol - _entry_eol) * _bd_s.direction
                                _exit_pnl_eol = 0.0  # approximate: direction * (last_price - entry)
                                _last_pos = next((p for p in positions_now if p["ticket"] == t), None)
                                if _last_pos:
                                    _exit_pnl_eol = (_last_pos.get("price_current", _entry_eol) - _entry_eol) * _bd_s.direction
                                _eol = (_mfe_peak_eol - _exit_pnl_eol) / (_mfe_peak_eol + 1)
                            # Log final EOL
                            if _bd_s.mfe_peak > 0:
                                logger.info(f"[BD_EOL] ticket={t} {_bd_s.symbol} mfe_peak={_bd_s.mfe_peak:.1f} eol={_eol:.4f}")
                            self._battle_decay.remove_ticket(t)
                    max_pos = SETTINGS.max_positions_active
                    active_pos = len(positions_now)
                    logger.info(f"[OCCUPANCY] active={active_pos}/{max_pos} slots={max_pos - active_pos}")
                    _bd_stats = self._battle_decay.state_stats()
                    if _bd_stats["abort_active"]:
                        logger.warning(f"[BD_ABORT_ACTIVE] BattleDecay+TPI disabled for rest of session")
                    logger.info(f"[BD_STATS] states={_bd_stats['active_states']} exits={_bd_stats['exit_log_count']} bad={_bd_stats['bad_exit_count']} abort={_bd_stats['abort_active']}")
                    self._battle_decay.persist_all()

                    # Phase E: cycle-end promotion checks
                    try:
                        self._validation.on_cycle_end()
                        _vs = self._validation.stats() or {}
                        logger.info(f"[VALIDATION_CYCLE] cycle={self._cycle_id} "
                                    f"evidence={_vs.get('evidence', 0)} resolved={_vs.get('resolved', 0)} "
                                    f"disagreements={_vs.get('disagreements', 0)}")
                    except Exception as ex:
                        logger.warning(f"[VALIDATION] cycle_end error: {ex}")

                    # Compute global ranks from prior cycle evaluations
                    self._global_rank_engine.compute()

                    # V6: Determine Top-3 qualified assets (execution universe only)
                    # Augment symbol thresholds with defaults for unconfigured symbols
                    augmented_thresholds = {}
                    for sym in self._execution_symbols:
                        augmented_thresholds[sym] = SETTINGS.symbol_thresholds.get(sym, SETTINGS.default_symbol_threshold)
                    top3_qualified = self._global_rank_engine.get_qualified_by_local(augmented_thresholds)
                    top3_qualified = [s for s in top3_qualified if s in self._execution_symbols]
                    # Bootstrap fallback: include symbols with active OSS prod_signal
                    for sym in self._execution_symbols:
                        if sym not in top3_qualified:
                            ps = prod_signals.get(sym, 0)
                            if ps != 0:
                                top3_qualified.append(sym)
                    # Filter out budget-blocked symbols (TTL blacklist)
                    submit_pool = [s for s in top3_qualified
                                   if self._budget_block_ttl.get(s, 0) <= self._cycle_id]
                    if not submit_pool:
                        submit_pool = top3_qualified  # fallback to full list
                    self._active_symbols = set(top3_qualified)
                    self._cycle_execution_set = set(self._execution_symbols)
                    if SETTINGS.portfolio_mode == "TOP_3_ROTATION":
                        self._top3_submit_set = self._select_balanced_top3(
                            submit_pool, self._execution_plan, n=3)
                    else:
                        self._top3_submit_set = set(top3_qualified)
                    # P1-C: TOP3 sticky TTL — force rotation if symbol stuck in TOP3
                    for sym in list(self._top3_sticky_cycles.keys()):
                        if sym in self._top3_submit_set:
                            self._top3_sticky_cycles[sym] = self._top3_sticky_cycles.get(sym, 0) + 1
                            if self._top3_sticky_cycles[sym] >= 3:
                                if sym in self._top3_submit_set:
                                    self._top3_submit_set.remove(sym)
                                self._top3_cooldown[sym] = 2
                                self._rotation_event_count += 1
                                self._sticky_removed_this_cycle.add(sym)
                                logger.info(f"[TOP3_STICKY] {sym} cycles={self._top3_sticky_cycles[sym]} -> REMOVED cooldown=2")
                        else:
                            self._top3_sticky_cycles[sym] = 0
                    # Apply cooldown decay before re-entry
                    for sym in list(self._top3_cooldown.keys()):
                        self._top3_cooldown[sym] -= 1
                        if self._top3_cooldown[sym] <= 0:
                            del self._top3_cooldown[sym]
                    logger.info(f"[TOP3 QUALIFIED] {top3_qualified}")
                    logger.info(f"[TOP3 BALANCED] pool={submit_pool} selected={self._top3_submit_set}")
                    logger.info(f"[CYCLE RANGE] full_set={self._cycle_execution_set} top3_submit={self._top3_submit_set}")
                    buy_ct = sum(1 for s in self._top3_submit_set if self._execution_plan.get(s, {}).get('direction') == 'BUY')
                    sell_ct = sum(1 for s in self._top3_submit_set if self._execution_plan.get(s, {}).get('direction') == 'SELL')
                    flat_ct = sum(1 for s in self._top3_submit_set if self._execution_plan.get(s, {}).get('direction') == 'FLAT')
                    logger.info(f"[TOP3_DIRECTION_AUDIT] symbols={sorted(self._top3_submit_set)} BUY={buy_ct} SELL={sell_ct} FLAT={flat_ct}")
                    # PLAN_FILTER: explain why execution_plan symbols didn't reach top3_submit_set
                    _ep_syms = set(self._execution_plan.keys())
                    _submit_syms = self._top3_submit_set
                    _qual_syms = set(top3_qualified)
                    _pool_syms = set(submit_pool)
                    _plan_excluded = _ep_syms - _submit_syms
                    if _plan_excluded:
                        for _ps in sorted(_plan_excluded):
                            _reasons = []
                            if _ps not in _qual_syms:
                                _reasons.append("NOT_QUALIFIED")
                            if _ps in _qual_syms and _ps not in _pool_syms:
                                _reasons.append(f"BUDGET_BLOCKED(ttl={self._budget_block_ttl.get(_ps, 0)} > cycle={self._cycle_id})")
                            if _ps in _pool_syms and _ps not in _submit_syms:
                                _reasons.append("TOP3_ROTATION_SKIPPED")
                            if _ps in self._sticky_removed_this_cycle:
                                _reasons.append("STICKY_REMOVED")
                            _ep_dir = self._execution_plan.get(_ps, {}).get("direction", "?")
                            logger.info(f"[PLAN_FILTER] {_ps} dir={_ep_dir} reasons={'|'.join(_reasons)}")

                    # V6: Track rotation events
                    if self._top3_history:
                        prev_top3 = self._top3_history[-1][1]
                        if set(top3_qualified) != set(prev_top3):
                            self._rotation_event_count += 1
                    self._top3_history.append((self._now_ts(), top3_qualified))
                    self._last_top3_qualified = top3_qualified

                    # P0.14: Build TPI snapshot from canonical per-symbol buffer (single source of truth)
                    self._cycle_tpi_snapshot = {}
                    for esym in self._cycle_execution_set:
                        _raw = self._canonical_tpi_buffer.get_tpi(esym)
                        if _raw is not None:
                            sig = {
                                "tpi": _raw["tpi"],
                                "direction": _raw["direction"],
                                "confidence": _raw["confidence"],
                                "n_ticks": _raw["n_ticks"],
                                "session_name": "CANONICAL",
                            }
                        else:
                            sig = {"tpi": 0, "direction": 0, "confidence": 0.0, "n_ticks": 0, "session_name": "CANONICAL"}
                        self._cycle_tpi_snapshot[esym] = sig
                        logger.info(f"[TPI_SOURCE] {esym} source=CANONICAL "
                                    f"direction={'LONG' if sig['direction']==1 else 'SHORT' if sig['direction']==-1 else 'FLAT'} "
                                    f"conf={sig['confidence']:.4f} n_ticks={sig.get('n_ticks', 0)}")
                    logger.info(f"[TPI_SNAPSHOT] built={len(self._cycle_tpi_snapshot)} symbols={list(self._cycle_tpi_snapshot.keys())}")

                    # Always run evaluation checks for all assets (even if paused) to log opportunities
                    # Only execution symbols go through signal evaluation

                    # Exploration mode dispatch: symbols for forced entry
                    self._exploration_dispatch = set()
                    if SETTINGS.exploration_mode:
                        _pool_raw = list(self._cycle_execution_set)
                        _pool_no_pos = [s for s in _pool_raw
                            if not any(self.mt5._get_broker_symbol(p["symbol"]) == self.mt5._get_broker_symbol(s) for p in positions_now)]
                        _pool_ready = [s for s in _pool_no_pos if self._exploration_cooldown.get(s, 0) <= 0]
                        exploration_pool = _pool_ready
                        if exploration_pool:
                            _fx_clusters = {"AUD": ["AUDJPY", "AUDCHF", "GBPAUD"], "EUR": ["EURUSD", "EURGBP", "EURJPY", "EURCAD", "EURCHF", "EURAUD"], "USD": ["USDJPY", "USDCAD", "GBPUSD"], "OTHER": []}
                            _sym_cluster = lambda s: next((k for k, v in _fx_clusters.items() if s in v), "OTHER")
                            _chosen_pool = list(exploration_pool)
                            _chosen = []
                            if _chosen_pool:
                                _primary = random.choice(_chosen_pool)
                                _chosen.append(_primary)
                                _chosen_pool.remove(_primary)
                                _clust = _sym_cluster(_primary)
                                _linked = [s for s in _chosen_pool if _sym_cluster(s) == _clust and self._exploration_cooldown.get(s, 0) <= 0]
                                if _linked:
                                    _chosen.append(random.choice(_linked))
                                _contrast = [s for s in _chosen_pool if _sym_cluster(s) != _clust and self._exploration_cooldown.get(s, 0) <= 0]
                                if _contrast and len(_chosen) < 3:
                                    _chosen.append(random.choice(_contrast))
                            self._exploration_dispatch = set(_chosen)
                            for s in _chosen:
                                self._exploration_cooldown[s] = SETTINGS.exploration_per_symbol_cooldown
                                logger.info(f"[EXPLORATION] {s}: selected for forced entry cycle={self._cycle_id} cluster={_sym_cluster(s)} role={'primary' if s==_chosen[0] else 'linked' if len(_chosen)>1 and _sym_cluster(s)==_sym_cluster(_chosen[0]) else 'contrast'}")
                    for s in list(self._exploration_cooldown.keys()):
                        self._exploration_cooldown[s] -= 1
                        if self._exploration_cooldown[s] <= 0:
                            del self._exploration_cooldown[s]

                    # ---- MOF: Market Observability Filter (pre-perception gate) ----
                    _signals_for_mof = []
                    _cluster_states_for_mof = {}
                    for sym in self._cycle_execution_set:
                        _sig_data = eval_data.get(sym, {})
                        _prod_dir = prod_signals.get(sym, 0)
                        if _prod_dir != 0:
                            _signals_for_mof.append({
                                "symbol": sym,
                                "direction": _prod_dir,
                                "confidence": _sig_data.get("oss_conf", 0.5),
                                "ecdf": _sig_data.get("ecdf_rank", 0.5),
                                "drift": _sig_data.get("exec_drift", 0),
                            })
                        _ee_state = self._entropy_compression.compute_state(sym) or {}
                        _ne = _ee_state.get("normalized_entropy", 0.5)
                        if _ne <= 0.30:
                            _cl_name = "CLUSTER_LOW_ENTROPY"
                        elif _ne <= 0.60:
                            _cl_name = "CLUSTER_MED_ENTROPY"
                        else:
                            _cl_name = "CLUSTER_HIGH_ENTROPY"
                        if _cl_name not in _cluster_states_for_mof:
                            _cluster_states_for_mof[_cl_name] = {"coherence": 0.0, "count": 0}
                        _cluster_states_for_mof[_cl_name]["coherence"] += (1.0 - _ne)
                        _cluster_states_for_mof[_cl_name]["count"] += 1
                    for _cl_data in _cluster_states_for_mof.values():
                        _cl_data["coherence"] /= max(_cl_data["count"], 1)
                    _mof_r = self._mof.evaluate(_cluster_states_for_mof, _signals_for_mof)
                    _mof_state = _mof_r["observability_state"]
                    self._last_mof_state = _mof_state
                    _mof_score = _mof_r["observability_score"]
                    _mof_perm = _mof_r["action_permission"]
                    logger.info(
                        f"[MOF] State: {_mof_state}, Score: {_mof_score:.4f}, "
                        f"Permission: {_mof_perm}, "
                        f"Clusters: {len(_cluster_states_for_mof)}, "
                        f"Signals: {len(_signals_for_mof)}"
                    )
                    if not ACCEPTANCE_MODE:
                        print(f"\n  {'MOF EVALUATION':^76s}")
                        print(f"  {'-' * 76}")
                        print(f"  State:       {_mof_state}")
                        print(f"  Score:       {_mof_score:.4f}")
                        print(f"  Permission:  {_mof_perm}")
                        print(f"  Components:  coherence={_mof_r['components']['coherence_quality']:.4f}  "
                              f"confidence={_mof_r['components']['oss_confidence_quality']:.4f}  "
                              f"stability={_mof_r['components']['stability_quality']:.4f}")
                        if _mof_state == ObservabilityState.INFORMATION_DEGRADED.value:
                            if SYSTEM_MODE.mof_policy == MOFPolicy.RELAXED_SIM:
                                self._mof_blocked = False
                                logger.info("[MOF_GATE] RELAXED_SIM mode — allowing simulated trades despite degraded observability")
                                print(f"\n  >>> MOF OVERRIDE: {SYSTEM_MODE.mof_policy.value} — allowing simulated trades (INFORMATION_DEGRADED)")
                            else:
                                self._mof_blocked = True
                                logger.warning("[MOF_GATE] INFORMATION_DEGRADED — blocking ALL execution")
                                print(f"\n  >>> MOF ENFORCEMENT: blocking ALL execution (INFORMATION_DEGRADED)")
                        elif _mof_state == ObservabilityState.STRUCTURE_LIMITED.value:
                            self._mof_blocked = False
                            logger.info(f"[MOF_GATE] STRUCTURE_LIMITED — reduced observability (score={_mof_score:.4f})")
                            print(f"\n  >>> MOF ENFORCEMENT: reduced observability (STRUCTURE_LIMITED)")
                        else:
                            self._mof_blocked = False
                            print(f"\n  >>> MOF ENFORCEMENT: full observability (INFORMATION_RICH)")
                        print()

                    # ── SYSTEM_MODE Governance: invariant check + snapshot ──
                    _invariant_checker.check(SYSTEM_MODE, self._mof_blocked, self._replay_mode)
                    _mode_snapshot_logger.maybe_log(SYSTEM_MODE, _mof_state, _mof_score)

                    # ── Restricted Execution Bridge — Gated Exit Authority ──
                    self._execution_bridge.shadow_instability_flag = self._mof_blocked
                    # Build minimal rfe_output from available signal/eval data
                    _bridge_rfe_evals = {}
                    for _bsym in self._cycle_execution_set:
                        _beval = eval_data.get(_bsym, {})
                        _bdir = prod_signals.get(_bsym, 0)
                        _bprice = _beval.get("price", 0.0)
                        _bconf = _beval.get("oss_conf", 0.5)
                        _brfe_score = min(1.0, abs(_bdir) * _bconf) if _bdir != 0 else 0.0
                        _bridge_rfe_evals[_bsym] = {
                            "score": _brfe_score,
                            "state": "INFO",
                            "current_price": _bprice,
                            "components": {"divergence": 0.0, "persistence": 0.0,
                                           "hysteresis_decay": 0.0, "pnl_regime": 0.0},
                            "exit_allowed": False,
                            "cycles_in_state": 1,
                        }
                    _bridge_rfe_output = {
                        "evaluations": _bridge_rfe_evals,
                        "summary": {"any_exit_allowed": False, "max_pressure": 0.0,
                                    "trades_at_risk": [], "dominant_state": "INFO"},
                        "transitions": {}, "temporal": {}, "breaches": [],
                        "timestamp": datetime.now().isoformat(),
                    }
                    _bridge_positions = self.positions.positions if hasattr(self, 'positions') else []
                    _raw = self._price_history if hasattr(self, '_price_history') else {}
                    _bridge_price_hist = {
                        sym: [r["close"] if isinstance(r, dict) else r for r in rates]
                        for sym, rates in _raw.items()
                    }
                    self._last_bridge_result = self._execution_bridge.evaluate(
                        _cluster_states_for_mof,
                        _bridge_rfe_output,
                        _bridge_price_hist,
                        _bridge_positions,
                    )
                    _bridge_summary = self._last_bridge_result.get("summary", {})
                    logger.info(
                        f"[BRIDGE_CYCLE] eligible={_bridge_summary.get('eligible_count',0)}/"
                        f"{_bridge_summary.get('total_symbols',0)} "
                        f"pending={_bridge_summary.get('pending_exits',[])} "
                        f"shadow={_bridge_summary.get('shadow_instability_flag',False)}"
                    )
                    if not ACCEPTANCE_MODE:
                        print(format_bridge_dashboard(self._last_bridge_result))

                    # Shadow Mirror — capture portfolio + context state (SZ5)
                    _sz5_account = self.mt5.get_account()
                    self._shadow_mirror.observe_portfolio(
                        current_positions=len(positions_now) if positions_now else 0,
            max_positions=SETTINGS.max_positions_active,
                        positions_by_symbol={p.get("symbol", "?"): p for p in (positions_now or [])},
                        account_balance=_sz5_account.get("balance", 0.0) if _sz5_account else 0.0,
                        session_pnl=sum(p.get("profit", 0) for p in (positions_now or [])),
                    )
                    # Capture market/risk context for shadow evaluation
                    self._shadow_mirror.observe_context({
                        "warmup_ticks": self._warmup_ticks,
                        "cycle_id": self._cycle_id,
                        "rf_gate_ready": any(self._rf_gate.ready(s) for s in self._execution_symbols),
                        "quarantine_active": hasattr(self, '_battle_decay') and self._battle_decay.is_quarantined(),
                        "kill_switch_active": self.kill_switch.is_active(),
                        "mof_blocked": self._mof_blocked,
                        "exploration_active": hasattr(self, '_exploration_dispatch') and bool(self._exploration_dispatch),
                    })
                    # Tap L2_Governor, L3_Intent, L4_CB, L5_VEL
                    for _sym_sz5 in self._execution_symbols:
                        _conv_sz5 = eval_data.get(_sym_sz5, {}).get("research_p_cont", 0.5)
                        self._shadow_orchestrator.registry.intercept(
                            "L2_Governor", _sym_sz5, {"conviction": _conv_sz5 * 0.9}
                        )
                        self._shadow_orchestrator.registry.intercept(
                            "L3_Intent", _sym_sz5, {"conviction": _conv_sz5 * 0.85}
                        )
                        self._shadow_orchestrator.registry.intercept(
                            "L4_CB", _sym_sz5, {"conviction": _conv_sz5 * 0.83}
                        )
                        self._shadow_orchestrator.registry.intercept(
                            "L5_VEL", _sym_sz5, {"conviction": _conv_sz5 * 0.80}
                        )
                        # Ground-truth captures alongside synthetic overlay
                        self._shadow_gt.capture("L2_GT", _sym_sz5, eval_data.get(_sym_sz5, {}))
                        self._shadow_gt.capture("L3_GT", _sym_sz5, eval_data.get(_sym_sz5, {}))
                        self._shadow_gt.capture("L4_GT", _sym_sz5, eval_data.get(_sym_sz5, {}))
                        self._shadow_gt.capture("L5_GT", _sym_sz5, eval_data.get(_sym_sz5, {}))
                    # Track per-symbol entry outcomes for divergence detection
                    self._sz5_entry_tracker = {}

                    for sym in self._cycle_execution_set:
                        broker_sym = self.mt5._get_broker_symbol(sym)
                        
                        is_traded = any(self.mt5._get_broker_symbol(p["symbol"]) == broker_sym for p in positions_now)
                        
                        rates = self._price_history.get(broker_sym)
                        if not rates or len(rates) < 524:
                            continue

                        # Extract feature arrays
                        prices = np.array([r["close"] for r in rates], dtype=np.float64)
                        highs = np.array([r["high"] for r in rates], dtype=np.float64)
                        lows = np.array([r["low"] for r in rates], dtype=np.float64)
                        volumes = np.array([r["volume"] for r in rates], dtype=np.float64)
                        returns = np.diff(np.log(prices), prepend=np.log(prices[0]))

                        data_dict = {
                            "price": prices, "returns": returns, "volume": volumes,
                            "high": highs, "low": lows
                        }

                        # Compute energy storage
                        res_ed = self.ed.compute(data_dict)
                        es_history = np.nan_to_num(res_ed.get("energy_storage", np.zeros(len(prices))), nan=0.0)
                        current_es = es_history[-1]
                        es_window = es_history[-504:]
                        es_percentile = float(np.sum(es_window <= current_es)) / len(es_window)
                        self._global_rank_engine.record_evaluation(sym, es_percentile, current_es)

                        # Extract live regime state from energy dynamics
                        energy_regime_arr = res_ed.get("energy_regime", np.array([2]))
                        energy_regime = int(energy_regime_arr[-1]) if len(energy_regime_arr) > 0 else None

                        # Compute adaptive time combined density
                        res_tt = self.tt.compute(data_dict)
                        time_density = np.nan_to_num(res_tt.get("time_density", np.zeros(len(prices))), nan=0.0)
                        event_density = np.nan_to_num(res_tt.get("event_density", np.zeros(len(prices))), nan=0.0)
                        info_density = np.nan_to_num(res_tt.get("information_density", np.zeros(len(prices))), nan=0.0)
                        behavior_density = np.nan_to_num(res_tt.get("behavior_density", np.zeros(len(prices))), nan=0.0)
                        combined_density = (time_density + event_density + info_density + behavior_density) / 4.0
                        current_density = combined_density[-1]
                        density_window = combined_density[-504:]
                        at_percentile = float(np.sum(density_window <= current_density)) / len(density_window)

                        # Extract live regime state from temporal topology
                        time_regime_arr = res_tt.get("time_regime", np.array([2]))
                        time_regime = int(time_regime_arr[-1]) if len(time_regime_arr) > 0 else None

                        # Combined regime: 0-8 (energy_regime * 3 + time_regime)
                        combined_regime = energy_regime * 3 + time_regime if (energy_regime is not None and time_regime is not None) else None

                        # Phase E: feed organism state to observer harness
                        try:
                            # --- Topology mapping ---
                            if combined_regime <= 2:
                                _topology = "DISTRIBUTED"
                            elif combined_regime <= 5:
                                _topology = "FRAGMENTED"
                            else:
                                _topology = "CONCENTRATED"

                            # --- Density statistics ---
                            _cd = np.asarray(combined_density, dtype=float)
                            _id = np.asarray(info_density, dtype=float)
                            _cd_mean = float(np.mean(_cd)) if len(_cd) else 0.0
                            _cd_std = float(np.std(_cd)) if len(_cd) else 0.0
                            _id_mean = float(np.mean(_id)) if len(_id) else 0.0

                            # --- RF warmup proxy ---
                            _rf_live = float(self._rf_gate.prob(sym)) if hasattr(self, '_rf_gate') else 0.5
                            _rf_ready = bool(self._rf_gate.ready(sym)) if hasattr(self, '_rf_gate') and hasattr(self._rf_gate, 'ready') else False
                            if _rf_ready:
                                _rf_prob = _rf_live
                            else:
                                _rf_prob = np.clip(0.30 + 0.40 * float(es_percentile) + 0.20 * (combined_regime / 8.0), 0.0, 1.0)

                            # --- Causal confidence ---
                            _causal = np.clip(_id_mean / max(_cd_mean, 1e-9), 0.0, 1.0)

                            # --- Counterfactual score ---
                            _r = np.asarray(returns, dtype=float)
                            if len(_r) >= 20:
                                _mu = np.mean(_r[-20:])
                                _sigma = np.std(_r[-20:]) + 1e-9
                                _z = abs((_r[-1] - _mu) / _sigma)
                                _counterfactual = np.clip(1.0 - (_z / 3.0), 0.0, 1.0)
                            else:
                                _counterfactual = 0.5

                            # --- Path probability ---
                            _path_prob = float(np.clip(at_percentile, 0.0, 1.0))

                            # --- Transition entropy proxy ---
                            _transition_entropy = np.clip(1.0 - (_cd_std / (_cd_mean + 1e-9)), 0.0, 1.0)

                            # --- Attractor strength ---
                            _attractor = np.clip(0.25 + 0.25 * energy_regime, 0.0, 1.0)

                            # --- Rupture probability ---
                            _rupture = np.clip((_cd_std / (_cd_mean + 1e-9)) * (1.0 - float(es_percentile)), 0.0, 1.0)

                            # --- Cohort instability ---
                            _es = np.asarray(es_history, dtype=float)
                            _cohort = np.clip(np.std(_es[-20:]), 0.0, 1.0) if len(_es) >= 20 else 0.5

                            # --- Memory weight ---
                            _memory = np.clip(0.60 * float(at_percentile) + 0.40 * (1.0 - abs(_cd_mean - float(at_percentile))), 0.0, 1.0)

                            # --- Organism coverage ---
                            _coverage = (0.30 * float(_rf_ready) + 0.20 * float(len(es_history) >= 50) + 0.20 * float(len(_r) >= 20) + 0.30)

                            # --- Fracture ---
                            _fracture = float(abs(es_history[-1])) if len(es_history) > 0 else 0.0

                            _os = OrganismState(
                                symbol=sym,
                                timestamp=int(now.timestamp()) if hasattr(now, 'timestamp') else int(time.time()),
                                cycle_id=self._cycle_id,
                                topology_state=_topology,
                                trust=float(es_percentile),
                                pressure=float(energy_regime / 2.0) if energy_regime else 0.5,
                                fracture=_fracture,
                                trust_band="HIGH" if es_percentile > 0.7 else "MEDIUM" if es_percentile > 0.4 else "LOW",
                                pressure_band="HIGH" if (energy_regime or 2) > 1 else "LOW",
                                rupture_flag=int(_rupture > 0.75),
                                rupture_probability=float(_rupture),
                                attractor_strength=float(_attractor),
                                transition_entropy=float(_transition_entropy),
                                causal_confidence=float(_causal),
                                path_probability=float(_path_prob),
                                counterfactual_score=float(_counterfactual),
                                cohort_instability=float(_cohort),
                                thesis_rf_probability=float(_rf_prob),
                                memory_weight=float(_memory),
                                es_percentile=float(es_percentile),
                                at_percentile=float(at_percentile),
                                rf_ready=_rf_ready,
                                coverage=float(_coverage),
                                production_action=eval_data.get(sym, {}).get("prod_signal_label", "FLAT") if eval_data.get(sym, {}).get("prod_signal_label") else "FLAT",
                            )
                            self._validation.on_feature_compute(_os)
                        except Exception as ex:
                            logger.warning(f"[VALIDATION] on_feature_compute error: {ex}")
                            import traceback
                            traceback.print_exc()

                        # P3.4: Track regime transitions
                        transition = self._regime_memory.update(sym, combined_regime)

                        # P4.1: Transition-conditioned entry sizing
                        transition_tis = None
                        if transition["status"] == "TRANSITION":
                            tf = transition["from"]
                            tt = transition["to"]
                            edge = self._regime_memory.transition_edge(tf, tt)
                            if edge is not None:
                                transition_tis = edge["tis"]
                                if edge["tis"] >= 0.05 and edge["n"] >= 5:
                                    entropy_state_s_tis = self._entropy_compression.compute_state(sym)
                                    norm_ent = entropy_state_s_tis.get("normalized_entropy", 1.0) or 1.0
                                    if norm_ent < 0.70:
                                        tis_mult = self._regime_memory.sizing_multiplier(edge["tis"])
                                        sizing_mult = min(sizing_mult * tis_mult, 1.5)

                        # Scale position size using AT overlay quintiles
                        thermo_mult = eval_data.get(sym, {}).get("thermo_sizing_mult", 1.0)
                        alpha_mult = 1.0
                        if self._warmup_ticks < 504:
                            alpha_mult = 0.50  # reduced size during topology cold-start; ramps up naturally
                        elif at_percentile <= 0.20:
                            alpha_mult = 0.50
                        elif at_percentile <= 0.40:
                            alpha_mult = 0.65
                        elif at_percentile <= 0.60:
                            alpha_mult = 0.80
                        elif at_percentile <= 0.80:
                            alpha_mult = 0.90
                        else:
                            alpha_mult = 1.00
                        # P9: Quality discount for symbols without OSS surface validation
                        if not self._oss_bootstrap.has_surface(sym):
                            alpha_mult = max(0.50, alpha_mult * 0.85)
                        final_mult = max(0.25, min(1.5, thermo_mult * alpha_mult))  # multiplicative synergy capped
                        phase6_mult = getattr(self, "_phase6_current_mult", 1.0)
                        final_mult = final_mult * phase6_mult
                        logger.info(
                            f"[VOLUME_TRACE] {sym} "
                            f"thermo={thermo_mult:.2f} "
                            f"alpha={alpha_mult:.2f} "
                            f"phase6={phase6_mult:.2f} "
                            f"final={final_mult:.2f}"
                        )

                        # Calculate signal, block status and reason
                        mode = SETTINGS.deployment_mode
                        activation_mode = SETTINGS.activation_mode
                        # Bootstrap trigger: if OSS prod_signal is active, trigger immediately
                        prod_sig = eval_data.get(sym, {}).get("prod_signal", 0)
                        triggered_from_oss = (prod_sig != 0)
                        if triggered_from_oss:
                            if prod_sig > 0:
                                self._dir_stats["BUY_TRIGGER"] = self._dir_stats.get("BUY_TRIGGER", 0) + 1
                            elif prod_sig < 0:
                                self._dir_stats["SELL_TRIGGER"] = self._dir_stats.get("SELL_TRIGGER", 0) + 1
                        # RF Gate: regime quality filter — gates ALL trigger paths uniformly
                        if self._rf_gate.ready(sym):
                            rf_p = self._rf_gate.prob(sym)
                            rf_below = rf_p < self._rf_gate.prob_thresh
                            if triggered_from_oss and rf_below:
                                triggered_from_oss = False
                                logger.info(f"[RF GATE] {sym}: blocked prod_signal={prod_sig} prob={rf_p:.3f} < {self._rf_gate.prob_thresh}")
                            elif triggered_from_oss:
                                logger.info(f"[RF GATE] {sym}: passed prob={rf_p:.3f} >= {self._rf_gate.prob_thresh}")
                        elif not self._rf_gate.ready(sym) and triggered_from_oss:
                            logger.info(f"[RF GATE] {sym}: warmup ({len(self._rf_gate._tpi_vals.get(sym, []))}/2000 ticks)")
                        # RF warmup: tighten rank threshold when RF not yet calibrated
                        rf_warming_up = not self._rf_gate.ready(sym)
                        # Compute rank-based trigger independently
                        rank_triggered = False
                        if activation_mode == "PER_SYMBOL_ECDF" and SETTINGS.symbol_thresholds:
                            sym_th = SETTINGS.symbol_thresholds.get(sym, SETTINGS.default_symbol_threshold)
                            # RF warmup: require stronger threshold when RF not yet calibrated
                            if rf_warming_up:
                                sym_th = min(sym_th + 0.15, 0.95)
                            rank_triggered = es_percentile >= sym_th
                            logger.info(f"V2.3 TriggerCheck: {sym} local={es_percentile:.3f} sym_threshold={sym_th:.3f} rank_triggered={rank_triggered} (oss={triggered_from_oss})")
                        elif mode == "GLOBAL_ALL_QUALIFIED":
                            gp = self._global_rank_engine.get_global_percentile(sym)
                            thr = SETTINGS.global_rank_threshold + (0.15 if rf_warming_up else 0.0)
                            rank_triggered = gp >= thr
                            logger.debug(f"V2.2 TriggerCheck: {sym} local={es_percentile:.3f} global_pct={gp:.1f} threshold={thr:.2f} rank_triggered={rank_triggered}")
                        elif mode == "GLOBAL_TOP1":
                            top = self._global_rank_engine.get_qualified_assets(100.0)
                            rank_triggered = bool(top and top[0] == sym)
                        else:
                            thr = SETTINGS.threshold + (0.15 if rf_warming_up else 0.0)
                            rank_triggered = es_percentile > thr
                        # RF gate applies to rank path uniformly (no bypass when OSS also active)
                        if self._rf_gate.ready(sym) and rank_triggered:
                            rf_p = self._rf_gate.prob(sym)
                            if rf_p < self._rf_gate.prob_thresh:
                                rank_triggered = False
                                logger.info(f"[RF GATE] {sym}: blocked rank-triggered entry prob={rf_p:.3f} < {self._rf_gate.prob_thresh}")
                        triggered = triggered_from_oss or rank_triggered
                        # Exploration mode: force entry for randomly selected symbols
                        exploration_active = sym in self._exploration_dispatch
                        if exploration_active and not triggered:
                            triggered = True
                            logger.info(f"[EXPLORATION] {sym}: forced triggered=True for data collection")
                        blocked = False
                        block_reason = ""
                        # Compute session and entropy state unconditionally for downstream
                        sess = get_session(datetime.utcnow().hour)
                        entropy_state_s = self._entropy_compression.compute_state(sym) or {}
                        
                        if triggered:
                            active_broker_symbols = [self.mt5._get_broker_symbol(s) for s in self._execution_symbols]
                            if self._paused:
                                blocked = True
                                block_reason = "RISK_LIMIT"
                            elif is_traded:
                                # P6: Continuation pyramiding check
                                entry_es = None
                                for ticket, meta in self._active_positions_metadata.items():
                                    if self.mt5._get_broker_symbol(meta.get("symbol", "")) == broker_sym:
                                        entry_es = meta.get("max_es_rank", 0.0) or 0.0
                                        meta["trigger_count_while_open"] = meta.get("trigger_count_while_open", 0) + 1
                                        meta["max_es_rank"] = max(meta.get("max_es_rank", 0.0) or 0.0, es_percentile)
                                        meta["max_at_rank"] = max(meta.get("max_at_rank", 0.0) or 0.0, at_percentile)
                                        break
                                self._save_active_positions_metadata()

                                can_pyramid = False
                                if self._pyramid_count.get(sym, 0) < 1:
                                    es_gain = es_percentile - entry_es if entry_es else 0.0
                                    if es_gain >= 0.15:
                                        pers = self._tpi_persistence.state(sym)
                                        curv_supportive = self._tpi_curvature.is_supportive(sym, 1)  # 1=LONG
                                        has_persistence = pers.get("streak", 0) >= 2
                                        if has_persistence and curv_supportive:
                                            can_pyramid = True

                                if can_pyramid:
                                    blocked = False
                                    block_reason = "PYRAMID_ADDON"
                                    logger.info(f"P6 Pyramid[{sym}]: ES_gain={es_gain:.3f} streak={pers.get('streak',0)} curv={curv.get('state','?')}")
                                else:
                                    # P4.3: Evaluate migration before static POSITION_EXISTS
                                    ck_migrate = cohort_key(
                                        transition.get("from"), transition.get("to"),
                                        self._cycle_tpi_snapshot.get(sym),
                                        (self._entropy_compression.compute_state(sym) or {}).get("normalized_entropy"),
                                        get_session(datetime.utcnow().hour))
                                    q_new = self._migration.quality_score(
                                        es_percentile, at_percentile, transition_tis,
                                        (self._signal_decay.dps(ck_migrate) or {}).get("dps"))
                                    # Find weakest held (argmin Q_held) for this symbol
                                    held_ticket = None
                                    held_age = 0
                                    held_pnl = 0.0
                                    held_q = float("inf")
                                    held_meta = None
                                    for ht, hm in self._active_positions_metadata.items():
                                        if self.mt5._get_broker_symbol(hm.get("symbol", "")) == broker_sym:
                                            hq = hm.get("q_held", 0.0)
                                            if hq < held_q:
                                                held_q = hq
                                                held_ticket = ht
                                                held_age = self._warmup_ticks - hm.get("entry_warmup_ticks", self._warmup_ticks)
                                                held_meta = hm
                                    for hp in positions_now:
                                        if hp["ticket"] == held_ticket:
                                            held_pnl = hp.get("profit", 0.0)
                                            break
                                    initial_r = (account["balance"] if account else 100000.0) * SETTINGS.risk_per_trade
                                    avg_rps = self._migration.rps_for_migration(sym, self._occupancy_audit)
                                    logger.info(
                                        f"[MIGRATION_CHECK] sym={sym} "
                                        f"q_new={q_new:.3f} q_held={held_q:.3f} "
                                        f"delta={q_new - held_q:.3f} "
                                        f"rps={avg_rps:.3f} age={held_age} "
                                        f"pnl={held_pnl:.2f} r={initial_r:.2f} "
                                        f"held_ticket={held_ticket}")
                                    # P1-D: History check before stale purge
                                    if held_ticket is not None and held_ticket not in {p["ticket"] for p in positions_now}:
                                        import MetaTrader5 as _mt5_stale
                                        deal_list = _mt5_stale.history_deals_get(ticket=held_ticket)
                                        if deal_list:
                                            d = deal_list[0]
                                            reason_map = {0: "BROKER_UNKNOWN", 1: "SL", 2: "TP", 3: "BROKER_STOP", 4: "MANUAL", 6: "SL", 8: "MANUAL"}
                                            mt5_reason = d.reason if hasattr(d, 'reason') else -1
                                            broker_exit = reason_map.get(mt5_reason, "BROKER_UNKNOWN")
                                            logger.warning(f"[STALE_HELD] {sym} ticket={held_ticket} closed by MT5 reason={mt5_reason} ({broker_exit}), syncing")
                                            self.trade_ledger.close_by_ticket(
                                                held_ticket,
                                                exit_reason=broker_exit,
                                                exit_detail=f"stale_mt5_reason_{mt5_reason}"
                                            )
                                        else:
                                            logger.warning(f"[STALE_HELD] {sym} ticket={held_ticket} true ghost, reconciling")
                                            self.trade_ledger.close_by_ticket(
                                                held_ticket,
                                                exit_reason="RECONCILE",
                                                exit_detail="stale_ghost_position"
                                            )
                                        held_ticket = None
                                        held_age = 0
                                        held_q = 0.0
                                        held_meta = None
                                    # P1-E: Position age telemetry
                                    logger.info(
                                        f"[POSITION_AGE] {sym} ticket={held_ticket} "
                                        f"age={held_age} q={held_q:.3f} pnl={held_pnl:.2f} "
                                        f"rps={avg_rps:.3f}")
                                    # Migration engine baseline
                                    should_migrate, migrate_reason = self._migration.should_migrate(
                                        q_new, held_q, avg_rps, held_age, held_pnl, initial_r)
                                    # P0-C: Delta threshold — can only BLOCK migration
                                    if held_ticket is not None:
                                        min_delta = max(0.08, abs(held_q) * 0.30)
                                        if q_new <= held_q + min_delta:
                                            should_migrate = False
                                            migrate_reason = f"INSUFFICIENT_DELTA delta={q_new-held_q:.3f} < min={min_delta:.3f}"
                                            logger.info(f"[MIGRATION_BLOCK] {sym} {migrate_reason}")
                                    # P0-B: Min hold freeze — can only BLOCK migration
                                    if held_ticket is not None and held_age < MIN_HOLD_TICKS_MIGRATION:
                                        should_migrate = False
                                        migrate_reason = f"MIN_HOLD_MIGRATION age={held_age} < {MIN_HOLD_TICKS_MIGRATION}"
                                        logger.info(f"[MIN_HOLD_MIGRATION] {sym} {migrate_reason}")
                                    # G3: EV sign flip — FORCES migration regardless of all blocks
                                    ev_sign_flip = False
                                    held_ev = held_meta.get("entry_ev", 0.0) if held_meta else 0.0
                                    current_ev = oss_info.get("ev", 0.0) if 'oss_info' in dir() else 0.0
                                    if held_ev != 0 and current_ev != 0 and (held_ev > 0) != (current_ev > 0):
                                        ev_sign_flip = True
                                        should_migrate = True
                                        migrate_reason = f"EV sign flip ({held_ev:.4f} -> {current_ev:.4f})"
                                        logger.info(f"[EV_SIGN_FLIP] {sym} {migrate_reason}")
                                    if should_migrate and held_ticket is not None:
                                        if self._paused:
                                            should_migrate = False
                                            migrate_reason = "PAUSED"
                                            logger.info(f"[MIGRATION_PAUSED] {sym} system paused, migration deferred")
                                        if should_migrate:
                                            min_delta_val = max(0.08, abs(held_q) * 0.30)
                                            logger.info(
                                                f"[MIGRATION_EXEC] {sym} "
                                                f"held_q={held_q:.3f} new_q={q_new:.3f} "
                                                f"delta={q_new-held_q:.3f} min={min_delta_val:.3f} "
                                                f"reason={migrate_reason}")
                                            logger.info(
                                                f"[MATURITY] {sym} "
                                                f"age={held_age} min_hold={MIN_HOLD_TICKS_MIGRATION} "
                                                f"ratio={held_age/max(1,MIN_HOLD_TICKS_MIGRATION):.2f}")
                                            held_profit = next((hp.get("profit", 0.0) for hp in positions_now if hp["ticket"] == held_ticket), 0.0)
                                            if held_ticket in self._active_positions_metadata:
                                                self._active_positions_metadata[held_ticket]["expected_exit_reason"] = "MIGRATION"
                                                self._active_positions_metadata[held_ticket]["last_profit"] = held_profit
                                                self._save_active_positions_metadata()
                                            # Defer close until route confirmed (two-phase atomicity)
                                            self._pending_close[broker_sym] = (held_ticket, self._cycle_id)
                                            # V6: Position lock after MIGRATION strategy exit
                                            broker_sym_lock = self.mt5._get_broker_symbol(sym)
                                            self._position_lock_until[broker_sym_lock] = SETTINGS.position_lock_bars
                                            self._lock_event_count += 1
                                            self._migration.record_event(
                                                from_symbol=sym, to_symbol=sym,
                                                from_ticket=held_ticket,
                                                q_new=q_new, q_held=held_q,
                                                reason=migrate_reason,
                                                avg_rps=avg_rps, age=held_age)
                                            blocked = False
                                            block_reason = "MIGRATION"
                                    else:
                                        # P3 + G1: Flip reversal check before static POSITION_EXISTS
                                        held_dir = None
                                        for hp in positions_now:
                                            if self.mt5._get_broker_symbol(hp["symbol"]) == broker_sym:
                                                held_dir = hp.get("type")  # "BUY" or "SELL"
                                                break
                                        sig_dir = self._execution_plan.get(sym, {}).get("direction_num", prod_signals.get(sym, 0))  # 1=BUY, -1=SELL
                                        held_sig = (1 if held_dir == "BUY" else -1 if held_dir == "SELL" else 0)
                                        is_flip = (held_sig != 0 and sig_dir != 0 and held_sig != sig_dir)

                                        if is_flip:
                                            flip_ticket = next((hp2["ticket"] for hp2 in positions_now if self.mt5._get_broker_symbol(hp2["symbol"]) == broker_sym), None)
                                            held_age = self._position_tick_age.get(flip_ticket, 0)
                                            if held_age < MIN_HOLD_TICKS_FLIP:
                                                blocked = True
                                                block_reason = "FLIP_TOO_YOUNG"
                                                logger.info(f"[FLIP_TOO_YOUNG] {sym} age={held_age} < {MIN_HOLD_TICKS_FLIP}")
                                                self._gate_audit_logger.log_rejection(sym, signal_id="?", gate="FLIP_TOO_YOUNG", value=None, threshold=None, context={"es_rank": es_percentile})
                                            else:
                                                cooldown = self._flip_cooldown.get(broker_sym, 0)
                                                if cooldown > 0:
                                                    self._flip_cooldown[broker_sym] = cooldown - 1
                                                    blocked = True
                                                    block_reason = "FLIP_COOLDOWN"
                                                    logger.info(f"[FLIP_COOLDOWN] {sym} cooldown={cooldown} ticks remaining")
                                                    self._gate_audit_logger.log_rejection(sym, signal_id="?", gate="FLIP_COOLDOWN", value=None, threshold=None, context={"es_rank": es_percentile})
                                                else:
                                                    # Close existing position for flip
                                                    held_ticket2 = None
                                                    for hp2 in positions_now:
                                                        if self.mt5._get_broker_symbol(hp2["symbol"]) == broker_sym:
                                                            held_ticket2 = hp2["ticket"]
                                                            break
                                                    if held_ticket2 is not None:
                                                        logger.info(f"[FLIP] {sym}: will close {held_ticket2} ({held_dir}) to open flip signal={sig_dir}")
                                                        flip_profit = next((hp2.get("profit", 0.0) for hp2 in positions_now if hp2["ticket"] == held_ticket2), 0.0)
                                                        if held_ticket2 in self._active_positions_metadata:
                                                            self._active_positions_metadata[held_ticket2]["expected_exit_reason"] = "FLIP"
                                                            self._active_positions_metadata[held_ticket2]["last_profit"] = flip_profit
                                                            self._save_active_positions_metadata()
                                                        # Defer close until route confirmed (two-phase atomicity)
                                                        self._pending_close[broker_sym] = (held_ticket2, self._cycle_id)
                                                        # V6: Position lock after FLIP strategy exit
                                                        broker_sym_lock = self.mt5._get_broker_symbol(sym)
                                                        self._position_lock_until[broker_sym_lock] = SETTINGS.position_lock_bars
                                                        self._lock_event_count += 1
                                                        self._flip_cooldown[broker_sym] = 3  # FLIP_COOLDOWN_TICKS
                                                        blocked = False
                                                        block_reason = "FLIP"
                                                    else:
                                                        blocked = True
                                                        block_reason = "POSITION_EXISTS"
                                                        self._gate_audit_logger.log_rejection(sym, signal_id="?", gate="POSITION_EXISTS", value=None, threshold=None, context={"es_rank": es_percentile})
                                                        if hasattr(self, '_lifecycle_mgr'):
                                                            sig_id = eval_data.get(sym, {}).get("signal_id", f"sig_{sym}_{self._warmup_ticks}")
                                                            self._lifecycle_mgr.record_rejected(sig_id, reason="POSITION_EXISTS")
                                        else:
                                            blocked = True
                                            block_reason = "POSITION_EXISTS"
                                            self._gate_audit_logger.log_rejection(sym, signal_id="?", gate="POSITION_EXISTS", value=None, threshold=None, context={"es_rank": es_percentile})
                                            if hasattr(self, '_lifecycle_mgr'):
                                                sig_id = eval_data.get(sym, {}).get("signal_id", f"sig_{sym}_{self._warmup_ticks}")
                                                self._lifecycle_mgr.record_rejected(sig_id, reason="POSITION_EXISTS")
                                        if block_reason == "POSITION_EXISTS":
                                            self._reinforcement_blocks += 1
                                            logger.info(f"[POS_EXISTS] {sym} held_ticket={held_ticket} held_dir={held_dir} sig_dir={sig_dir}")
                                        elif block_reason in ("FLIP_COOLDOWN",):
                                            self._flip_blocks += 1
                                        # P1.1: Occupancy audit — capture blocked signal with full context
                                        entry_px = rates[-1].get("close", rates[-1].get("open", 0.0)) if rates else 0.0
                                        bar_ts = rates[-1]["time"] if rates else self._now_ts()
                                        self._occupancy_audit.record_blocked(
                                            symbol=sym, bar_time=bar_ts, side=1,
                                            es_rank=es_percentile, at_rank=at_percentile,
                                            held_ticket_ids=[p["ticket"] for p in positions_now],
                                            regime=f"E{energy_regime}_T{time_regime}" if energy_regime is not None else "UNKNOWN",
                                            block_reason=block_reason if blocked else "FLIP",
                                            entry_price=entry_px,
                                        )
                            elif len([p for p in positions_now if p["symbol"] in active_broker_symbols]) >= SETTINGS.max_positions_active:
                                blocked = True
                                block_reason = "MAX_POSITIONS"
                                self._gate_audit_logger.log_rejection(sym, signal_id="?", gate="MAX_POSITIONS", value=None, threshold=None, context={"es_rank": es_percentile})
                            elif mt5_status["symbols"].get(sym, {}).get("spread_invalid", False):
                                blocked = True
                                block_reason = "INVALID_SPREAD"
                                self._gate_audit_logger.log_rejection(sym, signal_id="?", gate="INVALID_SPREAD", value=None, threshold=None, context={"es_rank": es_percentile})
                            elif mt5_status["symbols"].get(sym, {}).get("spread_stale", False):
                                fallback = SETTINGS.min_spread_fallback.get(sym, 2)
                                if sym in self._spread_normalizer._session_baselines:
                                    baselines = self._spread_normalizer._session_baselines[sym]
                                    if baselines:
                                        hist_spread = max(baselines.values())
                                        if hist_spread > 0:
                                            fallback = max(int(hist_spread), 1)
                                eval_data[sym]["spread"] = fallback
                                logger.debug(f"[STALE SPREAD] {sym}: spread=0, applying fallback={fallback}")
                            else:
                                tick = self.tick_source.next_tick(sym) if self.tick_source else (self._tick_cache.get_tick(sym) if self._tick_cache else self.mt5.get_tick(sym))
                                if tick is None:
                                    blocked = True
                                    block_reason = "NO_TICK"
                                    self._gate_audit_logger.log_rejection(sym, signal_id="?", gate="NO_TICK", value=None, threshold=None, context={"es_rank": es_percentile})
                                else:
                                    # Points-aligned with _dispatch_tick
                                    spread_raw = self._spread_points(tick)
                                    # P1.3: Adaptive spread normalization
                                    utc_hour_s = datetime.utcnow().hour
                                    sess = get_session(utc_hour_s)
                                    entropy_state_s = self._entropy_compression.compute_state(sym)
                                    entropy_score_s = entropy_state_s.get("entropy_score", 0) if entropy_state_s else 0
                                    expected_move = eval_data.get(sym, {}).get("expected_move", 0.0)
                                    spread_result = self._spread_normalizer.evaluate(
                                        symbol=sym, spread=spread_raw, rates=rates,
                                        entropy_score=entropy_score_s,
                                        session=sess,
                                        es_rank=es_percentile,
                                        expected_move=expected_move,
                                    )
                                    # P7: Inject spread decay into eval_data
                                    eval_data[sym]["spread_decay"] = spread_result.get("spread_decay", 1.0)
                                    eval_data[sym]["econ_ratio"] = spread_result.get("econ_ratio", 0.0)
                                    eval_data[sym]["expected_move"] = spread_result.get("expected_move", expected_move)
                                    self._spread_normalizer.update_session_baseline(sym, sess, spread_raw)
                                    # Microstructure observation
                                    if hasattr(self, '_micro_calibrator') and spread_raw > 0:
                                        self._micro_calibrator.observe_spread(sym, spread_raw)
                                    if not spread_result["passed"]:
                                        blocked = True
                                        block_reason = "SPREAD_NORM"
                                        logger.debug(f"Spread norm block[{sym}]: raw={spread_raw} "
                                                     f"atr_r={spread_result['atr_ratio']:.4f} "
                                                     f"ent_r={spread_result['entropy_ratio']:.4f} "
                                                     f"sess_r={spread_result['session_ratio']:.4f} "
                                                     f"es={es_percentile:.3f}")
                                        self._gate_audit_logger.log_rejection(sym, signal_id="?", gate="SPREAD_NORM", value=None, threshold=None, context={"es_rank": es_percentile})
                                    else:
                                        # Legacy ES-based elastic check as secondary guard
                                        if not self.mt5.verify_spread(sym, es_rank=es_percentile):
                                            blocked = True
                                            block_reason = "SPREAD"
                                            logger.debug(f"Spread elastic block[{sym}]: raw={spread_raw} es={es_percentile:.3f}")
                                            self._gate_audit_logger.log_rejection(sym, signal_id="?", gate="SPREAD", value=None, threshold=None, context={"es_rank": es_percentile})
                        else:
                            block_reason = "THRESHOLD_NOT_MET"
                            self._gate_audit_logger.log_rejection(sym, signal_id="?", gate="THRESHOLD_NOT_MET", value=None, threshold=None, context={"es_rank": es_percentile})


                        # Signal lifecycle audit
                        signal_id = self.audit.record_generated(
                            sym, es_percentile, current_es, at_percentile,
                            "DYNAMIC")
                        # Record in lifecycle manager
                        if hasattr(self, '_lifecycle_mgr'):
                            self._lifecycle_mgr.record_generated(signal_id, sym, prod_signals.get(sym, 0), es_percentile, eval_data[sym].get("price", 0.0))
                        if triggered:
                            self.audit.record_threshold_passed(signal_id)
                            self.audit.record_triggered(signal_id)
                            if hasattr(self, '_lifecycle_mgr'):
                                self._lifecycle_mgr.record_threshold_passed(signal_id)
                                self._lifecycle_mgr.record_triggered(signal_id)
                        eval_data[sym]["signal_id"] = signal_id

                        # P1.4: Funnel starvation audit — record generated and stage transitions
                        self._funnel_audit.record_generated(signal_id, sym, es_percentile)
                        passed_threshold = triggered
                        passed_tpi = not blocked or block_reason not in ("TPI_GATE", "PERSISTENCE_GATE", "CURVATURE_GATE", "TPI_HARD_GATE")
                        passed_spread = not blocked or block_reason not in ("SPREAD", "SPREAD_NORM", "INVALID_SPREAD", "NO_TICK")
                        passed_occupancy = not blocked or block_reason not in ("POSITION_EXISTS", "MAX_POSITIONS", "POSITION_LOCK", "FLIP_COOLDOWN")
                        passed_risk = not blocked or block_reason not in ("RISK_LIMIT", "RHL_BLOCKED", "NOT_IN_TOP3")
                        self._funnel_audit.update(signal_id,
                            passed_threshold=passed_threshold,
                            passed_tpi=passed_tpi,
                            passed_spread=passed_spread,
                            passed_occupancy=passed_occupancy,
                            passed_risk=passed_risk,
                        )
                        # Observe spread for model calibration
                        spread_val = eval_data.get(sym, {}).get("spread")
                        if spread_val is not None:
                            self._spread_model.observe_spread(sym, spread_val)

                        # P4.2: Build cohort key for signal decay velocity
                        rf_ck = transition.get("from")
                        rt_ck = transition.get("to")
                        tpi_sig_ck = self._cycle_tpi_snapshot.get(sym)
                        ne_ck = entropy_state_s.get("normalized_entropy")
                        sess_ck = sess
                        ck = cohort_key(rf_ck, rt_ck, tpi_sig_ck, ne_ck, sess_ck)

                        # RCL-1A context needs tpi_sig — get from snapshot
                        tpi_sig = self._cycle_tpi_snapshot.get(sym)

                        # RCL-1A: Record signal with current feature state
                        self._outcome_ledger.record_signal(
                            signal_id=signal_id, symbol=sym,
                            timestamp=self._now_ts(),
                            direction=prod_signals.get(sym, 0),
                            features={
                                "es_rank": es_percentile,
                                "at_rank": at_percentile,
                                "triggered": triggered,
                                "blocked": blocked,
                                "block_reason": block_reason if blocked else None,
                                "transition_tis": transition_tis,
                                "cohort_key": ck,
                                "cohort_dps": (self._signal_decay.dps(ck) or {}).get("dps", 0.0),
                            },
                            context={
                                "symbol": sym,
                                "price": eval_data[sym].get("price"),
                                "spread": eval_data[sym].get("spread"),
                                "regime": eval_data[sym].get("regime"),
                                "combined_regime": combined_regime,
                                "regime_from": transition.get("from"),
                                "regime_to": transition.get("to"),
                                "session": sess,
                                "normalized_entropy": entropy_state_s.get("normalized_entropy") if entropy_state_s else None,
                                "tpi_direction": tpi_sig.get("direction") if tpi_sig else None,
                                "tpi_confidence": tpi_sig.get("confidence") if tpi_sig else 0,
                            },
                            tpi_arm=self._tpi_arm,
                        )

                        # Parity ledger: capture raw signal event
                        if self._env and hasattr(self._env, 'ledger') and self._env.ledger is not None:
                            self._env.ledger.add_signal({
                                "symbol": sym,
                                "ts": self._now_ts(),
                                "phase": "raw",
                                "triggered": triggered,
                                "blocked": blocked,
                                "block_reason": block_reason if blocked else None,
                                "es_rank": es_percentile,
                                "at_rank": at_percentile,
                                "tpi_direction": tpi_sig.get("direction") if tpi_sig else None,
                                "tpi_confidence": tpi_sig.get("confidence") if tpi_sig else 0,
                                "regime": combined_regime,
                                "session": sess,
                                "signal_id": signal_id,
                            })

                        # P2.5: Track session balance
                        session_name = tpi_sig.get("session_name", "UNKNOWN") if tpi_sig else "UNKNOWN"
                        if session_name in self._session_balance:
                            self._session_balance[session_name] += 1

                        execution_success = False
                        result = None

                        # V6: Position lock check
                        broker_sym_lc = broker_sym
                        if broker_sym_lc in self._position_lock_until:
                            if self._position_lock_until[broker_sym_lc] > 0:
                                self._position_lock_until[broker_sym_lc] -= 1
                                if triggered:
                                    blocked = True
                                    block_reason = "POSITION_LOCK"
                            else:
                                del self._position_lock_until[broker_sym_lc]

                        # V6: Migration observation
                        if sym in top3_qualified:
                            was_in_top3 = False
                            if self._top3_history and len(self._top3_history) >= 2:
                                was_in_top3 = sym in self._top3_history[-2][1]
                            if not was_in_top3 and len(self._top3_history) >= 2:
                                self._migration_event_count += 1

                        # P0.13: Track rejected signals for ObserverDecay selection bias analysis
                        if triggered and blocked and sym not in self._active_signal:
                            _rej_tpi = tpi_sig.get("confidence", 0) if tpi_sig else 0
                            _rej_pers = self._tpi_persistence.state(sym).get("streak", 0) if hasattr(self, '_tpi_persistence') else 0
                            _rej_entropy = (self._entropy_compression.compute_state(sym) or {}).get("normalized_entropy", 0.5) if hasattr(self, '_entropy_compression') else 0.5
                            if hasattr(self, 'observer_decay'):
                                self.observer_decay.reject(
                                    signal_id, block_reason or "UNKNOWN",
                                    tpi_confidence=_rej_tpi,
                                    persistence_streak=_rej_pers,
                                    normalized_entropy=_rej_entropy,
                                    regime=combined_regime)
                        if triggered and not blocked and not self._paused:
                            # MOF: Market Observability Filter gate
                            if getattr(self, '_mof_blocked', False):
                                blocked = True
                                block_reason = "MOF_DEGRADED"
                                _mof_ss = getattr(self, '_last_mof_state', 'UNKNOWN')
                                logger.info(f"[MOF_GATE] {sym}: blocked, state={_mof_ss}")
                                self._gate_audit_logger.log_rejection(sym, signal_id="?", gate="MOF_DEGRADED", value=None, threshold=None, context={"es_rank": es_percentile})
                                self._freq_pipeline.record_blocked(sym, es_percentile, at_percentile, 0.90, block_reason, 0.0, energy_regime=energy_regime, time_regime=time_regime, combined_regime=combined_regime)
                                if signal_id:
                                    self.audit.record_blocked(signal_id, block_reason)
                                if self._env and hasattr(self._env, 'ledger') and self._env.ledger is not None:
                                    self._env.ledger.add_signal({
                                        "symbol": sym, "ts": self._now_ts(),
                                        "phase": "gate_reject", "reason": block_reason,
                                        "signal_id": signal_id,
                                    })
                                continue
                            # TPI calibration gates (regime-aware, adaptive thresholds)
                            gate_result = {"gate_blocked": False, "reasons": [], "regime": "UNKNOWN"}
                            if sym in TPI_ELIGIBLE:
                                tpi_sig_early = self._cycle_tpi_snapshot.get(sym)
                                if tpi_sig_early is None:
                                    raise RuntimeError(f"Missing cycle TPI snapshot for {sym}")
                                tpi_dir_early = tpi_sig_early.get("direction", 0)
                                pers_early = self._tpi_persistence.state(sym)
                                curv_state = self._tpi_curvature.state(sym).get("state", "NEUTRAL")
                                # Feed ECDF regime to calibration layer
                                ecdf_for_regime = eval_data.get(sym, {}).get("ecdf_rank", 0.5)
                                self._tpi_calibration.update_regime(sym, ecdf_for_regime)
                                # Evaluate gates with regime-aware thresholds
                                gate_result = self._tpi_calibration.evaluate(
                                    sym, tpi_dir_early, pers_early.get("streak", 0),
                                    curv_state, 1)  # position_dir=1 (LONG)
                                if not exploration_active and gate_result["gate_blocked"]:
                                    blocked = gate_result["blocked"]
                                    block_reason = gate_result["reasons"][0]
                                    logger.info(f"TPI gate[{sym}]: {gate_result['reasons']} regime={gate_result['regime']} blocked={blocked}")
                                    self._freq_pipeline.record_blocked(sym, es_percentile, at_percentile, 0.90, block_reason, 0.0, energy_regime=energy_regime, time_regime=time_regime, combined_regime=combined_regime)
                                    # B4: record_blocked removed here — unified blocker at funnel transition handles it once
                                    if self._env and hasattr(self._env, 'ledger') and self._env.ledger is not None:
                                        self._env.ledger.add_signal({
                                            "symbol": sym, "ts": self._now_ts(),
                                            "phase": "gate_reject", "reason": block_reason,
                                            "signal_id": signal_id,
                                        })

                            if blocked:
                                pass  # TPI gate or other gate blocked execution
                            elif not exploration_active and SETTINGS.portfolio_mode == "TOP_3_ROTATION" and sym not in self._top3_submit_set:
                                blocked = True
                                block_reason = "NOT_IN_TOP3"
                                logger.info(f"[BLOCKED] {sym} reason={block_reason} top3_set={self._top3_submit_set}")
                                self._freq_pipeline.record_blocked(
                                    sym, es_percentile, at_percentile,
                                    0.90, block_reason, 0.0,
                                    energy_regime=energy_regime,
                                    time_regime=time_regime,
                                    combined_regime=combined_regime)
                                if self._env and hasattr(self._env, 'ledger') and self._env.ledger is not None:
                                    self._env.ledger.add_signal({
                                        "symbol": sym, "ts": self._now_ts(),
                                        "phase": "gate_reject", "reason": block_reason,
                                        "signal_id": signal_id,
                                    })
                                continue
                            else:
                                risk_pct = SETTINGS.risk_per_trade * final_mult
                                logger.info(f"[PHASE6] {sym} risk_scaled={risk_pct:.4f} mult={phase6_mult:.4f} rollout={self._phase6_rollout.state}")
                                print(f"[EXEC FLOW] {sym}: entering execute block risk_pct={risk_pct:.4f} final_mult={final_mult:.2f} phase6_mult={phase6_mult:.4f}", flush=True)
                                tick = self.tick_source.next_tick(sym) if self.tick_source else (self._tick_cache.get_tick(sym) if self._tick_cache else self.mt5.get_tick(sym))
                                if tick is None:
                                    blocked = True
                                    block_reason = "SPREAD"
                                    print(f"[EXEC FLOW] {sym}: tick=None blocked=SPREAD", flush=True)
                                    logger.info(f"[BLOCKED] {sym} reason={block_reason} (tick=None)")
                                elif not self.mt5.verify_spread(sym, es_rank=es_percentile):
                                    blocked = True
                                    block_reason = "SPREAD"
                                    logger.info(f"[BLOCKED] {sym} reason={block_reason} spread_verify_failed es={es_percentile:.3f}")
                                else:
                                    # Refresh tick for execution pricing (named tuple -> dict)
                                    try:
                                        import MetaTrader5 as _mt5_refresh
                                        _fresh_tick = _mt5_refresh.symbol_info_tick(sym)
                                        if _fresh_tick is not None:
                                            tick = {
                                                "bid": _fresh_tick.bid,
                                                "ask": _fresh_tick.ask,
                                                "time": _fresh_tick.time,
                                                "spread": getattr(_fresh_tick, "spread", 0),
                                            }
                                    except Exception:
                                        pass
                                    _ex_dir = self._execution_plan.get(sym, {}).get("direction", "FLAT")
                                    # Exploration mode: force direction from TPI signal
                                    if exploration_active and _ex_dir == "FLAT":
                                        _tpi_for_dir = self._cycle_tpi_snapshot.get(sym, {}).get("direction", 0)
                                        if _tpi_for_dir > 0:
                                            _ex_dir = "BUY"
                                        elif _tpi_for_dir < 0:
                                            _ex_dir = "SELL"
                                        else:
                                            _ex_dir = "BUY" if hash(f"{sym}_{self._cycle_id}") % 2 == 0 else "SELL"
                                        logger.info(f"[EXPLORATION] {sym}: forced direction={_ex_dir} (was FLAT)")
                                    _ask = tick["ask"] if tick else 0.0
                                    _bid = tick["bid"] if tick else 0.0
                                    exec_price = _bid if _ex_dir == "SELL" else _ask
                                    volume = self.orders.calculate_volume(sym, exec_price,
                                        account["balance"] if account else 100000.0, risk_pct)
                                    # P0.26: Restart quarantine — halve position size if still in quarantine
                                    if self._quarantine_cycles_remaining > 0:
                                        volume = max(volume * 0.5, 0.01)
                                        logger.info(f"[QUARANTINE] {sym}: reducing vol {volume*2:.3f}→{volume:.3f} "
                                                    f"({self._quarantine_cycles_remaining} cycles remaining)")
                                    price = exec_price
                                    balance = account["balance"] if account else 100000.0
                                    # Wave 3: Update drawdown tracker
                                    self.drawdown_manager.update(balance)

                                    print(f"[EXEC FLOW] {sym}: pre_order_check vol={volume:.2f} price={price:.4f} bal={balance:.0f} pos={len(positions_now)}", flush=True)
                                    rhl_check = self._risk.pre_order_check(
                                        sym, volume, price, balance, positions_now, risk_pct=risk_pct, direction=_ex_dir)
                                    print(f"[EXEC FLOW] {sym}: rhl_check.allowed={rhl_check.get('allowed')} reason={rhl_check.get('reason','?')}", flush=True)
                                    if not rhl_check["allowed"]:
                                        blocked = True
                                        block_reason = rhl_check.get("reason", "RHL_BLOCKED")
                                        # Budget-block TTL: skip this symbol for N cycles
                                        if "budget_fit_zero_volume" in block_reason:
                                            self._budget_block_ttl[sym] = self._cycle_id + 5
                                            logger.info(f"[BUDGET_BLOCK] {sym}: blocked for 5 cycles until cycle {self._cycle_id + 5}")
                                        self._freq_pipeline.record_blocked(
                                            sym, es_percentile, at_percentile,
                                            0.90, block_reason, price,
                                            energy_regime=energy_regime,
                                            time_regime=time_regime,
                                            combined_regime=combined_regime)
                                        self.audit.record_blocked(signal_id, block_reason)
                                        if self._env and hasattr(self._env, 'ledger') and self._env.ledger is not None:
                                            self._env.ledger.add_signal({
                                                "symbol": sym, "ts": self._now_ts(),
                                                "phase": "gate_reject", "reason": block_reason,
                                                "signal_id": signal_id,
                                            })
                                        continue

                                    ex_dir = _ex_dir
                                    print(f"[SUBMIT] {sym}: recording submitted ex_dir={ex_dir}", flush=True)
                                    self.audit.record_submitted(signal_id)
                                    self._dir_stats["BUY_SUBMIT" if ex_dir == "BUY" else "SELL_SUBMIT"] = self._dir_stats.get("BUY_SUBMIT", 0) + 1
                                    self._funnel_audit.update(signal_id, submitted=True)
                                    self.order_tracker.record_attempt(
                                        signal_id, sym, ex_dir, volume,
                                        price, es_percentile, at_percentile)

                                    self.exec_stats.record_order_attempt()
                                    _tpi_sig_obs = self._cycle_tpi_snapshot.get(sym, {})
                                    _pers_obs = self._tpi_persistence.state(sym)
                                    _curv_obs = self._tpi_curvature.state(sym).get("state", "NEUTRAL")
                                    _entropy_obs = (self._entropy_compression.compute_state(sym) or {}).get("normalized_entropy", 0.5)
                                    _obs_state = self._compute_observer_state(
                                        tpi_confidence=_tpi_sig_obs.get("confidence", 0),
                                        persistence_streak=_pers_obs.get("streak", 0),
                                        curvature_state=_curv_obs,
                                        normalized_entropy=_entropy_obs)
                                    if sym not in self._active_signal:
                                        self._observer_seq[sym] = self._observer_seq.get(sym, 0) + 1
                                        _sid = f"{sym}:{self._observer_seq[sym]}"
                                        self._active_signal[sym] = _sid
                                        self.observer_decay.birth(
                                            _sid, _obs_state.get("reality_score", 0.5),
                                            _pers_obs.get("streak", 0), _entropy_obs, combined_regime)
                                    _tick_count_est = max(1, len(self._cycle_tpi_snapshot))
                                    self.observer_decay.tick(_tick_count_est)
                                    _decayed = self.observer_decay.compute(
                                        self._active_signal[sym],
                                        _pers_obs.get("streak", 0), _entropy_obs, combined_regime)
                                    _obs_state["observer_state"] = _decayed["state"]
                                    _obs_state["reality_score"] = _decayed["confidence"]
                                    self.weak_day.record_entropy_compression(
                                        _entropy_obs < 0.3, _curv_obs == "ACCELERATION")
                                    self.weak_day.record_persistence_event(
                                        _pers_obs.get("streak", 0), _pers_obs.get("streak", 0) < 2)
                                    self.weak_day.record_regime(combined_regime)
                                    _weak = self.weak_day.compute()
                                    _obs_state["reality_score"] *= _weak["trade_multiplier"]
                                    # SymbolTrust removed from forward execution path — all symbols evaluated on live signal only
                                    _obs_state["reality_score"] = max(0.0, min(1.0, _obs_state["reality_score"]))
                                    # Execute pending close (migration/flip) — atomic: close only if route will proceed
                                    pending_info = self._pending_close.pop(broker_sym, None)
                                    if pending_info is not None:
                                        pending_ticket, pending_cycle = pending_info
                                        if pending_cycle != self._cycle_id:
                                            logger.warning(f"[ATOMIC_CLOSE] {sym}: stale pending close cycle={pending_cycle} != current={self._cycle_id}, discarding")
                                        else:
                                            # Bridge gate — centralised exit authority
                                            if not self._bridge_allows_exit(sym, pending_ticket, "MIGRATION/FLIP"):
                                                logger.info(f"[BRIDGE_SKIP] {sym} ticket={pending_ticket} — MIGRATION/FLIP blocked, deferring")
                                                continue
                                            close_ok = self.orders.close(pending_ticket)
                                            if not close_ok:
                                                logger.warning(f"[ATOMIC_CLOSE] {sym}: failed to close ticket {pending_ticket}, aborting re-entry")
                                                blocked = True
                                                block_reason = "CLOSE_FAILED"
                                                continue
                                            logger.info(f"[ATOMIC_CLOSE] {sym}: closed ticket {pending_ticket} for re-entry")
                                            self._sdl.release(sym)
                                            self._reconcile_broker_positions()
                                    # E7: Spread net-alpha gate — expected spread cost must leave positive edge
                                    point_val = 0.01 if "JPY" in sym else (0.1 if "XAU" in sym or "XAG" in sym else 0.0001)
                                    spread_cost_pts = (_ask - _bid) / max(point_val, 1e-9)
                                    expected_alpha_pts = abs(price - rhl_check.get("sl", price)) * 0.3  # rough edge estimate
                                    net_alpha_pts = expected_alpha_pts - spread_cost_pts * 2  # entry + exit spread
                                    if not exploration_active and net_alpha_pts <= 0:
                                        logger.info(f"[NET_ALPHA_BLOCK] {sym}: expected alpha={expected_alpha_pts:.2f} <= spread_cost={spread_cost_pts*2:.2f}, skipping")
                                        self.rejection_engine.reject(sym, RejectionType.NET_ALPHA, net_alpha_pts, time.time())
                                        continue
                                    # Phase 2: RealityVector execution mode gates
                                    rv = self.exec_stats.reality_vector(sym)
                                    self.exec_stats.log_reality_vector(sym, rv)
                                    exec_tier = classify_execution_mode(rv)
                                    if not exploration_active and exec_tier == "PASSIVE":
                                        logger.info(f"[PASSIVE_SKIP] {sym}: rv(E_exec={rv.E_exec:.3f} E_pred={rv.E_pred:.3f} E_contam={rv.E_contam:.3f}) -> PASSIVE")
                                        self.rejection_engine.reject(sym, RejectionType.PASSIVE_MODE, rv.E_exec, time.time())
                                        continue
                                    # Wave 1: Quarantine mode blocks all execution
                                    if hasattr(self, '_battle_decay') and self._battle_decay.is_quarantined():
                                        logger.info(f"[QUARANTINE_BLOCK] {sym}: system in quarantine, skipping entry")
                                        self.rejection_engine.reject(sym, RejectionType.QUARANTINE, 1.0, time.time())
                                        continue
                                    # Wave 3: Kill switch check (redundant with global, safety net)
                                    if self.kill_switch.is_active():
                                        self.rejection_engine.reject(sym, RejectionType.KILL_SWITCH, 1.0, time.time())
                                        continue
                                    # Wave 3: Lock manager directional symmetry
                                    ex_dir_int = 1 if _ex_dir == "BUY" else -1 if _ex_dir == "SELL" else 0
                                    if not self.lock_manager.can_trade(sym, ex_dir_int):
                                        logger.info(f"[LOCK_BLOCK] {sym}: direction {_ex_dir} locked, skipping entry")
                                        self.rejection_engine.reject(sym, RejectionType.SYMBOL_LOCK, float(ex_dir_int), time.time())
                                        continue
                                    # Wave 3: Unified risk gate
                                    if not risk_gate(
                                        drawdown=self.drawdown_manager.drawdown(),
                                        cf_block_rate=self._cf_gate_block_counter / max(len(self.exec_stats.get_cf_records(20) if hasattr(self, 'exec_stats') and self.exec_stats else [1]), 1),
                                        tpi_collapse=getattr(self, '_battle_decay', None) and self._battle_decay.is_quarantined(),
                                        kill_switch=self.kill_switch,
                                    ):
                                        logger.info(f"[RISK_GATE] {sym}: risk gate blocked, skipping entry")
                                        self.rejection_engine.reject(sym, RejectionType.RISK_GATE,
                                            self.drawdown_manager.drawdown(), time.time())
                                        continue
                                    # Phase 3.2: CFEngine hard gating with hysteresis
                                    CF_GATE_THRESHOLD = 0.35
                                    _recent_cf = self.exec_stats.get_cf_records(20) if hasattr(self, 'exec_stats') and self.exec_stats else []
                                    if _recent_cf:
                                        _avg_cf_err = sum(r.get("cf_error", 0.0) for r in _recent_cf) / len(_recent_cf)
                                    else:
                                        _avg_cf_err = 0.0
                                    if _avg_cf_err > CF_GATE_THRESHOLD:
                                        self._cf_gate_block_counter += 1
                                    else:
                                        self._cf_gate_block_counter = max(0, self._cf_gate_block_counter - 1)
                                    if not exploration_active and self._cf_gate_block_counter >= 3:
                                        logger.info(f"[CF_GATE_BLOCK] {sym}: avg_cf_err={_avg_cf_err:.3f} block_counter={self._cf_gate_block_counter}, skipping entry")
                                        self._cf_gate_block_counter = min(self._cf_gate_block_counter, 10)
                                        self.rejection_engine.reject(sym, RejectionType.CF_BLOCK, _avg_cf_err, time.time())
                                        continue
                                    # Gate audit record for exploration mode
                                    if exploration_active:
                                        _audit_rf = self._rf_gate.prob(sym) if self._rf_gate.ready(sym) else None
                                        _would_block_by = []
                                        if _audit_rf is not None and _audit_rf < self._rf_gate.prob_thresh:
                                            _would_block_by.append("RF_GATE")
                                        if gate_result.get("gate_blocked", False):
                                            _would_block_by.append("TPI_GATE")
                                        if SETTINGS.portfolio_mode == "TOP_3_ROTATION" and sym not in self._top3_submit_set:
                                            _would_block_by.append("NOT_IN_TOP3")
                                        if net_alpha_pts <= 0:
                                            _would_block_by.append("NET_ALPHA")
                                        if exec_tier == "PASSIVE":
                                            _would_block_by.append("REALITY_VECTOR")
                                        if self._cf_gate_block_counter >= 3:
                                            _would_block_by.append("CF_GATE")
                                        logger.info(f"[EXPLORATION_AUDIT] {sym}: exploration=True rf_score={_audit_rf} cf_err={_avg_cf_err:.3f} tpi_gated={gate_result.get('gate_blocked', False)} would_block_by={_would_block_by}")
                                    _entropy_tags = {}
                                    if exploration_active:
                                        _rates_tag = self._price_history.get(broker_sym, [])
                                        if len(_rates_tag) >= 10:
                                            _c_tag = np.array([r.get("close", r.get("open", 0)) for r in _rates_tag[-10:]])
                                            _r_tag = np.diff(_c_tag) / np.maximum(np.abs(_c_tag[:-1]), 1e-9)
                                            _vol_comp = np.std(_r_tag[-5:]) / max(np.std(_r_tag), 1e-9)
                                            _drift_pers = abs(np.mean(_r_tag[-3:])) / max(np.std(_r_tag[-3:]), 1e-9)
                                            _micro_stab = np.std(_r_tag) / max(np.std(np.diff(_r_tag)), 1e-9)
                                            _eb = "LOW" if _vol_comp < 0.5 else "MED" if _vol_comp < 1.0 else "HIGH"
                                            _entropy_tags = {"entropy_bucket": _eb, "vol_compression": round(_vol_comp, 3), "drift_persistence": round(_drift_pers, 3), "micro_stability": round(_micro_stab, 3)}
                                        else:
                                            _entropy_tags = {"entropy_bucket": "UNKNOWN", "vol_compression": 0.0, "drift_persistence": 0.0, "micro_stability": 0.0}
                                        logger.info(f"[ENTROPY_TAG] {sym}: bucket={_entropy_tags['entropy_bucket']} vc={_entropy_tags['vol_compression']} dp={_entropy_tags['drift_persistence']} ms={_entropy_tags['micro_stability']}")
                                    # STR-E shadow gate — block when shadow/legacy dominates GT
                                    _stre_gate_state = self._stre_engine.compute()
                                    if _stre_gate_state["samples"] >= 10:
                                        _stas = _stre_gate_state.get("stas", 0.0)
                                        if _stas < -0.1:
                                            logger.info(f"[STR-E_GATE] {sym}: stas={_stas:.4f} samples={_stre_gate_state['samples']} — shadow winning, blocking")
                                            self.rejection_engine.reject(sym, RejectionType.STR_E_GATE, _stas, time.time())
                                            continue
                                    if not self._sdl.is_allowed(sym, ex_dir):
                                        logger.warning(f"[SDL_BLOCK] {sym}: {ex_dir} blocked by direction lock (current={self._sdl.get_current(sym)})")
                                        continue

                                    t_start = _wall_perf_counter()
                                    if exploration_active:
                                        _obs_state["observer_state"] = "EXECUTE"
                                        _obs_state["reality_score"] = max(_obs_state.get("reality_score", 0.5), 0.3)
                                        _explore_vol = self.orders.calculate_volume(sym, price, balance, SETTINGS.risk_per_trade * 0.08)
                                        logger.info(f"[ORDER EXECUTING] {sym} {ex_dir} price={price} vol={_explore_vol:.2f} tier={exec_tier} [EXPLORATION]")
                                        _exec_result = {"executed": True, "order_result": None}
                                        try:
                                            if ex_dir == "BUY":
                                                _exec_result["order_result"] = self.orders.execute_buy(sym, price, balance, SETTINGS.risk_per_trade * 0.08, sl=rhl_check.get("sl", 0.0), tp=rhl_check.get("tp", 0.0), comment="PROXIMA_EXPLORATION")
                                            else:
                                                _exec_result["order_result"] = self.orders.execute_sell(sym, price, balance, SETTINGS.risk_per_trade * 0.08, sl=rhl_check.get("sl", 0.0), tp=rhl_check.get("tp", 0.0), comment="PROXIMA_EXPLORATION")
                                        except Exception as _e:
                                            logger.error(f"[EXPLORE_ORDER_FAIL] {sym}: {_e}")
                                    else:
                                        logger.info(f"[ORDER EXECUTING] {sym} {ex_dir} price={price} vol={volume} tier={exec_tier}")
                                        _exec_result = self.execution_router.route(
                                            symbol=sym,
                                            direction=1 if ex_dir == "BUY" else -1 if ex_dir == "SELL" else 0,
                                            volume=volume,
                                            entry_price=price,
                                            sl_price=rhl_check.get("sl", 0.0),
                                            tp_price=rhl_check.get("tp", 0.0),
                                            observer_state=_obs_state["observer_state"],
                                            reality_score=_obs_state.get("reality_score", 0.5),
                                            calibration_ok=True,
                                            account_balance=balance,
                                            open_positions=positions_now)
                                    latency_ms = (_wall_perf_counter() - t_start) * 1000.0
                                    result = _exec_result.get("order_result") if _exec_result.get("executed") else None
                                    if not _exec_result.get("executed"):
                                        logger.warning(f"[ROUTER REJECT] {sym}: {_exec_result.get('rejection_reason','?')}")
                                    success = result is not None
                                    if success and hasattr(self, '_sz5_entry_tracker'):
                                        self._sz5_entry_tracker[sym] = "ENTERED"
                                    slippage_pts = 0.0
                                    dfad_pts = 0.0
                                    if success and tick:
                                        point_val = 0.01 if "JPY" in sym else (0.1 if "XAU" in sym or "XAG" in sym else 0.0001)
                                        fill_price = result.get("price", tick["ask"])
                                        signal_price = price
                                        slippage_pts = max(0.0, (fill_price - tick["ask"]) / point_val) if point_val > 0 else 0.0
                                        # E5: DFAD — execution delta ground truth
                                        dfad_pts = (fill_price - signal_price) / point_val if point_val > 0 else 0.0
                                        if ex_dir == "SELL":
                                            dfad_pts = -dfad_pts  # negative slippage on sells = worse
                                        self.exec_stats.record_dfad(sym, ex_dir, dfad_pts, _obs_state.get("reality_score", 0.5))
                                    
                                    self.exec_stats.record_order_result(
                                        success=success,
                                        latency_ms=latency_ms,
                                        slippage_pts=slippage_pts
                                    )

                                    if result:
                                        self._sdl.lock(sym, ex_dir)
                                        execution_success = True
                                        ticket = result["ticket"]
                                        self._reconcile_broker_positions()
                                        self._exit_state[ticket] = {
                                            "direction": 1 if ex_dir == "BUY" else -1,
                                            "entry_tpi": _tpi_sig_obs.get("tpi", 0.0),
                                            "inversion_count": 0,
                                            "symbol": sym,
                                        }
                                        if exploration_active:
                                            self._exit_state[ticket]["entry_mode"] = "EXPLORATION"
                                            self._exit_state[ticket].update(_entropy_tags)
                                        _entry_px_bd = (tick.get("bid", 0) + tick.get("ask", 0)) / 2.0 if tick else 0.0
                                        self._battle_decay.reset_for_ticket(
                                            ticket, sym,
                                            1 if ex_dir == "BUY" else -1,
                                            _entry_px_bd)
                                        self.audit.record_accepted(signal_id, ticket)
                                        self._dir_stats["BUY_ACCEPT" if ex_dir == "BUY" else "SELL_ACCEPT"] = self._dir_stats.get("BUY_ACCEPT", 0) + 1
                                        # B4: Record thesis at trade acceptance
                                        _ed = eval_data.get(sym, {})
                                        _tdir = 1 if ex_dir == "BUY" else -1 if ex_dir == "SELL" else 0
                                        _ttype = _ed.get("arb_reason", "UNKNOWN")
                                        _tconf = _ed.get("oss_conf", 0.5)
                                        _oss_sig = _ed.get("oss_ev_signal", 0) or prod_signals.get(sym, 0)
                                        _sh_sig = shadow_signals.get(sym, 0) if isinstance(shadow_signals, dict) else 0
                                        _exhaust_active = self._fusion._exhaustion_hist.get(sym, {}).get("exhausted", False)
                                        _topo = self._entropy_compression.topology(sym, all_symbols=list(eval_data.keys()))
                                        _topo_name = _topo.get("topology", "UNKNOWN") if _topo.get("status") == "ACTIVE" else "UNKNOWN"
                                        _rf_p = self._rf_gate.prob(sym)
                                        _thesis_rf_p = self._thesis_trainer.predict([
                                            _oss_sig, _sh_sig, int(_exhaust_active),
                                            _ed.get("ecdf_rank", 0.5), _ed.get("entropy", 0.5),
                                            _ed.get("research_p_cont", 0.5), _ed.get("research_drift", 0),
                                            _rf_p, _tconf,
                                            {"STRUCTURED_DOMINANCE": 20, "DIRECTED_TURBULENCE": 10,
                                             "DIFFUSE_CHAOS": 3, "TRANSITIONAL_COMPRESSION": 10}.get(_topo_name, 10)])
                                        _tid = self._thesis_buffer.record(
                                            symbol=sym, ticket=ticket,
                                            thesis_direction=_tdir, thesis_type=_ttype, thesis_confidence=_tconf,
                                            regime=_regime, cycle_id=self._cycle_id,
                                            oss_sig=_oss_sig, shadow_sig=_sh_sig,
                                            exhaustion_active=_exhaust_active,
                                            ecdf=_ed.get("ecdf_rank", 0.5),
                                            entropy=_ed.get("entropy", 0.5),
                                            topology=_topo_name,
                                            p_cont=_ed.get("research_p_cont", 0.5),
                                            drift=_ed.get("research_drift", 0),
                                            rf_prob=_rf_p,
                                            thesis_rf_prob=_thesis_rf_p)
                                        logger.info(f"[B4_RECORD] id={_tid} ticket={ticket} "
                                                    f"{sym} dir={_tdir:+d} type={_ttype}")
                                        _entry_px = (tick["bid"] + tick["ask"]) / 2.0 if tick else 0.0
                                        self._thesis_graph.register(_tid, sym, _entry_px)
                                        # Phase E: record trade entry in validation framework
                                        try:
                                            _dc = DecisionContext(
                                                decision_id="",
                                                timestamp=int(now.timestamp()) if hasattr(now, 'timestamp') else int(time.time()),
                                                cycle_id=self._cycle_id,
                                                symbol=sym,
                                                direction=_tdir,
                                                entry_price=_entry_px,
                                                ticket=ticket,
                                                thesis_id=str(_tid),
                                                organism_at_entry=_os,
                                                observer_recommendation="EXECUTE",
                                                observer_quality=0.5,
                                            )
                                            self._validation.on_trade_entry(_dc)
                                        except Exception as ex:
                                            logger.warning(f"[VALIDATION] on_trade_entry error: {ex}")
                                        self._funnel_audit.update(signal_id, accepted=True, opened=True)
                                        if self._env and hasattr(self._env, 'ledger') and self._env.ledger is not None:
                                            self._env.ledger.add_signal({
                                                "symbol": sym, "ts": self._now_ts(),
                                                "phase": "executed", "ticket": ticket,
                                                "signal_id": signal_id,
                                            })
                                        entry_bar_time = rates[-1]["time"]
                                        entry_px = tick["ask"]
                                        current_ev = oss_info.get("ev", 0.0) if 'oss_info' in dir() else 0.0
                                        self._position_tick_age[ticket] = 0
                                        if self._position_registry_lock:
                                            with self._position_registry_lock:
                                                self._active_positions_metadata[ticket] = {
                                                    "entry_bar_time": entry_bar_time,
                                                    "entry_warmup_ticks": self._warmup_ticks,
                                                    "entry_ev": current_ev,
                                                    "symbol": broker_sym,
                                                    "signal_id": signal_id,
                                                    "cohort_key": ck,
                                                    "q_held": self._migration.quality_score(
                                                        es_percentile, at_percentile, transition_tis,
                                                        (self._signal_decay.dps(ck) or {}).get("dps")),
                                                    "entry_es_rank": es_percentile,
                                                    "entry_at_rank": at_percentile,
                                                    "trigger_count_while_open": 0,
                                                    "max_es_rank": es_percentile,
                                                    "max_at_rank": at_percentile,
                                                    "entry_price": entry_px,
                                                    "min_price": entry_px,
                                                    "max_price": entry_px,
                                                    "entry_time": self._now_dt().strftime("%Y-%m-%d %H:%M:%S")
                                                }
                                        else:
                                            self._active_positions_metadata[ticket] = {
                                                "entry_bar_time": entry_bar_time,
                                                "entry_warmup_ticks": self._warmup_ticks,
                                                "entry_ev": current_ev,
                                                "symbol": broker_sym,
                                                "signal_id": signal_id,
                                                "cohort_key": ck,
                                                "q_held": self._migration.quality_score(
                                                    es_percentile, at_percentile, transition_tis,
                                                    (self._signal_decay.dps(ck) or {}).get("dps")),
                                                "entry_es_rank": es_percentile,
                                                "entry_at_rank": at_percentile,
                                                "trigger_count_while_open": 0,
                                                "max_es_rank": es_percentile,
                                                "max_at_rank": at_percentile,
                                                "entry_price": entry_px,
                                                "min_price": entry_px,
                                                "max_price": entry_px,
                                                "entry_time": self._now_dt().strftime("%Y-%m-%d %H:%M:%S")
                                            }
                                        self._save_active_positions_metadata()
                                        # P6: Log pyramid event
                                        if block_reason == "PYRAMID_ADDON":
                                            self._pyramid_count[sym] = self._pyramid_count.get(sym, 0) + 1
                                            self._pyramid_log.append({
                                                "time": self._now_dt().isoformat(),
                                                "symbol": sym,
                                                "ticket": ticket,
                                                "es_percentile": es_percentile,
                                                "entry_es": entry_es,
                                                "pyramid_number": self._pyramid_count[sym],
                                            })
                                            logger.info(f"P6 Pyramid EXECUTED[{sym}]: #{self._pyramid_count[sym]} add-on at ES={es_percentile:.3f}")
                                        self.trade_ledger.record(
                                            broker_sym, "LONG", es_percentile, current_es, at_percentile, current_density,
                                            "DYNAMIC", 0.90, "NORMAL",
                                            f"Quintile {int(at_percentile * 5)}", tick["ask"], 0.0, 0.0,
                                            ticket
                                        )
                                        self.audit.record_opened(signal_id, ticket)
                                        self.order_tracker.record_result(
                                            signal_id, True, retcode=10009, ticket=ticket)
                                        self._freq_pipeline.record_executed(
                                            sym, es_percentile, at_percentile,
                                            0.90, tick["ask"], ticket,
                                            energy_regime=energy_regime,
                                            time_regime=time_regime,
                                            combined_regime=combined_regime)

                                        # DRL: latency recording
                                        sig_rec = self.audit.get_signal(signal_id)
                                        if sig_rec:
                                            self._drl_latency.record(
                                                signal_id, sym,
                                                sig_rec.get("timestamp_generated"),
                                                sig_rec.get("timestamp_triggered"),
                                                sig_rec.get("timestamp_submitted"),
                                                sig_rec.get("timestamp_accepted"),
                                                sig_rec.get("timestamp_opened"))

                                        # DRL: execution quality
                                        ideal = tick["ask"] if tick else 0.0
                                        actual = result.get("price", ideal)
                                        pt = 0.01 if "JPY" in sym else 0.0001
                                        self._drl_exec_q.record(signal_id, sym, ideal, actual, pt)

                                        # DRL: signal decay (outcome-based via DelayedOutcomeEngine, not pseudo forward)
                                        future = {}
                                        if future:
                                            self._drl_decay.record(signal_id, sym,
                                                es_percentile, at_percentile, future, "EXECUTED")

                                        # DRL: spread reality at entry
                                        spread_val = mt5_status["symbols"].get(sym, {}).get("spread", 0)
                                        self._drl_spread.record(signal_id, sym, spread_val, spread_val)
                                    else:
                                        blocked = True
                                        block_reason = "REJECTED"
                                        err = self.mt5.last_error or "unknown"
                                        self.audit.record_rejected(signal_id, 0, err)
                                        self.order_tracker.record_result(
                                            signal_id, False, retcode=-1, comment=err)
                                        self._freq_pipeline.record_blocked(
                                            sym, es_percentile, at_percentile,
                                            0.90, "ORDER_REJECTED", tick["ask"] if tick else 0.0,
                                            energy_regime=energy_regime,
                                            time_regime=time_regime,
                                            combined_regime=combined_regime)

                        if triggered and blocked:
                            self._funnel_failures[block_reason] = self._funnel_failures.get(block_reason, 0) + 1
                            self.audit.record_blocked(signal_id, block_reason)
                            price = tick["ask"] if (triggered and not is_traded and 'tick' in locals() and tick is not None) else 0.0
                            self._freq_pipeline.record_blocked(
                                sym, es_percentile, at_percentile, 0.90,
                                block_reason, price,
                                energy_regime=energy_regime,
                                time_regime=time_regime,
                                combined_regime=combined_regime)
                            # DRL: signal decay for blocked signals
                            future = {}
                            if future:
                                self._drl_decay.record(signal_id, sym,
                                    es_percentile, at_percentile, future, "BLOCKED")

                        # Log Observability metrics
                        self.opp_tracker.record_evaluation(
                            symbol=sym,
                            es_value=current_es,
                            es_rank=es_percentile,
                            at_rank=at_percentile,
                            threshold=0.90,
                            triggered=triggered,
                            blocked=blocked,
                            block_reason=block_reason
                        )
                        self.trig_stats.record_evaluation(
                            symbol=sym,
                            triggered=triggered,
                            blocked=blocked,
                            block_reason=block_reason
                        )
                        self.sig_stats.record_evaluation(
                            symbol=sym,
                            es_rank=es_percentile,
                            at_rank=at_percentile
                        )

                        # Update eval data dict
                        eval_data[sym]["es_val"] = current_es
                        eval_data[sym]["es_rank"] = es_percentile
                        eval_data[sym]["at_rank"] = at_percentile

                        if is_traded:
                            eval_data[sym]["status"] = "TRADED"
                        elif triggered:
                            if blocked:
                                eval_data[sym]["status"] = f"BLOCKED ({block_reason})"
                            else:
                                eval_data[sym]["status"] = "TRIGGER"
                        else:
                            eval_data[sym]["status"] = "WATCH"

                        # ————————————————————————————————————————
                        # TPI FLOW OVERLAY (Layer 7) — SHADOW MODE
                        # Computes TPI direction, checks alignment with
                        # existing signal, and logs for dashboard.
                        # NO execution impact (deployment freeze).
                        # ————————————————————————————————————————
                        if sym in TPI_ELIGIBLE:
                            existing_dir = None
                            existing_dir_str = "NONE"
                            if triggered and not blocked:
                                existing_dir = 1
                                existing_dir_str = "LONG"
                            elif is_traded:
                                existing_dir = 1
                                existing_dir_str = "LONG_IN_POS"
                            tpi_sig = self._cycle_tpi_snapshot.get(sym)
                            if tpi_sig is None:
                                logger.warning(f"Missing cycle TPI snapshot for {sym} — skipping TPI gate")
                                tpi_sig = {"direction": 0, "confidence": 0, "tpi": 0, "alignment": "NO_DATA"}
                            if tpi_sig and existing_dir is not None:
                                alignment = None
                                ex_dir = existing_dir
                                if ex_dir == 0:
                                    alignment = "NEUTRAL"
                                elif tpi_sig.get("direction") == ex_dir:
                                    alignment = "MATCH"
                                else:
                                    alignment = "CONFLICT"
                                tpi_sig = {**tpi_sig, "alignment": alignment}
                            tpi_cache_signal(sym, tpi_sig)
                            alignment = tpi_sig.get("alignment") or "NO_SIGNAL"
                            tpi_record_shadow(sym, tpi_sig, alignment, existing_dir_str)

                            # Create Tick→Bar anchored observation
                            try:
                                bar = rates[-1] if rates else None
                                if bar:
                                    bar_time = datetime.fromtimestamp(bar["time"])
                                    obs = TPIObservation(
                                        obs_id=f"tpi_{sym}_{int(self._now_ts())}",
                                        symbol=sym,
                                        timestamp=self._now_dt(),
                                        tpi=tpi_sig.get("tpi", 0),
                                        direction=tpi_sig.get("direction_label", "FLAT"),
                                        confidence=tpi_sig.get("confidence", 0),
                                        percentile=tpi_sig.get("percentile", 0) or 0,
                                        session=tpi_sig.get("session_name", "ANY"),
                                        eligible=tpi_sig.get("eligible", False),
                                        aligned_with_signal=tpi_sig.get("direction") == existing_dir if existing_dir else None,
                                        bar_open_time=bar_time,
                                        entry_price=bar.get("close", bar.get("open", 0)),
                                    )
                                    self._tpi_tracker.register(obs)
                            except Exception:
                                pass

                            logger.debug(f"TPI[{sym}]: dir={tpi_sig.get('direction', '?')} | TPI={tpi_sig.get('tpi', '?')} | P{tpi_sig.get('percentile', 'N/A')} | Eligible={tpi_sig.get('eligible', '?')} | Alignment={alignment}")

                            # P4: TPI persistence — consecutive same-direction prints
                            tpi_dir = tpi_sig.get("direction", 0)
                            pers_state = self._tpi_persistence.update(sym, tpi_sig.get("tpi", 0), tpi_dir)
                            eval_data[sym]["tpi_persistence"] = pers_state

                            # DPL-22: Entropy compression
                            self._entropy_compression.update(sym, tpi_sig.get("tpi", 0))

                            # P5: TPI curvature — dTPI / d2TPI classification
                            curv_state = self._tpi_curvature.update(sym, tpi_sig.get("tpi", 0))
                            eval_data[sym]["tpi_curvature"] = curv_state

                            # DPL-18: Cross-asset propagation
                            self._tpi_propagation.update(sym, tpi_sig.get("tpi", 0), tpi_dir, self._now_ts())

                            # DPL-21: Meta-State Fusion (compute after all sub-signals ready)
                            decay_stats = self._tpi_tracker.live_decay_stats()
                            sym_decay = decay_stats.get("per_symbol", {}).get(sym, {})
                            prop_matrix = self._tpi_propagation.compute([sym])
                            prop_score = 0.0
                            for (src, tgt), m in prop_matrix.items():
                                if src == sym and m["propagation_score"] is not None:
                                    prop_score = m["propagation_score"]
                            thermo_state = self._tick_thermo.compute_pressure(sym)
                            meta = self._meta_fusion.compute(
                                symbol=sym,
                                tpi=tpi_sig.get("tpi", 0),
                                persistence=pers_state,
                                curvature=curv_state,
                                decay=sym_decay,
                                propagation=prop_score,
                                pressure=thermo_state,
                            )
                            self._last_meta_scores[sym] = meta

                            # DPL-20: Session-conditional recording
                            utc_hour = datetime.utcnow().hour
                            sess = get_session(utc_hour)
                            tpi_dir = tpi_sig.get("direction", 0)
                            meta_dir = meta.get("meta_direction", 0)
                            self._session_cond.record(sess, sym, "tpi", tpi_dir, 0)
                            self._session_cond.record(sess, sym, "meta", meta_dir, 0)
                            self._session_cond.record(sess, sym, "persistence",
                                                      pers_state.get("direction", 0), 0)
                            self._session_cond.record(sess, sym, "pressure",
                                                      thermo_state.get("pressure_direction", 0), 0)

                            # RCL-1A: Update ledger with full feature state
                            sig_id = eval_data[sym].get("signal_id")
                            if sig_id:
                                self._outcome_ledger.update_features(sig_id, {
                                    "tpi_value": tpi_sig.get("tpi", 0),
                                    "tpi_sign": tpi_sig.get("direction", 0),
                                    "persistence_rank": pers_state.get("persistence_rank", 0),
                                    "curvature_state": curv_state.get("state", "NEUTRAL"),
                                    "decay_ema": sym_decay.get("ema_confidence", 0),
                                    "entropy_score": self._entropy_compression.fusion_values([sym]).get(sym, 0),
                                    "propagation_score": prop_score,
                                    "pressure_score": thermo_state.get("pressure", 0),
                                    "burst_ratio": thermo_state.get("burst_ratio", 0),
                                    "asymmetry": thermo_state.get("asymmetry", 0),
                                    "tempo_compression": thermo_state.get("tempo_compression", 0),
                                    "meta_score": meta.get("meta_score", 0),
                                    "meta_rank": 1,
                                    "session": sess,
                                    "pyramid_flag": block_reason == "PYRAMID_ADDON",
                                })
                                self._outcome_ledger.update_context(sig_id, {"entry_price": eval_data[sym].get("price")})

                            # Store persistence + curvature in cached signal for dashboard
                            tpi_sig["persistence"] = pers_state
                            tpi_sig["curvature"] = curv_state

                            # Resolve pending TPI observations for this symbol
                            try:
                                bar = rates[-1] if rates else None
                                if bar:
                                    resolved = self._tpi_tracker.resolve_bar(
                                        sym,
                                        datetime.fromtimestamp(bar["time"]),
                                        bar.get("close", 0),
                                    )
                                    if resolved > 0:
                                        logger.debug(f"TPI resolve: {resolved} observations for {sym}")
                            except Exception:
                                pass
                        # ————————————————————————————————————————

                    # Pipeline audit
                    if hasattr(self, '_cycle_candidates') or '_cycle_candidates' in dir():
                        _ranked = len(getattr(self, '_cycle_candidates', []))
                    else:
                        _ranked = len(self._cycle_execution_set)
                    _submitted = self._dir_stats.get("BUY_SUBMIT", 0) + self._dir_stats.get("SELL_SUBMIT", 0)
                    logger.info(f"[PIPELINE_AUDIT] "
                        f"ranked={_ranked} "
                        f"triggered={self._dir_stats.get('BUY_TRIGGER',0)+self._dir_stats.get('SELL_TRIGGER',0)} "
                        f"buy_trigger={self._dir_stats.get('BUY_TRIGGER',0)} "
                        f"sell_trigger={self._dir_stats.get('SELL_TRIGGER',0)} "
                        f"submitted={_submitted} "
                        f"buy_submit={self._dir_stats.get('BUY_SUBMIT',0)} "
                        f"sell_submit={self._dir_stats.get('SELL_SUBMIT',0)} "
                        f"accepted={self._dir_stats.get('BUY_ACCEPT',0)+self._dir_stats.get('SELL_ACCEPT',0)} "
                        f"top3={len(getattr(self,'_top3_submit_set',set()))}")
                    logger.info(f"[DIRECTIONAL PARITY] BUY: {self._dir_stats.get('BUY_SUBMIT',0)}/{self._dir_stats.get('BUY_TRIGGER',0)}={self._dir_stats.get('BUY_SUBMIT',0)/max(1,self._dir_stats.get('BUY_TRIGGER',1)):.1%} "
                        f"SELL: {self._dir_stats.get('SELL_SUBMIT',0)}/{self._dir_stats.get('SELL_TRIGGER',0)}={self._dir_stats.get('SELL_SUBMIT',0)/max(1,self._dir_stats.get('SELL_TRIGGER',1)):.1%}")

                # Process matured delayed outcomes (true forward returns)
                matured = self._freq_pipeline.process_matured_outcomes()
                if matured > 0:
                    logger.info(f"DelayedOutcome: resolved {matured} forward return(s)")

                    # P2.3: RCL dual-horizon resolution (H5 fast, H20 thesis)
                    now_ts = self._now_ts()
                    for rec in list(self._outcome_ledger.unresolved()):
                        sig_id = rec["signal_id"]
                        sym_r = rec["symbol"]
                        age_bars = int((now_ts - rec["timestamp"]) / 3600)
                        broker_sym_r = self.mt5._get_broker_symbol(sym_r)
                        rates_r = self._price_history.get(broker_sym_r)
                        if not rates_r:
                            continue
                        current_price = rates_r[-1].get("close", rates_r[-1].get("open", 0.0))
                        outcomes = rec.get("outcomes", {})
                        # Get cohort key from features (stored at signal time)
                        ck = rec.get("features", {}).get("cohort_key")
                        if not ck:
                            ctx_r = rec.get("context", {})
                            rf = ctx_r.get("regime_from")
                            rt = ctx_r.get("regime_to")
                            tpi_dir_r = ctx_r.get("tpi_direction")
                            tpi_conf_r = ctx_r.get("tpi_confidence", 0)
                            tpi_sig_r = {"direction": tpi_dir_r, "confidence": tpi_conf_r} if tpi_dir_r is not None else None
                            ne_r = ctx_r.get("normalized_entropy")
                            sess_r = ctx_r.get("session", "UNKNOWN")
                            ck = cohort_key(rf, rt, tpi_sig_r, ne_r, sess_r)
                        # H5: fast directional signal (resolve once at 5 bars)
                        if age_bars >= 5 and "h5" not in outcomes:
                            entry_r = rec.get("context", {}).get("entry_price") or rec.get("features", {}).get("entry_price")
                            if not entry_r and current_price:
                                entry_r = rates_r[0].get("close", current_price)
                            if entry_r and entry_r != current_price:
                                outcome_h5 = self._outcome_ledger.resolve_outcome(sig_id, "h5", current_price, entry_r)
                                if outcome_h5:
                                    self._signal_decay.record_outcome(ck, "h5", outcome_h5["win"], outcome_h5["return"])
                        # H20: thesis horizon (resolve once at 20 bars)
                        if age_bars >= 20 and "h20" not in outcomes:
                            entry_r = rec.get("context", {}).get("entry_price") or rec.get("features", {}).get("entry_price")
                            if not entry_r and current_price:
                                entry_r = rates_r[0].get("close", current_price)
                            if entry_r and entry_r != current_price:
                                outcome_h20 = self._outcome_ledger.resolve_outcome(sig_id, "h20", current_price, entry_r)
                                if outcome_h20:
                                    self._signal_decay.record_outcome(ck, "h20", outcome_h20["win"], outcome_h20["return"])
                                self._outcome_ledger.mark_resolved(sig_id)

                # P1.1: Resolve shadow H20 outcomes for occupancy audit
                now = self._now_ts()
                for occ_rec in list(self._occupancy_audit.unresolved()):
                    if now - occ_rec["bar_time"] >= 20 * 3600:  # 20 H1 bars
                        broker_sym = self.mt5._get_broker_symbol(occ_rec["symbol"])
                        rates = self._price_history.get(broker_sym)
                        if rates:
                            current_price = rates[-1].get("close", rates[-1].get("open", 0.0))
                            # Determine held outcome: check if held tickets still open with positive PnL
                            held_tickets = occ_rec.get("held_ticket_ids", [])
                            held_win = None
                            if held_tickets:
                                pos_by_ticket = {p["ticket"]: p for p in open_positions}
                                held_results = []
                                for tid in held_tickets:
                                    p = pos_by_ticket.get(tid)
                                    if p:
                                        held_results.append(p["profit"] > 0)
                                if held_results:
                                    held_win = any(held_results)
                            self._occupancy_audit.try_resolve(
                                occ_rec["record_id"], current_price,
                                held_outcome_win=held_win,
                            )

                # Compute global ranks for next cycle
                # Lifecycle manager tick — updates bar counts, detects stuck positions
                if hasattr(self, '_lifecycle_mgr'):
                    self._lifecycle_mgr.tick()
                self._global_rank_engine.compute()
                logger.debug(f"V2.2 Cycle GlobalRanks:\n{self._global_rank_engine.summary()}")

                # Update performance metrics
                self.positions.refresh()
                open_positions = self.positions.positions

                # RHL: equity protection + MT5 health checks
                ai = self.mt5.get_account() or {}
                eq = ai.get("equity", 0.0)
                if eq > 0 and (self._risk.governor._peak_equity == 0 or self._risk.governor._start_equity <= 0):
                    self._risk.governor.set_start_equity(eq)
                if eq > 0:
                    unrealized = sum(p.get("profit", 0) for p in open_positions)
                    self._risk.governor.update_unrealized(unrealized, eq)
                    dd = self._risk.governor.check_equity_drawdown(eq)
                    if dd.get("triggered"):
                        logger.warning(f"EQUITY PROTECTION: closing all at {dd['drawdown_pct']:.1%} drawdown")
                        for tick_ep in open_positions:
                            m_ep = self._active_positions_metadata.get(tick_ep["ticket"])
                            if m_ep:
                                m_ep["expected_exit_reason"] = "EQUITY_PROTECTION"
                        self._save_active_positions_metadata()
                        self._sdl.reset()
                        self.orders.close_all()

                mt5_status = self.mt5_monitor.check()
                sym_avail = {s: True for s in self._execution_symbols}
                rhl_health = self._risk.health_check(
                    mt5_status.get("connected", False),
                    mt5_status.get("connected", False), sym_avail)
                if rhl_health.get("entries_disabled"):
                    logger.warning(f"RHL health block: {rhl_health.get('issues', [])}")

                for pos in open_positions:
                    # Track excursion (min/max price) — NOT PnL, that's recorded on close only
                    pt_meta = self._active_positions_metadata.get(pos["ticket"])
                    if pt_meta:
                        cp = pos["price_current"]
                        pt_meta["min_price"] = min(pt_meta.get("min_price", cp) or cp, cp)
                        pt_meta["max_price"] = max(pt_meta.get("max_price", cp) or cp, cp)
                        pt_meta["current_price"] = cp

                # Handle daily report snapshot logic in the background
                if last_daily_snapshot != today_str:
                    last_daily_snapshot = today_str
                    perf = self.perf.summary()
                    ds = self.score.compute(
                        perf["sharpe"], perf["pp"], perf["max_dd"],
                        0.3, perf["n_trades"], self.signal_ledger.total_signals,
                        effective_n=self._outcome_ledger.resolved_count())
                    account_info = self.mt5.get_account()
                    self.deployment_ledger.record(
                        ds, perf["pp"], perf["sharpe"], perf["max_dd"],
                        self.signal_ledger.total_signals, perf["n_trades"],
                        0.3, len(open_positions),
                        account_info["balance"] if account_info else 0.0,
                        account_info["equity"] if account_info else 0.0)
                    report = self._daily_report.generate()
                    logger.info(f"\n{report}")
                    if int(now) - last_hourly_check > 3600:
                        self.telegram.send_sync(report)
                        last_hourly_check = int(now)

                # Update reality dashboard
                if hasattr(self, '_reality_dashboard'):
                    # Wire lifecycle manager if available
                    if hasattr(self, '_lifecycle_mgr') and self._reality_dashboard._lifecycle_mgr is None:
                        self._reality_dashboard.register_lifecycle_manager(self._lifecycle_mgr)
                    # Wire gate audit if available
                    if hasattr(self, '_gate_audit_logger') and self._reality_dashboard._gate_audit is None:
                        self._reality_dashboard.register_gate_audit(self._gate_audit_logger)
                    # Compute MAI
                    if hasattr(self, '_micro_calibrator'):
                        self._micro_calibrator.compute_mai()
                    # Update all metrics
                    dashboard_snapshot = self._reality_dashboard.update()

                # Render terminal dashboard (skip full output in acceptance mode for speed)
                account_info = self.mt5.get_account() or account
                ds_info = self.score.summary()
                seconds_since_eval = int(self._now_ts() - last_signal_check)
                seconds_to_next_eval = max(0, 60 - seconds_since_eval)
                # Shadow evaluation — runs every cycle (including acceptance/fast mode)
                _shadow_cycle_report = self._shadow_orchestrator.process_cycle(self._warmup_ticks, self._execution_symbols)
                self._shadow_orchestrator.clear()

                _avg_similarity_val = 1.0
                _max_suppression_val = 0.0
                if _shadow_cycle_report:
                    _all_sims = [s_data["lkg_similarity_score"] for s_data in _shadow_cycle_report.get("symbols", {}).values()]
                    _avg_similarity_val = sum(_all_sims) / len(_all_sims) if _all_sims else 1.0
                    _max_suppression_val = max([s_data["suppression_delta"] for s_data in _shadow_cycle_report["symbols"].values()]) if _shadow_cycle_report["symbols"] else 0.0

                # STR-E + SOF
                try:
                    _gt_latest = self._shadow_gt_worker.get_latest()
                    _gt_sim = _gt_latest.get("state", {}).get("research_p_cont", 0.5) if _gt_latest else 0.5
                    _sy_sim = _avg_similarity_val
                    _pnl_proxy = sum(p.get("profit", 0) for p in (positions_now or [])) if 'positions_now' in dir() and positions_now else 0.0
                    _stre_result = self._stre_coordinator.step(_gt_sim, _sy_sim, _pnl_proxy)
                    self._last_stre_result = _stre_result
                    try:
                        _gt_latest_state = _gt_latest.get("state", {}) if _gt_latest else {}
                        _sy_signal = {"expected_move": 0, "p_cont": 0.5}
                        _sof_result = self._sof_evaluate(_gt_latest_state, _sy_signal, _pnl_proxy, _stre_result)
                        _stre_result["SOF"] = _sof_result["SOF"]
                        _stre_result["execution_efficiency"] = _sof_result["execution_efficiency"]
                        _stre_result["edge_preservation"] = _sof_result["edge_preservation"]
                    except Exception:
                        pass
                except Exception:
                    self._last_stre_result = None

                # Phase 2 GT suppression tracking
                try:
                    if hasattr(self, '_gt_suppression') and 'eval_data' in dir() and eval_data:
                        for _sym in self._execution_symbols:
                            self._gt_suppression.ingest("L0_GT", _sym, eval_data.get(_sym, {}))
                except Exception:
                    pass

                if not ACCEPTANCE_MODE:
                    reconcile_result = self.reconciler.reconcile()
                    if not reconcile_result["healthy"]:
                        for m in reconcile_result["mismatches"]:
                            logger.warning(f"Sync: {m['issue']} for ticket {m['ticket']} ({m['symbol']})")

                    perf_s = self.perf.summary() if hasattr(self.perf, 'summary') else {}
                    rce_rpt = self._rce_pipeline.summary() if hasattr(self._rce_pipeline, 'summary') else {}
                    try:
                        self._ard.collector.collect_freq_reality(
                            self._freq_blocked.blocked_count() if hasattr(self._freq_blocked, 'blocked_count') else 0,
                            len(self._freq_executed._executed) if hasattr(self._freq_executed, '_executed') else 0,
                            self._freq_analysis.leakage_rate() if hasattr(self._freq_analysis, 'leakage_rate') else 0,
                            self._freq_analysis.alpha_destruction_ratio() if hasattr(self._freq_analysis, 'alpha_destruction_ratio') else 0)
                        self._ard.collector.collect_live(
                            perf_s.get('n_trades', 0), perf_s.get('sharpe', 0.0), perf_s.get('pp', 0.5),
                            perf_s.get('today_pnl', 0.0), ds_info.get('current_score', 0.0), ds_info.get('classification', 'NO_DATA'))
                        self._ard.collector.collect_rce(
                            rce_rpt.get('ate', 0), rce_rpt.get('health_index', 0),
                            rce_rpt.get('divergence_score', 0), rce_rpt.get('friction_index', 0),
                            rce_rpt.get('classification', 'N/A'))
                    except Exception as e:
                        self._exception_dashboard.record(e, "ard_collect")
                        logger.warning(f"ARD evidence collect: {e}")

                    perf_summary = self.perf.summary()
                    average_hold_bars = perf_summary.get('avg_hold_bars', 0) if hasattr(self.perf, 'summary') else 0
                    top3_for_dash = getattr(self, '_last_top3_qualified', [])
                    # Log gate audit and spread model summary periodically
                    if self._warmup_ticks % 60 == 0:
                        gate_summary = self._gate_audit_logger.summary()
                        logger.info(f"[GATE_AUDIT] Summary: {gate_summary.get('total_evaluated', 0)} evaluated, "
                                    f"{gate_summary.get('acceptance_rate', 0)}% acceptance")
                        spread_summary = self._spread_model.summary()
                        logger.info(f"[SPREAD_MODEL] {spread_summary.get('total_evaluations', 0)} evaluations, "
                                    f"{spread_summary.get('hard_rejection_rate', 0)}% hard rejection")

                    self._print_dashboard(
                        eval_data, open_positions, account_info, ds_info, seconds_to_next_eval,
                        rotation_events=self._rotation_event_count,
                        lock_events=self._lock_event_count,
                        migration_events=self._migration_event_count,
                        avg_hold_bars=average_hold_bars,
                        top3_ranked=top3_for_dash)
                else:
                    _stre = self._last_stre_result or {}
                    _samples = _stre.get("samples", 0)
                    _sof = _stre.get("SOF", 0)
                    _gc = _stre.get("gt_corr", 0)
                    _sc = _stre.get("sy_corr", 0)
                    _p2 = "P2" if _stre.get("phase2_blocked") is False else "p2"
                    mai_str = ""
                    if hasattr(self, '_reality_dashboard'):
                        mai_val = self._reality_dashboard.snapshot().get('overall', {}).get('value', 0)
                        mai_str = f" integrity={mai_val:.3f}"
                    print(f"[CYCLE {self._warmup_ticks}] "
                          f"open_pos={len(open_positions)} "
                          f"balance=${account_info.get('balance',0):.2f}"
                          f" STR-E:n={_samples} gt={_gc} sy={_sc} SOF={_sof} {_p2}"
                          f"{mai_str}", flush=True)

                # Lifecycle summary (periodic)
                if hasattr(self, '_lifecycle_mgr') and self._warmup_ticks % 60 == 0:
                    lc = self._lifecycle_mgr.summary()
                    logger.info(f"[LIFECYCLE] generated={lc['total_generated']} opened={lc['total_opened']} "
                                f"closed={lc['total_closed']} h20_pending={lc['h20_pending_close']}")

                # P4.1: Save regime snapshot for transition edge dashboard
                self._regime_snapshot = dict(self._regime_memory._prev_regime)

                # Persist shadow state for dashboard visibility
                _shadow_payload = {
                    "cycle_id": self._warmup_ticks,
                    "timestamp": time.time(),
                    "avg_lkg_similarity": _avg_similarity if '_avg_similarity' in dir() else 1.0,
                    "max_suppression": _max_suppression if '_max_suppression' in dir() else 0.0,
                    "suppression_graph": _shadow_cycle_report.get("suppression_graph", {}),
                    "stre": _stre_result if isinstance(_stre_result, dict) else {},
                    "symbols": {}
                }
                for _sym, _sd in _shadow_cycle_report.get("symbols", {}).items():
                    _shadow_payload["symbols"][_sym] = {
                        "suppression_delta": _sd.get("suppression_delta", 0),
                        "lkg_similarity_score": _sd.get("lkg_similarity_score", 0),
                        "raw_conviction": _sd.get("raw_signal_state", {}).get("conviction", 0.5),
                        "final_conviction": _sd.get("post_governance_state", {}).get("conviction", 0.5),
                    }
                try:
                    self._atomic_write_json(os.path.join("state", "shadow_state.json"), _shadow_payload)
                except Exception as _spe:
                    logger.warning(f"[SHADOW] Failed to persist shadow state: {_spe}")

                # Clean up per-cycle state
                if hasattr(self, '_sz5_entry_tracker'):
                    del self._sz5_entry_tracker
                self._shadow_mirror.clear_cycle()

                # Track metrics for three-tier telemetry
                _all_sims = [s_data["lkg_similarity_score"] for s_data in _shadow_cycle_report["symbols"].values()]
                _avg_similarity = sum(_all_sims) / len(_all_sims) if _all_sims else 1.0
                _max_suppression = max([s_data["suppression_delta"] for s_data in _shadow_cycle_report["symbols"].values()]) if _shadow_cycle_report["symbols"] else 0.0

                # Initialize tracking attributes if not present
                if not hasattr(self, '_prev_avg_similarity'):
                    self._prev_avg_similarity = _avg_similarity
                if not hasattr(self, '_prev_max_suppression'):
                    self._prev_max_suppression = _max_suppression

                # 1. MICRO-tier: Minimal cycle fingerprint log
                logger.info(
                    f"[SHADOW_MICRO] Cycle={self._warmup_ticks} "
                    f"symbols={len(self._execution_symbols)} "
                    f"lkg_sim={_avg_similarity:.4f} "
                    f"max_supp={_max_suppression:.4f}"
                )

                # 2. EVENT-tier: Triggered deep dump
                _sim_drop = (_self_prev_sim := getattr(self, '_prev_avg_similarity', 1.0)) - _avg_similarity > 0.05
                _supp_spike = _max_suppression - getattr(self, '_prev_max_suppression', 0.0) > 0.10
                _lkg_low = _avg_similarity < 0.90
                
                if _sim_drop or _supp_spike or _lkg_low:
                    logger.warning(
                        f"[SHADOW_EVENT] Trigger active: sim_drop={_sim_drop} supp_spike={_supp_spike} lkg_low={_lkg_low} "
                        f"Avg Sim: {getattr(self, '_prev_avg_similarity', 1.0):.4f} -> {_avg_similarity:.4f} "
                        f"Max Supp: {getattr(self, '_prev_max_suppression', 0.0):.4f} -> {_max_suppression:.4f}"
                    )
                    # Dump full suppression details for each symbol
                    for _s, _s_data in _shadow_cycle_report["symbols"].items():
                        logger.warning(
                            f"  Symbol={_s} raw_conv={_s_data['raw_signal_state'].get('conviction', 0.5):.4f} "
                            f"final_conv={_s_data['post_governance_state'].get('conviction', 0.5):.4f} "
                            f"delta={_s_data['suppression_delta']:.4f} similarity={_s_data['lkg_similarity_score']:.4f}"
                        )

                # Update tracking states
                self._prev_avg_similarity = _avg_similarity
                self._prev_max_suppression = _max_suppression

                # 3. MACRO-tier: Periodic heartbeat (every 60 ticks)
                if self._warmup_ticks % 60 == 0:
                    _sz7_summary = self._shadow_mirror.summary()
                    logger.info(f"[SHADOW_MIRROR] total={_sz7_summary['total_evaluations']} "
                                f"agree={_sz7_summary['agreements']} diverge={_sz7_summary['divergences']} "
                                f"rate={_sz7_summary['agreement_rate']} "
                                f"top_block={_sz7_summary['dominant_blocking_gate'][0]}"
                                f"({_sz7_summary['dominant_blocking_gate'][1]})")
                    
                    _g_data = _shadow_cycle_report["suppression_graph"]
                    logger.info(f"[SHADOW_ORCHESTRATOR_MACRO] Cycle={self._warmup_ticks} Suppression Cascade:")
                    for edge in _g_data["edges"]:
                        logger.info(f"  {edge['source']} -> {edge['target']}: {edge['suppression_magnitude']:.4f} average conviction loss")
                    logger.info(f"[SHADOW_ORCHESTRATOR_MACRO] Average LKG Similarity Score: {_avg_similarity:.4f}")

                # Runtime and tick-limit check inside the try block
                if self._runtime_limit > 0 and (_wall_perf_counter() - _wall_start) >= self._runtime_limit:
                    logger.info(f"Runtime limit ({self._runtime_limit}s) reached — shutting down")
                    RUNNING = False
                    break
                if self._tick_limit > 0 and self._replay_feed and self._replay_feed.cursor >= self._tick_limit:
                    logger.info(f"Tick limit ({self._tick_limit}) reached — shutting down")
                    RUNNING = False
                    break

                if not self._replay_mode:
                    time.sleep(check_interval)

                # Consistency validation: MT5 vs internal state
                if hasattr(self, '_position_sync') and hasattr(self, '_lifecycle_mgr'):
                    active_positions = len([p for p in (open_positions or []) if p.get("ticket")])
                    lifecycle_open = self._lifecycle_mgr.open_position_count()
                    if active_positions != lifecycle_open and abs(active_positions - lifecycle_open) > 2:
                        logger.warning(f"[CONSISTENCY] MT5 positions={active_positions} vs lifecycle={lifecycle_open} — drift detected")

            except KeyboardInterrupt:
                break
            except Exception as e:
                self._exception_dashboard.record(e, "main_loop")
                logger.error(f"Main loop error: {e}", exc_info=True)
                raise

        if self._wfv_records:
            wfv_results = self._wfv.run(self._wfv_records)
            print("[WFV RESULTS]")
            for k, v in wfv_results.items():
                if k == "windows_detail":
                    continue
                print(f"  {k}: {v}")
            edge = StatisticalEdgeTest.run(wfv_results)
            print(f"[EDGE SUMMARY] Accuracy={edge['accuracy']} PnL={edge['pnl_proxy']} Verdict={edge['verdict']}")
            svr_fitness = self._svr.compute_fitness(
                doa_results={r["sym"]: r["outcome"] for r in self._wfv_records[-10:]},
                drift_scores={r["sym"]: r["drift"] for r in self._wfv_records[-10:]},
                lct_score=self._lct.convergence_score() if hasattr(self, '_lct') else 0.0,
                stability_score=0.5,
            )
            print(f"[SVR FITNESS] {svr_fitness:.4f}")
            redundancy = self._svr.redundancy_map()
            print("[SVR REDUNDANCY MAP]")
            for k, v in redundancy.items():
                print(f"  {k}: {v}")
            contributions = self._edge_trace.trace(self._wfv_records)
            print("[EDGE TRACE CONTRIBUTIONS]")
            for k, v in sorted(contributions.items(), key=lambda x: -abs(x[1])):
                print(f"  {k}: {v:+.6f}")
            keep_map = self._edge_trace.core_extraction(contributions)
            keepers = [k for k, v in keep_map.items() if v == "KEEP"]
            print(f"[CORE SYSTEM] {len(keepers)} modules retained: {keepers}")
            for s in self._edge_trace.reduction_suggestions(keep_map):
                print(f"  SUGGESTION: {s}")
        else:
            print("[WFV SKIPPED] No validation records collected")

        # Phase 3 — FTMO firm-risk survivability: does this session survive a
        # real firm challenge (daily loss / max drawdown / max lot / min days)?
        # Replay-only additive summary — never gates live execution.
        try:
            from proxima_ops.risk.firm_risk import FirmRiskEvaluator, FirmRiskConfig
            from datetime import date as _date
            completed = None
            if hasattr(self, 'trade_ledger'):
                completed = self.trade_ledger.get_completed()
            if completed:
                start_balance = float(self._risk_dashboard_balance() if hasattr(self, '_risk_dashboard_balance') else 100000.0)
                cfg = FirmRiskConfig(initial_balance=start_balance)
                evaluator = FirmRiskEvaluator(cfg)
                snapshots = []
                running = start_balance
                for t in completed:
                    pm = float(t.get("profit_money") or 0.0)
                    running += pm
                    ts = t.get("entry_time") or t.get("timestamp") or 0
                    try:
                        day = _date.fromtimestamp(int(ts))
                    except Exception:
                        day = _date.today()
                    snapshots.append((day, running, float(t.get("volume") or 0.0)))
                verdict = evaluator.evaluate(snapshots)
                print(f"[FIRM RISK] {'SURVIVED' if verdict.survived else 'FAILED'} — {verdict.reason or 'all FTMO rules met'}")
                print(f"  final_equity={verdict.final_equity:.2f} peak={verdict.peak_equity:.2f} "
                      f"days={verdict.trading_days} max_dd={verdict.max_drawdown_pct_reached:.2%} "
                      f"max_daily_loss={verdict.max_daily_loss_pct_reached:.2%} "
                      f"max_lot={verdict.max_lot_seen} target_hit={verdict.target_hit}")
            else:
                print("[FIRM RISK SKIPPED] No completed trades in ledger")
        except ImportError:
            pass  # firm_risk module unavailable — non-blocking

        self.shutdown()

    def run_monitor(self):
        logger.info("Starting monitoring-only mode...")
        self.mt5.connect()
        while RUNNING:
            status = self.mt5_monitor.check()
            logger.info(f"MT5: {status['connected']}, "
                        f"Positions: {len(self.positions.positions)}")
            time.sleep(60)
        self.mt5.disconnect()

    def run_report(self):
        print(self._daily_report.generate())

    def shutdown(self):
        logger.info("Shutting down Proxima Ops...")
        # Stop ground-truth shadow observer
        try:
            if hasattr(self, '_shadow_gt_worker'):
                self._shadow_gt_worker.stop()
                logger.info("[SHADOW_GT] Observer stopped")
        except Exception as _sgt_s:
            logger.warning(f"[SHADOW_GT] Stop error: {_sgt_s}")
        # Finalize parity ledger on shutdown
        if self._env and hasattr(self._env, 'ledger') and self._env.ledger is not None:
            try:
                positions = self.positions.positions if hasattr(self.positions, 'positions') else []
                account = self.mt5.get_account() or {}
                self._env.ledger.finalize({
                    "positions": sorted(positions, key=lambda p: p.get("ticket", 0)),
                    "balance": account.get("balance", 0),
                    "equity": account.get("equity", 0),
                })
                self._env.ledger.save("replay_shutdown")
            except Exception as e:
                logger.warning(f"Parity ledger finalize failed: {e}")
        self.mt5.disconnect()
        # Save STR-E state for follow.py
        try:
            if hasattr(self, '_stre_engine') and self._stre_engine is not None:
                self._stre_engine.save()
                logger.info("[STR-E] State saved")
        except Exception as _se:
            logger.warning(f"[STR-E] Save error: {_se}")
        logger.info("Shutdown complete.")


def _repair_duckdb():
    db = SETTINGS.db_path
    try:
        import duckdb
        conn = duckdb.connect(db)
        conn.execute("SELECT 1")
        conn.close()
    except Exception:
        logger.warning(f"DuckDB corrupted, resetting database files")
        for ext in ["", ".wal"]:
            p = db + ext
            if os.path.exists(p):
                os.unlink(p)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Proxima Ops — MT5 Demo Deployment Platform")
    parser.add_argument("--mode", choices=["live", "paper", "replay", "demo", "monitor", "report"],
                        default="demo", help="Execution mode")
    parser.add_argument("--replay-start", default="2026-03-12", help="Replay start date YYYY-MM-DD (need >=55d for 550 H1 bars)")
    parser.add_argument("--replay-end", default="2026-05-12", help="Replay end date YYYY-MM-DD")
    parser.add_argument("--speed", type=float, default=1000.0, help="Replay speed multiplier")
    parser.add_argument("--burst", action="store_true", help="Burst mode (no sleeping)")
    parser.add_argument("--no-latency", action="store_true", help="Disable latency simulation")
    parser.add_argument("--no-slippage", action="store_true", help="Disable slippage simulation")
    parser.add_argument("--symbols", default="", help="Comma-separated symbols for replay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for replay")
    parser.add_argument("--tsv", action="store_true", help="Enable temporal shuffle validation")
    parser.add_argument("--runtime", type=int, default=0,
                        help="Auto-stop after N seconds of wall time (0 = run until Ctrl+C)")
    parser.add_argument("--replay-until-ticks", type=int, default=0,
                        help="Stop replay after N merged ticks (deterministic cutoff)")
    parser.add_argument("--acceptance", action="store_true",
                        help="Acceptance test mode: minimal telemetry, auto-stop via PROXIMA_MAX_CYCLES")
    parser.add_argument("--stop-on-first-thesis", action="store_true",
                        help="B4.1b: shutdown after first thesis resolution")
    parser.add_argument("--paper-exec", action="store_true",
                        help="Paper execution mode: bypass MOF gating in replay for virtual trades")
    parser.add_argument("--trader-view", action="store_true",
                        help="Simplified dashboard view: PnL, positions, signals, risk only")

    args, remaining = parser.parse_known_args()
    has_mode_flag = any(a.startswith("--mode") for a in sys.argv[1:])
    if has_mode_flag:
        mode = args.mode
    else:
        positional = [a for a in sys.argv[1:] if not a.startswith("-")]
        mode = positional[0] if positional else "demo"

    # Configure SYSTEM_MODE from CLI args
    if mode == "live":
        SYSTEM_MODE.plane = Plane.EXECUTION
        SYSTEM_MODE.mof_policy = MOFPolicy.STRICT
    elif mode == "replay":
        SYSTEM_MODE.plane = Plane.SIMULATION
        SYSTEM_MODE.mof_policy = MOFPolicy.STRICT
    elif mode == "paper":
        SYSTEM_MODE.plane = Plane.EXECUTION
        SYSTEM_MODE.execution = ExecutionMode.PAPER
    else:
        SYSTEM_MODE.plane = Plane.SIMULATION
        SYSTEM_MODE.mof_policy = MOFPolicy.STRICT
    if args.paper_exec:
        SYSTEM_MODE.execution = ExecutionMode.PAPER
    if args.trader_view:
        SYSTEM_MODE.ui = UIMode.TRADER_VIEW
    if args.acceptance:
        SYSTEM_MODE.ui = UIMode.ACCEPTANCE
    # MOF policy: SIMULATION plane + PAPER execution = RELAXED_SIM
    if SYSTEM_MODE.plane == Plane.SIMULATION and SYSTEM_MODE.execution == ExecutionMode.PAPER:
        SYSTEM_MODE.mof_policy = MOFPolicy.RELAXED_SIM
    # Validate and correct mode
    violations = ModeValidator.validate(SYSTEM_MODE)
    if violations:
        logger.warning(f"[MAIN] SYSTEM_MODE violations: {violations}")
        _corrected = ModeValidator.correct(SYSTEM_MODE)
        for _attr in ["plane", "execution", "ui", "mof_policy"]:
            setattr(SYSTEM_MODE, _attr, getattr(_corrected, _attr))
        logger.info(f"[MAIN] Corrected SYSTEM_MODE: {SYSTEM_MODE}")

    env = None
    if mode == "replay":
        from replay.environment import build_replay_environment, ReplayConfig
        from replay.clock_patcher import patch_clock
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else \
            ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD", "EURUSD"]
        # Override SETTINGS.symbols for replay
        SETTINGS.symbols = symbols
        config = ReplayConfig(
            symbols=symbols,
            start=args.replay_start,
            end=args.replay_end,
            speed=args.speed,
            burst=args.burst,
            latency=not args.no_latency,
            slippage=not args.no_slippage,
            seed=args.seed,
            tsv_enabled=args.tsv,
        )
        env = build_replay_environment(config)
        patch_clock(env.clock)
        # Isolate database path so replay doesn't conflict with live demo's duckdb lock
        SETTINGS.db_path = os.path.join(
            os.path.dirname(SETTINGS.db_path), "replay_proxima_ops.duckdb")
        logger.info(f"Replay mode: {config.start} -> {config.end} speed={config.speed}x "
                     f"burst={config.burst} symbols={config.symbols}")

    _repair_duckdb()
    tick_source = getattr(env, 'tick_source', None) if env else None
    broker = getattr(env, 'broker', None) if env else None
    # Phase 5 ship overlay: live/demo/paper run through the SAME canonical
    # tick contract as replay (LiveTickSource wraps MT5Connector) so shipped
    # live consumes byte-identical ticks to the validated backtest.
    if tick_source is None and mode in ("demo", "live", "paper"):
        from data.live_tick_source import LiveTickSource
        tick_source = LiveTickSource(MT5Connector())
    demo = ProximaDemo(env=env, tick_source=tick_source, broker=broker, replay_mode=(mode == "replay"))
    demo._runtime_limit = args.runtime
    demo._tick_limit = args.replay_until_ticks
    demo._stop_on_first_thesis = args.stop_on_first_thesis

    if mode == "replay":
        demo.warmup(5000)
        demo.run_demo()
    elif mode == "demo" or mode == "live":
        demo.run_demo()
    elif mode == "paper":
        demo.run_demo()
    elif mode == "monitor":
        demo.run_monitor()
    elif mode == "report":
        demo.run_report()
    else:
        print(f"Unknown mode: {mode}")
        print("Available: demo, monitor, report, replay")


if __name__ == "__main__":
    main()
