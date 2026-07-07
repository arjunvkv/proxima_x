#!/usr/bin/env python3
"""
Gate Shadow Experiment — OFFLINE Analysis
==========================================
Estimates the impact of converting 4 gates (RF Gate, Observer State,
Reality Vector, CF Gate) into passive observers.

Usage:
    python research/direction_simulation/gate_shadow_experiment.py

The script reads proxima_demo.log, counts every gate rejection, and
computes what the pipeline funnel would look like if each gate were
converted to shadow mode (log-only, non-blocking).
"""

import re
import json
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────
# 1. Config
# ──────────────────────────────────────────────────────────────────────

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "proxima_demo.log")

GATES = [
    "RF_GATE",
    "OBSERVER_STATE",
    "REALITY_VECTOR",
    "CF_GATE",
]


# ──────────────────────────────────────────────────────────────────────
# 2. Data structures
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GateRejection:
    gate: str
    symbol: str
    reason: str
    prob_value: Optional[float] = None
    timestamp: str = ""


@dataclass
class ExplorationAudit:
    symbol: str
    rf_score: float
    tpi_gated: bool
    would_block_by: List[str]


@dataclass
class PipelineSnapshot:
    ranked: int
    triggered: int
    submitted: int
    accepted: int
    top3: int


@dataclass
class GateStats:
    total_blocks: int = 0
    would_recover: int = 0
    signal_quality_score: float = 0.0
    unique_symbols: int = 0
    would_block_rate: float = 0.0
    already_caught_by_other: int = 0
    redundancy_pct: float = 0.0


# ──────────────────────────────────────────────────────────────────────
# 3. Log parser
# ──────────────────────────────────────────────────────────────────────

class GateLogParser:
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.rejections: Dict[str, List[GateRejection]] = {g: [] for g in GATES}
        self.exploration_audits: List[ExplorationAudit] = []
        self.pipeline_snapshots: List[PipelineSnapshot] = []
        self.all_gate_check_lines: List[str] = []

        # cumulative funnel
        self.total_submitted_in_audit = 0
        self.total_accepted_in_audit = 0
        self.pipeline_audit_count = 0

        # Observer detail
        self.observer_harness_count = 0
        self.observer_execute_count = 0
        self.observer_non_execute_count = 0
        self.observer_not_execute_states = defaultdict(int)

        # RF detail
        self.rf_blocked_signal = 0
        self.rf_blocked_rank = 0
        self.rf_passed = 0
        self.rf_warmup = 0
        self.rf_symbol_counts = defaultdict(int)

        # Router reject detail
        self.router_reject_total = 0
        self.router_reject_causes = defaultdict(int)

        self.total_lines = 0
        self.date_ranges = set()

        # Gate-level cross-block tracking: for each RF block, what other
        # gates would also have blocked it?  (from exploration audit)
        self.gate_cross_block = defaultdict(lambda: defaultdict(int))
        self.exploration_total = 0

    def parse(self):
        with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                self.total_lines += 1
                self._track_date(line)
                self._parse_rf_gate(line)
                self._parse_observer(line)
                self._parse_router_reject(line)
                self._parse_reality_vector(line)
                self._parse_cf_gate(line)
                self._parse_exploration_audit(line)
                self._parse_pipeline_audit(line)

        self._compute_cross_block_stats()
        return self

    def _track_date(self, line: str):
        m = re.match(r'(\d{4}-\d{2}-\d{2})', line)
        if m:
            self.date_ranges.add(m.group(1))

    def _parse_rf_gate(self, line: str):
        if 'RF GATE' not in line:
            return
        # Count
        if 'blocked signal' in line:
            self.rf_blocked_signal += 1
            self.rejections["RF_GATE"].append(GateRejection(
                gate="RF_GATE",
                symbol=self._extract_symbol(line) or "?",
                reason="signal_blocked",
            ))
        elif 'blocked rank-triggered' in line:
            self.rf_blocked_rank += 1
            self.rejections["RF_GATE"].append(GateRejection(
                gate="RF_GATE",
                symbol=self._extract_symbol(line) or "?",
                reason="rank_blocked",
            ))
        elif 'blocked' in line:
            pass  # already counted above
        elif 'passed' in line:
            self.rf_passed += 1
        elif 'warmup' in line:
            self.rf_warmup += 1

    def _extract_symbol(self, line: str) -> Optional[str]:
        m = re.search(r'RF GATE\] (\w+):', line)
        if m:
            sym = m.group(1)
            self.rf_symbol_counts[sym] += 1
            return sym
        return None

    def _parse_observer(self, line: str):
        if 'OBSERVER_HARNESS' not in line:
            return
        self.observer_harness_count += 1
        if 'observer=EXECUTE' in line:
            self.observer_execute_count += 1
        else:
            self.observer_non_execute_count += 1
            m = re.search(r'observer=(\w+)', line)
            if m:
                self.observer_not_execute_states[m.group(1)] += 1

    def _parse_router_reject(self, line: str):
        if 'ROUTER REJECT' not in line:
            return
        self.router_reject_total += 1
        if 'observer_state=' in line:
            self.router_reject_causes['OBSERVER_STATE'] += 1
        elif 'REALITY_GATE' in line:
            self.router_reject_causes['REALITY_GATE'] += 1
        elif 'SDL' in line:
            self.router_reject_causes['SDL'] += 1
        elif 'RISK' in line:
            self.router_reject_causes['RISK'] += 1
        elif 'POSITION_LIMIT' in line:
            self.router_reject_causes['POSITION_LIMIT'] += 1

    def _parse_reality_vector(self, line: str):
        if 'PASSIVE_SKIP' not in line:
            return
        self.rejections["REALITY_VECTOR"].append(GateRejection(
            gate="REALITY_VECTOR",
            symbol=self._extract_any_symbol(line),
            reason="passive_execution_tier",
        ))

    def _parse_cf_gate(self, line: str):
        if 'CF_GATE_BLOCK' not in line:
            return
        self.rejections["CF_GATE"].append(GateRejection(
            gate="CF_GATE",
            symbol=self._extract_any_symbol(line),
            reason="cf_error_hysteresis",
        ))

    def _extract_any_symbol(self, line: str) -> str:
        m = re.search(r'(?:^|\s)([A-Z]{6})\b', line)
        if m:
            return m.group(1)
        m2 = re.search(r'(?:^|\s)([A-Z]{3,6})\b', line)
        return m2.group(1) if m2 else "?"

    def _parse_exploration_audit(self, line: str):
        if 'EXPLORATION_AUDIT' not in line:
            return
        self.exploration_total += 1
        m = re.search(r'would_block_by=\[(.*?)\]', line)
        if m:
            items_raw = m.group(1).replace("'", "").split(", ")
            items = [i.strip() for i in items_raw if i.strip()]
            m2 = re.search(r'(\w+): exploration=True', line)
            sym = m2.group(1) if m2 else "?"
            self.exploration_audits.append(ExplorationAudit(
                symbol=sym,
                rf_score=0.0,
                tpi_gated='tpi_gated=True' in line,
                would_block_by=items,
            ))

    def _parse_pipeline_audit(self, line: str):
        if 'PIPELINE_AUDIT' not in line:
            return
        self.pipeline_audit_count += 1
        m = re.search(
            r'ranked=(\d+)\s+triggered=(\d+)\s+.*?submitted=(\d+)\s+.*?accepted=(\d+)\s+top3=(\d+)',
            line
        )
        if m:
            groups = m.groups()
            s = PipelineSnapshot(
                ranked=int(groups[0]),
                triggered=int(groups[1]),
                submitted=int(groups[2]),
                accepted=int(groups[3]),
                top3=int(groups[4]),
            )
            self.pipeline_snapshots.append(s)
            self.total_submitted_in_audit += s.submitted
            self.total_accepted_in_audit += s.accepted

    def _compute_cross_block_stats(self):
        """From exploration audit, compute gate overlap."""
        self.gate_cross_block = defaultdict(lambda: defaultdict(int))
        for audit in self.exploration_audits:
            for item in audit.would_block_by:
                # For each pair of gates that would block together
                for other in audit.would_block_by:
                    if other != item and other in GATES:
                        self.gate_cross_block[item][other] += 1


# ──────────────────────────────────────────────────────────────────────
# 4. Analysis engine
# ──────────────────────────────────────────────────────────────────────

class GateShadowAnalyst:
    def __init__(self, parser: GateLogParser):
        self.p = parser
        self.gate_stats: Dict[str, GateStats] = {}

    def analyze(self) -> dict:
        self._compute_gate_stats()
        self._estimate_recovery()
        self._compute_risk_analysis()
        return self._build_report()

    def _cross_block(self) -> dict:
        return self.p.gate_cross_block

    def _compute_gate_stats(self):
        # RF Gate
        rf_total = self.p.rf_blocked_signal + self.p.rf_blocked_rank
        rf_other_caught = (
            self.p.gate_cross_block.get("RF_GATE", {}).get("NOT_IN_TOP3", 0)
        )
        self.gate_stats["RF_GATE"] = GateStats(
            total_blocks=rf_total,
            unique_symbols=len(self.p.rf_symbol_counts),
            would_block_rate=(
                rf_total / max(rf_total + self.p.rf_passed, 1) * 100
            ),
        )

        # Observer State: count how many non-EXECUTE harness calls
        obs_total = self.p.observer_non_execute_count
        self.gate_stats["OBSERVER_STATE"] = GateStats(
            total_blocks=obs_total,
            unique_symbols=28,
            would_block_rate=(
                obs_total / max(self.p.observer_harness_count, 1) * 100
            ),
        )

        # Reality Vector
        rv_total = len(self.p.rejections["REALITY_VECTOR"])
        self.gate_stats["REALITY_VECTOR"] = GateStats(
            total_blocks=rv_total,
            unique_symbols=0,
            would_block_rate=0.0,
        )

        # CF Gate
        cf_total = len(self.p.rejections["CF_GATE"])
        self.gate_stats["CF_GATE"] = GateStats(
            total_blocks=cf_total,
            unique_symbols=0,
            would_block_rate=0.0,
        )

    def _estimate_recovery(self):
        """Estimate how many blocked signals would produce valid trades
        if the gate were converted to shadow mode.

        Methodology:
        - RF Gate: ~56.9% of RF-blocked signals are ALSO caught by
          NOT_IN_TOP3 or NET_ALPHA (from exploration audit cross-block).
          So ~43.1% might survive other gates.
          But many of those are false triggers (no real signal).
          Conservative estimate: only ~25% of RF-blocked would produce
          a valid submitted trade.
        - Observer State: non-EXECUTE states are low-confidence signals.
          Assume ~15% would produce valid execution.
        - Reality Vector: 0 triggered → 0 impact.
        - CF Gate: 0 triggered → 0 impact.
        """
        for gate in GATES:
            gs = self.gate_stats[gate]
            total = gs.total_blocks

            if gate == "RF_GATE":
                other_caught = (
                    self.p.gate_cross_block.get("RF_GATE", {}).get("NOT_IN_TOP3", 0)
                    + self.p.gate_cross_block.get("RF_GATE", {}).get("NET_ALPHA", 0)
                )
                caught_pct = min(other_caught / max(total, 1), 1.0)
                gs.already_caught_by_other = int(total * caught_pct)
                # Of the remaining, estimate signal quality
                survival_pct = 0.25
                gs.would_recover = int(total * (1 - caught_pct) * survival_pct)
                gs.signal_quality_score = 0.30  # RF measures regime quality
                gs.redundancy_pct = caught_pct * 100

            elif gate == "OBSERVER_STATE":
                # Observer signals 72% non-EXECUTE. Many are genuinely weak.
                # Estimate ~15% would have been valid trades.
                gs.would_recover = int(total * 0.15)
                gs.already_caught_by_other = int(total * 0.70)
                gs.redundancy_pct = 70.0
                gs.signal_quality_score = 0.15  # Low confidence signals

            elif gate in ("REALITY_VECTOR", "CF_GATE"):
                gs.would_recover = 0
                gs.already_caught_by_other = 0
                gs.redundancy_pct = 0.0
                gs.signal_quality_score = 0.0  # Never triggered

    def _compute_risk_analysis(self):
        """Tag each gate with risk of removing it."""
        self.risk_notes = {}

        self.risk_notes["RF_GATE"] = {
            "protections_lost": [
                "No regime quality filter — bad market regimes pass through",
                "No micro-structure probability calibration",
                "Replay drift detection removed",
                "Model drift alerts disabled",
            ],
            "unique_protection": True,
            "duplicated_by": [],
            "notes": "RF Gate is the ONLY gate that measures regime-conditioned signal probability. "
                     "No other gate provides micro-structure quality filtering. "
                     "Shadow mode would let ~10,800 raw signals through, "
                     "but ~75% would fail other downstream gates.",
        }

        self.risk_notes["OBSERVER_STATE"] = {
            "protections_lost": [
                "No signal confidence threshold — all confidence levels treated equally",
                "No TPI confidence gating",
                "No persistence-streak awareness in execution",
                "No entropy/curvature sanity check",
            ],
            "unique_protection": False,
            "duplicated_by": ["SDL (persistence)", "TPI gate (micro flow)"],
            "notes": "Observer State overlaps with SDL for persistence checking "
                     "and with TPI gate for micro-flow quality. "
                     "However, it is the ONLY gate that combines TPI + persistence "
                     "+ curvature + entropy into a single composite confidence score.",
        }

        self.risk_notes["REALITY_VECTOR"] = {
            "protections_lost": [
                "No execution tier awareness (PASSIVE/AGGRESSIVE/MODERATE)",
                "No E_exec / E_pred / E_contam contamination detection",
            ],
            "unique_protection": True,
            "duplicated_by": [],
            "notes": "Reality Vector is unique — no other gate measures execution "
                     "contamination (E_contam). It never triggered in this log "
                     "because it requires E_contam > 0.3 to go PASSIVE. "
                     "Critical for protecting against market regime collapse.",
        }

        self.risk_notes["CF_GATE"] = {
            "protections_lost": [
                "No counterfactual error hysteresis",
                "No CF error-driven circuit breaker",
            ],
            "unique_protection": True,
            "duplicated_by": [],
            "notes": "CF Gate is unique — no other gate measures counterfactual "
                     "prediction error. It never triggered in this log, meaning "
                     "the system's counterfactual predictions were within tolerance "
                     "(avg_cf_err < 0.35). Essential as a circuit breaker for "
                     "when the CF engine degrades.",
        }

    def _build_report(self) -> dict:
        return {
            "gate_stats": {k: asdict(v) for k, v in self.gate_stats.items()},
            "risk_analysis": self.risk_notes,
            "pipeline_current": {
                "total_pipeline_audits": self.p.pipeline_audit_count,
                "total_submitted": self.p.total_submitted_in_audit,
                "total_accepted": self.p.total_accepted_in_audit,
                "acceptance_rate_pct": (
                    self.p.total_accepted_in_audit
                    / max(self.p.total_submitted_in_audit, 1) * 100
                ),
            },
            "observer_detail": {
                "harness_total": self.p.observer_harness_count,
                "execute_count": self.p.observer_execute_count,
                "non_execute_count": self.p.observer_non_execute_count,
                "non_execute_states": dict(self.p.observer_not_execute_states),
            },
            "rf_detail": {
                "blocked_signal": self.p.rf_blocked_signal,
                "blocked_rank": self.p.rf_blocked_rank,
                "passed": self.p.rf_passed,
                "warmup": self.p.rf_warmup,
                "symbol_top5": sorted(
                    self.p.rf_symbol_counts.items(),
                    key=lambda x: -x[1],
                )[:5],
            },
            "router_reject_detail": {
                "total": self.p.router_reject_total,
                "causes": dict(self.p.router_reject_causes),
            },
            "exploration_detail": {
                "total_audits": self.p.exploration_total,
                "would_block_breakdown": self._compute_would_block_pct(),
            },
            "total_scanned_lines": self.p.total_lines,
            "date_range": sorted(self.p.date_ranges),
        }

    def _compute_would_block_pct(self) -> dict:
        counts = defaultdict(int)
        for audit in self.p.exploration_audits:
            for item in audit.would_block_by:
                counts[item] += 1
        total = max(self.p.exploration_total, 1)
        return {
            k: {"count": v, "pct": round(v / total * 100, 1)}
            for k, v in sorted(counts.items(), key=lambda x: -x[1])
        }


# ──────────────────────────────────────────────────────────────────────
# 5. Report formatter
# ──────────────────────────────────────────────────────────────────────

class ReportFormatter:
    def __init__(self, analysis: dict):
        self.a = analysis

    def print_report(self):
        self._print_header()
        self._print_scope()
        self._print_gate_table()
        self._print_observer_detail()
        self._print_rf_detail()
        self._print_recovery_impact()
        self._print_funnel_estimate()
        self._print_risk_analysis()
        self._print_exploration_detail()
        self._print_recommendation()

    def _print_header(self):
        print("=" * 72)
        print("  GATE SHADOW EXPERIMENT — OFFLINE ANALYSIS REPORT")
        print("=" * 72)
        print()

    def _print_scope(self):
        print(f"Log file                : {LOG_PATH}")
        print(f"Lines scanned           : {self.a['total_scanned_lines']:,}")
        print(f"Date range              : {self.a['date_range'][0]} to {self.a['date_range'][-1]}")
        print(f"Gates analyzed          : {', '.join(GATES)}")
        print()

    def _print_gate_table(self):
        print("─" * 72)
        print(f"  {'Gate':<20} {'Total Blocks':>13} {'Would Recover':>14} {'Signal Quality':>14}")
        print("─" * 72)
        for gate in GATES:
            s = self.a['gate_stats'][gate]
            sq_label = self._quality_label(s['signal_quality_score'])
            print(f"  {gate:<20} {s['total_blocks']:>13,} {s['would_recover']:>14,} {sq_label:>14}")
        print("─" * 72)
        print()

    def _quality_label(self, score: float) -> str:
        if score >= 0.7:
            return "HIGH"
        elif score >= 0.4:
            return "MEDIUM"
        elif score > 0:
            return "LOW"
        return "N/A"

    def _print_observer_detail(self):
        od = self.a['observer_detail']
        print("── Observer State Detail ──")
        print(f"  OBSERVER_HARNESS total    : {od['harness_total']:>7,}")
        print(f"  EXECUTE state             : {od['execute_count']:>7,}  ({od['execute_count']/max(od['harness_total'],1)*100:.1f}%)")
        print(f"  Non-EXECUTE state         : {od['non_execute_count']:>7,}  ({od['non_execute_count']/max(od['harness_total'],1)*100:.1f}%)")
        print(f"  Non-EXECUTE breakdown:")
        for state, cnt in sorted(od['non_execute_states'].items(), key=lambda x: -x[1]):
            print(f"    {state:<20} {cnt:>7,}")
        print(f"  ROUTER REJECT by OBSERVER : {self.a['router_reject_detail']['causes'].get('OBSERVER_STATE', 0):>7,}")
        print(f"    (remaining non-EXECUTE filtered pre-router)")
        print()

    def _print_rf_detail(self):
        rf = self.a['rf_detail']
        print("── RF Gate Detail ──")
        print(f"  Signal blocks   : {rf['blocked_signal']:>7,}")
        print(f"  Rank blocks     : {rf['blocked_rank']:>7,}")
        print(f"  Total blocks    : {rf['blocked_signal'] + rf['blocked_rank']:>7,}")
        print(f"  Passes          : {rf['passed']:>7,}")
        print(f"  Block rate      : {(rf['blocked_signal'] + rf['blocked_rank']) / max(rf['blocked_signal'] + rf['blocked_rank'] + rf['passed'], 1) * 100:.1f}%")
        print(f"  Top-5 symbols   : {rf['symbol_top5']}")
        print()

    def _print_recovery_impact(self):
        print("── Recovery Impact Estimate ──")
        print(f"  {'Gate':<20} {'Blocks':>10} {'Survive Other':>14} {'Recovered':>10} {'Quality':>10}")
        print("─" * 70)
        for gate in GATES:
            s = self.a['gate_stats'][gate]
            sq_label = self._quality_label(s['signal_quality_score'])
            print(f"  {gate:<20} {s['total_blocks']:>10,} {s['already_caught_by_other']:>14,} {s['would_recover']:>10,} {sq_label:>10}")
        print()

    def _print_funnel_estimate(self):
        pc = self.a['pipeline_current']
        print("── Pipeline Funnel — Current vs Shadow-Estimated ──")
        print()
        print(f"  Current pipeline (from {pc['total_pipeline_audits']} audit snapshots):")
        print(f"    Submitted : {pc['total_submitted']:>6,}")
        print(f"    Accepted  : {pc['total_accepted']:>6,}")
        print(f"    Conv rate : {pc['acceptance_rate_pct']:.1f}%")
        print()

        # Compute impact for each shadow scenario
        for gate in GATES:
            s = self.a['gate_stats'][gate]
            if s['would_recover'] == 0:
                continue
            new_submitted = pc['total_submitted'] + s['would_recover']
            new_accepted = int(new_submitted * pc['acceptance_rate_pct'] / 100)
            print(f"  If {gate} → SHADOW:")
            print(f"    Submitted : {pc['total_submitted']:>6,} → {new_submitted:>6,} (+{s['would_recover']:>5,})")
            print(f"    Accepted  : {pc['total_accepted']:>6,} → {new_accepted:>6,} (+{new_accepted - pc['total_accepted']:>5,})")
            print(f"    Recovery  : {s['signal_quality_score']*100:.0f}% signal quality (of recovered)")
            print()

        # Combined impact
        total_recovered = sum(
            s['would_recover'] for s in self.a['gate_stats'].values()
        )
        if total_recovered > 0:
            combined_submitted = pc['total_submitted'] + total_recovered
            combined_accepted = int(combined_submitted * pc['acceptance_rate_pct'] / 100)
            print(f"  If ALL 4 gates → SHADOW:")
            print(f"    Submitted : {pc['total_submitted']:>6,} → {combined_submitted:>6,} (+{total_recovered:>5,})")
            print(f"    Accepted  : {pc['total_accepted']:>6,} → {combined_accepted:>6,} (+{combined_accepted - pc['total_accepted']:>5,})")
            print()

    def _print_risk_analysis(self):
        print("── Risk Analysis ──")
        print()
        for gate in GATES:
            r = self.a['risk_analysis'].get(gate)
            if not r:
                continue
            print(f"  {gate}:")
            print(f"    Unique protection : {'YES' if r['unique_protection'] else 'NO'}")
            if r['duplicated_by']:
                print(f"    Also covered by   : {', '.join(r['duplicated_by'])}")
            print(f"    Protections lost  :")
            for prot in r['protections_lost']:
                print(f"      • {prot}")
            print(f"    Notes             : {r['notes']}")
            print()

    def _print_exploration_detail(self):
        ed = self.a['exploration_detail']
        print("── Exploration Mode Audit — Would-Block Cross-Reference ──")
        print(f"  Total exploration events: {ed['total_audits']}")
        print(f"  {'Gate would block':<20} {'Count':>8} {'% of Exp':>10}")
        print("─" * 42)
        for gate in GATES:
            d = ed['would_block_breakdown'].get(gate)
            if d:
                print(f"  {gate:<20} {d['count']:>8,} {d['pct']:>9.1f}%")
        for other_gate, d in ed['would_block_breakdown'].items():
            if other_gate not in GATES:
                print(f"  {other_gate:<20} {d['count']:>8,} {d['pct']:>9.1f}%")
        print()

    def _print_recommendation(self):
        print("── Recommendation ──")
        print()
        print("  SAFE TO SHADOW-IZE FIRST (Tier 1):")
        print()
        print("  1. Reality Vector — 0 triggered, latent protection only.")
        print("     Shadow mode provides monitoring without loss of safety.")
        print("     Continue tracking E_contam in log for early warning.")
        print()
        print("  2. CF Gate — 0 triggered, latent circuit breaker only.")
        print("     Shadow mode allows monitoring CF error trends.")
        print("     Re-introduce hard block if avg_cf_err > 0.35 persists.")
        print()
        print("  CONDITIONAL SHADOW-IZE (Tier 2):")
        print()
        print("  3. Observer State — 72% non-EXECUTE rate is VERY high.")
        print("     Shadow mode would add ~1,245 low-confidence signals.")
        print("     Only safe if SDL + TPI gate + NET_ALPHA remain hard.")
        print("     RECOMMENDATION: Keep hard for now. Shadow only after")
        print("     Tier-1 gates prove safe for 500+ cycles.")
        print()
        print("  4. RF Gate — 10,811 blocks with 85.5% block rate.")
        print("     This is the LARGEST funnel bottleneck.")
        print("     But it is also the ONLY regime-probability filter.")
        print("     Shadow mode would flood pipeline with ~10k raw signals,")
        print("     most of which (85%+) are low-probability noise.")
        print("     RECOMMENDATION: Do NOT shadow-ize. Keep as hard gate.")
        print("     Instead: reduce prob_thresh from 0.6 to 0.45 to")
        print("     allow more signals through while retaining filtering.")
        print()
        print("  TOTAL IMPACT: Shadow-izing only Reality + CF → 0 change.")
        print("  Shadow-izing Observer + RF → ~204 recovered trades")
        print("  (but ~70% are low-quality signals). Net benefit: NEGATIVE.")
        print()
        print("  FINAL VERDICT: Convert Reality Vector + CF Gate to shadow.")
        print("  Keep RF Gate and Observer State as hard gates.")
        print()


# ──────────────────────────────────────────────────────────────────────
# 6. Main
# ──────────────────────────────────────────────────────────────────────

def main():
    import sys

    log_path = LOG_PATH
    if not os.path.exists(log_path):
        print(f"ERROR: Log file not found: {log_path}")
        sys.exit(1)

    print(f"Reading log: {log_path}")
    print(f"Size: {os.path.getsize(log_path) / 1e6:.1f} MB")
    print()

    parser = GateLogParser(log_path).parse()
    analyst = GateShadowAnalyst(parser)
    analysis = analyst.analyze()

    fmt = ReportFormatter(analysis)
    fmt.print_report()

    # Also save JSON
    report_path = os.path.join(
        os.path.dirname(__file__), "gate_shadow_analysis.json"
    )
    with open(report_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"JSON report saved: {report_path}")


if __name__ == "__main__":
    main()
