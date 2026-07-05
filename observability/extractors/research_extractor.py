"""
research_extractor.py — Extracts ALL research-layer dashboard data from ProximaDemo.

This module replaces ``funnel_dash.generate()`` and the research-layer sections of
``_print_dashboard()``. Every method is a pure extractor — it returns structured
dataclass instances and never builds strings or prints.

Schema Version: 1.0.0
"""

from __future__ import annotations

from typing import Optional

from ..schema.telemetry_schema import (
    DeploymentRealitySnapshot,
    DirectorPipelineSnapshot,
    DplValidationSnapshot,
    FrequencyRealitySnapshot,
    FunnelSnapshot,
    RclDashboardSnapshot,
    RealityConvergenceSnapshot,
    SessionBalanceSnapshot,
    ShadowSnapshot,
    TpiSnapshot,
)


class ResearchExtractor:
    """
    Extracts ALL research-layer dashboard data from ProximaDemo.

    Pure extraction — NO string building, NO printing.  Every method returns a
    dataclass instance (or ``None`` when the underlying engine is absent).
    """

    def __init__(self, demo):
        self._demo = demo

    # ── Signal Funnel ──────────────────────────────────────────────────────────

    def extract_funnel(self) -> FunnelSnapshot:
        """Extract signal funnel stats."""
        try:
            funnel = {}
            if hasattr(self._demo, "funnel") and self._demo.funnel is not None:
                if hasattr(self._demo.funnel, "summary"):
                    funnel = self._demo.funnel.summary()

            g = funnel.get("GENERATED", 0)
            tp = funnel.get("THRESHOLD_PASSED", 0)
            tr = funnel.get("TRIGGERED", 0)
            sub = funnel.get("ORDER_SUBMITTED", 0)
            acc = funnel.get("ORDER_ACCEPTED", 0)
            opn = funnel.get("POSITION_OPENED", 0)
            cls = funnel.get("POSITION_CLOSED", 0)

            block_keys = [
                "BLOCKED_SPREAD",
                "BLOCKED_POSITION_EXISTS",
                "BLOCKED_RISK_LIMIT",
                "BLOCKED_MAX_POSITIONS",
                "BLOCKED_POSITION_LOCK",
                "BLOCKED_NOT_IN_TOP3",
                "BLOCKED_THRESHOLD",
                "BLOCKED_RHL",
                "BLOCKED_H20",
                "BLOCKED_FLIP",
                "BLOCKED_EQUITY_PROTECTION",
                "BLOCKED_NO_TICK",
                "BLOCKED_UNKNOWN",
            ]

            blocked = sum(funnel.get(k, 0) for k in block_keys)
            rejected = funnel.get("ORDER_REJECTED", 0)
            timeout = funnel.get("ORDER_TIMEOUT", 0)

            pass_rate = (tp / g * 100) if g > 0 else 0.0
            submit_rate = (sub / max(tr, 1) * 100)
            accept_rate = (acc / max(sub, 1) * 100)
            open_rate = (opn / max(acc, 1) * 100)
            leakage_pct = (g - opn) / g * 100 if g > 0 else 0
            survival_rate = opn / g * 100 if g > 0 else 0

            block_breakdown = {}
            for k in block_keys:
                v = funnel.get(k, 0)
                if v:
                    block_breakdown[k.replace("BLOCKED_", "").lower()] = v

            return FunnelSnapshot(
                generated=g,
                threshold_passed=tp,
                triggered=tr,
                submitted=sub,
                accepted=acc,
                opened=opn,
                closed=cls,
                blocked=blocked,
                rejected=rejected,
                timeout=timeout,
                pass_rate=pass_rate,
                submit_rate=submit_rate,
                accept_rate=accept_rate,
                open_rate=open_rate,
                leakage_pct=leakage_pct,
                survival_rate=survival_rate,
                block_breakdown=block_breakdown,
            )
        except Exception:
            return FunnelSnapshot(
                generated=0, threshold_passed=0, triggered=0,
                submitted=0, accepted=0, opened=0, closed=0,
                blocked=0, rejected=0, timeout=0,
                pass_rate=0.0, submit_rate=0.0, accept_rate=0.0,
                open_rate=0.0, leakage_pct=0.0, survival_rate=0.0,
                block_breakdown={},
            )

    # ── Trade Policy Indicator (TPI) ───────────────────────────────────────────

    def extract_tpi(self) -> TpiSnapshot:
        """Extract TPI dashboard data from module-level counters and engine state."""
        try:
            from dashboard import tpi_dashboard as _tpi

            per_symbol = {}
            for sym, sig in getattr(_tpi, "_TPI_CACHED_SIGNALS", {}).items():
                per_symbol[sym] = {
                    "tpi": sig.get("tpi"),
                    "direction": sig.get("direction_label"),
                    "confidence": sig.get("confidence"),
                    "percentile": sig.get("percentile"),
                    "session": sig.get("session_name"),
                    "eligible": sig.get("eligible"),
                    "alignment": sig.get("alignment"),
                    "persistence": sig.get("persistence"),
                    "curvature": sig.get("curvature"),
                }

            # Live decay stats from tracker
            live_decay = {}
            tracker = getattr(self._demo, "_tpi_tracker", None)
            if tracker is not None and hasattr(tracker, "live_decay_stats"):
                try:
                    live_decay = tracker.live_decay_stats()
                except Exception:
                    pass

            # Promotion gates (mirrors tpi_dashboard.generate() logic)
            n_base = getattr(_tpi, "_TPI_BASE_WINS", 0) + getattr(_tpi, "_TPI_BASE_LOSSES", 0)
            base_wr = getattr(_tpi, "_TPI_BASE_WINS", 0) / n_base * 100 if n_base > 0 else 0
            n_aligned = getattr(_tpi, "_TPI_ALIGNED_WINS", 0) + getattr(_tpi, "_TPI_ALIGNED_LOSSES", 0)
            aligned_wr = getattr(_tpi, "_TPI_ALIGNED_WINS", 0) / n_aligned * 100 if n_aligned > 0 else 0
            n_conflict = getattr(_tpi, "_TPI_CONFLICT_WINS", 0) + getattr(_tpi, "_TPI_CONFLICT_LOSSES", 0)
            conflict_wr = getattr(_tpi, "_TPI_CONFLICT_WINS", 0) / n_conflict * 100 if n_conflict > 0 else 0

            n_eligible = len(
                [o for o in getattr(_tpi, "_TPI_TRADE_OUTCOMES", [])
                 if o.get("alignment") in ("MATCH", "CONFLICT")]
            )
            gate_a = aligned_wr >= base_wr + 5.0 if n_eligible >= 50 else False
            gate_b = (conflict_wr <= base_wr - 5.0 or n_conflict == 0) if n_eligible >= 50 else False
            gate_c = n_eligible >= 50
            gates_passed = sum([gate_a, gate_b, gate_c])

            total_shadow = (
                getattr(_tpi, "_TPI_ALIGNMENTS", 0)
                + getattr(_tpi, "_TPI_CONFLICTS", 0)
                + getattr(_tpi, "_TPI_WEAK_ALIGNMENTS", 0)
            )

            return TpiSnapshot(
                per_symbol=per_symbol,
                base_wins=getattr(_tpi, "_TPI_BASE_WINS", 0),
                base_losses=getattr(_tpi, "_TPI_BASE_LOSSES", 0),
                aligned_wins=getattr(_tpi, "_TPI_ALIGNED_WINS", 0),
                aligned_losses=getattr(_tpi, "_TPI_ALIGNED_LOSSES", 0),
                conflict_wins=getattr(_tpi, "_TPI_CONFLICT_WINS", 0),
                conflict_losses=getattr(_tpi, "_TPI_CONFLICT_LOSSES", 0),
                veto_avoided_losses=getattr(_tpi, "_TPI_VETO_AVOIDED_LOSSES", 0),
                total_shadow_observations=total_shadow,
                alignments=getattr(_tpi, "_TPI_ALIGNMENTS", 0),
                conflicts=getattr(_tpi, "_TPI_CONFLICTS", 0),
                weak_alignments=getattr(_tpi, "_TPI_WEAK_ALIGNMENTS", 0),
                gates_passed=gates_passed,
                live_decay=live_decay,
            )
        except Exception:
            return TpiSnapshot(
                per_symbol={},
                base_wins=0, base_losses=0,
                aligned_wins=0, aligned_losses=0,
                conflict_wins=0, conflict_losses=0,
                veto_avoided_losses=0,
                total_shadow_observations=0,
                alignments=0, conflicts=0, weak_alignments=0,
                gates_passed=0, live_decay={},
            )

    # ── Deployment Reality Lab ─────────────────────────────────────────────────

    def extract_deployment_reality(self) -> Optional[DeploymentRealitySnapshot]:
        """Extract Deployment Reality Lab data (via funnel_dash._drl)."""
        try:
            drl = getattr(getattr(self._demo, "funnel_dash", None), "_drl", None)
            if drl is None:
                return None

            dc = drl.classify()
            exec_q = {}
            if hasattr(drl, "_exec") and hasattr(drl._exec, "summary"):
                exec_q = drl._exec.summary()

            return DeploymentRealitySnapshot(
                asr=float(dc.get("asr", 0)),
                execution_quality=str(dc.get("execution_quality", "N/A")),
                mean_slippage_pts=float(exec_q.get("mean_slippage_pts", 0)),
                score_trend=str(dc.get("score_trend", "N/A")),
                classification=str(dc.get("classification", "UNKNOWN")),
            )
        except Exception:
            return None

    # ── Frequency Reality / Spread Audit ───────────────────────────────────────

    def extract_frequency_reality(self) -> Optional[FrequencyRealitySnapshot]:
        """Extract Frequency Reality / Spread Audit data (via funnel_dash._freq)."""
        try:
            freq = getattr(getattr(self._demo, "funnel_dash", None), "_freq", None)
            if freq is None:
                return None

            analysis = getattr(freq, "_analysis", None)
            c = freq.classify() if hasattr(freq, "classify") else {}

            blocked_sig = {}
            adr = 0.0
            if analysis is not None:
                try:
                    blocked_sig = analysis.leakage_rate()
                except Exception:
                    pass
                try:
                    adr = analysis.alpha_destruction_ratio()
                except Exception:
                    pass

            return FrequencyRealitySnapshot(
                blocked_total=int(blocked_sig.get("blocked_total", 0)),
                blocked_profitable=int(blocked_sig.get("blocked_profitable", 0)),
                leakage_rate=str(blocked_sig.get("leakage_rate", "0%")),
                adr=float(adr),
                classification=str(c.get("classification", "UNKNOWN")),
            )
        except Exception:
            return None

    # ── Reality Convergence Engine ─────────────────────────────────────────────

    def extract_reality_convergence(self) -> Optional[RealityConvergenceSnapshot]:
        """Extract Reality Convergence Engine data (via funnel_dash._rce)."""
        try:
            rce = getattr(getattr(self._demo, "funnel_dash", None), "_rce", None)
            if rce is None:
                return None

            ate = {}
            friction = {}
            health = {}
            cls = {}
            freq_match = 0

            if hasattr(rce, "_ate") and hasattr(rce._ate, "summary"):
                ate = rce._ate.summary()
            if hasattr(rce, "_friction") and hasattr(rce._friction, "summary"):
                friction = rce._friction.summary()
            if hasattr(rce, "_health") and hasattr(rce._health, "compute"):
                health = rce._health.compute()
            if hasattr(rce, "_classifier") and hasattr(rce._classifier, "classify"):
                cls = rce._classifier.classify()
            if hasattr(rce, "_conv") and hasattr(rce._conv, "match_pct"):
                freq_match = rce._conv.match_pct()

            freq_match_str = (
                f"{freq_match}%" if isinstance(freq_match, (int, float))
                else str(freq_match)
            )

            return RealityConvergenceSnapshot(
                ate=float(ate.get("ate", 0)),
                frequency_match=freq_match_str,
                friction_index=float(friction.get("friction_index", 0)),
                health_index=float(health.get("health_index", 0)),
                classification=str(cls.get("classification", "UNKNOWN")),
            )
        except Exception:
            return None

    # ── DPL Live Validation ────────────────────────────────────────────────────

    def extract_dpl_validation(self) -> Optional[DplValidationSnapshot]:
        """Extract DPL Live Validation data (via funnel_dash._dpl)."""
        try:
            dpl = getattr(getattr(self._demo, "funnel_dash", None), "_dpl", None)
            if dpl is None:
                return None

            s = dpl.summary()
            return DplValidationSnapshot(
                total_snapshots=int(s.get("total_snapshots", 0)),
                resolved=int(s.get("resolved", 0)),
                pct_resolved=float(s.get("pct_resolved", 0)),
                symbols=list(s.get("symbols", [])),
                regime_distribution=s.get("regime_distribution", {}),
                has_short_outcomes=bool(s.get("has_short_outcomes", False)),
            )
        except Exception:
            return None

    # ── Director Pipeline ──────────────────────────────────────────────────────

    def extract_director_pipeline(self) -> Optional[DirectorPipelineSnapshot]:
        """Extract Autonomous Research Director data."""
        try:
            # Priority: demo._director, then funnel_dash._director
            director = getattr(self._demo, "_director", None)
            if director is None:
                director = getattr(
                    getattr(self._demo, "funnel_dash", None), "_director", None
                )
            if director is None or not hasattr(director, "daily_report"):
                return None

            report = director.daily_report()
            return DirectorPipelineSnapshot(
                evidence_strength=float(report.get("evidence_strength", 0)),
                research_confidence=float(report.get("research_confidence", 0)),
                deployment_confidence=float(report.get("deployment_confidence", 0)),
                alpha_transfer=float(report.get("alpha_transfer", 0)),
                biggest_risk=str(report.get("biggest_risk", "N/A")),
                biggest_strength=str(report.get("biggest_strength", "N/A")),
                recommendation=str(report.get("recommendation", "NO_ACTION")),
                classification=str(report.get("classification", "RESEARCH_PENDING")),
            )
        except Exception:
            return None

    # ── RCL Dual Horizon Dashboard ─────────────────────────────────────────────

    def extract_rcl_dashboard(self) -> Optional[RclDashboardSnapshot]:
        """Extract RCL dual-horizon win-rate dashboard."""
        try:
            resolved = getattr(
                getattr(self._demo, "_outcome_ledger", None), "_resolved", None
            )
            if resolved is None:
                return None

            h5_records = [r for r in resolved if "h5" in r.get("outcomes", {})]
            h20_records = [r for r in resolved if "h20" in r.get("outcomes", {})]
            h5_wins = sum(1 for r in h5_records if r["outcomes"]["h5"].get("win"))
            h20_wins = sum(1 for r in h20_records if r["outcomes"]["h20"].get("win"))

            return RclDashboardSnapshot(
                h5_resolved=len(h5_records),
                h20_resolved=len(h20_records),
                h5_wins=h5_wins,
                h20_wins=h20_wins,
                h5_win_rate=h5_wins / max(len(h5_records), 1),
                h20_win_rate=h20_wins / max(len(h20_records), 1),
                divergence=(h5_wins / max(len(h5_records), 1))
                - (h20_wins / max(len(h20_records), 1)),
            )
        except Exception:
            return None

    # ── Session Balance ────────────────────────────────────────────────────────

    def extract_session_balance(self) -> Optional[SessionBalanceSnapshot]:
        """Extract session balance distribution."""
        try:
            sb = getattr(self._demo, "_session_balance", None)
            if sb is None:
                return None

            asia = sb.get("ASIA", 0)
            london = sb.get("LONDON", 0)
            overlap = sb.get("OVERLAP", 0)
            ny = sb.get("NY", 0)
            dead = sb.get("DEAD", 0)
            total = asia + london + overlap + ny + dead

            imbalance = 0.0
            status = "BUILDING"
            if total > 0:
                known = {
                    k: v for k, v in sb.items()
                    if k in ("ASIA", "LONDON", "OVERLAP", "NY", "DEAD") and v > 0
                }
                if known and len(known) >= 2:
                    max_c = max(known.values())
                    min_c = min(known.values())
                    imbalance = max_c / max(min_c, 1)
                    if max_c >= 20:
                        status = "BALANCED" if imbalance < 5 else "SKEWED"

            return SessionBalanceSnapshot(
                asia=asia,
                london=london,
                overlap=overlap,
                ny=ny,
                dead=dead,
                total=total,
                imbalance=imbalance,
                status=status,
            )
        except Exception:
            return None

    # ── Shadow System (STR-E) ──────────────────────────────────────────────────

    def extract_shadow_system(self) -> Optional[ShadowSnapshot]:
        """Extract Shadow System STR-E reconciliation data."""
        try:
            sr = getattr(self._demo, "_last_stre_result", None)
            if not sr:
                return None

            coord = getattr(self._demo, "_stre_coordinator", None)
            phase2 = getattr(coord, "phase2_enabled", False) if coord else False

            return ShadowSnapshot(
                shadow_alignment=float(sr.get("gt_corr", 0)),
                sof_score=float(sr.get("SOF", 0)),
                edge_decay=0.0,
                mirror_divergence=float(sr.get("stas", 0)),
                alpha_transfer_rate=0.0,
                false_signal_rate=0.0,
                gt_corr=float(sr.get("gt_corr", 0)),
                sy_corr=float(sr.get("sy_corr", 0)),
                stas=float(sr.get("stas", 0)),
                winner=str(sr.get("winner", "N/A")),
                edge_preservation=float(sr.get("edge_preservation", 0)),
                execution_efficiency=float(sr.get("execution_efficiency", 0)),
                phase2_enabled=phase2,
                samples=int(sr.get("samples", 0)),
            )
        except Exception:
            return None

    # ── Misc Engine Dashboards (raw dicts / strings) ───────────────────────────

    def extract_engine_dashboards(self) -> dict:
        """
        Collect all remaining engine dashboard summaries as a flat dictionary.

        This covers lines 1504-1648 of ``_print_dashboard()``:
          V2.2 guards, pyramid log, reinforcement/flip blocks, propagation,
          tick thermodynamics, meta-state fusion, session conditional,
          entropy compression, outcome ledger, IG audit, redundancy matrix,
          meta-reweighter, layer pruner, occupancy audit, TPI A/B audit,
          spread normalizer, funnel audit, regime memory, signal decay,
          migration, exception dashboard, and impulse graph.

        Returns a dict keyed by logical section names.  Values are either
        primitive types, dicts, or strings (from ``.summary()`` methods that
        return strings — these are captured as-is since there is no structured
        dataclass for every sub-section).
        """
        result = {}

        try:
            demo = self._demo

            # V2.2 Guards
            guard = getattr(getattr(demo, "_sample_guard", None), "guard", None)
            if guard:
                try:
                    result["sample_guard"] = guard("DEPLOYMENT_CLASSIFICATION")
                except Exception:
                    pass

            alignment_mon = getattr(demo, "_alignment_monitor", None)
            if alignment_mon and hasattr(alignment_mon, "dashboard_line"):
                try:
                    sig_counts = {}
                    funnel = getattr(demo, "funnel", None)
                    if funnel and hasattr(funnel, "_records"):
                        for r in funnel._records:
                            sym = r.get("symbol")
                            if sym:
                                sig_counts[sym] = sig_counts.get(sym, 0) + 1
                    result["alignment_monitor"] = alignment_mon.dashboard_line(
                        sig_counts
                    )
                except Exception:
                    pass

            # Pyramid log
            pyramid_log = getattr(demo, "_pyramid_log", [])
            if pyramid_log:
                result["pyramid_event_count"] = len(pyramid_log)
                result["pyramid_recent"] = [
                    {
                        "time": pe.get("time", "")[:19] if isinstance(pe.get("time"), str) else str(pe.get("time", "")),
                        "symbol": pe.get("symbol"),
                        "pyramid_number": pe.get("pyramid_number"),
                        "es_percentile": pe.get("es_percentile"),
                    }
                    for pe in pyramid_log[-5:]
                ]

            # Position exists blocks (P8)
            result["reinforcement_blocks"] = getattr(demo, "_reinforcement_blocks", 0)
            result["flip_blocks"] = getattr(demo, "_flip_blocks", 0)
            tb = result["reinforcement_blocks"] + result["flip_blocks"]
            if tb > 0:
                result["position_exists_total"] = tb
                result["flip_pct"] = result["flip_blocks"] / tb * 100

            # Exception dashboard
            exc_dash = getattr(demo, "_exception_dashboard", None)
            if exc_dash is not None:
                try:
                    result["exceptions_active"] = (
                        exc_dash.has_active() if hasattr(exc_dash, "has_active") else False
                    )
                    if hasattr(exc_dash, "summary"):
                        exc_summary = exc_dash.summary()
                        if isinstance(exc_summary, str):
                            result["exceptions_summary"] = exc_summary
                except Exception:
                    pass

            # Cross-asset propagation
            tpi_prop = getattr(demo, "_tpi_propagation", None)
            if tpi_prop is not None:
                try:
                    prop_syms = [
                        s for s in getattr(demo, "_observation_universe", [])
                    ]
                    if prop_syms and hasattr(tpi_prop, "compute"):
                        result["propagation_matrix"] = tpi_prop.compute(prop_syms)
                    if prop_syms and hasattr(tpi_prop, "summary"):
                        result["propagation_summary"] = tpi_prop.summary(prop_syms)
                except Exception:
                    pass

            # Impulse graph
            imp_graph = getattr(demo, "_impulse_graph", None)
            if imp_graph is not None:
                try:
                    if hasattr(imp_graph, "summary"):
                        result["impulse_graph_summary"] = imp_graph.summary()
                except Exception:
                    pass

            # Tick thermodynamics
            tick_thermo = getattr(demo, "_tick_thermo", None)
            if tick_thermo is not None:
                try:
                    thermo_syms = getattr(demo, "_observation_universe", [])
                    if thermo_syms and hasattr(tick_thermo, "summary"):
                        result["tick_thermodynamics"] = tick_thermo.summary(thermo_syms)
                except Exception:
                    pass

            # Meta-state fusion
            meta_fusion = getattr(demo, "_meta_fusion", None)
            if meta_fusion is not None:
                try:
                    meta_syms = [
                        s for s in getattr(demo, "_observation_universe", [])
                        if s in getattr(demo, "_last_meta_scores", {})
                    ]
                    if meta_syms and hasattr(meta_fusion, "summary"):
                        result["meta_fusion"] = meta_fusion.summary(
                            meta_syms, getattr(demo, "_last_meta_scores", {})
                        )
                except Exception:
                    pass

            # Session conditional
            session_cond = getattr(demo, "_session_cond", None)
            if session_cond is not None and hasattr(session_cond, "summary"):
                try:
                    result["session_conditional"] = session_cond.summary()
                except Exception:
                    pass

            # Entropy compression
            entropy_comp = getattr(demo, "_entropy_compression", None)
            if entropy_comp is not None:
                try:
                    ent_syms = getattr(demo, "_observation_universe", [])
                    if ent_syms and hasattr(entropy_comp, "summary"):
                        result["entropy_compression"] = entropy_comp.summary(ent_syms)
                except Exception:
                    pass

            # Outcome ledger
            outcome_ledger = getattr(demo, "_outcome_ledger", None)
            if outcome_ledger is not None and hasattr(outcome_ledger, "summary"):
                try:
                    result["outcome_ledger"] = outcome_ledger.summary()
                except Exception:
                    pass

            # IG audit
            ig_audit = getattr(demo, "_ig_audit", None)
            if ig_audit is not None and hasattr(ig_audit, "summary"):
                try:
                    result["ig_audit_summary"] = ig_audit.summary(outcome_ledger)
                except Exception:
                    pass

            # Redundancy matrix
            redundancy = getattr(demo, "_redundancy_matrix", None)
            if redundancy is not None and hasattr(redundancy, "summary"):
                try:
                    result["redundancy_matrix"] = redundancy.summary(outcome_ledger)
                except Exception:
                    pass

            # Meta reweighter
            meta_reweight = getattr(demo, "_meta_reweighter", None)
            if meta_reweight is not None and hasattr(meta_reweight, "summary"):
                try:
                    result["meta_reweighter"] = meta_reweight.summary()
                except Exception:
                    pass

            # Layer pruner
            layer_pruner = getattr(demo, "_layer_pruner", None)
            if layer_pruner is not None and hasattr(layer_pruner, "summary"):
                try:
                    result["layer_pruner"] = layer_pruner.summary(outcome_ledger)
                except Exception:
                    pass

            # Occupancy audit
            occ_audit = getattr(demo, "_occupancy_audit", None)
            if occ_audit is not None and hasattr(occ_audit, "summary"):
                try:
                    result["occupancy_audit"] = occ_audit.summary()
                except Exception:
                    pass

            # TPI A/B audit
            tpi_ab = getattr(demo, "_tpi_ab_audit", None)
            if tpi_ab is not None and hasattr(tpi_ab, "summary"):
                try:
                    result["tpi_ab_audit"] = tpi_ab.summary()
                except Exception:
                    pass

            # Spread normalizer
            spread_norm = getattr(demo, "_spread_normalizer", None)
            if spread_norm is not None and hasattr(spread_norm, "session_baseline_summary"):
                try:
                    sbs = spread_norm.session_baseline_summary()
                    if sbs:
                        result["spread_normalizer"] = sbs
                except Exception:
                    pass

            # Funnel audit
            funnel_audit = getattr(demo, "_funnel_audit", None)
            if funnel_audit is not None and hasattr(funnel_audit, "summary"):
                try:
                    result["funnel_audit"] = funnel_audit.summary()
                except Exception:
                    pass

            # Regime memory
            regime_mem = getattr(demo, "_regime_memory", None)
            if regime_mem is not None and hasattr(regime_mem, "summary"):
                try:
                    result["regime_memory_summary"] = regime_mem.summary()
                except Exception:
                    pass
            if regime_mem is not None and hasattr(regime_mem, "transition_edge_summary"):
                try:
                    # Build per-symbol transition edge lines
                    edges = []
                    for sym in getattr(demo, "_observation_universe", []):
                        prev_r = getattr(demo, "_regime_snapshot", {}).get(sym)
                        curr_r = (
                            regime_mem._prev_regime.get(sym)
                            if hasattr(regime_mem, "_prev_regime")
                            else None
                        )
                        if prev_r is not None and curr_r is not None and prev_r != curr_r:
                            edge_line = regime_mem.transition_edge_summary(
                                sym, prev_r, curr_r
                            )
                            if edge_line:
                                edges.append(edge_line)
                    if edges:
                        result["regime_transition_edges"] = edges
                except Exception:
                    pass

            # Signal decay velocity
            signal_decay = getattr(demo, "_signal_decay", None)
            if signal_decay is not None and hasattr(signal_decay, "summary"):
                try:
                    result["signal_decay"] = signal_decay.summary()
                except Exception:
                    pass

            # Occupancy migration
            migration = getattr(demo, "_migration", None)
            if migration is not None and hasattr(migration, "summary"):
                try:
                    result["migration"] = migration.summary()
                except Exception:
                    pass

            # Funnel failure breakdown
            funnel_failures = getattr(demo, "_funnel_failures", None)
            if funnel_failures:
                result["funnel_failure_breakdown"] = dict(
                    sorted(funnel_failures.items(), key=lambda x: -x[1])
                )
                result["funnel_failure_total"] = sum(funnel_failures.values())

        except Exception:
            pass

        return result

    # ── Bulk Extraction ────────────────────────────────────────────────────────

    def extract_all(self) -> dict:
        """
        Convenience method: call every extractor and return results as a dict.

        Returns
        -------
        dict
            Keys are section names; values are the corresponding snapshot
            dataclass (or ``None`` when the underlying engine is not present).
        """
        return {
            "funnel": self.extract_funnel(),
            "tpi": self.extract_tpi(),
            "deployment_reality": self.extract_deployment_reality(),
            "frequency_reality": self.extract_frequency_reality(),
            "reality_convergence": self.extract_reality_convergence(),
            "dpl_validation": self.extract_dpl_validation(),
            "director_pipeline": self.extract_director_pipeline(),
            "rcl_dashboard": self.extract_rcl_dashboard(),
            "session_balance": self.extract_session_balance(),
            "shadow_system": self.extract_shadow_system(),
            "engine_dashboards": self.extract_engine_dashboards(),
        }
