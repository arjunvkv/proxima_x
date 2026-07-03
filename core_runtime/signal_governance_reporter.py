"""
Signal Governance Reporter — produces human-readable reports about which
signal system controls execution, how often they agree, where divergence
causes loss, and what the system "believes" about the market overall.

Collects observations from ALL SAAL modules:
  - SignalAuthorityArbiter       (via feed_authority)
  - SignalConsensusModel         (via feed_consensus_result)
  - FinalExecutionDecisionEngine (via feed_execution_decision)
  - ExecutionPolicySwitcher      (via feed_policy_change)
  - SignalEconomicValueRanker    (via feed_economic_value)
  - AuthorityStabilityTracker    (via feed_stability)
"""

import logging
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

_MAXLEN = 10_000


class _SignalGovernanceReporter:
    """Store observations from SAAL modules and produce governance reports."""

    def __init__(self, instance_id: str = "default") -> None:
        self._instance_id = instance_id

        # --- sliding-window stores -------------------------------------------
        self._authority_obs: deque = deque(maxlen=_MAXLEN)       # (authority, confidence, ts)
        self._consensus_obs: deque = deque(maxlen=_MAXLEN)       # (oss_sig, alt_sig, consensus_sig, strategy)
        self._execution_obs: deque = deque(maxlen=_MAXLEN)       # (decision, signal, value, skip_reason)
        self._policy_obs: deque = deque(maxlen=_MAXLEN)          # (old_policy, new_policy, reason)
        self._economic_obs: deque = deque(maxlen=_MAXLEN)        # (oss_value, alt_value, best_source)
        self._stability_obs: deque = deque(maxlen=_MAXLEN)       # (verdict, volatility, flip_rate)

        # --- current (latest) state ------------------------------------------
        self._current_authority: Optional[str] = None
        self._current_confidence: float = 0.0
        self._current_policy: Optional[str] = None
        self._current_regime: Optional[str] = None
        self._current_stability_verdict: Optional[str] = None
        self._current_volatility: float = 0.0
        self._current_flip_rate: float = 0.0

        # cached latest consensus info for market-belief
        self._last_oss: Optional[int] = None
        self._last_alt: Optional[int] = None
        self._last_consensus: Optional[int] = None

        logger.info("SignalGovernanceReporter '%s' initialised (maxlen=%d)",
                     instance_id, _MAXLEN)

    # ------------------------------------------------------------------
    # Feed methods — called by the respective SAAL modules
    # ------------------------------------------------------------------

    def feed_authority(self, authority: str, confidence: float,
                       timestamp: Any = None) -> None:
        """Record a signal-authority verdict from SignalAuthorityArbiter."""
        self._authority_obs.append((authority, confidence, timestamp))
        self._current_authority = authority
        self._current_confidence = confidence
        logger.debug("feed_authority: %s (%.2f) @ %s", authority, confidence, timestamp)

    def feed_consensus_result(self, oss_sig: int, alt_sig: int,
                              consensus_sig: int, strategy: str) -> None:
        """Record a consensus result from SignalConsensusModel."""
        self._consensus_obs.append((oss_sig, alt_sig, consensus_sig, strategy))
        self._last_oss = oss_sig
        self._last_alt = alt_sig
        self._last_consensus = consensus_sig
        logger.debug("feed_consensus_result: OSS=%d ALT=%d consensus=%d (%s)",
                     oss_sig, alt_sig, consensus_sig, strategy)

    def feed_execution_decision(self, decision: str, signal: int,
                                value: float,
                                skip_reason: Optional[str] = None) -> None:
        """Record an execution decision from FinalExecutionDecisionEngine."""
        self._execution_obs.append((decision, signal, value, skip_reason))
        logger.debug("feed_execution_decision: %s sig=%d val=%.2f reason=%s",
                     decision, signal, value, skip_reason)

    def feed_policy_change(self, old_policy: Optional[str],
                           new_policy: str, reason: str) -> None:
        """Record a policy change from ExecutionPolicySwitcher."""
        self._policy_obs.append((old_policy, new_policy, reason))
        self._current_policy = new_policy
        logger.debug("feed_policy_change: %s -> %s because %s",
                     old_policy, new_policy, reason)

    def feed_economic_value(self, oss_value: float, alt_value: float,
                            best_source: str) -> None:
        """Record an economic-value ranking from SignalEconomicValueRanker."""
        self._economic_obs.append((oss_value, alt_value, best_source))
        logger.debug("feed_economic_value: OSS=%.4f ALT=%.4f best=%s",
                     oss_value, alt_value, best_source)

    def feed_stability(self, verdict: str, volatility: float,
                       flip_rate: float) -> None:
        """Record an authority-stability verdict from AuthorityStabilityTracker."""
        self._stability_obs.append((verdict, volatility, flip_rate))
        self._current_stability_verdict = verdict
        self._current_volatility = volatility
        self._current_flip_rate = flip_rate
        logger.debug("feed_stability: %s (vol=%.2f flip=%.2f)",
                     verdict, volatility, flip_rate)

    # ------------------------------------------------------------------
    # Helper breakdowns
    # ------------------------------------------------------------------

    def _agreement_pct(self) -> float:
        """Percentage of consensus observations where OSS == ALT."""
        n = len(self._consensus_obs)
        if n == 0:
            return 0.0
        agree = sum(1 for oss, alt, _, _ in self._consensus_obs if oss == alt)
        return round(agree / n * 100, 1)

    def _conviction_rate(self, source: str) -> float:
        """Fraction of non-zero signals for 'oss' or 'alt'."""
        n = len(self._consensus_obs)
        if n == 0:
            return 0.0
        idx = 0 if source == "oss" else 1
        non_zero = sum(1 for obs in self._consensus_obs if obs[idx] != 0)
        return round(non_zero / n * 100, 1)

    def _conflict_breakdown(self) -> Dict[str, int]:
        """Count conflict resolutions: oss_wins, alt_wins, no_trades."""
        oss_wins = 0
        alt_wins = 0
        no_trades = 0
        for oss, alt, consensus, _ in self._consensus_obs:
            if oss == alt:
                continue  # not a conflict
            if consensus == oss:
                oss_wins += 1
            elif consensus == alt:
                alt_wins += 1
            else:
                no_trades += 1
        return {"oss_wins": oss_wins, "alt_wins": alt_wins,
                "no_trades": no_trades}

    def _execution_breakdown(self) -> Tuple[int, Dict[str, int], int]:
        """Return (execute_count, skip_reasons_dict, total_decisions)."""
        execute = 0
        skip_reasons: Dict[str, int] = {}
        for decision, _signal, _value, reason in self._execution_obs:
            if decision == "execute":
                execute += 1
            else:
                key = reason or "unknown"
                skip_reasons[key] = skip_reasons.get(key, 0) + 1
        return execute, skip_reasons, len(self._execution_obs)

    def _conflict_count(self) -> int:
        """Number of consensus observations where OSS != ALT."""
        return sum(1 for oss, alt, _, _ in self._consensus_obs if oss != alt)

    def _estimate_divergence_loss(self) -> float:
        """Heuristic: estimate loss from divergence events.

        Each conflict reduced to no-trade costs 0.5 x average |value|,
        and each 'alt_wins' when oss was correct costs 1.0 x average |value|.
        """
        values = [v for _, _, v, _ in self._execution_obs if v != 0.0]
        avg_abs_value = sum(abs(v) for v in values) / len(values) if values else 1.0
        cb = self._conflict_breakdown()
        loss = 0.0
        loss += cb["no_trades"] * 0.5 * avg_abs_value
        # Assume alt_wins that went against OSS have a 20% chance of being
        # the wrong choice → heuristic penalty
        loss += cb["alt_wins"] * 0.20 * avg_abs_value
        return round(-loss, 2)

    def _system_belief(self) -> str:
        """Describe what the system believes about the market right now."""
        oss = self._last_oss
        alt = self._last_alt
        consensus = self._last_consensus

        if oss is None or alt is None or consensus is None:
            return "UNKNOWN (no data yet)"

        # Determine sentiment
        oss_label = {1: "BUY", -1: "SELL", 0: "NEUTRAL"}.get(oss, "?")
        alt_label = {1: "BUY", -1: "SELL", 0: "NEUTRAL"}.get(alt, "?")
        consensus_label = {1: "BUY", -1: "SELL", 0: "NEUTRAL"}.get(consensus, "?")

        # Build overall belief
        if oss == alt == consensus:
            base = f"STRONGLY {consensus_label}"
        elif consensus == oss and consensus != alt:
            base = f"CAUTIOUSLY {consensus_label}"
        elif consensus == alt and consensus != oss:
            base = f"ALT-LEANING {consensus_label}"
        elif consensus == 0 and oss != 0 and alt != 0:
            base = "NEUTRAL (conflict suspended)"
        elif consensus == 0:
            base = "NEUTRAL"
        else:
            base = consensus_label

        return (
            f"{base}\n"
            f"(OSS signals {oss_label}, ALT signals {alt_label}, "
            f"hybrid consensus = {consensus_label})"
        )

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_market_belief_report(self) -> str:
        """Produce a multi-line human-readable market-belief report."""
        total_obs = len(self._consensus_obs)
        agreement_pct = self._agreement_pct()
        oss_rate = self._conviction_rate("oss")
        alt_rate = self._conviction_rate("alt")

        cb = self._conflict_breakdown()
        conflicts = sum(cb.values())
        exec_count, skip_reasons, exec_total = self._execution_breakdown()
        loss_est = self._estimate_divergence_loss()
        belief = self._system_belief()

        # Format skip reasons
        skip_lines = []
        for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
            pct = round(count / exec_total * 100, 1) if exec_total else 0.0
            skip_lines.append(f"       - Skip ({reason}): {count} ({pct}%)")

        regime = self._current_regime or "UNKNOWN"
        stability = self._current_stability_verdict or "UNKNOWN"
        vol_str = f"volatility: {self._current_volatility:.1f} changes/100 decisions" if self._current_volatility else "N/A"

        lines = [
            "=" * 48,
            "MARKET BELIEF REPORT",
            "=" * 48,
            f"Current Authority: {self._current_authority or 'UNKNOWN'} "
            f"(confidence: {self._current_confidence:.2f})",
            f"Active Policy: {self._current_policy or 'UNKNOWN'}",
            "",
            f"Signal Agreement: {agreement_pct}% "
            f"(over {total_obs} decisions)" if total_obs else "Signal Agreement: N/A (no data)",
            f"OSS conviction rate: {oss_rate}%",
            f"ALT conviction rate: {alt_rate}%",
            "",
            "Divergence Impact:",
            f"   - Conflicts resolved: {conflicts}",
            f"   - OSS wins: {cb['oss_wins']} "
            f"({round(cb['oss_wins'] / conflicts * 100, 1) if conflicts else 0}%)",
            f"   - ALT wins: {cb['alt_wins']} "
            f"({round(cb['alt_wins'] / conflicts * 100, 1) if conflicts else 0}%)",
            f"   - Estimated loss from divergence: ${loss_est}",
            "",
            f"Regime Context: {regime}",
            "Execution Stats:",
            f"   - Total decisions: {exec_total}",
            f"   - Execute: {exec_count} "
            f"({round(exec_count / exec_total * 100, 1) if exec_total else 0}%)",
        ]
        lines.extend(skip_lines)

        lines.extend([
            "",
            f"Stability: {stability} ({vol_str})",
            "",
            f"System Belief: {belief}",
            "=" * 48,
        ])

        return "\n".join(lines)

    def generate_governance_summary(self) -> Dict[str, Any]:
        """Return a structured dict summary of governance state."""
        total_obs = len(self._consensus_obs)
        agreement_pct = self._agreement_pct()
        cb = self._conflict_breakdown()
        exec_count, skip_reasons, exec_total = self._execution_breakdown()

        return {
            "current_authority": self._current_authority,
            "authority_confidence": self._current_confidence,
            "active_policy": self._current_policy,
            "total_observations": total_obs,
            "oss_alt_agreement_pct": agreement_pct,
            "conflict_breakdown": {
                "oss_wins": cb["oss_wins"],
                "alt_wins": cb["alt_wins"],
                "no_trades": cb["no_trades"],
            },
            "execution_breakdown": {
                "execute": exec_count,
                "skip_reasons": dict(skip_reasons),
            },
            "stability_verdict": self._current_stability_verdict,
            "market_belief": self._system_belief(),
            "regime_context": self._current_regime,
        }

    def generate_divergence_report(self) -> str:
        """Detailed analysis of where OSS and ALT diverge and estimated impact."""
        total_obs = len(self._consensus_obs)
        agreements = sum(1 for oss, alt, _, _ in self._consensus_obs if oss == alt)
        conflicts = total_obs - agreements
        cb = self._conflict_breakdown()
        loss_est = self._estimate_divergence_loss()

        # Per-strategy divergence
        strategy_div: Dict[str, Dict[str, int]] = {}
        for oss, alt, consensus, strategy in self._consensus_obs:
            if oss == alt:
                continue
            if strategy not in strategy_div:
                strategy_div[strategy] = {"conflicts": 0, "oss_wins": 0,
                                          "alt_wins": 0, "no_trades": 0}
            strategy_div[strategy]["conflicts"] += 1
            if consensus == oss:
                strategy_div[strategy]["oss_wins"] += 1
            elif consensus == alt:
                strategy_div[strategy]["alt_wins"] += 1
            else:
                strategy_div[strategy]["no_trades"] += 1

        # Per-signal-pair divergence patterns
        pattern_counts: Dict[str, int] = {}
        for oss, alt, consensus, _ in self._consensus_obs:
            if oss == alt:
                continue
            pattern = f"OSS={oss} ALT={alt} → consensus={consensus}"
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        lines = [
            "=" * 48,
            "DIVERGENCE REPORT",
            "=" * 48,
            f"Total observations: {total_obs}",
            f"Agreements: {agreements}",
            f"Conflicts: {conflicts}",
            f"Conflict rate: {round(conflicts / total_obs * 100, 1) if total_obs else 0}%",
            "",
            "Conflict Resolution:",
            f"   OSS wins: {cb['oss_wins']}",
            f"   ALT wins: {cb['alt_wins']}",
            f"   No-trade outcomes: {cb['no_trades']}",
            "",
            f"Estimated divergence loss: ${loss_est}",
            "",
            "Divergence by Strategy:",
        ]

        if strategy_div:
            for strat, counts in sorted(strategy_div.items()):
                lines.append(
                    f"   {strat}: {counts['conflicts']} conflicts "
                    f"(OSS wins={counts['oss_wins']}, "
                    f"ALT wins={counts['alt_wins']}, "
                    f"no-trade={counts['no_trades']})"
                )
        else:
            lines.append("   (no conflicting observations recorded)")

        lines.extend([
            "",
            "Top Divergence Patterns:",
        ])

        if pattern_counts:
            for pattern, cnt in sorted(pattern_counts.items(),
                                       key=lambda x: -x[1])[:10]:
                lines.append(f"   {pattern}: {cnt} occurrences")
        else:
            lines.append("   (none)")

        lines.append("=" * 48)
        return "\n".join(lines)

    def generate_system_identity_report(self) -> str:
        """Describe what the system 'believes' about the market right now."""
        belief = self._system_belief()
        oss_rate = self._conviction_rate("oss")
        alt_rate = self._conviction_rate("alt")
        cb = self._conflict_breakdown()
        conflicts = sum(cb.values())
        agreement_pct = self._agreement_pct()

        # Determine identity labels
        if conflicts == 0:
            conflict_style = "HARMONIOUS"
        elif cb["oss_wins"] > cb["alt_wins"] * 1.5:
            conflict_style = "OSS-DOMINANT"
        elif cb["alt_wins"] > cb["oss_wins"] * 1.5:
            conflict_style = "ALT-DOMINANT"
        else:
            conflict_style = "BALANCED"

        if agreement_pct >= 80:
            trust_label = "HIGH-TRUST"
        elif agreement_pct >= 50:
            trust_label = "MODERATE-TRUST"
        else:
            trust_label = "LOW-TRUST"

        oss_personality = "DECISIVE" if oss_rate >= 50 else "CAUTIOUS"
        alt_personality = "DECISIVE" if alt_rate >= 50 else "CAUTIOUS"

        lines = [
            "=" * 48,
            "SYSTEM IDENTITY REPORT",
            "=" * 48,
            f"Current Authority: {self._current_authority or 'UNKNOWN'}",
            f"Active Policy: {self._current_policy or 'UNKNOWN'}",
            "",
            "System Personality:",
            f"   Trust Profile: {trust_label} "
            f"(agreement={agreement_pct}%)",
            f"   Conflict Style: {conflict_style} "
            f"({cb['oss_wins']} OSS / {cb['alt_wins']} ALT / "
            f"{cb['no_trades']} no-trade)",
            f"   OSS Personality: {oss_personality} (conviction={oss_rate}%)",
            f"   ALT Personality: {alt_personality} (conviction={alt_rate}%)",
            "",
            "Market Belief:",
            f"   {belief}",
            "",
            "Stability:",
            f"   Verdict: {self._current_stability_verdict or 'UNKNOWN'}",
            f"   Volatility: {self._current_volatility:.2f}",
            f"   Flip rate: {self._current_flip_rate:.2f}",
            "",
            "Economic Value Preference:",
        ]

        # Last few economic observations
        eco_list = list(self._economic_obs)
        if eco_list:
            oss_vals = [e[0] for e in eco_list[-20:]]
            alt_vals = [e[1] for e in eco_list[-20:]]
            best_counts: Dict[str, int] = {}
            for _, _, best in eco_list[-100:]:
                best_counts[best] = best_counts.get(best, 0) + 1
            lines.append(
                f"   OSS avg value (last 20): "
                f"{round(sum(oss_vals) / len(oss_vals), 4) if oss_vals else 0}"
            )
            lines.append(
                f"   ALT avg value (last 20): "
                f"{round(sum(alt_vals) / len(alt_vals), 4) if alt_vals else 0}"
            )
            best_str = ", ".join(
                f"{src}={cnt}" for src, cnt in
                sorted(best_counts.items(), key=lambda x: -x[1])
            )
            lines.append(f"   Best-source distribution (last 100): {best_str}")
        else:
            lines.append("   (no economic data recorded)")

        lines.append("=" * 48)
        return "\n".join(lines)

    def set_regime_context(self, regime: str) -> None:
        """Allow external callers to set the current market regime."""
        self._current_regime = regime
        logger.debug("Regime context set to '%s'", regime)

    def reset(self) -> None:
        """Clear all stored observations (keeps current-state fields)."""
        self._authority_obs.clear()
        self._consensus_obs.clear()
        self._execution_obs.clear()
        self._policy_obs.clear()
        self._economic_obs.clear()
        self._stability_obs.clear()
        logger.info("SignalGovernanceReporter '%s' reset", self._instance_id)


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------

_instances: Dict[str, _SignalGovernanceReporter] = {}


def SignalGovernanceReporter(instance_id: str = "default") -> _SignalGovernanceReporter:
    """Factory that returns a shared _SignalGovernanceReporter singleton per instance_id."""
    if instance_id not in _instances:
        _instances[instance_id] = _SignalGovernanceReporter(instance_id)
    return _instances[instance_id]


# ===========================================================================
# Self-test
# ===========================================================================

def _self_test() -> None:
    """Feed synthetic data and exercise all report types."""
    print("=" * 60)
    print("SignalGovernanceReporter :: Self-Test")
    print("=" * 60)

    reporter = SignalGovernanceReporter("self_test")

    # --- Phase 1: feed authority ---
    print("\n--- Feeding authority observations ---")
    for i in range(200):
        if i < 100:
            reporter.feed_authority("OSS", 0.82 + (i * 0.0001), timestamp=i)
        else:
            reporter.feed_authority("OSS", 0.78 - (i * 0.0001), timestamp=i)

    # --- Phase 2: feed consensus results ---
    print("--- Feeding consensus results ---")
    import random
    random.seed(42)
    for i in range(1000):
        # Mostly agree, some divergence
        if i < 600:
            oss = random.choice([-1, 0, 1])
            alt = oss  # agree
        elif i < 850:
            oss = random.choice([-1, 1])
            alt = -oss  # conflict
        else:
            oss = 0
            alt = random.choice([-1, 0, 1])

        strategy = random.choice(["MAJORITY", "OSS_PRIORITY", "ALT_PRIORITY",
                                  "CONFIDENCE_WEIGHTED", "SPLIT"])
        if oss == alt:
            consensus = oss
        elif strategy == "OSS_PRIORITY":
            consensus = oss
        elif strategy == "ALT_PRIORITY":
            consensus = alt
        elif oss == 0:
            consensus = alt
        elif alt == 0:
            consensus = oss
        else:
            consensus = random.choice([oss, alt, 0])

        reporter.feed_consensus_result(oss, alt, consensus, strategy)

    # --- Phase 3: feed execution decisions ---
    print("--- Feeding execution decisions ---")
    for i in range(1000):
        r = random.random()
        if r < 0.12:
            decision = "execute"
            signal = random.choice([-1, 1])
            value = random.uniform(-5, 10)
            reason = None
        elif r < 0.72:
            decision = "skip"
            signal = 0
            value = 0.0
            reason = "no_signal"
        elif r < 0.92:
            decision = "skip"
            signal = random.choice([-1, 0, 1])
            value = random.uniform(-3, -0.5)
            reason = "negative_ev"
        elif r < 0.97:
            decision = "skip"
            signal = random.choice([-1, 0, 1])
            value = 0.0
            reason = "disabled"
        else:
            decision = "skip"
            signal = random.choice([-1, 0, 1])
            value = 0.0
            reason = "unstable"
        reporter.feed_execution_decision(decision, signal, value, reason)

    # --- Phase 4: feed policy changes ---
    print("--- Feeding policy changes ---")
    policies = [
        ("MAJORITY", "OSS_PRIORITY", "authority_shift"),
        ("OSS_PRIORITY", "CONFIDENCE_WEIGHTED", "volatility_increase"),
        ("CONFIDENCE_WEIGHTED", "HYBRID", "regime_change"),
        ("HYBRID", "ALT_PRIORITY", "alt_confidence_surge"),
        ("ALT_PRIORITY", "HYBRID", "rebalance"),
    ]
    for old, new, reason in policies:
        reporter.feed_policy_change(old, new, reason)

    # --- Phase 5: feed economic values ---
    print("--- Feeding economic values ---")
    for i in range(200):
        oss_v = random.uniform(-0.5, 2.0)
        alt_v = random.uniform(-0.3, 1.8)
        best = "OSS" if oss_v >= alt_v else "ALT"
        reporter.feed_economic_value(oss_v, alt_v, best)

    # --- Phase 6: feed stability ---
    print("--- Feeding stability observations ---")
    reporter.feed_stability("STABLE", 3.0, 0.05)
    reporter.feed_stability("STABLE", 2.8, 0.04)
    reporter.feed_stability("WARNING", 8.0, 0.15)

    # --- Set regime ---
    reporter.set_regime_context("TRENDING")

    # --- Phase 7: generate reports ---
    print("\n" + "=" * 60)
    print("GENERATING MARKET BELIEF REPORT")
    print("=" * 60)
    market_report = reporter.generate_market_belief_report()
    print(market_report)

    print("\n" + "=" * 60)
    print("GENERATING GOVERNANCE SUMMARY (dict)")
    print("=" * 60)
    summary = reporter.generate_governance_summary()
    for k, v in summary.items():
        print(f"  {k:30s} = {v}")

    print("\n" + "=" * 60)
    print("GENERATING DIVERGENCE REPORT")
    print("=" * 60)
    div_report = reporter.generate_divergence_report()
    print(div_report)

    print("\n" + "=" * 60)
    print("GENERATING SYSTEM IDENTITY REPORT")
    print("=" * 60)
    identity_report = reporter.generate_system_identity_report()
    print(identity_report)

    # --- Phase 8: verify structured data ---
    print("\n--- Structured-data assertions ---")
    gs = reporter.generate_governance_summary()
    assert isinstance(gs, dict), "governance_summary must be dict"
    assert "current_authority" in gs
    assert "authority_confidence" in gs
    assert "active_policy" in gs
    assert "total_observations" in gs
    assert "oss_alt_agreement_pct" in gs
    assert "conflict_breakdown" in gs
    assert "execution_breakdown" in gs
    assert "stability_verdict" in gs
    assert "market_belief" in gs
    assert "regime_context" in gs
    assert "oss_wins" in gs["conflict_breakdown"]
    assert "alt_wins" in gs["conflict_breakdown"]
    assert "no_trades" in gs["conflict_breakdown"]
    assert "execute" in gs["execution_breakdown"]
    assert "skip_reasons" in gs["execution_breakdown"]
    print("  All structured-data keys present ✓")

    # --- Phase 9: singleton test ---
    print("\n--- Singleton test ---")
    a = SignalGovernanceReporter("self_test_singleton")
    b = SignalGovernanceReporter("self_test_singleton")
    assert a is b
    c = SignalGovernanceReporter("self_test_other")
    assert c is not a
    print("  Singleton pattern verified ✓")

    # --- Phase 10: reset test ---
    print("\n--- Reset test ---")
    reporter.reset()
    assert len(reporter._authority_obs) == 0
    assert len(reporter._consensus_obs) == 0
    assert len(reporter._execution_obs) == 0
    assert len(reporter._policy_obs) == 0
    assert len(reporter._economic_obs) == 0
    assert len(reporter._stability_obs) == 0
    # current state fields should survive reset
    assert reporter._current_authority is not None
    print("  reset() clears deques ✓")

    print("\n" + "=" * 60)
    print("All self-tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    _self_test()
