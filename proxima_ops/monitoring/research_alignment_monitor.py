"""
RESEARCH ALIGNMENT MONITOR — Phase 5 Deliverable

Tracks alignment between validated expected asset weights
and live execution distribution.

Alerts if alignment < 0.80.

Metrics:
- alignment_score (cosine similarity)
- concentration_score (HHI)
- asset_participation_score (fraction of expected assets present)
"""

import os
import math
from collections import defaultdict
from proxima_ops.config.settings import SETTINGS

CORE_ASSETS = {"EURJPY", "USDJPY", "GBPJPY", "XAUUSD", "EURUSD"}

# Build expected weights from SETTINGS.symbols (all tradable assets)
AAE_ASSETS = set(SETTINGS.symbols)
# Core assets get research-derived weights; others get equal small fallback
CORE_WEIGHTS = {
    "EURJPY": 183 / 904,
    "USDJPY": 182 / 904,
    "GBPJPY": 183 / 904,
    "XAUUSD": 176 / 904,
    "EURUSD": 180 / 904,
}
CORE_TOTAL = sum(CORE_WEIGHTS.values())
FALLBACK_WEIGHT = 0.001  # minimal weight for non-core symbols


class ResearchAlignmentMonitor:
    def __init__(self):
        self._history = []

    def compute(self, live_executions: dict) -> dict:
        """Compute alignment metrics.
        
        Args:
            live_executions: dict of {symbol: execution_count}
        """
        total_live = sum(live_executions.values()) or 1
        all_assets = sorted(set(list(AAE_ASSETS) + list(live_executions.keys())))

        # Cosine similarity
        expected_vec = []
        live_vec = []
        for a in all_assets:
            w = CORE_WEIGHTS.get(a, FALLBACK_WEIGHT)
            expected_vec.append(w)
            live_vec.append(live_executions.get(a, 0) / total_live)

        dot = sum(av * lv for av, lv in zip(expected_vec, live_vec))
        norm_exp = sum(av ** 2 for av in expected_vec) ** 0.5
        norm_live = sum(lv ** 2 for lv in live_vec) ** 0.5
        alignment = dot / (norm_exp * norm_live) if norm_exp * norm_live > 0 else 0.0

        # Asset participation: how many expected assets appear in live?
        live_assets = set(live_executions.keys())
        participation = len(CORE_ASSETS & live_assets) / len(CORE_ASSETS)

        # Concentration (HHI)
        hhi = sum((live_executions.get(a, 0) / total_live * 100) ** 2 for a in all_assets)

        result = {
            "alignment_score": round(alignment, 4),
            "asset_participation": round(participation, 4),
            "concentration_hhi": round(hhi, 1),
            "n_assets_live": len(live_assets),
            "n_assets_expected_but_missing": len(CORE_ASSETS - live_assets),
            "alert": alignment < 0.80,
            "live_assets": sorted(live_assets),
            "missing_assets": sorted(CORE_ASSETS - live_assets),
        }
        self._history.append(result)
        return result

    def dashboard_line(self, live_executions: dict) -> str:
        r = self.compute(live_executions)
        alert = " *** ALERT ***" if r["alert"] else ""
        return (f"Research Alignment: {r['alignment_score']:.2f} | "
                f"Target: {len(AAE_ASSETS)} assets | "
                f"Live: {r['n_assets_live']} assets | "
                f"Missing: {r['missing_assets']}{alert}")

    def summary(self) -> str:
        lines = []
        lines.append("Research Alignment Monitor")
        lines.append("=" * 40)
        lines.append(f"AAE target assets: {sorted(AAE_ASSETS)}")
        lines.append(f"Expected weights:")
        for sym, v in sorted(AAE_EXPECTED.items()):
            lines.append(f"  {sym}: {v['weight']:.3f} (sharpe={v['sharpe']:.3f})")
        if self._history:
            latest = self._history[-1]
            lines.append("")
            lines.append(f"Latest alignment score: {latest['alignment_score']:.4f}")
            lines.append(f"Latest asset participation: {latest['asset_participation']:.1%}")
            lines.append(f"Latest HHI: {latest['concentration_hhi']:.1f}")
            lines.append(f"Live assets: {latest['live_assets']}")
            lines.append(f"Missing: {latest['missing_assets']}")
            if latest['alert']:
                lines.append("*** ALIGNMENT ALERT ***")
        return "\n".join(lines)


def demo():
    monitor = ResearchAlignmentMonitor()
    # Test with current deployment (EURUSD only)
    r = monitor.compute({"EURUSD": 6})
    print("Test 1 — EURUSD only:")
    for k, v in r.items():
        print(f"  {k}: {v}")
    print()
    print(monitor.dashboard_line({"EURUSD": 6}))
    print()
    # Test with ideal deployment
    r2 = monitor.compute({"EURJPY": 30, "USDJPY": 25, "GBPJPY": 28, "XAUUSD": 20, "EURUSD": 10})
    print("Test 2 — Multi-asset:")
    for k, v in r2.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    demo()
