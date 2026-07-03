"""
regime_trade_alignment.py — Map RSI / ATR / SIL regime to actual trade
occurrence probability to identify best trade zones and dead zones.

Reads:
  - state/wave12_cycle_log.jsonl     (cycle records with decision, active_symbol)
  - state/live_pipeline_trace.jsonl  (RSI / ATR per symbol via md_summary)

Output:
  rsi_band_success_rate, atr_band_success_rate, best_trade_zone,
  dead_zones, data_quality_warning
"""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from typing import Any

logger = logging.getLogger("proxima_ops.analytics.regime_trade_alignment")

DEFAULT_CYCLE_LOG = "state/wave12_cycle_log.jsonl"
DEFAULT_PIPELINE_TRACE_LOG = "state/live_pipeline_trace.jsonl"

# RSI band definitions: (label, lower_incl, upper_excl)
RSI_BANDS: list[tuple[str, float, float]] = [
    ("under_20",    0.0,  20.0),
    ("20_to_30",   20.0,  30.0),
    ("30_to_40",   30.0,  40.0),
    ("40_to_60",   40.0,  60.0),
    ("60_to_70",   60.0,  70.0),
    ("70_to_80",   70.0,  80.0),
    ("over_80",    80.0,  float("inf")),
]

# ATR percentile thresholds for banding
ATR_LOW_THRESHOLD: float = 0.3   # below this -> low_volatility
ATR_HIGH_THRESHOLD: float = 0.7  # above this -> high_volatility

# Decisions that count as a "trade" (opposite of HOLD / SKIP)
TRADE_DECISIONS: set[str] = {
    "BUY", "SELL", "ENTER", "EXIT", "CLOSE",
    "LONG", "SHORT", "MODIFY",
}

ZONE_NAME_MAP: dict[str, str] = {
    "under_20": "RSI <20",
    "20_to_30": "RSI 20-30",
    "30_to_40": "RSI 30-40",
    "40_to_60": "RSI 40-60",
    "60_to_70": "RSI 60-70",
    "70_to_80": "RSI 70-80",
    "over_80": "RSI >80",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _classify_rsi(value: float) -> str:
    """Return the RSI band label for a given RSI value."""
    for label, lo, hi in RSI_BANDS:
        if lo <= value < hi:
            return label
    return "over_80"  # fallback for exactly 80.0 — should not happen


def _classify_atr_percentile(pctile: float) -> str:
    """Classify an ATR percentile rank into a volatility band."""
    if pctile < ATR_LOW_THRESHOLD:
        return "low_volatility"
    if pctile < ATR_HIGH_THRESHOLD:
        return "medium_volatility"
    return "high_volatility"


def _is_trade(decision: str) -> bool:
    """Return True if the decision represents an actual trade."""
    return decision in TRADE_DECISIONS


# ===================================================================
# Main class
# ===================================================================


class RegimeTradeAlignment:
    """Map regime bands to trade-occurrence probability.

    Parameters
    ----------
    log_path : str
        Path to the cycle log (JSONL).
    pipeline_trace_path : str
        Path to the pipeline trace log (JSONL) that contains md_summary
        with per-symbol RSI / ATR data.
    """

    def __init__(
        self,
        log_path: str = DEFAULT_CYCLE_LOG,
        pipeline_trace_path: str = DEFAULT_PIPELINE_TRACE_LOG,
    ) -> None:
        self._log_path = log_path
        self._pipeline_trace_path = pipeline_trace_path

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, n_recent_cycles: int = 500) -> dict[str, Any]:
        """Analyse the trace logs and return a regime-trade alignment report.

        Parameters
        ----------
        n_recent_cycles : int
            Only consider this many of the most recent cycles (default 500).
            Pass a large number (e.g. 1_000_000) to analyse all available.

        Returns
        -------
        dict with keys:

            rsi_band_success_rate : dict
                Per RSI band: {"occurrences": int, "trades": int, "rate": float}
            atr_band_success_rate : dict
                Per ATR volatility band: {"occurrences": int, "trades": int, "rate": float}
            best_trade_zone : str
                Human-readable description of the single best RSI band.
            dead_zones : list[str]
                Bands with trade rate < 1 %.
            data_quality_warning : str | None
                Warning if insufficient data is available.
        """
        try:
            return self._build_report(n_recent_cycles)
        except Exception:
            logger.exception("RegimeTradeAlignment.analyze failed")
            return {
                "rsi_band_success_rate": {},
                "atr_band_success_rate": {},
                "best_trade_zone": "unknown",
                "dead_zones": [],
                "data_quality_warning": "Error during analysis — see logs",
            }

    # ------------------------------------------------------------------
    # Internal: report builder
    # ------------------------------------------------------------------

    def _build_report(self, n_recent_cycles: int) -> dict[str, Any]:
        cycle_records = self._load_cycle_records()
        pipeline_trace = self._load_pipeline_trace()

        if not cycle_records:
            return self._empty_report("no cycle records found in log")

        # Build a lookup: cycle_num -> {symbol: {"rsi": ..., "atr": ...}}
        md_by_cycle: dict[int, dict[str, dict[str, float]]] = {}
        for pt in pipeline_trace:
            cyc = pt.get("cycle")
            md = pt.get("md_summary") or {}
            if cyc is not None:
                md_by_cycle[cyc] = md

        # Slice to most recent N cycles
        if n_recent_cycles <= 0:
            return self._empty_report(
                f"n_recent_cycles={n_recent_cycles} is not positive"
            )
        if n_recent_cycles < len(cycle_records):
            cycle_records = cycle_records[-n_recent_cycles:]

        # --- First pass: collect RSI + raw ATR values ---
        # Store (cycle_num, symbol, is_trade, rsi_val, atr_val) for second pass
        aligned: list[dict[str, Any]] = []
        atr_values_by_symbol: dict[str, list[float]] = defaultdict(list)

        for entry in cycle_records:
            cycle_num: int | None = entry.get("cycle")
            active_symbol: str = entry.get("active_symbol") or ""
            decision: str = entry.get("decision", "HOLD")
            trade_flag: bool = _is_trade(decision)

            md = md_by_cycle.get(cycle_num, {}) if cycle_num is not None else {}
            symbol_md = md.get(active_symbol, {})
            if not isinstance(symbol_md, dict):
                symbol_md = {}

            rsi_val: float | None = symbol_md.get("rsi")
            atr_val: float | None = symbol_md.get("atr")

            aligned.append({
                "cycle_num": cycle_num,
                "symbol": active_symbol,
                "is_trade": trade_flag,
                "rsi": rsi_val if isinstance(rsi_val, (int, float)) else None,
                "atr": atr_val if isinstance(atr_val, (int, float)) else None,
            })

            # Accumulate ATR values per-symbol for percentile computation
            if atr_val is not None and isinstance(atr_val, (int, float)) and active_symbol:
                atr_values_by_symbol[active_symbol].append(atr_val)

        # --- Compute per-symbol ATR percentile ranks ---
        # For each symbol, map ATR value -> percentile rank (0..1)
        # Identical values get the same rank (midpoint of their index range).
        atr_percentile_lookup: dict[str, dict[float, float]] = {}
        for sym, vals in atr_values_by_symbol.items():
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            lookup: dict[float, float] = {}
            i = 0
            while i < n:
                # Find the run of identical values
                j = i
                while j < n and sorted_vals[j] == sorted_vals[i]:
                    j += 1
                # Midpoint percentile for this group: (i+1 + j) / 2 / n
                pct = ((i + 1) + j) / 2.0 / n
                for k in range(i, j):
                    lookup[sorted_vals[k]] = pct
                i = j
            atr_percentile_lookup[sym] = lookup

        # --- Second pass: classify into bands ---
        rsi_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"occurrences": 0, "trades": 0}
        )
        atr_stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"occurrences": 0, "trades": 0}
        )

        rsi_cycles_used = 0
        atr_cycles_used = 0

        for row in aligned:
            # --- RSI band ---
            rsi_val = row["rsi"]
            if rsi_val is not None:
                band = _classify_rsi(rsi_val)
                rsi_stats[band]["occurrences"] += 1
                if row["is_trade"]:
                    rsi_stats[band]["trades"] += 1
                rsi_cycles_used += 1

            # --- ATR band (via per-symbol percentile) ---
            atr_val = row["atr"]
            sym = row["symbol"]
            if atr_val is not None and sym in atr_percentile_lookup:
                pctile = atr_percentile_lookup[sym].get(atr_val)
                if pctile is not None:
                    band = _classify_atr_percentile(pctile)
                    atr_stats[band]["occurrences"] += 1
                    if row["is_trade"]:
                        atr_stats[band]["trades"] += 1
                    atr_cycles_used += 1

        # --- Compute rates ---
        rsi_band_rate: dict[str, dict[str, int | float]] = {}
        for label, _lo, _hi in RSI_BANDS:
            s = rsi_stats.get(label, {"occurrences": 0, "trades": 0})
            occ = s["occurrences"]
            trd = s["trades"]
            rate = round(trd / occ, 4) if occ > 0 else 0.0
            rsi_band_rate[label] = {
                "occurrences": occ,
                "trades": trd,
                "rate": rate,
            }

        atr_band_labels = ["low_volatility", "medium_volatility", "high_volatility"]
        atr_band_rate: dict[str, dict[str, int | float]] = {}
        for label in atr_band_labels:
            s = atr_stats.get(label, {"occurrences": 0, "trades": 0})
            occ = s["occurrences"]
            trd = s["trades"]
            rate = round(trd / occ, 4) if occ > 0 else 0.0
            atr_band_rate[label] = {
                "occurrences": occ,
                "trades": trd,
                "rate": rate,
            }

        # --- Determine best trade zone & dead zones ---
        data_quality_warning: str | None = None

        if rsi_cycles_used < 10:
            data_quality_warning = (
                f"Insufficient RSI data: only {rsi_cycles_used} cycles have "
                f"RSI values (need >=10 for meaningful analysis)."
            )
        elif atr_cycles_used < 10:
            data_quality_warning = (
                f"Insufficient ATR data: only {atr_cycles_used} cycles have "
                f"ATR values (need >=10 for meaningful analysis)."
            )

        # Best trade zone = RSI band with the highest trade rate
        best_zone = "unknown"
        best_rate = -1.0
        best_band = ""
        for label, data in rsi_band_rate.items():
            occ: int = data["occurrences"]  # type: ignore[assignment]
            rate: float = data["rate"]  # type: ignore[assignment]
            if rate > best_rate and occ > 0:
                best_rate = rate
                best_band = label

        if best_band in ZONE_NAME_MAP:
            best_zone = ZONE_NAME_MAP[best_band]

        # Dead zones = bands with occurrence-adjusted rate < 0.01 (below 1 %)
        dead_zones: list[str] = []
        for label, data in rsi_band_rate.items():
            occ = data["occurrences"]  # type: ignore[assignment]
            rate = data["rate"]  # type: ignore[assignment]
            if occ > 0 and rate < 0.01:
                dead_zones.append(ZONE_NAME_MAP.get(label, label))
        for label, data in atr_band_rate.items():
            occ = data["occurrences"]  # type: ignore[assignment]
            rate = data["rate"]  # type: ignore[assignment]
            if occ > 0 and rate < 0.01:
                dead_zones.append(label)

        if not dead_zones and rsi_cycles_used > 0:
            dead_zones.append("none detected (all bands have >=1% trade rate)")

        return {
            "rsi_band_success_rate": rsi_band_rate,
            "atr_band_success_rate": atr_band_rate,
            "best_trade_zone": best_zone,
            "dead_zones": dead_zones,
            "data_quality_warning": data_quality_warning,
        }

    # ------------------------------------------------------------------
    # Internal: empty report
    # ------------------------------------------------------------------

    def _empty_report(self, reason: str) -> dict[str, Any]:
        logger.warning("RegimeTradeAlignment: %s", reason)
        return {
            "rsi_band_success_rate": {},
            "atr_band_success_rate": {},
            "best_trade_zone": "unknown",
            "dead_zones": [],
            "data_quality_warning": reason,
        }

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_cycle_records(self) -> list[dict[str, Any]]:
        return self._load_jsonl(self._log_path)

    def _load_pipeline_trace(self) -> list[dict[str, Any]]:
        return self._load_jsonl(self._pipeline_trace_path)

    @staticmethod
    def _load_jsonl(path: str) -> list[dict[str, Any]]:
        """Load all JSON objects from a newline-delimited JSON file."""
        if not os.path.exists(path):
            logger.warning("File not found: %s", path)
            return []

        records: list[dict[str, Any]] = []
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.debug("Skipping unparseable line in %s", path)
        except Exception:
            logger.exception("Failed to read %s", path)
            return []

        return records


# ===================================================================
# CLI convenience
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    import sys

    n = 500
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print("Usage: python regime_trade_alignment.py [n_recent_cycles]")
            sys.exit(1)

    runner = RegimeTradeAlignment()
    report = runner.analyze(n_recent_cycles=n)
    print(json.dumps(report, indent=2))
