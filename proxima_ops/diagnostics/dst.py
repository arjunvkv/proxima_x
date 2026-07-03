"""
DST — Decision Stagnation Thermodynamics

Model system as thermodynamic: Signal = energy, Gates = entropy, HOLD = equilibrium.

Reads from state/wave12_cycle_log.jsonl and computes thermodynamic-style metrics
to detect decision stagnation, metastable equilibria, and phase transitions.
"""

import json
from collections import deque


class DecisionStagnationThermo:
    """Thermodynamic model of decision stagnation."""

    def __init__(self, log_path: str = "state/wave12_cycle_log.jsonl", window: int = 20):
        self.log_path = log_path
        self.window = window

    # ------------------------------------------------------------------
    # Signal Energy (SE) — per cycle
    # ------------------------------------------------------------------
    def _signal_energy(self, row: dict, max_signals: float) -> float:
        """
        Compute Signal Energy for one cycle.

        Weighted components:
          40% — total_signals / max_signals_in_window  (0 if max=0)
          30% — best_signal confidence via active_confidence (0 if none)
          30% — confirm_count / 2, capped at 1.0
        """
        total_signals = row.get("total_signals", 0) or 0
        active_conf = row.get("active_confidence", 0) or 0.0
        confirm = row.get("confirm_cycles", 0) or 0

        c1 = (total_signals / max_signals) if max_signals > 0 else 0.0
        c1 = max(0.0, min(1.0, c1))

        c2 = float(active_conf)  # already 0..1 usually
        c2 = max(0.0, min(1.0, c2))

        c3 = min(1.0, confirm / 2.0)

        return 0.40 * c1 + 0.30 * c2 + 0.30 * c3

    # ------------------------------------------------------------------
    # Gate Entropy (GE) — per cycle
    # ------------------------------------------------------------------
    def _gate_entropy(self, row: dict) -> float:
        """
        Compute Gate Entropy for one cycle.

        Weighted components (25% each):
          1) mof_state uncertainty:
               INFORMATION_RICH -> 0.1
               STRUCTURE_LIMITED -> 0.5
               NOISE / anything else -> 0.9
          2) segl_state: ARMED -> 0.2, OBSERVE -> 0.8
          3) circuit_breaker: 0.9 if triggered via cb_decision, else 0.0
          4) VEL block rate: 0.5 default, 0.0 if vel_decision == "allowed"
        """
        # --- mof_state ---
        mof = str(row.get("mof_state", "")).upper()
        if "INFORMATION_RICH" in mof or "RICH" in mof:
            mof_val = 0.1
        elif "STRUCTURE_LIMITED" in mof or "LIMITED" in mof:
            mof_val = 0.5
        else:
            mof_val = 0.9  # NOISE or unknown

        # --- segl_state ---
        segl = str(row.get("segl_state", "")).upper()
        segl_val = 0.2 if "ARMED" in segl else 0.8  # OBSERVE default

        # --- circuit_breaker ---
        cb = str(row.get("cb_decision", "")).lower()
        cb_val = 0.9 if ("triggered" in cb or "denied cb" in cb) else 0.0

        # --- VEL block rate ---
        vel = str(row.get("vel_decision", "")).lower()
        vel_val = 0.0 if vel == "allowed" else 0.5

        return 0.25 * mof_val + 0.25 * segl_val + 0.25 * cb_val + 0.25 * vel_val

    # ------------------------------------------------------------------
    # Main analysis
    # ------------------------------------------------------------------
    def analyze(self, n_recent_cycles: int = 500) -> dict:
        """
        Read the cycle log and compute thermodynamic metrics.

        Parameters
        ----------
        n_recent_cycles : int
            How many of the most recent cycles to analyse (default 500).

        Returns
        -------
        dict with keys:
          energy_trajectory, entropy_trajectory, net_energy_trajectory,
          metastable_cycles, phase_transition_windows, execution_threshold,
          equilibrium_hold_rate, by_segl_state
        """
        try:
            rows = self._load_rows(n_recent_cycles)
        except Exception as exc:
            return {"error": f"Failed to load cycle log: {exc}"}

        if not rows:
            return {"error": "No cycle data found"}

        # ---- compute per-cycle metrics ----
        window_deque: deque = deque(maxlen=self.window)
        energy: dict[int, float] = {}
        entropy: dict[int, float] = {}
        net: dict[int, float] = {}
        segl_states: dict[int, str] = {}

        for row in rows:
            cycle = row.get("cycle")
            if cycle is None:
                continue
            # track rolling max of total_signals
            window_deque.append(row.get("total_signals", 0) or 0)
            max_sig = max(window_deque) if window_deque else 0.0

            se = self._signal_energy(row, float(max_sig))
            ge = self._gate_entropy(row)
            nt = max(-1.0, min(1.0, se - ge))

            energy[cycle] = round(se, 6)
            entropy[cycle] = round(ge, 6)
            net[cycle] = round(nt, 6)
            segl_states[cycle] = str(row.get("segl_state", "OBSERVE"))

        # ---- metastable cycles (|net| < 0.15) ----
        metastable_cycles = sum(1 for v in net.values() if abs(v) < 0.15)

        # ---- execution threshold E* ----
        # Default 0.3, or mean SE of cycles that had an execute intent
        exec_se = [
            energy[c]
            for c, s in segl_states.items()
            if s.upper() == "ARMED" and energy.get(c, 0) > 0
        ]
        execution_threshold = round(
            (sum(exec_se) / len(exec_se)) if exec_se else 0.3, 6
        )

        # ---- phase transition windows ----
        # net crosses from < E* to >= E* or back
        sorted_cycles = sorted(net.keys())
        phase_windows: list[dict] = []
        prev_below = True  # start "below" threshold
        window_start = None

        for i, c in enumerate(sorted_cycles):
            val = net.get(c, 0.0)
            below = val < execution_threshold
            if i == 0:
                prev_below = below
                continue
            # crossing detected
            if below != prev_below:
                if window_start is None:
                    window_start = sorted_cycles[i - 1]
            else:
                if window_start is not None:
                    phase_windows.append({
                        "start": window_start,
                        "end": sorted_cycles[i - 1],
                        "crossed": True,
                    })
                    window_start = None
            prev_below = below

        # close any open window
        if window_start is not None:
            phase_windows.append({
                "start": window_start,
                "end": sorted_cycles[-1],
                "crossed": True,
            })

        # ---- equilibrium hold rate (fraction of cycles with HOLD decision) ----
        hold_count = sum(
            1 for row in rows if str(row.get("decision", "")).upper() == "HOLD"
        )
        equilibrium_hold_rate = round(hold_count / len(rows), 6) if rows else 0.0

        # ---- by segl_state aggregates ----
        by_segl: dict[str, dict] = {}
        for c in sorted_cycles:
            s = segl_states[c]
            if s not in by_segl:
                by_segl[s] = {"energy": [], "entropy": [], "net": []}
            by_segl[s]["energy"].append(energy[c])
            by_segl[s]["entropy"].append(entropy[c])
            by_segl[s]["net"].append(net[c])

        by_segl_agg: dict[str, dict] = {}
        for state, vals in by_segl.items():
            by_segl_agg[state] = {
                "mean_energy": round(
                    sum(vals["energy"]) / len(vals["energy"]), 6
                ) if vals["energy"] else 0.0,
                "mean_entropy": round(
                    sum(vals["entropy"]) / len(vals["entropy"]), 6
                ) if vals["entropy"] else 0.0,
                "mean_net": round(
                    sum(vals["net"]) / len(vals["net"]), 6
                ) if vals["net"] else 0.0,
            }

        return {
            "energy_trajectory": {str(k): v for k, v in energy.items()},
            "entropy_trajectory": {str(k): v for k, v in entropy.items()},
            "net_energy_trajectory": {str(k): v for k, v in net.items()},
            "metastable_cycles": metastable_cycles,
            "phase_transition_windows": phase_windows,
            "execution_threshold": execution_threshold,
            "equilibrium_hold_rate": equilibrium_hold_rate,
            "by_segl_state": by_segl_agg,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_rows(self, n_recent: int) -> list[dict]:
        """Load at most *n_recent* rows from the JSONL log (newest first)."""
        rows: list[dict] = []
        with open(self.log_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # keep only the *n_recent* most recent entries
        return rows[-n_recent:] if len(rows) > n_recent else rows


# ------------------------------------------------------------------
# Quick CLI demo
# ------------------------------------------------------------------
if __name__ == "__main__":
    dst = DecisionStagnationThermo()
    result = dst.analyze(n_recent_cycles=500)
    print(json.dumps(result, indent=2, default=str))
