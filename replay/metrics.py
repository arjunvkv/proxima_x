import logging
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger("proxima.replay.metrics")


class ReplayMetrics:
    def __init__(self):
        self._signals: list[dict] = []
        self._trades: list[dict] = []
        self._tpi_states: list[dict] = []
        self._entropy_states: list[dict] = []
        self._freeze_ages: list[float] = []
        self._session_map: list[dict] = []
        self._spreads: list[dict] = []
        self._timestamps: list[float] = []

    def record_signal(self, signal: dict):
        self._signals.append(signal)

    def record_trade(self, trade: dict):
        self._trades.append(trade)

    def record_tpi(self, state: dict):
        self._tpi_states.append(state)

    def record_entropy(self, state: dict):
        self._entropy_states.append(state)

    def record_freeze_age(self, age: float):
        self._freeze_ages.append(age)

    def record_session(self, sym: str, session: str):
        self._session_map.append({"symbol": sym, "session": session, "time": datetime.now().isoformat()})

    def record_spread(self, sym: str, spread: float):
        self._spreads.append({"symbol": sym, "spread": spread, "time": datetime.now().isoformat()})

    def record_timestamp(self, ts: float):
        self._timestamps.append(ts)

    def parity_report(self, reference: "ReplayMetrics" = None) -> dict:
        report = {
            "n_signals": len(self._signals),
            "n_trades": len(self._trades),
            "n_tpi_states": len(self._tpi_states),
            "n_entropy_states": len(self._entropy_states),
            "signal_parity": None,
            "tpi_parity": None,
            "entropy_parity": None,
        }
        if reference:
            report["signal_parity"] = self._compare_signals(reference._signals)
            report["tpi_parity"] = len(self._tpi_states) == len(reference._tpi_states)
            report["entropy_parity"] = len(self._entropy_states) == len(reference._entropy_states)
            report["position_parity"] = len(self._trades) == len(reference._trades)
        return report

    def _compare_signals(self, other: list[dict]) -> dict:
        if len(self._signals) != len(other):
            return {"match": False, "reason": "count_mismatch"}
        matches = sum(1 for a, b in zip(self._signals, other) if a.get("direction") == b.get("direction"))
        return {"match": matches == len(self._signals), "matches": matches, "total": len(self._signals)}

    def summary(self) -> str:
        lines = ["REPLAY METRICS", "=" * 52]
        lines.append(f"  Signals:      {len(self._signals)}")
        lines.append(f"  Trades:       {len(self._trades)}")
        lines.append(f"  TPI States:   {len(self._tpi_states)}")
        lines.append(f"  Entropy:      {len(self._entropy_states)}")
        lines.append(f"  Spreads:      {len(self._spreads)}")
        if self._freeze_ages:
            lines.append(f"  Freeze Ages:  {min(self._freeze_ages):.1f}-{max(self._freeze_ages):.1f}")
        return "\n".join(lines)
