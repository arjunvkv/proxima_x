import logging
from datetime import datetime, date

logger = logging.getLogger("proxima_ops.ard.evidence")


class EvidenceCollector:
    def __init__(self):
        self._evidence: list[dict] = []

    def collect_freq_reality(self, blocked_count: int, executed_count: int,
                              leakage_rate: float, adr: float) -> dict:
        rec = {
            "source": "freq_reality", "timestamp": datetime.now().isoformat(),
            "blocked_signals": blocked_count, "executed_signals": executed_count,
            "leakage_rate": leakage_rate, "adr": adr}
        self._evidence.append(rec)
        return rec

    def collect_drl(self, asr: float, exec_quality: str, mean_slippage: float,
                    latency_ms: float, n_trades: int) -> dict:
        rec = {
            "source": "drl", "timestamp": datetime.now().isoformat(),
            "asr": asr, "execution_quality": exec_quality,
            "mean_slippage": mean_slippage, "latency_ms": latency_ms,
            "n_trades": n_trades}
        self._evidence.append(rec)
        return rec

    def collect_rce(self, ate: float, health_index: float,
                     divergence_score: float, friction_index: float,
                     rce_classification: str) -> dict:
        rec = {
            "source": "rce", "timestamp": datetime.now().isoformat(),
            "ate": ate, "health_index": health_index,
            "divergence_score": divergence_score,
            "friction_index": friction_index,
            "rce_classification": rce_classification}
        self._evidence.append(rec)
        return rec

    def collect_live(self, n_trades: int, sharpe: float, pp: float,
                      today_pnl: float, score: float,
                      score_classification: str) -> dict:
        rec = {
            "source": "live", "timestamp": datetime.now().isoformat(),
            "n_trades": n_trades, "sharpe": sharpe, "pp": pp,
            "today_pnl": today_pnl, "score": score,
            "score_classification": score_classification}
        self._evidence.append(rec)
        return rec

    def collect_research(self, expected_sharpe: float, expected_pp: float,
                          expected_frequency: int) -> dict:
        rec = {
            "source": "research", "timestamp": datetime.now().isoformat(),
            "expected_sharpe": expected_sharpe, "expected_pp": expected_pp,
            "expected_frequency": expected_frequency}
        self._evidence.append(rec)
        return rec

    def recent(self, source: str = None, n: int = 5) -> list[dict]:
        filtered = self._evidence
        if source:
            filtered = [e for e in self._evidence if e.get("source") == source]
        return filtered[-n:]

    def all_evidence(self) -> list[dict]:
        return list(self._evidence)

    def summary(self) -> dict:
        if not self._evidence:
            return {"total_records": 0}
        sources = {}
        for e in self._evidence:
            s = e.get("source", "unknown")
            sources[s] = sources.get(s, 0) + 1
        return {"total_records": len(self._evidence), "sources": sources,
                "latest": self._evidence[-1] if self._evidence else None}
