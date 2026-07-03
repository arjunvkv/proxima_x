class AnomalyDetector:
    def __init__(self, perf_monitor=None, mt5_monitor=None, drl_module=None, freq_reality=None):
        self._perf = perf_monitor
        self._mt5 = mt5_monitor
        self._drl = drl_module
        self._freq = freq_reality

    def check(self) -> dict:
        anomalies = []
        severity = "NORMAL"

        if self._perf:
            dd = self._perf.get("max_drawdown_pct", 0) if isinstance(self._perf, dict) else 0
            if dd > 0.10:
                anomalies.append({"type": "HIGH_DRAWDOWN", "detail": f"DD={dd:.2%}", "severity": "CRITICAL"})

        freq = self._freq.check() if hasattr(self._freq, 'check') else {}
        total = freq.get("total_signals", 0) if isinstance(freq, dict) else 0
        if total == 0:
            anomalies.append({"type": "SIGNAL_COLLAPSE", "detail": "Zero signals", "severity": "CRITICAL"})

        if self._mt5:
            mt5_status = self._mt5.check() if hasattr(self._mt5, 'check') else {}
            symbols = mt5_status.get("symbols", {}) if isinstance(mt5_status, dict) else {}
            for sym, info in symbols.items():
                sp = info.get("spread", 0) if isinstance(info, dict) else 0
                if sp >= 500:
                    anomalies.append({"type": "SPREAD_EXPLOSION", "detail": f"{sym}={sp}", "severity": "WARNING"})

        if anomalies:
            severities = [a.get("severity", "NORMAL") for a in anomalies]
            if "CRITICAL" in severities:
                severity = "CRITICAL"
            elif "WARNING" in severities:
                severity = "WARNING"
            else:
                severity = "WATCH"

        return {
            "anomaly_count": len(anomalies),
            "severity": severity,
            "anomalies": anomalies}

    def is_healthy(self) -> bool:
        c = self.check()
        return c["severity"] == "NORMAL" and c["anomaly_count"] == 0
