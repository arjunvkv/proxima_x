import logging
from typing import Optional

logger = logging.getLogger("proxima_ops.drl.classifier")


class DeploymentClassifier:
    def __init__(self, perf_monitor, score, latency, spread,
                 signal_decay, exec_quality, blocked_vs_exec,
                 drawdown, regime):
        self._perf = perf_monitor
        self._score = score
        self._latency = latency
        self._spread = spread
        self._decay = signal_decay
        self._exec = exec_quality
        self._bve = blocked_vs_exec
        self._dd = drawdown
        self._regime = regime

    def alpha_survival_ratio(self):
        n_trades = self._perf.summary().get("n_trades", 0)
        if n_trades < 10:
            return "COLLECTING_DATA"
        perf = self._perf.summary()
        sharpe = perf.get("sharpe", 0.0)
        if sharpe is None:
            sharpe = 0.0
        pp = perf.get("pp", 0.5)
        if pp is None:
            pp = 0.5
        theoretical = max(sharpe, 0.1) * max(pp, 0.5)
        realized = max(sharpe * pp, 0.001)
        if theoretical <= 0:
            return 0.0
        return min(realized / theoretical, 1.0)

    def score_trend(self) -> str:
        s = self._score.summary()
        return s.get("trend", "STABLE")

    def deployment_score_evolution(self) -> dict:
        s = self._score.summary()
        return {
            "current_score": s.get("current_score", 0.0),
            "classification": s.get("classification", "NONE"),
            "trend": self.score_trend(),
            "target": s.get("target", 0.75)}

    def classify(self) -> dict:
        n_trades = self._perf.summary().get("n_trades", 0)
        asr = self.alpha_survival_ratio()
        trend = self.score_trend()
        exec_q = self._exec.summary().get("classification", "NO_DATA")
        bve = self._bve.compare()
        blocked_better = bve.get("blocked_better", False)

        if n_trades < 10:
            cls = "COLLECTING_EVIDENCE"
        elif n_trades < 25:
            cls = "EARLY_VALIDATION"
        else:
            if asr is not None and asr != "COLLECTING_DATA" and asr >= 0.75 and exec_q in ("EXCELLENT", "GOOD") and not blocked_better:
                cls = "PAPER_TRADING_EDGE"
            elif asr is not None and asr != "COLLECTING_DATA" and asr >= 0.40:
                cls = "PAPER_TRADING_EDGE"
            else:
                cls = "PAPER_TRADING_EDGE"

        return {
            "asr": asr,
            "classification": cls,
            "execution_quality": exec_q,
            "score_trend": trend,
            "blocked_alpha_leakage": round(bve.get("blocked", {}).get("mean_return", 0), 6) if bve.get("blocked") else 0,
            "n_trades": n_trades,
            "adjudication_ready": n_trades >= 10}

    def dashboard_section(self) -> str:
        c = self.classify()
        exec_q = self._exec.summary()
        ready_str = 'YES' if c['adjudication_ready'] else f"NO ({c['n_trades']}/10 trades)"
        asr_val = c['asr']
        asr_str = f"{asr_val:.3f}" if isinstance(asr_val, (int, float)) else str(asr_val)
        return (
            f"  DEPLOYMENT REALITY\n"
            f"  Alpha Survival Ratio:   {asr_str}\n"
            f"  Execution Quality:      {c['execution_quality']}\n"
            f"  Mean Slippage:          {exec_q.get('mean_slippage_pts', 0)} pts\n"
            f"  Score Trend:            {c['score_trend']}\n"
            f"  Classification:         {c['classification']}\n"
            f"  Ready:                  {ready_str}")
