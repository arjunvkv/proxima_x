import logging
from datetime import datetime

logger = logging.getLogger("proxima_ops.drl.pipeline")


class DeploymentRealityPipeline:
    def __init__(self, latency, spread, signal_decay, exec_quality,
                 blocked_vs_exec, drawdown, regime, classifier):
        self._latency = latency
        self._spread = spread
        self._decay = signal_decay
        self._exec = exec_quality
        self._bve = blocked_vs_exec
        self._dd = drawdown
        self._regime = regime
        self._classifier = classifier

    def record_trade_close(self, signal_id: str, symbol: str, regime: str,
                           es_rank: float, at_rank: float,
                           entry_price: float, exit_price: float,
                           pnl_points: float, pnl_money: float,
                           duration: int, persistence_forecast: str = "N/A",
                           spread_at_entry: int = 0):
        if pnl_money < 0:
            self._dd.record(symbol, regime, es_rank, at_rank,
                           persistence_forecast, duration, pnl_points, pnl_money)
        self._regime.record(symbol, regime, pnl_points, pnl_money,
                           duration, es_rank, at_rank)

    def report(self) -> str:
        asr = self._classifier.alpha_survival_ratio()
        c = self._classifier.classify()
        exec_q = self._exec.summary()
        latency_s = self._latency.summary()
        spread_s = self._spread.summary()
        decay_s = self._decay.summary()
        dd_s = self._dd.summary()
        regime_s = self._regime.summary()
        bve = self._bve.compare()

        lines = []
        lines.append("=" * 52)
        lines.append(f"  DEPLOYMENT REALITY LAB — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("=" * 52)
        lines.append("")
        lines.append("  ALPHA SURVIVAL")
        lines.append(f"  ASR:                    {asr:.3f}")
        lines.append(f"  Classification:         {c['classification']}")
        lines.append(f"  Trades:                 {c['n_trades']}")
        lines.append("")
        lines.append("  EXECUTION QUALITY")
        lines.append(f"  Quality:                {exec_q.get('classification', 'N/A')}")
        lines.append(f"  Mean Slippage:          {exec_q.get('mean_slippage_pts', 0)} pts")
        lines.append(f"  Max Slippage:           {exec_q.get('max_slippage_pts', 0)} pts")
        lines.append("")
        lines.append("  LATENCY")
        lines.append(f"  Mean Total:             {latency_s.get('mean_latency_ms', 0)} ms")
        lines.append(f"  Signal→Submit:          {latency_s.get('mean_signal_to_submit_ms', 0)} ms")
        lines.append(f"  Submit→Accept:          {latency_s.get('mean_submit_to_accept_ms', 0)} ms")
        lines.append("")
        lines.append("  SPREAD REALITY")
        lines.append(f"  Mean Spread (Wins):     {spread_s.get('mean_spread_wins', 'N/A')}")
        lines.append(f"  Mean Spread (Losses):   {spread_s.get('mean_spread_losses', 'N/A')}")
        lines.append("")
        lines.append("  BLOCKED vs EXECUTED")
        b = bve.get("blocked", {})
        e = bve.get("executed", {})
        lines.append(f"  Blocked PP:             {b.get('pp', 0):.2%}")
        lines.append(f"  Executed PP:            {e.get('pp', 0):.2%}")
        lines.append(f"  Blocked Mean Ret:       {b.get('mean_return', 0):.6f}")
        lines.append(f"  Executed Mean Ret:      {e.get('mean_return', 0):.6f}")
        if b.get("count", 0) > 0:
            lines.append(f"  Blocked Better:         {bve.get('blocked_better', False)}")
        lines.append("")
        lines.append("  SIGNAL DECAY")
        for h in ["return_h1", "return_h5", "return_h20"]:
            ek = f"{h}_executed_mean"
            bk = f"{h}_blocked_mean"
            if decay_s.get(ek) is not None:
                lines.append(f"  {h}: Exec={decay_s[ek]:.6f} Blocked={decay_s.get(bk, 'N/A')}")
        lines.append("")
        lines.append("  DRAWDOWN FORENSICS")
        if dd_s.get("count", 0) > 0:
            lines.append(f"  Total Losses:           {dd_s['count']} (${dd_s.get('total_loss', 0):.2f})")
            for reg, data in dd_s.get("regime_breakdown", {}).items():
                lines.append(f"  {reg}: count={data['count']}, mean=${data['mean_loss']:.2f}")
        lines.append("")
        lines.append("  REGIME REALITY")
        if regime_s.get("count", 0) > 0:
            for reg, data in regime_s.get("regimes", {}).items():
                lines.append(f"  {reg}: PP={data['pp']:.2%} Sharpe={data['sharpe']:.2f} n={data['count']}")
        lines.append("")
        lines.append("  ADJUDICATION")
        lines.append(f"  ASR:                    {asr:.3f}")
        lines.append(f"  Trend:                  {c['score_trend']}")
        ready_str = 'YES' if c['adjudication_ready'] else f"NO ({c['n_trades']}/10 trades)"
        lines.append(f"  Ready:                  {ready_str}")
        lines.append(f"  Final:                  {c['classification']}")
        lines.append("=" * 52)
        return "\n".join(lines)
