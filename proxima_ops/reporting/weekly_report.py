from datetime import datetime, date, timedelta
from proxima_ops.ledger.trade_ledger import TradeLedger
from proxima_ops.ledger.signal_ledger import SignalLedger
from proxima_ops.ledger.deployment_ledger import DeploymentLedger
from proxima_ops.monitoring.deployment_score import DeploymentScore
from proxima_ops.monitoring.performance_monitor import OpsPerformanceMonitor
from proxima_ops.config.settings import SETTINGS


class WeeklyReport:
    def __init__(self, trade_ledger: TradeLedger, signal_ledger: SignalLedger,
                 deployment_ledger: DeploymentLedger,
                 score: DeploymentScore, perf: OpsPerformanceMonitor):
        self._trades = trade_ledger
        self._signals = signal_ledger
        self._deploy = deployment_ledger
        self._score = score
        self._perf = perf

    def generate(self) -> str:
        ds = self._score.summary()
        perf = self._perf.summary()
        dep = self._deploy.summary()
        lines = []
        lines.append("===========================================")
        lines.append("PROXIMA OPS - WEEKLY REPORT")
        lines.append(f"Week Ending: {date.today().isoformat()}")
        lines.append(f"Days Active: {dep['days_recorded']}")
        lines.append("===========================================")
        lines.append("")
        lines.append("-- DEPLOYMENT SCORE --")
        lines.append(f" Current:   {ds['current_score']:.3f}")
        lines.append(f" 7d Avg:    {dep['avg_7d_score']:.3f}")
        lines.append(f" Trend:     {dep['trend']}")
        lines.append(f" Class:     {ds['classification']}")
        lines.append(f" Target:    {ds['target']} - {'HIT' if ds['hits_target'] else 'MISS'}")
        lines.append("")
        sharpe_val = perf['sharpe']
        sharpe_str = f"{sharpe_val:.3f}" if isinstance(sharpe_val, (int, float)) else str(sharpe_val)
        
        pp_val = perf['pp']
        pp_str = f"{pp_val:.3f}" if isinstance(pp_val, (int, float)) else str(pp_val)
        
        dd_val = perf['max_dd']
        dd_str = f"{dd_val:.2%}" if isinstance(dd_val, (int, float)) else str(dd_val)

        lines.append("-- PERFORMANCE --")
        lines.append(f" Sharpe:    {sharpe_str} -> target {SETTINGS.min_sharpe}")
        lines.append(f" PP:        {pp_str} -> target {SETTINGS.min_pp}")
        lines.append(f" Return:    ${perf['total_return']:.2f}")
        lines.append(f" Trades:    {perf['n_trades']}")
        lines.append(f" DD:        {dd_str} -> max {SETTINGS.max_drawdown_pct:.0%}")
        lines.append("")
        lines.append("-- RISKS --")
        
        has_risks = False
        if isinstance(dd_val, (int, float)) and dd_val > SETTINGS.max_drawdown_pct * 0.7:
            lines.append(" [!] Drawdown approaching limit")
            has_risks = True
        if isinstance(sharpe_val, (int, float)) and sharpe_val < SETTINGS.min_sharpe:
            lines.append(" [!] Sharpe below target")
            has_risks = True
        if isinstance(pp_val, (int, float)) and pp_val < SETTINGS.min_pp:
            lines.append(" [!] PP below target")
            has_risks = True
        if ds['current_score'] < SETTINGS.deployment_score_target:
            lines.append(" [!] Deployment score below target")
            has_risks = True
            
        if not has_risks:
            lines.append(" [OK] No significant risks detected")
        lines.append("")
        lines.append("-- PERSISTENCE FORECAST ACCURACY --")
        lines.append(" (requires RLVL integration)")
        lines.append("")
        lines.append("-- FREQUENCY STABILITY --")
        lines.append(" (requires RLVL integration)")
        lines.append("")
        lines.append("===========================================")
        return "\n".join(lines)

    def send_telegram(self, telegram_bot) -> str:
        report = self.generate()
        telegram_bot.send_sync(report)
        return report
