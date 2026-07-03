from datetime import datetime, date
from proxima_ops.ledger.trade_ledger import TradeLedger
from proxima_ops.ledger.signal_ledger import SignalLedger
from proxima_ops.ledger.deployment_ledger import DeploymentLedger
from proxima_ops.monitoring.deployment_score import DeploymentScore
from proxima_ops.monitoring.performance_monitor import OpsPerformanceMonitor
from proxima_ops.monitoring.mt5_monitor import MT5Monitor
from proxima_ops.config.settings import SETTINGS


class DailyReport:
    def __init__(self, trade_ledger: TradeLedger, signal_ledger: SignalLedger,
                 deployment_ledger: DeploymentLedger,
                 score: DeploymentScore, perf: OpsPerformanceMonitor,
                 mt5_monitor: MT5Monitor):
        self._trades = trade_ledger
        self._signals = signal_ledger
        self._deploy = deployment_ledger
        self._score = score
        self._perf = perf
        self._mt5 = mt5_monitor

    def generate(self) -> str:
        ds = self._score.summary()
        perf = self._perf.summary()
        deploy = self._deploy.summary()
        signal_s = self._signals.summary()
        mt5_h = self._mt5.health_summary
        today_signals = self._signals.get_today()
        today_trades = [t for t in self._trades.get_recent(100)
                        if date.fromisoformat(str(t.get("entry_time", "")).split()[0]) == date.today()]
        lines = []
        lines.append("===================================")
        lines.append("PROXIMA OPS - DAILY REPORT")
        lines.append(f"Date: {date.today().isoformat()}")
        lines.append("===================================")
        lines.append("")
        lines.append("-- DEPLOYMENT --")
        lines.append(f" Score:     {ds['current_score']:.3f}")
        lines.append(f" Class:     {ds['classification']}")
        lines.append(f" Trend:     {ds['trend']}")
        lines.append(f" Target:    {ds['target']}")
        lines.append("")
        
        components = ds.get("components", {})
        lines.append("Deployment Score Components")
        lines.append("")
        lines.append(f"Frequency:\n{components.get('Frequency', 0.0):.2f}\n")
        lines.append(f"Execution:\n{components.get('Execution', 0.0):.2f}\n")
        lines.append(f"Performance:\n{components.get('Performance', 0.0):.2f}\n")
        lines.append(f"Persistence:\n{components.get('Persistence', 0.0):.2f}\n")
        lines.append(f"Trade Count:\n{components.get('Trade Count', 0.0):.2f}")
        lines.append("")
        sharpe_val = perf['sharpe']
        sharpe_str = f"{sharpe_val:.3f}" if isinstance(sharpe_val, (int, float)) else str(sharpe_val)
        
        pp_val = perf['pp']
        pp_str = f"{pp_val:.3f}" if isinstance(pp_val, (int, float)) else str(pp_val)
        
        dd_val = perf['max_dd']
        dd_str = f"{dd_val:.2%}" if isinstance(dd_val, (int, float)) else str(dd_val)

        lines.append("-- PERFORMANCE --")
        lines.append(f" Sharpe:    {sharpe_str}")
        lines.append(f" PP:        {pp_str}")
        lines.append(f" Return:    ${perf['total_return']:.2f}")
        lines.append(f" Points:    {perf['total_points']:.1f}")
        lines.append(f" DD:        {dd_str}")
        lines.append(f" Trades:    {perf['n_trades']}")
        lines.append(f" Today PnL: ${perf['today_pnl']:.2f}")
        lines.append("")
        lines.append("-- SIGNALS --")
        lines.append(f" Today:     {len(today_signals)}")
        lines.append(f" Total:     {signal_s['total_signals']}")
        lines.append(f" Executed:  {signal_s['executed']}")
        lines.append("")
        lines.append("-- MT5 --")
        lines.append(f" Status:    {mt5_h['mt5_status']}")
        lines.append(f" Uptime:    {mt5_h['uptime_minutes']}m")
        lines.append("")
        lines.append("-- VPL REGIMES --")
        try:
            from proxima_ops.monitoring.deployment_dashboard import _refresh_vpl_cache, _VPL_CACHE
            _refresh_vpl_cache()
            for sym in ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD"]:
                sig = _VPL_CACHE.get(sym)
                if sig:
                    rm = sig.get("risk_multiplier", "?")
                    perm = sig.get("trade_permission", "?")
                    lines.append(f" {sym:<10s} {sig['regime']:<20s} risk={rm}x  perm={perm}")
                else:
                    lines.append(f" {sym:<10s} N/A")
        except Exception:
            lines.append(" VPL engine unavailable")
        lines.append("")
        lines.append("-- OPEN POSITIONS --")
        for pos in self._trades.get_open():
            lines.append(f" {pos['symbol']} | {pos['signal_type']} | {pos['entry_price']}")
        if not self._trades.get_open():
            lines.append(" None")
        lines.append("")
        lines.append("===================================")
        return "\n".join(lines)

    def send_telegram(self, telegram_bot) -> str:
        report = self.generate()
        telegram_bot.send_sync(report)
        return report
