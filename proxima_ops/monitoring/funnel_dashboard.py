from proxima_ops.monitoring.signal_funnel import SignalFunnel
from proxima_ops.monitoring.trade_reconciler import TradeReconciler


class FunnelDashboard:
    def __init__(self, funnel: SignalFunnel, reconciler: TradeReconciler,
                 deployment_score, performance_monitor,
                 freq_classifier=None, drl_classifier=None,
                 rce_pipeline=None, director=None,
                 dpl_live=None):
        self._funnel = funnel
        self._reconciler = reconciler
        self._score = deployment_score
        self._perf = performance_monitor
        self._freq = freq_classifier
        self._drl = drl_classifier
        self._rce = rce_pipeline
        self._director = director
        self._dpl = dpl_live

    def generate(self, order_attempts: list[dict] = None,
                 paper_metrics: dict = None,
                 research_metrics: dict = None) -> str:
        funnel = self._funnel.summary()
        reconcile = self._reconciler.reconcile()
        perf = self._perf.summary()
        ds = self._score.summary()

        g = funnel.get("GENERATED", 0)
        tp = funnel.get("THRESHOLD_PASSED", 0)
        tr = funnel.get("TRIGGERED", 0)
        sub = funnel.get("ORDER_SUBMITTED", 0)
        acc = funnel.get("ORDER_ACCEPTED", 0)
        opn = funnel.get("POSITION_OPENED", 0)
        cls = funnel.get("POSITION_CLOSED", 0)
        blocked = sum(funnel.get(k, 0) for k in
                      ["BLOCKED_SPREAD", "BLOCKED_POSITION_EXISTS",
                       "BLOCKED_RISK_LIMIT", "BLOCKED_MAX_POSITIONS",
                       "BLOCKED_POSITION_LOCK", "BLOCKED_NOT_IN_TOP3",
                       "BLOCKED_THRESHOLD", "BLOCKED_RHL", "BLOCKED_H20",
                       "BLOCKED_FLIP", "BLOCKED_EQUITY_PROTECTION",
                       "BLOCKED_NO_TICK", "BLOCKED_UNKNOWN"])
        rejected = funnel.get("ORDER_REJECTED", 0)
        timeout = funnel.get("ORDER_TIMEOUT", 0)

        pass_rate = (tp / g * 100) if g > 0 else 0.0
        submit_rate = (sub / max(tr, 1) * 100)
        accept_rate = (acc / max(sub, 1) * 100)
        open_rate = (opn / max(acc, 1) * 100)

        lines = []
        lines.append("=" * 52)
        lines.append("  SIGNAL FUNNEL")
        lines.append("=" * 52)
        lines.append("")
        lines.append(f"  Generated:          {g}")
        lines.append(f"  Threshold Passed:   {tp}")
        lines.append(f"  Triggered:          {tr}")
        lines.append(f"  Submitted:          {sub}")
        lines.append(f"  Accepted:           {acc}")
        lines.append(f"  Opened:             {opn}")
        lines.append(f"  Closed:             {cls}")
        lines.append(f"  Blocked:            {blocked}")
        if rejected:
            lines.append(f"  Rejected:           {rejected}")
        if timeout:
            lines.append(f"  Timeout:            {timeout}")
        lines.append("")
        lines.append("-" * 52)
        lines.append("  Conversion Rates")
        lines.append("-" * 52)
        lines.append(f"  Pass Rate:     {pass_rate:5.1f}%")
        lines.append(f"  Submit Rate:   {submit_rate:5.1f}%")
        lines.append(f"  Accept Rate:   {accept_rate:5.1f}%")
        lines.append(f"  Open Rate:     {open_rate:5.1f}%")
        lines.append("")

        leakage_pct = (g - opn) / g * 100 if g > 0 else 0
        survival_rate = opn / g * 100 if g > 0 else 0
        lines.append("=" * 52)
        lines.append("  ALPHA LEAKAGE")
        lines.append("=" * 52)
        lines.append("")
        lines.append(f"  Generated -> Threshold Passed -> Executed -> Closed")
        lines.append(f"  Leakage: {leakage_pct:.1f}%  ({(g - opn)} signals lost between generation and execution)")
        lines.append(f"  Execution Survival: {survival_rate:.1f}%  ({opn}/{g})")
        lines.append("")

        _block_labels = [
            ("BLOCKED_SPREAD", "Spread"),
            ("BLOCKED_POSITION_EXISTS", "Position Exists"),
            ("BLOCKED_RISK_LIMIT", "Risk Limit"),
            ("BLOCKED_MAX_POSITIONS", "Max Positions"),
            ("BLOCKED_POSITION_LOCK", "Position Lock"),
            ("BLOCKED_NOT_IN_TOP3", "Not In Top3"),
            ("BLOCKED_THRESHOLD", "Threshold"),
            ("BLOCKED_RHL", "RHL"),
            ("BLOCKED_H20", "H20"),
            ("BLOCKED_FLIP", "Flip"),
            ("BLOCKED_EQUITY_PROTECTION", "Equity Protection"),
            ("BLOCKED_NO_TICK", "No Tick"),
            ("BLOCKED_UNKNOWN", "Unknown"),
        ]
        if blocked + rejected + timeout > 0:
            lines.append("=" * 52)
            lines.append("  REJECTION BREAKDOWN")
            lines.append("=" * 52)
            for key, label in _block_labels:
                val = funnel.get(key, 0)
                if val:
                    lines.append(f"  {label:20s} {val}")
            if rejected:
                lines.append(f"  {'Broker':20s} {rejected}")
            if timeout:
                lines.append(f"  {'Timeout':20s} {timeout}")
            lines.append("")

        lines.append("=" * 52)
        lines.append("  POSITION AUDIT")
        lines.append("=" * 52)
        lines.append(f"  MT5 Open:    {reconcile['mt5_open']}")
        lines.append(f"  Ledger Open: {reconcile['ledger_open']}")
        lines.append(f"  Diff:        {reconcile['mt5_open'] - reconcile['ledger_open']}")
        lines.append(f"  Status:      {'HEALTHY' if reconcile['healthy'] else 'MISMATCH'}")
        lines.append("")

        for m in reconcile.get("mismatches", []):
            lines.append(f"  MISMATCH: ticket {m['ticket']} ({m['issue']}) - {m['symbol']}")
        if reconcile.get("mismatches"):
            lines.append("")

        lines.append("=" * 52)
        lines.append("  DEPLOYMENT COMPONENTS")
        lines.append("=" * 52)
        sharpe = perf.get("sharpe")
        pp = perf.get("pp")
        dd = perf.get("max_dd")
        n_trades = perf.get("n_trades", 0)

        if sharpe is None or sharpe == "COLLECTING_DATA":
            sharpe_score = 0.0
            sharpe_str = "COLLECTING_DATA"
        else:
            sharpe_score = max(0.0, min(1.0, (sharpe - 0.5) / 2.0)) if sharpe > 0 else 0.0
            sharpe_str = f"Sharpe={sharpe:.2f}"

        if pp is None or pp == "COLLECTING_DATA":
            pp_score = 0.0
            pp_str = "COLLECTING_DATA"
        else:
            pp_score = max(0.0, min(1.0, (pp - 0.45) / 0.20))
            pp_str = f"PP={pp:.2f}"

        if dd is None or dd == "COLLECTING_DATA":
            dd_score = 0.0
            dd_str = "COLLECTING_DATA"
        else:
            dd_score = 1.0 - min(dd / 0.12, 1.0)
            dd_str = f"DD={dd:.2%}"

        vol_score = min(n_trades / 100.0, 1.0) if n_trades > 0 else 0.0
        lines.append(f"  Performance:  {sharpe_score:.2f}  ({sharpe_str})")
        lines.append(f"  Execution:    {pp_score:.2f}        ({pp_str})")
        lines.append(f"  DD Control:   {dd_score:.2f}        ({dd_str})")
        lines.append(f"  Trade Count:  {vol_score:.2f}       (Trades={n_trades})")
        lines.append(f"  ------------------------")
        lines.append(f"  Score:        {ds['current_score']:.3f}    ({ds['classification']})")
        if order_attempts:
            lines.append("")
            lines.append("=" * 52)
            lines.append("  LAST ORDER")
            lines.append("=" * 52)
            o = order_attempts[-1]
            lines.append(f"  [{o.get('timestamp', '')}]")
            lines.append(f"  Signal:   {o.get('signal_id', 'N/A')}")
            lines.append(f"  Action:   {o.get('action', 'N/A')}  {o.get('symbol', '')}")
            lines.append(f"  Vol:      {o.get('volume', 0):.2f}  @ {o.get('price', 0):.5f}")
            lines.append(f"  Status:   {o.get('submission', 'N/A')}")
            if o.get("retcode"):
                lines.append(f"  Retcode:  {o['retcode']}")
            if o.get("ticket"):
                lines.append(f"  Ticket:   {o['ticket']}")

        if self._drl is not None:
            dc = self._drl.classify()
            exec_q = self._drl._exec.summary()
            lines.append("")
            lines.append("=" * 52)
            lines.append("  DEPLOYMENT REALITY")
            lines.append("=" * 52)
            asr_val = dc.get('asr')
            asr_str = f"{asr_val:.3f}" if isinstance(asr_val, (int, float)) else str(asr_val)
            lines.append(f"  ASR:                      {asr_str}")
            lines.append(f"  Execution Quality:        {dc['execution_quality']}")
            lines.append(f"  Mean Slippage:            {exec_q.get('mean_slippage_pts', 0)} pts")
            lines.append(f"  Trend:                    {dc['score_trend']}")
            lines.append(f"  Classification:           {dc['classification']}")
            lines.append("")

        if self._freq is not None:
            blocked_sig = self._freq._analysis.leakage_rate()
            c = self._freq.classify()
            adr = self._freq._analysis.alpha_destruction_ratio()
            lines.append("")
            lines.append("=" * 52)
            lines.append("  INVALID SPREAD AUDIT")
            lines.append("=" * 52)
            lines.append(f"  Blocked Signals:          {blocked_sig['blocked_total']}")
            lines.append(f"  Profitable Blocked:       {blocked_sig['blocked_profitable']}")
            lines.append(f"  Leakage Rate:             {blocked_sig['leakage_rate']}%")
            lines.append(f"  ADR:                      {adr:.3f}")
            lines.append(f"  Classification:           {c['classification']}")
            lines.append("")

        if self._dpl is not None:
            s = self._dpl.summary()
            t = s.get("tournament", {})
            lines.append("")
            lines.append("=" * 52)
            lines.append("  DPL LIVE VALIDATION (REAL OUTCOMES)")
            lines.append("=" * 52)
            lines.append(f"  Snapshots:                {s['total_snapshots']}")
            lines.append(f"  Resolved:                 {s['resolved']} ({s['pct_resolved']}%)")
            lines.append(f"  Symbols:                  {', '.join(s['symbols'])}")
            rd = s.get("regime_distribution", {})
            if rd:
                rd_str = ", ".join(f"R{k}={v}" for k, v in sorted(rd.items()))
                lines.append(f"  Regime Dist:              {rd_str}")
            lines.append(f"  Short Outcomes:           {'YES' if s.get('has_short_outcomes') else 'NO'}")
            if t:
                # Special handling: regime entries are dicts with "regimes" key
                regime_entry = t.get("regime_h20", {})
                regime_data = regime_entry.get("regimes", {}) if isinstance(regime_entry, dict) else {}
                if regime_data:
                    lines.append(f"  Regime P(up):")
                    for r_val, r_info in sorted(regime_data.items()):
                        lines.append(f"    R{r_val}: {r_info.get('p_up', 0.5):.4f} (n={r_info.get('n', 0)}, z={r_info.get('z_score', 0):.2f})")
                best_candidate = max((k for k, v in t.items() if isinstance(v, dict) and "accuracy" in v),
                                     key=lambda k: t[k].get("accuracy", 0), default=None)
                if best_candidate:
                    best = t[best_candidate]
                    lines.append(f"  Best Feature:             {best_candidate}")
                    lines.append(f"    Accuracy:               {best.get('accuracy', 0.5):.4f}")
                    lines.append(f"    Info Gain:              {best.get('info_gain', 0):.6f}")
                    lines.append(f"    P(up|high):             {best.get('p_up_high', 0.5):.4f}")
                    lines.append(f"    N:                      {best.get('n', 0)}")
            lines.append("")

        if self._rce is not None:
            ate = self._rce._ate.summary()
            conv = self._rce._conv.check()
            friction = self._rce._friction.summary()
            health = self._rce._health.compute()
            cls = self._rce._classifier.classify()
            lines.append("")
            lines.append("=" * 52)
            lines.append("  REALITY CONVERGENCE")
            lines.append("=" * 52)

            ate_val = ate.get('ate', 0.0)
            ate_str = f"{ate_val:.2f}" if isinstance(ate_val, (int, float)) else str(ate_val)

            freq_match = self._rce._conv.match_pct()
            freq_match_str = f"{freq_match}%" if isinstance(freq_match, (int, float)) else str(freq_match)

            health_idx = health.get('health_index', 0.0)
            health_idx_str = f"{health_idx:.1f}" if isinstance(health_idx, (int, float)) else str(health_idx)

            lines.append(f"  ATE:                      {ate_str}")
            lines.append(f"  Frequency Match:          {freq_match_str}")
            lines.append(f"  Friction Index:           {friction.get('friction_index', 0.0)}")
            lines.append(f"  Health Index:             {health_idx_str}")
            lines.append(f"  Classification:           {cls.get('classification', 'UNKNOWN')}")
            lines.append("")

        if self._director is not None:
            report = self._director.daily_report()
            lines.append("")
            lines.append("=" * 52)
            lines.append("  AUTONOMOUS RESEARCH DIRECTOR")
            lines.append("=" * 52)

            es_val = report.get('evidence_strength', 0.0)
            es_str = f"{es_val:.2f}" if isinstance(es_val, (int, float)) else str(es_val)

            rc_val = report.get('research_confidence', 0.0)
            rc_str = f"{rc_val:.2f}" if isinstance(rc_val, (int, float)) else str(rc_val)

            dc_val = report.get('deployment_confidence', 0.0)
            dc_str = f"{dc_val:.2f}" if isinstance(dc_val, (int, float)) else str(dc_val)

            at_val = report.get('alpha_transfer', 0.0)
            at_str = f"{at_val:.2f}" if isinstance(at_val, (int, float)) else str(at_val)

            lines.append(f"  Evidence:         {es_str}")
            lines.append(f"  Research Conf:    {rc_str}")
            lines.append(f"  Deployment Conf:  {dc_str}")
            lines.append(f"  Alpha Transfer:   {at_str}")
            lines.append(f"  Risk:             {report.get('biggest_risk', 'N/A')}")
            lines.append(f"  Strength:         {report.get('biggest_strength', 'N/A')}")
            lines.append(f"  Recommendation:   {report.get('recommendation', 'NO_ACTION')}")
            lines.append(f"  Classification:   {report.get('classification', 'RESEARCH_PENDING')}")
            lines.append("")

        if paper_metrics:
            lines.append("=" * 52)
            lines.append("  RESEARCH vs PAPER")
            lines.append("=" * 52)
            pp_val = paper_metrics.get('pp', 0)
            pp_val = pp_val if isinstance(pp_val, (int, float)) else 0
            hold_val = paper_metrics.get('avg_hold', 0)
            hold_val = hold_val if isinstance(hold_val, (int, float)) else 0
            sharpe_val = paper_metrics.get('sharpe', 0)
            sharpe_val = sharpe_val if isinstance(sharpe_val, (int, float)) else 0
            assets_val = paper_metrics.get('active_assets', 0)
            assets_val = assets_val if isinstance(assets_val, (int, float)) else 0
            # Research values from pipeline — fallback to COLLECTING_DATA when unavailable
            r_pp = research_metrics.get('pp', 'COLLECTING_DATA') if research_metrics else 'COLLECTING_DATA'
            r_hold = research_metrics.get('avg_hold', 'COLLECTING_DATA') if research_metrics else 'COLLECTING_DATA'
            r_sharpe = research_metrics.get('sharpe', 'COLLECTING_DATA') if research_metrics else 'COLLECTING_DATA'
            r_assets = research_metrics.get('active_assets', 'COLLECTING_DATA') if research_metrics else 'COLLECTING_DATA'
            r_pp_str = f"{r_pp:.3f}" if isinstance(r_pp, (int, float)) else str(r_pp)
            r_hold_str = f"{r_hold:.1f}" if isinstance(r_hold, (int, float)) else str(r_hold)
            r_sharpe_str = f"{r_sharpe:.2f}" if isinstance(r_sharpe, (int, float)) else str(r_sharpe)
            r_assets_str = str(r_assets) if isinstance(r_assets, (int, float)) else str(r_assets)
            lines.append(f"  PP:    Research {r_pp_str} | Paper {pp_val:.3f}")
            lines.append(f"  Hold:  Research {r_hold_str}  | Paper {hold_val:.1f} bars")
            lines.append(f"  Sharpe:Research {r_sharpe_str}  | Paper {sharpe_val:.2f}")
            lines.append(f"  Mix:   Research {r_assets_str} | Paper {assets_val} assets")
            lines.append("")

        lines.append("=" * 52)
        return "\n".join(lines)
