import time
import os
import json
from pathlib import Path
from data.models import HealthStatus


_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "dashboard_log.jsonl"
# Full spec terminal dashboard — verbose, exhaustive monitoring output.
# CLEANUP: This is a dense all-in-one dashboard built for active debugging.
# Future: migrate to a structured UI (Streamlit/fastAPI) and strip console spam.


class Dashboard:
    def __init__(self):
        self._last_update = 0.0
        self._last_log_ts = 0.0
        self._header_printed = False
        self._uptime_start = time.time()
        self._log_handle = None
        self._init_log()
        self.latest_event: dict = None

    def _init_log(self):
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = _LOG_FILE.open("a", encoding="utf-8")
        self._log_handle = fh

    def _log_jsonl(self, record: dict):
        try:
            self._log_handle.write(json.dumps(record, default=str) + "\n")
            self._log_handle.flush()
        except Exception:
            pass

    def render(self, mode: str, health: HealthStatus, currency_strengths: dict,
               positions: list, top_hypothesis: dict = None,
               pnl: float = 0.0, trade_count: int = 0,
               factor_exposure: dict = None,
               observability: dict = None,
               z_scores: dict = None,
               stability: dict = None,
               edge_balance: dict = None,
               missing_symbols: list = None,
               missing_impact: dict = None,
               health_report: dict = None,
               concentration: dict = None,
               production_ready: bool = False,
               stress_test: dict = None,
               recent_failures: list = None,
                pipeline_metrics: dict = None,
                exec_fail: str = None,
                unavailable_symbols: list = None,
                total_lots: float = 0.0, max_total_lots: float = 0.0,
               lot_size: float = 0.0, profit_target: float = 0.0,
               cooldown_active: bool = False, cooldown_remaining: int = 0,
                currency_bursts: dict = None,
                 persistence: dict = None,
                 strength_persistence: dict = None,
                 currency_der: dict = None,
                 der_persistence: dict = None,
                 top_der_pairs: list = None,
            wls_direct: bool = False,
                   top_burst_pairs: list = None,
                   burst_state: str = None,
                   bar_state_summary: str = None,
                   bar_output: str = None,
                 available_symbols_count: int = 0,
                 configured_symbols_count: int = 0,
                 nme_output: str = None,
                 nme_trade_snapshots: list = None) -> None:
        now = time.time()
        if now - self._last_update < 1.0:
            return
        self._last_update = now
        uptime_s = now - self._uptime_start
        uptime_str = f"{int(uptime_s // 3600):02d}:{int((uptime_s % 3600) // 60):02d}:{int(uptime_s % 60):02d}"
        ts = time.strftime("%H:%M:%S")

        os.system("cls" if os.name == "nt" else "clear")

        if nme_output:
            print(nme_output)
        if bar_output:
            print(bar_output)

        health_icon = "OK" if health.state == "OK" else ("DEG" if health.state == "DEGRADED" else "ERR")
        health_color = "\033[92m" if health.state == "OK" else ("\033[93m" if health.state == "DEGRADED" else "\033[91m")
        reset = "\033[0m"
        ready_str = "WLS DIRECT" if wls_direct else ("READY" if production_ready else "BLOCKED")
        ready_color = "\033[92m" if (wls_direct or production_ready) else "\033[91m"

        # ── HEADER ──────────────────────────────────────────────
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print(f"║  PROXIMA CURRENCY DECOMPOSITION ENGINE   MODE: {mode:<8s}              ║")
        print(f"║  {ts}  │  UPTIME: {uptime_str}  │  "
              f"{health_color}HEALTH: {health_icon}{reset}  │  "
              f"{ready_color}EXEC: {ready_str}{reset}         ║")
        print("╠══════════════════════════════════════════════════════════════════════╣")

        # ── CONNECTION ───────────────────────────────────────────
        mt5_str = f"MT5:  {'CONNECTED' if health.mt5_ok else 'DISCONNECTED'}"
        tick_str = f"TICK QUALITY: {health.tick_quality:.2f}"
        snap_str = f"SNAPSHOT: {'OK' if health.last_snapshot_ok else 'FAIL'}"
        mem_str = f"MEM: {health.memory_mb:.1f} MB"
        solve_str = f"SOLVE LATENCY: {health.solve_latency_ms:.1f} ms"

        col_w = 32
        print(f"║  {mt5_str:<{col_w}}{tick_str:<{col_w}}║")
        print(f"║  {snap_str:<{col_w}}{solve_str:<{col_w}}║")
        print(f"║  {mem_str:<{col_w}}{'TRADES: ' + str(trade_count):<{col_w}}║")
        print("╠══════════════════════════════════════════════════════════════════════╣")

        # ── CURRENCY STRENGTHS ──────────────────────────────────
        print("║  CURRENCY STRENGTHS (latent WLS decomposition / streak)                ║")
        if currency_strengths:
            items = sorted(currency_strengths.items(), key=lambda x: abs(x[1]), reverse=True)
            bar_max = 14
            n = len(items)
            for rank, (ccy, val) in enumerate(items):
                bar_len = int(abs(val) * bar_max / 0.02)
                bar_len = min(bar_len, bar_max)
                bar = "█" * bar_len if bar_len > 0 else ""
                sign = "+" if val >= 0 else ""
                sp = (strength_persistence or {}).get(ccy, {})
                direction = sp.get("direction", 0)
                streak = sp.get("streak", 0)
                peak = sp.get("peak", 0.0)
                trough = sp.get("trough", 0.0)
                arrow = "▲" if direction > 0 else "▼" if direction < 0 else "○"
                ext = f"pk={peak:.5f}" if direction > 0 else f"tr={trough:.5f}" if direction < 0 else "─"
                line = f"    {ccy}: {sign}{val:+.5f}  {bar:<{bar_max}}  {arrow}{streak:<3d}  {ext}  "
                if rank < 2:
                    print(f"║\033[92m{line}\033[0m║")
                elif rank >= n - 2:
                    print(f"║\033[91m{line}\033[0m║")
                else:
                    print(f"║{line}║")
        print("╠══════════════════════════════════════════════════════════════════════╣")

        # ── PARTICIPATION BURST ────────────────────────────────────
        if currency_bursts:
            print("║  BURST & PERSISTENCE (volume activity / streak)                        ║")
            if top_burst_pairs:
                pairs_str = " ".join(top_burst_pairs[:3])
                print(f"║    ACTIVE: {pairs_str:<62}║")
            items = sorted(currency_bursts.items(), key=lambda x: abs(x[1]), reverse=True)
            bar_max = 14
            n = len(items)
            for rank, (ccy, val) in enumerate(items):
                bar_len = int(abs(val) * bar_max / 3.0)
                bar_len = min(bar_len, bar_max)
                bar = "█" * bar_len if bar_len > 0 else ""
                sign = "+" if val >= 0 else ""
                p = (persistence or {}).get(ccy, {})
                direction = p.get("direction", 0)
                streak = p.get("streak", 0)
                peak = p.get("peak", 0.0)
                trough = p.get("trough", 0.0)
                gap = p.get("neutral_gap", 0)
                arrow = "▲" if direction > 0 else "▼" if direction < 0 else "○"
                gap_str = f" g={gap}" if gap else ""
                ext = f"pk={peak:.3f}" if direction > 0 else f"tr={trough:.3f}" if direction < 0 else "─"
                line = f"    {ccy}: {sign}{val:+.3f}  {bar:<{bar_max}}  {arrow}{streak:<3d}{gap_str:<4s}  {ext}  "
                if rank < 2:
                    print(f"║\033[92m{line}\033[0m║")
                elif rank >= n - 2:
                    print(f"║\033[91m{line}\033[0m║")
                else:
                    print(f"║{line}║")
            print("╠══════════════════════════════════════════════════════════════════════╣")

        # ── DIRECTIONAL EFFICIENCY ───────────────────────────────
        if currency_der:
            print("║  DIRECTIONAL EFFICIENCY (DER — price movement quality / streak)            ║")
            if top_der_pairs:
                pairs_str = " ".join(top_der_pairs[:3])
                print(f"║    EFFICIENT: {pairs_str:<62}║")
            items = sorted(currency_der.items(), key=lambda x: abs(x[1]), reverse=True)
            bar_max = 14
            n = len(items)
            for rank, (ccy, val) in enumerate(items):
                bar_len = int(abs(val) * bar_max / 1.0)
                bar_len = min(bar_len, bar_max)
                bar = "█" * bar_len if bar_len > 0 else ""
                sign = "+" if val >= 0 else ""
                p = (der_persistence or {}).get(ccy, {})
                pval = p.get("value", 0.0)
                streak = p.get("streak", 0)
                peak = p.get("peak", 0.0)
                trough = p.get("trough", 0.0)
                arrow = "▲" if pval > 0 else "▼" if pval < 0 else "○"
                ext = f"pk={peak:.3f}" if pval > 0 else f"tr={trough:.3f}" if pval < 0 else "─"
                line = f"    {ccy}: {sign}{val:.3f}  {bar:<{bar_max}}  {arrow}{streak:<3d}   {ext}  "
                if rank < 2:
                    print(f"║\033[92m{line}\033[0m║")
                elif rank >= n - 2:
                    print(f"║\033[91m{line}\033[0m║")
                else:
                    print(f"║{line}║")
            print("╠══════════════════════════════════════════════════════════════════════╣")

        # ── GRAPH STATE ──────────────────────────────────────────
        qual = health.graph_quality
        qual_bar = int(qual * 20)
        print(f"║  GRAPH: quality={qual:.3f} {'█' * qual_bar:<20}  ║")
        if configured_symbols_count:
            uni_color = "\033[92m" if available_symbols_count == configured_symbols_count else "\033[93m"
            print(f"║  UNIVERSE: {uni_color}{available_symbols_count}/{configured_symbols_count}{reset} symbols available                         ║")
        if observability:
            obs_sorted = sorted(observability.items(), key=lambda x: x[1], reverse=True)
            obs_str = "  ".join(f"{c}={v:.2f}" for c, v in obs_sorted)
            print(f"║  CURRENCY RELIABILITY: {obs_str:<52}  ║")
        if z_scores:
            z_sorted = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)
            z_str = "  ".join(f"{c}={v:+.1f}" for c, v in z_sorted)
            print(f"║  CURRENCY Z-SCORES:     {z_str:<52}  ║")
        if stability:
            stab_sorted = sorted(stability.items(), key=lambda x: x[1], reverse=True)
            stab_str = "  ".join(f"{c}={v:.2f}" for c, v in stab_sorted)
            print(f"║  STRENGTH STABILITY:    {stab_str:<52}  ║")
        if edge_balance:
            bal_sorted = sorted(edge_balance.items(), key=lambda x: x[1], reverse=True)
            bal_str = "  ".join(f"{c}={v*100:.0f}%" for c, v in bal_sorted)
            print(f"║  EDGE DISTRIBUTION:     {bal_str:<52}  ║")
        print("╠══════════════════════════════════════════════════════════════════════╣")

        # ── GRAPH HOLES (missing symbols) ────────────────────────
        if missing_symbols:
            print(f"║  MISSING SYMBOLS: {len(missing_symbols)} — no M1 data from broker               ║")
            if len(missing_symbols) <= 4:
                print(f"║    {', '.join(missing_symbols):<64}  ║")
            else:
                mid = (len(missing_symbols) + 1) // 2
                for i in range(mid):
                    left = missing_symbols[i] if i < len(missing_symbols) else ""
                    ri = i + mid
                    right = missing_symbols[ri] if ri < len(missing_symbols) else ""
                    row = f"  {left:<10s}  {right:<10s}"
                    print(f"║{row:<64}  ║")
            if missing_impact:
                imp = sorted(missing_impact.items(), key=lambda x: x[1], reverse=True)
                imp_str = "  ".join(f"{c}: -{v}" for c, v in imp if v > 0)
                print(f"║  CURRENCIES AFFECTED: {imp_str:<48}  ║")
        if unavailable_symbols:
            print(f"║  EXCLUDED SYMBOLS ({len(unavailable_symbols)} not in MT5):               ║")
            exc_str = ", ".join(unavailable_symbols)
            print(f"║    {exc_str:<62}  ║")
        print("╠══════════════════════════════════════════════════════════════════════╣")

        # ── GRAPH HEALTH REPORT ──────────────────────────────────
        if health_report:
            hr = health_report
            print(f"║  GRAPH HEALTH: {hr.get('active_pairs',0)}/{hr.get('total_pairs',0)} pairs  "
                  f"conn={hr.get('connectivity',0):.2f}  "
                  f"worst={hr.get('worst_currency','?')}  "
                  f"conf={hr.get('confidence_level','?')}  ║")
        if stress_test:
            stable_currencies = [c for c, s in stress_test.items() if s is not None]
            if stable_currencies:
                print(f"║  STRESS TEST: all {len(stable_currencies)} currencies stable on removal              ║")
        if exec_fail:
            print(f"║  ! EXEC FAIL: {exec_fail:<62s}  ║")
        if recent_failures:
            for f in recent_failures:
                sym = f.get("symbol", "?")
                ev = f.get("event", "?")
                reason = f.get("reason", "")
                print(f"║  ! EXEC FAIL: {sym:<6s} {ev:<12s} {reason:<44s}  ║")
        if pipeline_metrics:
            gen = pipeline_metrics.get("generated", 0)
            bst = pipeline_metrics.get("burst_hyp", 0)
            con = pipeline_metrics.get("confirmed", 0)
            rej = pipeline_metrics.get("burst_rejected", 0)
            rnk = pipeline_metrics.get("ranked", 0)
            sel = pipeline_metrics.get("selected", 0)
            rsk = pipeline_metrics.get("risk_approved", 0)
            exe = pipeline_metrics.get("executed", 0)
            bar = pipeline_metrics.get("bar_aligned", 0)
            print(f"║  PIPELINE: gen={gen:<2d} burst={bst:<2d} confirm={con:<2d} reject={rej:<2d} ranked={rnk:<2d} select={sel:<2d} risk={rsk:<2d} exe={exe:<2d} bar={bar:<2d}║")

        print("╠══════════════════════════════════════════════════════════════════════╣")

        # ── RISK STATE ───────────────────────────────────────────
        cooldown_str = f"COOLDOWN: {cooldown_remaining}s" if cooldown_active else "COOLDOWN: INACTIVE"
        print(f"║  RISK: lot={lot_size:.2f}  open={total_lots:.2f}/{max_total_lots:.2f}  "
              f"target=${profit_target:.0f}  {cooldown_str:<21s}  ║")

        # ── POSITIONS ────────────────────────────────────────────
        pnl_str = f"${pnl:.2f}"
        print(f"║  POSITIONS: {len(positions)} open     Unrealised PnL: {pnl_str:>10s}             ║")
        if positions:
            for p in positions:
                sym = p.get("symbol", "?")
                d = p.get("direction", "?")
                entry = p.get("entry", 0.0)
                curr = p.get("current", entry)
                pnl_p = p.get("pnl", 0.0)
                age_s = p.get("age_s", 0)
                age_str = f"{int(age_s // 3600):02d}h{int((age_s % 3600) // 60):02d}m"
                pnl_sign = "+" if pnl_p >= 0 else ""
                chg = ((curr - entry) / entry * 10000) if entry != 0 else 0
                print(f"║    {sym:<6s} {d:<4s}  entry={entry:.5f}  "
                      f"chg={chg:+.1f} pip  PnL={pnl_sign}{pnl_p:.2f}  age={age_str}    ║")
        else:
            print(f"║    (no open positions)                                            ║")
        # ── EVENT LOG ─────────────────────────────────────────────
        if self.latest_event:
            ev = self.latest_event
            ev_name = ev.get("event", "")
            ev_time = ev.get("time", 0)
            ev_age = now - ev_time
            if ev_age < 60:
                print(f"║  ! EVENT: {ev_name:<20s}  ago={ev_age:.1f}s                              ║")
            elif ev_age < 300:
                print(f"║  ! EVENT: {ev_name:<20s}  ago={int(ev_age)}s                              ║")
            else:
                self.latest_event = None
        # ── FACTOR EXPOSURE ──────────────────────────────────────
        if factor_exposure:
            print("║  CURRENCY FACTOR EXPOSURE (net portfolio)                          ║")
            for ccy, val in sorted(factor_exposure.items(), key=lambda x: abs(x[1]), reverse=True):
                if val != 0:
                    bar = "█" * min(abs(val) * 8, 16)
                    sign = "+" if val >= 0 else ""
                    print(f"║    {ccy}: {sign}{val:+.0f}  {bar:<16}  ║")
        if concentration:
            print("║  FACTOR CONCENTRATION (% of total absolute exposure)                ║")
            conc_sorted = sorted(concentration.items(), key=lambda x: x[1], reverse=True)
            conc_str = "  ".join(f"{c}={v*100:.0f}%" for c, v in conc_sorted)
            print(f"║    {conc_str:<60}  ║")
        print("╠══════════════════════════════════════════════════════════════════════╣")

        # ── TOP HYPOTHESIS ───────────────────────────────────────
        if top_hypothesis:
            sym = top_hypothesis.get("symbol", "")
            dir_str = "BUY " if top_hypothesis.get("direction", 0) > 0 else "SELL"
            conf = top_hypothesis.get("confidence", 0)
            drs = top_hypothesis.get("drs_score", 0)
            base_strength = top_hypothesis.get("base_strength", 0)
            quote_strength = top_hypothesis.get("quote_strength", 0)
            spread = (base_strength - quote_strength)

            print(f"║  TOP HYPOTHESIS: {sym:<6s} {dir_str}  "
                  f"conf={conf:.2f}  drs={drs:.2f}  "
                  f"base={base_strength:+.4f}  quote={quote_strength:+.4f}  "
                  f"Δ={spread:+.4f}  ║")
        else:
            print(f"║  TOP HYPOTHESIS: none (below MIN_CONFIDENCE or no signal)        ║")
        print("╠══════════════════════════════════════════════════════════════════════╣")

        # ── NME TRADE SNAPSHOTS ──────────────────────────────────
        if nme_trade_snapshots:
            print(f"║  NME AT ENTRY {' ' * 57}║")
            for s in nme_trade_snapshots[-5:]:
                t = time.strftime("%H:%M:%S", time.localtime(s.get("time", 0)))
                sym = s.get("symbol", "")
                d = s.get("direction", "")
                ldr = s.get("leader", "")
                nmi = s.get("nmi", 0)
                ph = s.get("phase", "")
                print(f"║   {t}  {sym:<6s} {d:<4s}  leader={ldr}  nmi={nmi:.2f}  {ph:<12s}            ║")
            print("╠══════════════════════════════════════════════════════════════════════╣")

        # ── SYSTEM ───────────────────────────────────────────────
        print(f"║  CYCLE: {trade_count:<8d}  │  "
              f"GRAPH QUALITY: {health.graph_quality:.3f}  │  "
              f"SOLVE LATENCY: {health.solve_latency_ms:.1f} ms              ║")
        print(f"║  MEMORY: {health.memory_mb:.1f} MB  │  "
              f"STATE: {health.state:<8s}  │  "
              f"UPTIME: {uptime_str}                               ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")

        # ── TRADE JOURNAL ────────────────────────────────────────
        import os as _os
        j_path = _os.path.join("logs", "trade_journal.jsonl")
        print(f"║  TRADE JOURNAL: {j_path:<48s}    ║")
        print(f"║  Tracks entry/exit WLS, reach, burst, streaks — analyze post-session         ║")
        print("╠══════════════════════════════════════════════════════════════════════╣")

        # ── RAW METRICS BAR (compact health line) ─────────────────
        m = "OK  " if health.mt5_ok else "FAIL"
        t = f"{health.tick_quality:.2f}".rjust(4)
        g = f"{health.graph_quality:.2f}".rjust(4)
        s = "OK" if health.last_snapshot_ok else "FAIL"
        l = f"{health.solve_latency_ms:.0f}".rjust(3)
        me = f"{health.memory_mb:.0f}".rjust(3)
        hr = health_report or {}
        conf = hr.get("confidence_level", "?").rjust(4)
        mode_str = "DIRECT" if wls_direct else conf
        print(f"  MT5={m}  TICK={t}  GRAPH={g}  SNAP={s}  SOLVE={l}ms  MEM={me}MB  MODE={mode_str}  |  "
              f"POS={len(positions)}  PnL={pnl_str}")


        # ── PERSISTENT LOG (JSONL, 30s interval) ─────────────────
        if now - self._last_log_ts >= 30.0:
            self._last_log_ts = now
            record = {
                "ts": ts,
                "uptime": uptime_s,
                "mode": mode,
                "health_state": health.state,
                "mt5_ok": health.mt5_ok,
                "tick_quality": health.tick_quality,
                "graph_quality": health.graph_quality,
                "solve_latency_ms": health.solve_latency_ms,
                "memory_mb": health.memory_mb,
                "positions": len(positions),
                "pnl": round(pnl, 2),
                "trade_count": trade_count,
                "top_symbol": top_hypothesis.get("symbol") if top_hypothesis else None,
                "top_conf": round(top_hypothesis.get("confidence", 0), 4) if top_hypothesis else None,
                "top_drs": round(top_hypothesis.get("drs_score", 0), 4) if top_hypothesis else None,
                "currency_strengths": {k: round(v, 5) for k, v in (currency_strengths or {}).items()},
                "factor_exposure": {k: v for k, v in (factor_exposure or {}).items() if v != 0},
                "observability": {k: round(v, 3) for k, v in (observability or {}).items()},
                "z_scores": {k: round(v, 2) for k, v in (z_scores or {}).items()},
                "stability": {k: round(v, 3) for k, v in (stability or {}).items()},
                "missing_symbols": len(missing_symbols or []),
                "health_conf": (health_report or {}).get("confidence_level"),
                "production_ready": production_ready,
                "max_concentration": round(max(concentration.values()), 3) if concentration else 0.0,
                "stress_test_stable": sum(1 for v in (stress_test or {}).values() if v is not None),
                "currency_bursts": {k: round(v, 3) for k, v in (currency_bursts or {}).items()},
                "strength_persistence": {
                    k: {"dir": v.get("direction", 0), "streak": v.get("streak", 0)}
                    for k, v in (strength_persistence or {}).items()
                },
                "burst_persistence": {
                    k: {"dir": v.get("direction", 0), "streak": v.get("streak", 0), "gap": v.get("neutral_gap", 0)}
                    for k, v in (persistence or {}).items()
                },
                "pipeline": pipeline_metrics,
                "universe_available": available_symbols_count,
                "universe_configured": configured_symbols_count,
            }
            self._log_jsonl(record)
