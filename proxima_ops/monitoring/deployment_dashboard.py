import time
from typing import List, Optional, Dict
from proxima_ops.monitoring.trigger_statistics import TriggerStatistics
from proxima_ops.monitoring.opportunity_tracker import OpportunityTracker
from proxima_ops.monitoring.deployment_context import DeploymentContext
import logging

logger = logging.getLogger("proxima_ops.dashboard")


# Lazy VPL regime signal cache
_VPL_CACHE: Dict[str, dict] = {}
_VPL_CACHE_TIME = 0.0


def _refresh_vpl_cache():
    global _VPL_CACHE, _VPL_CACHE_TIME
    now = time.time()
    if now - _VPL_CACHE_TIME < 60.0 and _VPL_CACHE:
        return
    try:
        from deployment.get_vpl_signal import get_current_signal
        symbols = ["EURJPY", "USDJPY", "GBPJPY", "XAUUSD", "EURUSD"]
        for sym in symbols:
            try:
                sig = get_current_signal(sym)
                if sig:
                    _VPL_CACHE[sym] = sig
                else:
                    _VPL_CACHE[sym] = {"state": "NO_DATA", "regime": "NO_DATA", "trade_permission": "NONE"}
                    logger.warning(f"VPL: {sym} returned no signal (insufficient bars?)")
            except Exception as e:
                _VPL_CACHE[sym] = {"state": "DATA_CORRUPT", "regime": "DATA_CORRUPT", "trade_permission": "NONE"}
                logger.error(f"VPL: {sym} failed to load: {e}")
        _VPL_CACHE_TIME = now
    except Exception as e:
        logger.error(f"VPL cache refresh failed: {e}")


class DeploymentDashboard:
    def __init__(self, start_time: float, trigger_stats: TriggerStatistics,
                 opp_tracker: OpportunityTracker,
                 deployment_context: Optional[DeploymentContext] = None):
        self._start_time = start_time
        self._stats = trigger_stats
        self._tracker = opp_tracker
        self._ctx = deployment_context

    def _format_runtime(self) -> str:
        elapsed_sec = time.time() - self._start_time
        hours = int(elapsed_sec // 3600)
        minutes = int((elapsed_sec % 3600) // 60)
        return f"{hours}h {minutes}m"

    def render(self, eval_data: dict, account_info: dict, score_data: dict,
               seconds_to_next_eval: int, spinner: str, closed_trades: int,
               rotation_events: int = 0, lock_events: int = 0,
               migration_events: int = 0, avg_hold_bars: float = 0.0,
               top3_ranked: list = None,
               paper_metrics: dict = None) -> str:
        import numpy as np
        runtime_str = self._format_runtime()

        # Lifetime
        lt_evaluated = self._stats.evaluated_count
        lt_triggered = self._stats.trigger_count
        lt_executed = self._stats.executed_count
        lt_blocked = self._stats.blocked_count
        lt_trig_rate = (lt_triggered / max(lt_evaluated, 1)) * 100.0
        lt_exec_rate = (lt_executed / max(lt_evaluated, 1)) * 100.0
        lt_block_rate = (lt_blocked / max(lt_evaluated, 1)) * 100.0

        # Session
        ss = self._stats.session_summary()
        ss_eval = ss["evaluated"]
        ss_trig = ss["triggered"]
        ss_exec = ss["executed"]
        ss_block = ss["blocked"]
        ss_trig_rate = (ss_trig / max(ss_eval, 1)) * 100.0
        ss_exec_rate = (ss_exec / max(ss_eval, 1)) * 100.0
        ss_block_rate = (ss_block / max(ss_eval, 1)) * 100.0

        if closed_trades < 10:
            phase = "COLLECTING_EVIDENCE"
        elif closed_trades < 25:
            phase = "EARLY_VALIDATION"
        elif closed_trades < 50:
            phase = "INTERMEDIATE_VALIDATION"
        else:
            phase = "FULL_VALIDATION"

        lines = []
        lines.append("=" * 76)
        header = f"   [ {spinner} ]  PROXIMA V6 OPS - TOP-3 ROTATION  |  PHASE: {phase}"
        lines.append(header)
        if self._ctx:
            lines.append(f"   Deployment: {self._ctx.deployment_id} | Session: {self._ctx.session_id}")
        lines.append("=" * 76)

        status = "ACTIVE"
        acc_num = account_info.get("login", "N/A")
        balance = account_info.get("balance", 0.0)
        score = score_data.get("current_score", 0.0)
        classification = score_data.get("classification", "UNKNOWN")

        lines.append(f"Status: {status} | Account: {acc_num} | Balance: ${balance:,.2f} | Score: {score:.3f} ({classification})")
        lines.append(f"Runtime: {runtime_str} | Next Signal Evaluation in: {seconds_to_next_eval}s")
        lines.append("-" * 76)
        lines.append("")
        lines.append("  SESSION (this deployment only)")
        lines.append(f"  Evaluated: {ss_eval} | Triggered: {ss_trig} ({ss_trig_rate:.2f}%) | Executed: {ss_exec} ({ss_exec_rate:.2f}%) | Blocked: {ss_block} ({ss_block_rate:.2f}%)")
        lines.append("")
        lines.append("  LIFETIME (all deployments)")
        lines.append(f"  Evaluated: {lt_evaluated} | Triggered: {lt_triggered} ({lt_trig_rate:.2f}%) | Executed: {lt_executed} ({lt_exec_rate:.2f}%) | Blocked: {lt_blocked} ({lt_block_rate:.2f}%)")
        lines.append("")
        other_blocks = max(0, lt_blocked - self._stats.spread_blocks - self._stats.position_blocks - self._stats.risk_blocks)
        lines.append(f"  Block Analysis -> Spread: {self._stats.spread_blocks} | Pos Exists: {self._stats.position_blocks} | Risk Limit: {self._stats.risk_blocks} | Other: {other_blocks}")
        lines.append("-" * 76)
        lines.append("RESEARCH -> PAPER TRANSFER:")
        if paper_metrics:
            raw_pp = paper_metrics.get("pp", 0)
            pp_value = raw_pp if isinstance(raw_pp, (int, float)) else 0.0
            raw_hold = paper_metrics.get("avg_hold", 0)
            avg_hold = raw_hold if isinstance(raw_hold, (int, float)) else 0.0
            raw_sharpe = paper_metrics.get("sharpe", 0)
            sharpe_value = raw_sharpe if isinstance(raw_sharpe, (int, float)) else 0.0
        else:
            pp_value = 0.0
            avg_hold = 0.0
            sharpe_value = 0.0
        lines.append(f"Target PP: 0.536 | Target Hold: 15.6 bars | Target Sharpe: ~6.0")
        lines.append(f"Current PP: {pp_value:.3f} | Current Hold: {avg_hold:.1f} bars | Current Sharpe: {sharpe_value:.2f}")
        lines.append("-" * 76)
        lines.append(f"ARCHITECTURE: Top-3 Rotation | Lock: 3 bars | Exit: H20 Cap | Risk: Emergency Stop")
        lines.append(f"ROTATION: {rotation_events} events | LOCKS: {lock_events} events | MIGRATIONS: {migration_events} observed")
        lines.append("-" * 76)
        lines.append("ASSET EVALUATION:")
        _refresh_vpl_cache()

        BASE_SPREAD = {"EURJPY": 15, "USDJPY": 15, "GBPJPY": 20, "XAUUSD": 50, "EURUSD": 10}

        lines.append(f"{'Symbol':<10s} {'Price':<10s} {'Spread':<8s} {'SpdPres':<9s} {'ES Value':<12s} {'ES Rank':<10s} {'AT Rank':<10s} {'Sizing':<8s} {'Regime':<14s} {'Status':<10s} {'History (E/T/Ex)'}")

        status_map = {
            "BLOCKED (SPREAD)": "BLK_SPREAD",
            "BLOCKED (POSITION_EXISTS)": "BLK_EXIST",
            "BLOCKED (MAX_POSITIONS)": "BLK_MAXPOS",
            "BLOCKED (RISK_LIMIT)": "BLK_RISK",
            "BLOCKED (INVALID_SPREAD)": "BLK_SPRD",
            "BLOCKED (POSITION_LOCK)": "BLK_LOCK",
            "BLOCKED (NOT_IN_TOP3)": "BLK_RANK",
            "BLOCKED (THRESHOLD_NOT_MET)": "BLK_THRES",
        }

        for sym, data in eval_data.items():
            price_val = data.get("price", float('nan'))
            price_str = f"{price_val:.3f}" if not np.isnan(price_val) else "N/A"
            spread_val = data.get("spread")
            spread_str = str(spread_val) if spread_val is not None else "N/A"
            es_val = data.get("es_val", float('nan'))
            es_str = f"{es_val:.6f}" if not np.isnan(es_val) else "N/A"

            es_rank = data.get("es_rank", float('nan'))
            es_rank_str = f"{es_rank:.1%}" if not np.isnan(es_rank) else "N/A"

            at_rank = data.get("at_rank", float('nan'))
            at_rank_str = f"{at_rank:.1%}" if not np.isnan(at_rank) else "N/A"

            sizing = data.get("sizing_mult", 0.0)
            sizing_str = f"{sizing:.2f}x" if not np.isnan(sizing) else "N/A"

            runtime_regime = data.get("regime")
            if runtime_regime:
                regime_str = str(runtime_regime)[:12]
            else:
                vpl_sig = _VPL_CACHE.get(sym)
                regime_str = vpl_sig["regime"][:12] if vpl_sig else "N/A"

            status_str = data.get("status", "WATCH")
            display_status = status_map.get(status_str, status_str)

            sym_stats = self._stats.symbol_stats.get(sym, {
                "evaluated": 0, "triggered": 0, "executed": 0
            })
            hist_str = f"[{sym_stats['evaluated']} / {sym_stats['triggered']} / {sym_stats['executed']}]"

            # Elastic spread pressure
            sp = spread_val if isinstance(spread_val, (int, float)) else None
            er = es_rank if not np.isnan(es_rank) else 0.0
            base = BASE_SPREAD.get(sym, 20)
            if sp is not None and er > 0 and base > 0:
                es_norm = min(er, 1.0)
                gamma = 1.5
                elastic_limit = int(base * (1.0 + pow(es_norm, gamma)))
                spread_pressure = round(sp / max(elastic_limit, 1), 3)
            else:
                spread_pressure = "N/A"
            sp_str = f"{spread_pressure:.2f}" if isinstance(spread_pressure, float) else "N/A"

            lines.append(f"{sym:<10s} {price_str:<10s} {spread_str:<8s} {sp_str:<9s} {es_str:<12s} {es_rank_str:<10s} {at_rank_str:<10s} {sizing_str:<8s} {regime_str:<14s} {display_status:<10s} {hist_str}")

        lines.append("-" * 76)
        lines.append("TOP CURRENT OPPORTUNITIES:")

        latest_evals = self._tracker.get_latest_evals()
        sorted_opps = sorted(latest_evals, key=lambda x: x["es_rank"], reverse=True)

        for idx, opp in enumerate(sorted_opps[:3]):
            status_opp = "WATCH"
            if opp["triggered"]:
                if opp["blocked"]:
                    status_opp = f"BLOCKED ({opp['block_reason']})"
                else:
                    status_opp = "EXECUTED"
            lines.append(f" {idx + 1}. {opp['symbol']:<10s} ES Rank: {opp['es_rank']:.1%} | AT Rank: {opp['at_rank']:.1%} | Status: {status_opp}")

        if not sorted_opps:
            lines.append(" No opportunities evaluated yet")

        if top3_ranked:
            lines.append("-" * 76)
            lines.append(f"TOP-3 RANKED: {', '.join(top3_ranked)}")

        return "\n".join(lines)

    def generate_alpha_snapshot(self) -> str:
        lines = []
        lines.append("=" * 57)
        lines.append("ALPHA SNAPSHOT")
        lines.append("=" * 57)
        lines.append("")
        lines.append("Current Strongest Signals")
        lines.append("")
        lines.append(f"{'Rank':<6s} {'Symbol':<10s} {'ES Rank':<10s} {'AT Rank':<10s} {'Status':<10s}")
        lines.append("")

        latest_evals = self._tracker.get_latest_evals()
        sorted_opps = sorted(latest_evals, key=lambda x: x["es_rank"], reverse=True)

        for idx, opp in enumerate(sorted_opps):
            status = "WATCH"
            if opp["triggered"]:
                if opp["blocked"]:
                    status = "BLOCKED"
                else:
                    status = "EXECUTED"
            lines.append(f"{idx + 1:<6d} {opp['symbol']:<10s} {opp['es_rank']:.1%:<10s} {opp['at_rank']:.1%:<10s} {status:<10s}")
        lines.append("=" * 57)
        return "\n".join(lines)
