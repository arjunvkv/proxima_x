class MetricsCollector:
    @staticmethod
    def collect(demo, env):
        broker = env.broker if hasattr(env, 'broker') else None
        ledger = env.ledger if hasattr(env, 'ledger') else None

        trades = broker.history if (broker and hasattr(broker, "history")) else []
        wins = sum(1 for t in trades if t.get("profit", 0) > 0)
        losses = sum(1 for t in trades if t.get("profit", 0) <= 0)
        pnl = sum(t.get("profit", 0) for t in trades)
        gross_profit = sum(t.get("profit", 0) for t in trades if t.get("profit", 0) > 0)
        gross_loss = abs(sum(t.get("profit", 0) for t in trades if t.get("profit", 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        raw_signals = len(ledger.signals) if (ledger and hasattr(ledger, "signals")) else 0
        executed_signals = len([s for s in (ledger.signals if (ledger and hasattr(ledger, "signals")) else []) if s.get("phase") == "executed"])
        rejected_signals = len([s for s in (ledger.signals if (ledger and hasattr(ledger, "signals")) else []) if s.get("phase") == "gate_reject"])
        execution_rate = executed_signals / raw_signals if raw_signals > 0 else 0.0

        h_ticks = ledger.h_ticks if (ledger and hasattr(ledger, "h_ticks")) else ""
        h_signals = ledger.h_signals if (ledger and hasattr(ledger, "h_signals")) else ""
        h_trades = ledger.h_trades if (ledger and hasattr(ledger, "h_trades")) else ""
        h_state = ledger.h_state if (ledger and hasattr(ledger, "h_state")) else ""

        doa_count = len(demo._wfv_records) if hasattr(demo, '_wfv_records') else 0
        rotation_changes = len(getattr(demo, '_top3_history', []))
        active_symbols = len(getattr(demo, '_active_symbols', set()))
        allocation_entropy = MetricsCollector._entropy(getattr(demo, '_allocations', {}))
        doa_ready = getattr(demo._doa, 'ready', False) if hasattr(demo, '_doa') else False
        lct_score = getattr(demo._lct, 'convergence_score', lambda: 0.0)() if hasattr(demo, '_lct') else 0.0

        return {
            # Legacy trade metrics
            "trade_count": len(trades),
            "win_rate": wins / max(1, len(trades)),
            "net_pnl": pnl,
            "profit_factor": profit_factor,
            "raw_signals": raw_signals,
            "rejected_signals": rejected_signals,
            "executed_signals": executed_signals,
            "execution_rate": execution_rate,
            # Determinism hashes
            "H_ticks": h_ticks,
            "H_signals": h_signals,
            "H_trades": h_trades,
            "H_state": h_state,
            # V3/V4 behavioral metrics
            "doa_evaluations": doa_count,
            "doa_ready": doa_ready,
            "rotation_history_entries": rotation_changes,
            "active_symbol_count": active_symbols,
            "allocation_entropy": allocation_entropy,
            "lct_score": round(lct_score, 4),
        }

    @staticmethod
    def _entropy(weights: dict) -> float:
        import math
        vals = [abs(w) for w in weights.values()]
        total = sum(vals)
        if total == 0:
            return 0.0
        probs = [v / total for v in vals]
        return -sum(p * math.log(p) for p in probs if p > 0)
