from typing import Any
from proxima_ops.decision.gate.mra_signal import MarketRealityAnchor
from proxima_ops.decision.gate.emd_signal import ExecutionMicrostructureDrift
from proxima_ops.decision.gate.recovery_policy import RecoveryPolicy, check_regime_failure


class MT5GateShadow:
    def __init__(self) -> None:
        self._mra = MarketRealityAnchor()
        self._emd = ExecutionMicrostructureDrift()
        self._recovery = RecoveryPolicy()
        self._decisions: list[dict] = []

    def evaluate(self, symbol: str, tick: dict[str, Any], cycle: int) -> dict:
        bid = tick.get("bid", 0)
        ask = tick.get("ask", 0)
        mid = (bid + ask) / 2 if bid and ask else 1.10
        spread = abs(ask - bid) if bid and ask else 0.0002
        latency = tick.get("latency", 0.05)
        exp_slip = tick.get("expected_slippage", 0.0001)
        act_slip = tick.get("actual_slippage", 0.0001)
        rv = tick.get("recovery_velocity", 0.5)
        rc = tick.get("recovery_confidence", 0.5)

        self._mra.update(symbol, mid, spread)
        self._emd.record_fill(symbol, latency, exp_slip, act_slip)
        self._recovery.update_rv(symbol, rv)
        self._recovery.update_rc(symbol, rc)
        reg_vol = self._mra.get_regime_volatility(symbol)
        self._recovery.set_regime_volatility(symbol, reg_vol)
        dampen = reg_vol > 0.7
        mra_result = self._mra.get_mra(symbol, dampen=dampen)
        emd_result = self._emd.get_emd(symbol, dampen=dampen)
        rp_result = self._recovery.resolve(symbol)
        rf_result = check_regime_failure(tick.get("ucf_alignment", 0.5))

        decision = {
            "cycle": cycle,
            "symbol": symbol,
            "mra_score": mra_result["mra_score"],
            "emd_score": emd_result["emd_score"],
            "atr_normalized": mra_result["atr_normalized"],
            "spread_stability": mra_result["spread_stability"],
            "latency_variance": emd_result["latency_variance"],
            "slippage_deviation": emd_result["slippage_deviation"],
            "rv_score": rp_result["rv_score"],
            "rc_score": rp_result["rc_score"],
            "classification": rp_result["classification"],
            "veto_applied": rp_result["veto_applied"],
            "veto_threshold": rp_result.get("veto_threshold", 0.5),
            "regime_vol": round(reg_vol, 4),
            "dampened": dampen,
            "regime_failure": rf_result,
        }
        self._decisions.append(decision)
        return decision

    def get_stats(self) -> dict:
        total = len(self._decisions)
        if total == 0:
            return {"total_decisions": 0}
        structural = sum(1 for d in self._decisions if d["classification"] == "STRUCTURAL")
        vetoes = sum(1 for d in self._decisions if d["veto_applied"])
        dampened = sum(1 for d in self._decisions if d["dampened"])
        degraded = sum(1 for d in self._decisions if d["regime_failure"] == "DEGRADED")
        return {
            "total_decisions": total,
            "structural_count": structural,
            "structural_rate": round(structural / max(1, total), 4),
            "veto_count": vetoes,
            "veto_rate": round(vetoes / max(1, total), 4),
            "dampened_count": dampened,
            "dampened_rate": round(dampened / max(1, total), 4),
            "degraded_count": degraded,
            "pass_rate": round((total - structural) / max(1, total), 4),
        }

    def reset(self) -> None:
        self._decisions.clear()
