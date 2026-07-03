import logging
from datetime import datetime
from typing import Optional, Callable

logger = logging.getLogger("proxima_ops.freq_reality.pipeline")


class FrequencyRealityPipeline:
    def __init__(self, blocked_tracker, executed_tracker,
                 future_return_engine, cost_analysis, classifier,
                 dpl_live=None):
        self._blocked = blocked_tracker
        self._executed = executed_tracker
        self._future = future_return_engine
        self._analysis = cost_analysis
        self._classifier = classifier
        self._dpl = dpl_live

    def record_blocked(self, symbol: str, es_rank: float, at_rank: float,
                       threshold: float, block_reason: str,
                       price: float = 0.0,
                       frequency_band: str = None, wavelet: str = None,
                       energy_regime: int = None, time_regime: int = None,
                       combined_regime: int = None):
        signal_id = f"BLK_{symbol}_{int(datetime.now().timestamp())}"
        self._blocked.record(symbol, es_rank, at_rank, threshold,
                             block_reason, price, future_returns=None,
                             signal_id=signal_id)
        self._tag_direction_metadata(signal_id, symbol, frequency_band, wavelet)
        if price > 0:
            self._future.record_snapshot(signal_id, symbol, price)
        if self._dpl and price > 0:
            self._dpl.record_features(
                signal_id, symbol, es_rank, at_rank, price,
                is_blocked=True, block_reason=block_reason,
                frequency_band=frequency_band, wavelet=wavelet,
                energy_regime=energy_regime, time_regime=time_regime,
                combined_regime=combined_regime)

    def record_executed(self, symbol: str, es_rank: float, at_rank: float,
                        threshold: float, entry_price: float,
                        ticket: int = None,
                        frequency_band: str = None, wavelet: str = None,
                        energy_regime: int = None, time_regime: int = None,
                        combined_regime: int = None):
        signal_id = f"EXE_{symbol}_{int(datetime.now().timestamp())}_{ticket or 'na'}"
        self._executed.record(symbol, es_rank, at_rank, threshold,
                              entry_price, future_returns=None, ticket=ticket,
                              signal_id=signal_id)
        self._tag_direction_metadata(signal_id, symbol, frequency_band, wavelet)
        if entry_price > 0:
            self._future.record_snapshot(signal_id, symbol, entry_price)
        if self._dpl and entry_price > 0:
            self._dpl.record_features(
                signal_id, symbol, es_rank, at_rank, entry_price,
                is_blocked=False, block_reason=None,
                frequency_band=frequency_band, wavelet=wavelet,
                energy_regime=energy_regime, time_regime=time_regime,
                combined_regime=combined_regime)

    def _tag_direction_metadata(self, signal_id: str, symbol: str,
                                 frequency_band: str = None, wavelet: str = None):
        if not frequency_band and not wavelet:
            return
        for rec in self._blocked.get_all() + self._executed.get_all():
            if rec.get("signal_id") == signal_id:
                if frequency_band:
                    rec["frequency_band"] = frequency_band
                if wavelet:
                    rec["wavelet"] = wavelet
                break

    def record_executed_result(self, ticket: int, pnl: float):
        for r in self._executed.get_all():
            if r.get("ticket") == ticket:
                r["actual_pnl"] = round(pnl, 2)
                break

    def process_matured_outcomes(self) -> int:
        count = self._future.process_matured()
        self._backfill_resolved_returns()
        return count

    def _backfill_resolved_returns(self):
        for rec in self._blocked.get_all():
            self._backfill_one(rec)
        for rec in self._executed.get_all():
            self._backfill_one(rec)
        if self._dpl:
            for rec in self._blocked.get_all() + self._executed.get_all():
                sig_id = rec.get("signal_id", "")
                if sig_id and rec.get("future_resolved"):
                    self._dpl.attach_outcome(sig_id, self._future.get_returns(sig_id))

    def _backfill_one(self, rec: dict):
        sig_id = rec.get("signal_id", "")
        if not sig_id:
            return
        if rec.get("return_h20") is not None:
            return
        returns = self._future.get_returns(sig_id)
        has_any = False
        for k, v in returns.items():
            if v is not None:
                rec[k] = v
                has_any = True
        if has_any:
            rec["future_resolved"] = True

    def report(self) -> str:
        blocked_summary = self._blocked.summary()
        leakage = self._analysis.leakage_rate()
        class_result = self._classifier.classify()
        oc = self._analysis.opportunity_cost("h20")

        lines = []
        lines.append("=" * 52)
        lines.append("  FREQUENCY REALITY AUDIT")
        lines.append("=" * 52)
        lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
        lines.append("  COLLECTION")
        lines.append(f"  Blocked Signals:    {blocked_summary['total']}")
        for reason, count in blocked_summary.get("reasons", {}).items():
            lines.append(f"    {reason}: {count}")
        lines.append(f"  Executed Signals:   {self._executed.count()}")
        lines.append("")
        lines.append("  OPPORTUNITY COST (H20)")
        lines.append(f"  Blocked Mean Ret:   {oc['blocked']['mean_return']:.6f}")
        lines.append(f"  Blocked PP:         {oc['blocked']['pp']:.2%}")
        lines.append(f"  Executed Mean Ret:  {oc['executed']['mean_return']:.6f}")
        lines.append(f"  Executed PP:        {oc['executed']['pp']:.2%}")
        lines.append("")
        lines.append("  LEAKAGE")
        lines.append(f"  Blocked Profitable: {leakage['blocked_profitable']}/{leakage['blocked_total']}")
        lines.append(f"  Leakage Rate:       {leakage['leakage_rate']}%")
        lines.append("")
        lines.append("  ADJUDICATION")
        lines.append(f"  Classification:     {class_result['classification']}")
        lines.append(f"  Confidence:         {class_result['confidence']}")
        lines.append(f"  ADR:                {class_result['adr']}")
        lines.append("=" * 52)
        return "\n".join(lines)
