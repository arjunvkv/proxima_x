"""Signal Aggregation Layer — combines RollingAccumulator, CrossSymbolConsensus, VolatilityGate.
Trade trigger:
  LONG:  agg_score >= +ENTRY_THRESHOLD AND consensus_aligned(+, strength) AND tradable
  SHORT: agg_score <= -ENTRY_THRESHOLD AND consensus_aligned(-, strength) AND tradable
Else: FLAT
No new entry if position already open for symbol.
"""
from .signal_aggregator import RollingAccumulator
from .consensus import CrossSymbolConsensus
from .volatility_gate import VolatilityGate


class SignalAggregationLayer:
    def __init__(self, entry_threshold=0.65, accum_window=100, accum_decay=0.977,
                 consensus_min=0.60, vol_window=20, vol_min_z=0.15, vol_max_z=2.0,
                 sym_stats=None):
        self.entry_threshold = entry_threshold
        self.accumulator = RollingAccumulator(window=accum_window, decay=accum_decay)
        self.consensus = CrossSymbolConsensus(min_strength=consensus_min)
        self.volgate = VolatilityGate(window=vol_window, vol_min_z=vol_min_z,
                                      vol_max_z=vol_max_z, sym_stats=sym_stats)
        self._active_symbols = set()
        self._scores = {}

    def update(self, sym, signal, confidence, price, all_scores=None):
        """Update state for a single symbol tick. Returns sal_signal: -1, 0, or +1."""
        self.accumulator.update(signal, confidence)
        self.volgate.update(sym, price)

        if all_scores:
            self._scores = all_scores
        self.consensus.update(self._scores)

        agg_score = self.accumulator.score()
        consensus_aligned = self.consensus.is_aligned()
        cdir = self.consensus.consensus_direction()
        tradable = self.volgate.tradable(sym)

        # Decision (no position blocking — accumulator naturally modulates)
        if (agg_score >= self.entry_threshold
                and consensus_aligned
                and cdir >= 0
                and tradable):
            self._active_symbols.add(sym)
            return 1

        if (agg_score <= -self.entry_threshold
                and consensus_aligned
                and cdir <= 0
                and tradable):
            self._active_symbols.add(sym)
            return -1

        return 0

    def close(self, sym):
        """Remove symbol from active set (no-op in v2)."""
        pass

    def reset(self):
        self.accumulator.reset()
        self.volgate.reset()
        self._active_symbols.clear()
        self._scores.clear()

    def agg_score(self):
        return self.accumulator.score()

    def consensus_strength(self):
        return self.consensus.consensus_strength()

    def vol_tradable(self, sym):
        return self.volgate.tradable(sym)

    def compute_ucf_metrics(self, aggregated: list[dict], ucf_field) -> list[dict]:
        if ucf_field is None or not hasattr(ucf_field, 'field'):
            return []
        feedback = []
        for signal in aggregated:
            sym = signal.get("symbol", "")
            ucf_data = ucf_field.field.get(sym, {})
            if not ucf_data:
                continue
            signal_conf = signal.get("confidence", 0.5)
            ucf_score = ucf_data.get("conviction_score", 0.0)
            signal_dir = signal.get("direction", 0)
            ucf_dir = ucf_data.get("direction", 0)
            agreement_delta = abs(signal_conf - ucf_score)
            divergence = 1.0 if signal_dir != 0 and ucf_dir != 0 and signal_dir != ucf_dir else 0.0
            confidence_tension = abs(signal_conf - ucf_score) / max(signal_conf, ucf_score, 0.01)
            feedback.append({
                "symbol": sym,
                "agreement_delta": round(agreement_delta, 4),
                "divergence": divergence,
                "confidence_tension": round(confidence_tension, 4),
            })
        return feedback
