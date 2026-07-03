import logging
from collections import defaultdict
from typing import Dict, List, Optional

logger = logging.getLogger("proxima_demo")

REGIME_HORIZONS = {"STRUCTURED": 20, "TRANSITION": 10, "CHAOTIC": 3}


class ThesisRecord:
    def __init__(self, thesis_id: int, symbol: str, ticket: int,
                 thesis_direction: int, thesis_type: str, thesis_confidence: float,
                 regime: str, cycle_id: int,
                 oss_sig: int, shadow_sig: int, exhaustion_active: bool,
                 ecdf: float, entropy: float, topology: str,
                 p_cont: float, drift: int, rf_prob: float,
                 thesis_rf_prob: float = 0.5):
        self.thesis_id = thesis_id
        self.symbol = symbol
        self.ticket = ticket
        self.thesis_direction = thesis_direction
        self.thesis_type = thesis_type
        self.thesis_confidence = thesis_confidence
        self.regime = regime
        self.cycle_id = cycle_id
        self.oss_sig = oss_sig
        self.shadow_sig = shadow_sig
        self.exhaustion_active = exhaustion_active
        self.ecdf = ecdf
        self.entropy = entropy
        self.topology = topology
        self.p_cont = p_cont
        self.drift = drift
        self.rf_prob = rf_prob
        self.thesis_rf_prob = thesis_rf_prob
        self.evaluation_horizon = REGIME_HORIZONS.get(regime, 10)
        self.entry_tick = cycle_id
        self.label = None
        self.resolved = False
        self.exit_reason = None
        self.horizon_probes = {}
        self.horizon_labels = None
        self.fracture_score = None

    def features(self) -> list:
        base = [self.oss_sig, self.shadow_sig, int(self.exhaustion_active),
                self.ecdf, self.entropy, self.p_cont, self.drift,
                self.rf_prob, self.thesis_confidence,
                self.evaluation_horizon]
        if self.horizon_labels is not None:
            base.extend([int(self.horizon_labels[0]), int(self.horizon_labels[1]),
                         int(self.horizon_labels[2])])
            base.append(self.fracture_score if self.fracture_score is not None else 0.0)
        return base

    def __repr__(self):
        return (f"Thesis({self.thesis_id}|{self.symbol}|d={self.thesis_direction:+d}|"
                f"t={self.thesis_type}|r={self.regime}|"
                f"{'RESOLVED' if self.resolved else 'PENDING'})")


class ThesisBuffer:
    def __init__(self):
        self._records: Dict[int, ThesisRecord] = {}
        self._pending: Dict[int, ThesisRecord] = {}
        self._resolved: List[ThesisRecord] = []
        self._next_id = 1

    def record(self, symbol: str, ticket: int,
               thesis_direction: int, thesis_type: str, thesis_confidence: float,
               regime: str, cycle_id: int,
               oss_sig: int, shadow_sig: int, exhaustion_active: bool,
               ecdf: float, entropy: float, topology: str,
               p_cont: float, drift: int, rf_prob: float,
               thesis_rf_prob: float = 0.5) -> int:
        tid = self._next_id
        self._next_id += 1
        rec = ThesisRecord(tid, symbol, ticket,
                           thesis_direction, thesis_type, thesis_confidence,
                           regime, cycle_id,
                           oss_sig, shadow_sig, exhaustion_active,
                           ecdf, entropy, topology,
                           p_cont, drift, rf_prob,
                           thesis_rf_prob=thesis_rf_prob)
        self._records[tid] = rec
        self._pending[tid] = rec
        logger.info(f"[THESIS_RECORD] id={tid} {symbol} dir={thesis_direction:+d} "
                    f"type={thesis_type} conf={thesis_confidence:.2f} "
                    f"regime={regime} cycle={cycle_id}")
        return tid

    def attach_horizons(self, thesis_id: int, horizons: list, entry_price: float = 0.0):
        rec = self._records.get(thesis_id)
        if rec is None:
            return
        rec.horizon_probes = {}
        for h in sorted(horizons):
            rec.horizon_probes[h] = {
                "ticks_elapsed": 0,
                "resolved": False,
                "entry_price": entry_price,
                "label": None,
            }
        logger.info(f"[HORIZON_ATTACH] id={thesis_id} {rec.symbol} probes={horizons}")

    def tick_horizons(self, symbol: str, price: float, epsilon: float = 0.0001):
        for rec in list(self._pending.values()):
            if rec.symbol != symbol:
                continue
            if not rec.horizon_probes:
                continue
            for h, probe in sorted(rec.horizon_probes.items()):
                if probe["resolved"]:
                    continue
                probe["ticks_elapsed"] += 1
                if probe["ticks_elapsed"] >= h:
                    ret = (price - probe["entry_price"]) / max(probe["entry_price"], 1e-12)
                    direction = rec.thesis_direction
                    if abs(ret) < epsilon:
                        label = 0
                    elif (ret > 0 and direction > 0) or (ret < 0 and direction < 0):
                        label = 1
                    else:
                        label = -1
                    probe["resolved"] = True
                    probe["label"] = label
                    logger.info(f"[HORIZON_RESOLVE] id={rec.thesis_id} "
                                f"{rec.symbol} horizon={h}t "
                                f"ret={ret:+.5f} label={label:+d}")
            if all(p["resolved"] for p in rec.horizon_probes.values()):
                hl = tuple(rec.horizon_probes[h]["label"] for h in sorted(rec.horizon_probes))
                pairs = [(0, 1), (1, 2), (0, 2)]
                hl_list = list(hl)
                disagreements = sum(1 for i, j in pairs if hl_list[i] != hl_list[j])
                rec.horizon_labels = hl
                rec.fracture_score = disagreements / len(pairs)
                logger.info(f"[HORIZON_ALL] id={rec.thesis_id} "
                            f"labels={hl} fracture={rec.fracture_score:.2f}")

    def resolve(self, ticket: int, exit_profit: float, exit_reason: str = "CLOSED") -> Optional[int]:
        for tid, rec in list(self._pending.items()):
            if rec.ticket == ticket:
                rec.resolved = True
                rec.exit_reason = exit_reason
                label = 1 if exit_profit > 0 else 0
                rec.label = label
                self._resolved.append(rec)
                del self._pending[tid]
                logger.info(f"[THESIS_RESOLVE] id={tid} {rec.symbol} "
                            f"profit={exit_profit:+.2f} label={label} reason={exit_reason}")
                return label
        return None

    def pending_count(self) -> int:
        return len(self._pending)

    def resolved_count(self) -> int:
        return len(self._resolved)

    def total_count(self) -> int:
        return len(self._records)

    def positive_rate(self) -> float:
        if not self._resolved:
            return 0.0
        return sum(1 for r in self._resolved if r.label == 1) / len(self._resolved)

    def get_pending(self) -> List[ThesisRecord]:
        return list(self._pending.values())

    def get_resolved(self) -> List[ThesisRecord]:
        return list(self._resolved)

    def get_training_batch(self, min_samples: int = 10) -> tuple:
        if len(self._resolved) < min_samples:
            return [], []
        X = [r.features() for r in self._resolved]
        y = [r.label for r in self._resolved]
        return X, y

    def stats(self) -> dict:
        return {
            "total": self.total_count(),
            "pending": self.pending_count(),
            "resolved": self.resolved_count(),
            "positive_rate": round(self.positive_rate(), 3),
        }
