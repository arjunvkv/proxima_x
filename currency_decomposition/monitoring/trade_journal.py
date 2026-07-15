import json
import time
from pathlib import Path
from collections import defaultdict


_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_FILE = _LOG_DIR / "trade_journal.jsonl"


class TradeJournal:
    def __init__(self):
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        self._fh = _LOG_FILE.open("a", encoding="utf-8")
        self._positions: dict[str, dict] = {}

    def record_open(self, position_id: str, symbol: str, direction: str,
                     entry_price: float, volume: float, confidence: float,
                     drs_score: float, strengths: dict, peaks: dict,
                     troughs: dict, streaks: dict, bursts: dict,
                     sl: float = 0.0, tp: float = 0.0) -> None:
        now = time.time()
        self._positions[position_id] = {
            "event": "open",
            "trade_id": position_id,
            "ts": now,
            "symbol": symbol,
            "direction": direction,
            "volume": volume,
            "entry_price": entry_price,
            "sl": sl,
            "tp": tp,
            "confidence": confidence,
            "drs_score": drs_score,
            "strengths_entry": dict(strengths),
            "peaks_entry": dict(peaks),
            "troughs_entry": dict(troughs),
            "burst_entry": dict(bursts),
            "streaks_entry": dict(streaks),
        }
        record = self._positions[position_id].copy()
        self._write(record)

    def record_close(self, position_id: str, exit_price: float, pnl: float,
                      reason: str, strengths: dict, peaks: dict,
                      troughs: dict, streaks: dict, bursts: dict) -> None:
        entry = self._positions.pop(position_id, {})
        if not entry:
            return
        duration = time.time() - entry["ts"]
        rotation = {}
        if strengths and entry.get("strengths_entry"):
            all_ccy = set(strengths) | set(entry["strengths_entry"])
            for ccy in all_ccy:
                s1 = entry["strengths_entry"].get(ccy, 0)
                s2 = strengths.get(ccy, 0)
                rotation[ccy] = round(s2 - s1, 6)
        base = entry.get("symbol", "")[:3]
        quote = entry.get("symbol", "")[3:6]
        reach_exit = self._calc_reach(strengths, peaks, troughs)
        record = {
            **entry,
            "event": "close",
            "exit_price": exit_price,
            "pnl": round(pnl, 2),
            "exit_reason": reason,
            "duration_sec": round(duration, 1),
            "strengths_exit": dict(strengths) if strengths else None,
            "burst_exit": dict(bursts) if bursts else None,
            "reach_exit": reach_exit,
            "reach_entry_base": self._get_reach(entry.get("strengths_entry", {}),
                                                  entry.get("peaks_entry", {}),
                                                  entry.get("troughs_entry", {}), base),
            "reach_entry_quote": self._get_reach(entry.get("strengths_entry", {}),
                                                   entry.get("peaks_entry", {}),
                                                   entry.get("troughs_entry", {}), quote),
            "wls_rotation": rotation if rotation else None,
        }
        self._write(record)

    def _calc_reach(self, strengths: dict, peaks: dict, troughs: dict) -> dict:
        result = {}
        for ccy in (strengths or {}):
            result[ccy] = self._get_reach(strengths, peaks, troughs, ccy)
        return result

    def _get_reach(self, strengths: dict, peaks: dict, troughs: dict, ccy: str) -> float:
        val = abs(strengths.get(ccy, 0))
        if val < 1e-8:
            return 0.0
        pk = peaks.get(ccy, 0)
        tr = troughs.get(ccy, 0)
        ext = pk if strengths.get(ccy, 0) > 0 else tr
        if abs(ext) < 1e-8:
            return 1.0
        return round(abs(val / ext), 3)

    def _write(self, record: dict) -> None:
        try:
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()
        except Exception:
            pass

    def pending_count(self) -> int:
        return len(self._positions)
