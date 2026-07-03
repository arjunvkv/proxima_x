from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple


class LayerMemory:
    __slots__ = ("_store",)

    def __init__(self) -> None:
        self._store: Dict[str, List[Dict]] = defaultdict(list)

    def record(self, symbol: str, regime: str, latency_chain: List[Tuple[str, int]],
               birth_capture_ratio: Optional[float],
               realization_capture_ratio: Optional[float],
               etl: Optional[int]) -> None:
        entry = {
            "regime": regime,
            "tpi_tick": None,
            "cal_tick": None,
            "obs_tick": None,
            "entry_tick": None,
            "birth_capture_ratio": birth_capture_ratio,
            "realization_capture_ratio": realization_capture_ratio,
            "etl": etl,
        }
        for name, tick in latency_chain:
            if name == "TPI":
                entry["tpi_tick"] = tick
            elif name == "CALIBRATION":
                entry["cal_tick"] = tick
            elif name == "OBSERVER":
                entry["obs_tick"] = tick
            elif name == "ENTRY":
                entry["entry_tick"] = tick

        self._store[symbol].append(entry)

    def get_symbol_stats(self, symbol: str) -> Dict:
        records = self._store.get(symbol, [])
        if not records:
            return {"symbol": symbol, "count": 0}

        regime_groups: Dict[str, List[Dict]] = defaultdict(list)
        for r in records:
            regime_groups[r["regime"]].append(r)

        per_regime = {}
        for regime, group in regime_groups.items():
            etls = [r["etl"] for r in group if r["etl"] is not None]
            bcr = [r["birth_capture_ratio"] for r in group if r["birth_capture_ratio"] is not None]
            rcr = [r["realization_capture_ratio"] for r in group if r["realization_capture_ratio"] is not None]

            obs_to_entry = []
            for r in group:
                if r["obs_tick"] is not None and r["entry_tick"] is not None:
                    obs_to_entry.append(r["entry_tick"] - r["obs_tick"])

            per_regime[regime] = {
                "count": len(group),
                "mean_etl": sum(etls) / len(etls) if etls else None,
                "std_etl": (sum((v - sum(etls)/len(etls))**2 for v in etls) / len(etls))**0.5 if len(etls) > 1 else None,
                "mean_birth_capture": sum(bcr) / len(bcr) if bcr else None,
                "mean_realization_capture": sum(rcr) / len(rcr) if rcr else None,
                "mean_observer_to_entry": sum(obs_to_entry) / len(obs_to_entry) if obs_to_entry else None,
            }

        all_etls = [r["etl"] for r in records if r["etl"] is not None]
        all_bcr = [r["birth_capture_ratio"] for r in records if r["birth_capture_ratio"] is not None]
        all_rcr = [r["realization_capture_ratio"] for r in records if r["realization_capture_ratio"] is not None]
        all_ote = []
        for r in records:
            if r["obs_tick"] is not None and r["entry_tick"] is not None:
                all_ote.append(r["entry_tick"] - r["obs_tick"])

        return {
            "symbol": symbol,
            "count": len(records),
            "regimes": list(per_regime.keys()),
            "mean_etl": sum(all_etls) / len(all_etls) if all_etls else None,
            "mean_birth_capture": sum(all_bcr) / len(all_bcr) if all_bcr else None,
            "mean_realization_capture": sum(all_rcr) / len(all_rcr) if all_rcr else None,
            "mean_observer_to_entry": sum(all_ote) / len(all_ote) if all_ote else None,
            "per_regime": per_regime,
        }

    def get_all_summary(self) -> Dict:
        symbols = list(self._store.keys())
        all_stats = {}
        for sym in symbols:
            all_stats[sym] = self.get_symbol_stats(sym)
        return {"symbols": symbols, "stats": all_stats}

    def clear(self) -> None:
        self._store.clear()
