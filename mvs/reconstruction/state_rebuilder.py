from __future__ import annotations

from typing import Dict, List

from mvs.reconstruction.entropy_rebuilder import EntropyRebuilder
from mvs.reconstruction.tpi_rebuilder import TpiRebuilder
from mvs.reconstruction.regime_rebuilder import RegimeRebuilder
from mvs.reconstruction.vpl_rebuilder import VplRebuilder
from mvs.reconstruction.drift_rebuilder import DriftRebuilder
from mvs.reconstruction.calibration_rebuilder import CalibrationRebuilder
from mvs.reconstruction.observer_rebuilder import ObserverRebuilder
from mvs.reconstruction.age_rebuilder import AgeRebuilder


class StateRebuilder:
    __slots__ = (
        "symbol", "entropy_rebuilder", "tpi_rebuilder", "regime_rebuilder",
        "vpl_rebuilder", "drift_rebuilder", "calibration_rebuilder",
        "observer_rebuilder", "age_rebuilder"
    )

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.entropy_rebuilder = EntropyRebuilder()
        self.tpi_rebuilder = TpiRebuilder()
        self.regime_rebuilder = RegimeRebuilder()
        self.vpl_rebuilder = VplRebuilder(symbol)
        self.drift_rebuilder = DriftRebuilder()
        self.calibration_rebuilder = CalibrationRebuilder()
        self.observer_rebuilder = ObserverRebuilder()
        self.age_rebuilder = AgeRebuilder()

    def on_tick(self, tick_id: int, symbol: str, ts_ns: int, tick_data: Dict, open_trades: List[dict]) -> Dict:
        entropy_data = self.entropy_rebuilder.on_tick(tick_id, symbol, ts_ns, tick_data["mid"])
        tpi_data = self.tpi_rebuilder.on_tick(tick_id, symbol, ts_ns, tick_data["mid"], tick_data["bid"], tick_data["ask"])
        regime_data = self.regime_rebuilder.on_tick(tick_id, symbol, ts_ns, tick_data["mid"], entropy_data, tpi_data)
        vpl_data = self.vpl_rebuilder.on_tick(tick_id, symbol, ts_ns, tick_data["mid"], tick_data["bid"], tick_data["ask"])
        drift_data = self.drift_rebuilder.on_tick(tick_id, symbol, ts_ns, tick_data["mid"])
        calibration_data = self.calibration_rebuilder.on_tick(tick_id, symbol, ts_ns, tpi_data["tpi"], entropy_data["entropy"], regime_data["regime"])
        self.observer_rebuilder.set_calibration_passed(calibration_data.get("calibration_ok", False))
        observer_data = self.observer_rebuilder.on_tick(tick_id, symbol, ts_ns, tpi_data["tpi"], tpi_data["tpi_sign"], entropy_data["entropy"], regime_data["regime"], 0.5)
        age_data = self.age_rebuilder.on_tick(tick_id, symbol, ts_ns, open_trades)
        merged = {
            "tick_id": tick_id,
            "symbol": symbol,
            "ts_ns": ts_ns,
            **tpi_data,
            "entropy": entropy_data["entropy"],
            "entropy_state": "COMPRESSED" if entropy_data["compression_ratio"] < 1.0 else "EXPANDED",
            **regime_data,
            **vpl_data,
            **observer_data,
            **drift_data,
            **calibration_data,
            **age_data,
        }
        return merged
