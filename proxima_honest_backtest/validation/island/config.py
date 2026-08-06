"""Island configuration — run_id, dataset, executor scenario wiring."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from proxima_honest_backtest.live.config import get_ship


@dataclass
class IslandConfig:
    strategy_key: str = "tokyo_h0"
    months: List[int] = None       # e.g. [1, 2, 3]
    year: int = 2026
    env: str = "offline"           # "offline" | "ftmo_demo"
    fake_seed: int = 0
    scenario: str = "instant"      # FakeBroker scenario for ALL pairs
    magic_base: int = 900000
    base_lot: float = 0.15
    out_dir: Optional[str] = None

    def __post_init__(self):
        if self.months is None:
            self.months = [1]  # default: fast PR feedback; --months 1,2,3 for full

    @property
    def run_id(self) -> str:
        ship = get_ship(self.strategy_key)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        base = f"{self.strategy_key}|{self.env}|{','.join(map(str, self.months))}|{self.scenario}|{self.fake_seed}"
        return f"{self.strategy_key}_{self.env}_{stamp}_{hashlib.sha1(base.encode()).hexdigest()[:6]}"

    def ship(self):
        return get_ship(self.strategy_key)