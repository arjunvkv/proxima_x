"""StrategySpec — the declarative contract for the generalized backtest->live engine.

A strategy is fully described by one Spec. The ENGINE (strategy-agnostic) consumes
the spec to: build the data feed, enforce the anti-lookahead fill contract, run the
simulated exits, apply tick-value-correct PnL + costs, run the validation battery, and
emit a live-port runtime. Adding a strategy = authoring one Spec; nothing else changes.

This encodes every hard-won constraint:
  * ANTI-LOOKAHEAD is engine-enforced (fill at next-bar open; never inside signal bar).
  * FEED is either bar-granular (MT5/audit cache) or tick-granular (tick archive);
    both produce the SAME canonical bar stream the live engine consumes.
  * SESSION gates + walk-forward + purple + determinism + server-clock are all
    properties the Engine/Validation own, not the strategy author.
  * CONFIG is declarative data (a dict/JSON), never code — portable to live unchanged.

A StrategySpec is a plain dict (JSON-serializable) with the keys below; `from_dict`
asserts the required shape so a malformed spec fails at registration, not at runtime.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable


REQUIRED_KEYS = {"name", "universe", "feed", "signal"}


@dataclass
class SignalSpec:
    # --- closed-bar signal rule (engine guarantees no lookahead) ---
    rule: str                # symbolic; the longer dim of signal
    lookback: int = 6        # bars of return used at the signal (closed bars)
    pick: str = "n_worst"    # how to pick among the cross-section: n_worst | n_best | all
    top_n: int = 5
    side: str = "long"       # long | short | both
    fill_bar: int = 1        # enter at the OPEN of signal_bar + fill_bar (anti-lookahead; default next bar)

    @classmethod
    def from_dict(cls, d: dict) -> "SignalSpec":
        return cls(
            rule=d.get("rule", "session_exhaustion"),
            lookback=d.get("lookback", 6),
            pick=d.get("pick", "n_worst"),
            top_n=d.get("top_n", 5),
            side=d.get("side", "long"),
            fill_bar=d.get("fill_bar", 1),
        )


@dataclass
class ExitSpec:
    mode: str = "sl_tp_hold"   # sl_tp_hold | hold_only | sl_tp
    hold_bars: int = 12
    jpy_sl_tp: tuple = (0.35, 0.45)      # SL/TP distance for JPY pairs (search units)
    non_jpy_sl_tp: tuple = (0.0035, 0.0045)  # SL/TP for non-JPY
    stop_first: bool = True              # MT5 stop-first convention (conservative)

    @classmethod
    def from_dict(cls, d: dict) -> "ExitSpec":
        c = cls(
            mode=d.get("mode", "sl_tp_hold"),
            hold_bars=d.get("hold_bars", 12),
            stop_first=d.get("stop_first", True),
        )
        if "jpy_sl_tp" in d: c.jpy_sl_tp = tuple(d["jpy_sl_tp"])
        if "non_jpy_sl_tp" in d: c.non_jpy_sl_tp = tuple(d["non_jpy_sl_tp"])
        return c


@dataclass
class FeedSpec:
    kind: str            # "bar" | "tick"
    timeframe: str = "M5"

    @classmethod
    def from_dict(cls, d: dict) -> "FeedSpec":
        return cls(kind=d.get("mode", "bar"), timeframe=d.get("timeframe", "M5"))


@dataclass
class StrategySpec:
    name: str
    universe: list[str]
    feed: dict
    signal: SignalSpec
    exit: ExitSpec
    sessions: Optional[list[int]] = None     # UTC hour(s) that fire; None = server clock gate any hour
    base_lot: float = 0.15
    comment: str = ""

    @property
    def session_hours(self) -> list[int]:
        return self.sessions or [0]

    @property
    def all_hours(self) -> list[int]:
        return list(range(24)) if self.sessions is None else self.sessions

    @classmethod
    def from_dict(cls, d: dict) -> "StrategySpec":
        for k in REQUIRED_KEYS:
            if k not in d:
                raise ValueError(f"StrategySpec missing required key: {k}")
        return cls(
            name=str(d["name"]),
            universe=list(d["universe"]),
            feed=dict(d["feed"]),
            signal=SignalSpec.from_dict(d.get("signal", {})),
            exit=ExitSpec.from_dict(d.get("exit", {})),
            sessions=d.get("sessions"),
            base_lot=d.get("base_lot", 0.15),
            comment=d.get("comment", ""),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name, "universe": self.universe, "feed": self.feed,
            "signal": asdict(self.signal), "exit": asdict(self.exit),
            "sessions": self.sessions, "base_lot": self.base_lot, "comment": self.comment,
        }