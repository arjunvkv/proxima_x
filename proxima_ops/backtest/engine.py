"""Engine — the strategy-agnostic no-lookahead backtest core.

The engine walks canonical bar streams (from any feed) and, given a StrategySpec,
produces a deterministic trade list. ANTI-LOOKAHEAD IS ENGINE-ENFORCED:

  * signal uses ONLY bars CLOSED at or before index i (never the forming bar);
  * entry always fills at the OPEN of signal_idx + fill_bar (default next bar),
    never inside the signal bar;
  * SL/TP are monitored intrabar from the ENTRY bar onward, stop-first
    (conservative MT5 convention);
  * one position per symbol per session-day (blocked-conflict parity);
  * exits: SL/TP touch intrabar, else held >= hold_bars closes at that bar's open.

Session support: `spec.sessions` (UTC hours) select WHICH bars may be signal bars
(the first closed bar of each listed hour per UTC day). If sessions is None, every
closed bar qualifies (any-hour / tick-to-bar strategies).

Determinism: pure function over the bar list — same feed + spec => same trades.
"""
from __future__ import annotations
from typing import Optional
from .spec import StrategySpec
from .pnl import trade_to_usd


def bar_hour(ts: int) -> int:
    return (ts // 3600) % 24


def session_signal_indices(bars: list[dict], spec: StrategySpec) -> list[int]:
    """Indices of closed bars that may signal (session-qualified, one per day/hour)."""
    lb = spec.signal.lookback
    fb = spec.signal.fill_bar
    if spec.sessions is None:
        return [i for i in range(lb, len(bars) - fb)]
    out, seen = [], set()
    for i in range(lb, len(bars) - fb):
        b = bars[i]
        d = b["ts"] // 86400
        key = (d, bar_hour(b["ts"]))
        if bar_hour(b["ts"]) in spec.sessions and key not in seen:
            out.append(i)
            seen.add(key)
    return out


def _ret(bars: list[dict], i: int, lb: int) -> float:
    return (bars[i]["close"] - bars[i - lb]["close"]) / bars[i - lb]["close"]


def _sl_tp_for(symbol: str, spec: StrategySpec) -> tuple[float, float]:
    return spec.exit.jpy_sl_tp if "JPY" in symbol else spec.exit.non_jpy_sl_tp


def simulate_exit(bars: list[dict], entry_idx: int, side: str,
                  spec: StrategySpec, symbol: str) -> dict:
    """Walk from entry (fill at entry_idx OPEN). Returns trade dict (raw pts)."""
    entry_idx = min(entry_idx, len(bars) - 1)
    entry = bars[entry_idx]["open"]
    dirn = 1 if side == "BUY" else -1
    sl_d, tp_d = _sl_tp_for(symbol, spec)
    sl = entry - sl_d if side == "BUY" else entry + sl_d
    tp = entry + tp_d if side == "BUY" else entry - tp_d
    last = min(entry_idx + spec.exit.hold_bars, len(bars) - 1)
    for k in range(entry_idx, last + 1):
        b = bars[k]
        hi, lo = b["high"], b["low"]
        t_sl = (lo <= sl) if dirn == 1 else (hi >= sl)
        t_tp = (hi >= tp) if dirn == 1 else (lo <= tp)
        if t_sl and t_tp:
            rs = (sl - entry) * dirn; rt = (tp - entry) * dirn
            pick = rs if spec.exit.stop_first else rt
            return {"symbol": symbol, "side": side, "entry": entry,
                    "entry_ts": bars[entry_idx]["ts"], "exit_ts": b["ts"],
                    "reason": "SL-sto" if spec.exit.stop_first else "TP", "pnl_pts": pick}
        if t_sl:
            return {"symbol": symbol, "side": side, "entry": entry,
                    "entry_ts": bars[entry_idx]["ts"], "exit_ts": b["ts"],
                    "reason": "SL", "pnl_pts": (sl - entry) * dirn}
        if t_tp:
            return {"symbol": symbol, "side": side, "entry": entry,
                    "entry_ts": bars[entry_idx]["ts"], "exit_ts": b["ts"],
                    "reason": "TP", "pnl_pts": (tp - entry) * dirn}
    exit_bar = bars[last]
    return {"symbol": symbol, "side": side, "entry": entry,
            "entry_ts": bars[entry_idx]["ts"], "exit_ts": exit_bar["ts"],
            "reason": "HOLD", "pnl_pts": (exit_bar["open"] - entry) * dirn}


def run_strategy(bars_map: dict[str, list[dict]], spec: StrategySpec,
                 tick_value_map: Optional[dict] = None, volume: float = 0.15,
                 raw: bool = False) -> list[dict]:
    """Run the spec over the bar universe -> list of USD trade dicts (raw if raw)."""
    sigs = {s: session_signal_indices(bars_map[s], spec) for s in bars_map}
    # chronological union of candidate (timestamp, symbol, idx), rank-per-day later
    by_day: dict[int, list[tuple[float, str, int]]] = {}
    for sym, idxs in sigs.items():
        for i in idxs:
            d = bars_map[sym][i]["ts"] // 86400
            by_day.setdefault(d, []).append((_ret(bars_map[sym], i, spec.signal.lookback), sym, i))
    trades: list[dict] = []
    entries: list[tuple[str, int]] = []      # (symbol, entry_idx) opened across the tape
    for day in sorted(by_day):
        cands = by_day[day]
        if spec.signal.pick == "n_worst":
            cands.sort(key=lambda x: x[0])
        else:
            cands.sort(key=lambda x: -x[0])
        opened = 0
        opened_today: set[str] = set()
        for r, sym, i in cands:
            if opened >= spec.signal.top_n or sym in opened_today:
                continue
            entry_idx = i + spec.signal.fill_bar
            if entry_idx >= len(bars_map[sym]):
                continue
            opened_today.add(sym)
            entries.append((sym, entry_idx))
            opened += 1
    # settle all opened positions. Positions per symbol are >= a session apart
    # and hold (12 bars = 1h) < session spacing, so no open-position conflict.
    for sym, eidx in entries:
        trades.append(simulate_exit(bars_map[sym], eidx, "BUY", spec, sym))
    if raw:
        return trades
    return [trade_to_usd(t, volume, tick_value_map) for t in trades]