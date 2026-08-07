"""Simulate run_tokyo_h0_live.live_loop — the exact MT5 daemon — offline.

A deliberately faithful "dummy live trade firer" so we can prove the batch backtest
engine and the live daemon are 1:1. Contract copied from run_tokyo_h0_live.py:
  * SIGNAL: first CLOSED bar of session hour (top-5 most-negative 6-bar M5 return).
  * FILL: at the OPEN of signal_bar + 1 (next M5 bar) — never inside signal bar.
  * EXIT: SL/TP intrabar stop-first; else close at the open of entry+hold_bars.
  * Only the fetch is incremental (server-clock gated); position economics identical
    to run_tokyo_h0_live._manage + broker SL/TP.

"Live" vs "engine": this firer's only job is to reproduce the same closed-bar
contract and same entry/exit/pnl the daemon would produce, so a 1:1 parity test
against the generalized engine (which is batch) is honest.
"""
from __future__ import annotations
from typing import Optional
from .spec import StrategySpec
from .pnl import trade_to_usd


def _dist(symbol: str) -> tuple[float, float]:
    return (0.35, 0.45) if "JPY" in symbol else (0.0035, 0.0045)


def fire_live(bars_map: dict[str, list[dict]], spec: StrategySpec,
              volume: float = 0.15, side: str = "BUY") -> list[dict]:
    """Offline live daemon fire over the whole tape. Returns USD trade dicts."""
    lb = spec.signal.lookback
    fb = spec.signal.fill_bar
    by_day: dict[int, list[tuple[float, str, int]]] = {}   # ret -> (symbol, sig_idx)
    for sym, bars in bars_map.items():
        if len(bars) < lb + fb:
            continue
        seen: set = set()   # one sig per (day, session-hour)
        for i in range(lb, len(bars) - fb):
            b = bars[i]
            d = b["ts"] // 86400
            h = (b["ts"] // 3600) % 24
            if spec.sessions is not None and h not in spec.sessions:
                continue
            if (d, h) in seen:
                continue
            seen.add((d, h))
            r = (b["close"] - bars[i - lb]["close"]) / bars[i - lb]["close"]
            by_day.setdefault(d, []).append((r, sym, i))
    trades: list[dict] = []
    for day in sorted(by_day):
        cands = by_day[day]
        if spec.signal.pick in ("n_worst", "all"):
            cands.sort(key=lambda x: x[0])
        else:
            cands.sort(key=lambda x: -x[0])
        opened: set[str] = set()
        for r, sym, sig_idx in cands:
            if len(opened) >= spec.signal.top_n or sym in opened:
                continue
            entry_idx = sig_idx + fb
            if entry_idx >= len(bars_map[sym]):
                continue
            opened.add(sym)
            trades.append(_run_position(bars_map[sym], entry_idx, spec, sym, side))
    return [trade_to_usd(t, volume, None) for t in trades]


def _run_position(bars, entry_idx: int, spec: StrategySpec, sym: str, side: str) -> dict:
    entry_idx = min(entry_idx, len(bars) - 1)
    entry = bars[entry_idx]["open"]
    d_sl, d_tp = _dist(sym)
    dirn = 1 if side == "BUY" else -1
    sl = entry - d_sl if side == "BUY" else entry + d_sl
    tp = entry + d_tp if side == "BUY" else entry - d_tp
    last = min(entry_idx + spec.exit.hold_bars, len(bars) - 1)
    base = {"symbol": sym, "side": side, "entry": entry, "entry_ts": bars[entry_idx]["ts"]}
    for k in range(entry_idx, last + 1):
        b = bars[k]
        hi, lo = b["high"], b["low"]
        t_sl = (lo <= sl) if dirn == 1 else (hi >= sl)
        t_tp = (hi >= tp) if dirn == 1 else (lo <= tp)
        if t_sl and t_tp:
            pick_reason = "SL" if spec.exit.stop_first else "TP"
            px = sl if spec.exit.stop_first else tp
            return {**base, "exit_ts": b["ts"], "reason": pick_reason,
                    "pnl_pts": (px - entry) * dirn}
        if t_sl:
            return {**base, "exit_ts": b["ts"], "reason": "SL", "pnl_pts": (sl - entry) * dirn}
        if t_tp:
            return {**base, "exit_ts": b["ts"], "reason": "TP", "pnl_pts": (tp - entry) * dirn}
    exit_bar = bars[last]
    return {**base, "exit_ts": exit_bar["ts"], "reason": "HOLD",
            "pnl_pts": (exit_bar["open"] - entry) * dirn}