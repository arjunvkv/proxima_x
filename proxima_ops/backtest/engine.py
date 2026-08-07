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

SIGNAL RULES (additive sender, default keeps legacy behavior):
Each rule is a function `(bars, i, spec) -> signed_score` where score > 0 means
"prefer LONG at bar i+1 open", score < 0 means "prefer SHORT". Magnitude orders
the cross-section on a given session-day. `side` picks which rejected side(s) to
keep. The default rule `session_exhaustion` is score = -(lookback return) so its
long side reproduces the original Tokyo_H0 mean-reversion exactly.

Determinism: pure function over the bar list — same feed + spec => same trades.
"""
from __future__ import annotations
from typing import Optional
from .spec import StrategySpec
from .pnl import trade_to_usd


def bar_hour(ts: int) -> int:
    return (ts // 3600) % 24


def _ret(bars: list[dict], i: int, lb: int) -> float:
    return (bars[i]["close"] - bars[i - lb]["close"]) / bars[i - lb]["close"]


def _prior_range(bars: list[dict], i: int, spec: StrategySpec):
    """Prior (pre-session) high/low/avg across a reference lookback of closed bars
    strictly before index i. Default = trailing `signal.lookback` closed bars.
    No lookahead by construction (only bars index < i)."""
    lb = spec.signal.lookback
    lo = max(0, i - lb)
    seg = bars[lo:i]
    if not seg:
        return None
    return {"hi": max(b["high"] for b in seg), "lo": min(b["low"] for b in seg),
            "prev": seg[-1]["close"]}


def _session_avg_price(bars: list[dict], i: int, spec: StrategySpec):
    """Trailing mean of typical price across `signal.lookback` closed bars.
    VWAP-style reversion anchor when volume is unavailable (OHLC-only tape)."""
    lb = spec.signal.lookback
    lo = max(0, i - lb)
    seg = bars[lo:i]
    if not seg:
        return float(bars[i]["open"])
    return sum((b["high"] + b["low"] + b["close"]) / 3.0 for b in seg) / len(seg)


def signal_score(bars: list[dict], i: int, spec: StrategySpec) -> float:
    """Signed cross-sectional edge score for the closed bar i. >0 long, <0 short.
    Uses only bars index <= i (closed) -> no lookahead. Rules are additive:
    add a branch here; it is shared by every Symbol automated via the Spec."""
    rule = spec.signal.rule
    lb = spec.signal.lookback
    b = bars[i]
    if rule in ("session_exhaustion", "session_momentum", "return"):
        return -_ret(bars, i, lb) if rule == "session_exhaustion" else _ret(bars, i, lb)
    if rule == "session_reversion":
        # fade a move that has gone too far: score = -normalized offset vs
        # trailing session avg (premium -> short, discount -> long)
        anchor = _session_avg_price(bars, i, spec)
        return (anchor - b["close"]) / anchor * 100.0 if anchor else 0.0
    if rule == "range_reversion":
        # fade extension outside the trailing range: price below range-low = long
        r = _AT_range(bars, i, spec)
        if r is None:
            return 0.0
        if b["close"] < r["lo"]:
            return (r["lo"] - b["close"]) / (r["hi"] - r["lo"] + 1e-12) + 1.0
        if b["close"] > r["hi"]:
            return -(b["close"] - r["hi"]) / (r["hi"] - r["lo"] + 1e-12) - 1.0
        # inside range too: only the score at the extremes fire; neutral below
        return 0.0
    if rule == "range_breakout":
        # momentum: sustained close beyond the trailing range-bound, enter continuation
        if (r := _AT_range(bars, i, spec)) is None:
            return 0.0
        rng = (r["hi"] - r["lo"]) + 1e-12
        if b["close"] > r["hi"]:
            return (b["close"] - r["hi"]) / rng
        if b["close"] < r["lo"]:
            return -(r["lo"] - b["close"]) / rng
        return 0.0
    if rule == "liquidity_sweep":
        # stop-hunt rejection: wick beyond the prior range then CLOSE back inside
        # (lower shadow long, upper shadow short)
        if (r := _AT_range(bars, i, spec)) is None:
            return 0.0
        if b["low"] < r["lo"] and b["close"] > r["lo"]:
            return (r["lo"] - b["low"]) / (r["hi"] - r["lo"] + 1e-12)
        if b["high"] > r["hi"] and b["close"] < r["hi"]:
            return -(b["high"] - r["hi"]) / (r["hi"] - r["lo"] + 1e-12)
        return 0.0
    if rule == "session_open_breakout":
        # London/NY open momentum: gap-level break of the very first bar of the
        # session (front-bar open) sustained on the current closed bar.
        hi0 = bars[i]["open"]
        return (b["close"] - hi0) / hi0 * 10000.0 if b["close"] != hi0 else 0.0
    # default / future rules: treat as session_exhaustion
    return -_ret(bars, i, lb)


def _AT_range(bars: list[dict], i: int, spec: StrategySpec):
    """High/low of the trailing `signal.lookback` closed bars strictly before i."""
    lb = spec.signal.lookback
    lo = max(0, i - lb)
    seg = bars[lo:i]
    if not seg:
        return None
    return {"hi": float(max(b["high"] for b in seg)),
            "lo": float(min(b["low"] for b in seg))}


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


def _sl_tp_for(symbol: str, spec: StrategySpec) -> tuple[float, float]:
    return spec.exit.jpy_sl_tp if "JPY" in symbol else spec.exit.non_jpy_sl_tp


def simulate_exit(bars: list[dict], entry_idx: int, side: str,
                  spec: StrategySpec, symbol: str) -> dict:
    """Walk from entry (fill at entry_idx OPEN). Returns trade dict (raw pts).

    stop_first (MT5 conservative): when both SL and TP are touched in one bar the
    SL is taken. SL/TP distances come from ExitSpec per symbol (jpy vs non-jpy)."""
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
                    "reason": "SL-sto", "pnl_pts": pick}
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


def _signal_side(score: float, side_pref: str) -> str:
    # >=0 treated as long (keeps the zero-return mean-reversion candidates that
    # the original Tokyo n_worst path included -> byte parity).
    if score >= 0:
        return "BUY" if side_pref in ("long", "both") else "NONE"
    return "SELL" if side_pref in ("short", "both") else "NONE"


def _legacy_rule(spec: StrategySpec) -> bool:
    """Legacy cross-sectional rules reproduce the ORIGINAL pick semantics exactly
    (rank by raw lookback return, pick n_worst/n_best, all BUY, top_n fills
    unconditional) — byte parity with the pre-extension engine."""
    return spec.signal.rule in ("session_exhaustion", "session_momentum", "return")


def run_strategy(bars_map: dict[str, list[dict]], spec: StrategySpec,
                 tick_value_map: Optional[dict] = None, volume: float = 0.15,
                 raw: bool = False,
                 commission_per_lot: Optional[float] = None) -> list[dict]:
    """Run the spec over the bar universe -> list of USD trade dicts (raw if raw)."""
    if _legacy_rule(spec):
        return _run_legacy(bars_map, spec, tick_value_map, volume, raw, commission_per_lot)
    return _run_signed(bars_map, spec, tick_value_map, volume, raw, commission_per_lot)


def _run_legacy(bars_map: dict[str, list[dict]], spec: StrategySpec,
                tick_value_map: Optional[dict], volume: float, raw: bool,
                commission_per_lot: Optional[float]) -> list[dict]:
    """EXACT original engine behavior (n_worst/n_best over lookback return, BUY
    only). Do not alter — byte parity is a hard gate."""
    sigs = {s: session_signal_indices(bars_map[s], spec) for s in bars_map}
    by_day: dict[int, list[tuple[float, str, int]]] = {}
    for sym, idxs in sigs.items():
        for i in idxs:
            d = bars_map[sym][i]["ts"] // 86400
            by_day.setdefault(d, []).append((_ret(bars_map[sym], i, spec.signal.lookback), sym, i))
    trades: list[dict] = []
    entries: list[tuple[str, int]] = []
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
    for sym, eidx in entries:
        trades.append(simulate_exit(bars_map[sym], eidx, "BUY", spec, sym))
    if raw:
        return trades
    return [trade_to_usd(t, volume, tick_value_map, commission_per_lot) for t in trades]


def _run_signed(bars_map: dict[str, list[dict]], spec: StrategySpec,
                tick_value_map: Optional[dict], volume: float, raw: bool,
                commission_per_lot: Optional[float]) -> list[dict]:
    """Signed-score path for the market-structure rules: signal_score() returns
    a signed edge (positive = long, negative = short); `side` keeps the selected
    direction(s); candidates rank by |score| and top_n fill per session-day."""
    sigs = {s: session_signal_indices(bars_map[s], spec) for s in bars_map}
    by_day: dict[int, list[tuple[float, str, int, str]]] = {}
    for sym, idxs in sigs.items():
        for i in idxs:
            score = signal_score(bars_map[sym], i, spec)
            side = _signal_side(score, spec.signal.side)
            if side == "NONE":
                continue
            d = bars_map[sym][i]["ts"] // 86400
            by_day.setdefault(d, []).append((score, sym, i, side))
    trades: list[dict] = []
    entries: list[tuple[str, int, str]] = []
    for day in sorted(by_day):
        cands = by_day[day]
        cands.sort(key=lambda x: -abs(x[0]))
        opened = 0
        opened_today: set[str] = set()
        for r, sym, i, side in cands:
            if opened >= spec.signal.top_n or sym in opened_today:
                continue
            entry_idx = i + spec.signal.fill_bar
            if entry_idx >= len(bars_map[sym]):
                continue
            opened_today.add(sym)
            entries.append((sym, entry_idx, side))
            opened += 1
    for sym, eidx, side in entries:
        trades.append(simulate_exit(bars_map[sym], eidx, side, spec, sym))
    if raw:
        return trades
    return [trade_to_usd(t, volume, tick_value_map, commission_per_lot) for t in trades]