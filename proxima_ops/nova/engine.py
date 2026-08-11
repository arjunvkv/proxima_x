"""NOVA interpreter — spec -> trade dicts, byte-parity with the legacy engine.

Replicates proxima_ops/backtest/engine.py semantics exactly (session_signal_
indices, _run_legacy, _run_signed, simulate_exit) using numpy. Costs delegate
to pnl.trade_to_usd so the USD path is byte-identical by construction.
Parity proof: scripts/verify_nova_parity.py.
"""
import numpy as np

from proxima_ops.backtest.spec import StrategySpec
from proxima_ops.backtest.pnl import trade_to_usd, FTMO_TICK_VALUES

from . import factors as F

LEGACY_RULES = ("session_exhaustion", "session_momentum", "return")


def signal_indices(ts, hour, day, lb, fb, sessions, wds):
    """First bar >= lb of each (day, hour) in `sessions` (one per day/hour),
    wds-filtered; sessions None -> every bar in [lb, len-fb). Matches
    engine session_signal_indices exactly."""
    n = len(ts)
    mask = np.ones(n, dtype=bool)
    mask[:lb] = False
    if fb > 0:
        mask[n - fb:] = False
    if wds is not None:
        mask &= np.isin(F.weekday(ts), wds)
    if sessions is None:
        return np.flatnonzero(mask)
    mask &= np.isin(hour, sessions)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return idx
    keys = day[idx] * 100 + hour[idx]
    _, first = np.unique(keys, return_index=True)
    return idx[first]


def _run_legacy_vec(bars_map, spec: StrategySpec):
    """EXACT legacy pick semantics: per day, rank candidates by lookback ret
    (asc for n_worst / desc for n_best), fill top_n, one entry per symbol per
    day, BUY only, entry at signal index + fill_bar. Returns raw trade dicts."""
    lb, fb, tn = spec.signal.lookback, spec.signal.fill_bar, spec.signal.top_n
    sess, wds = spec.sessions, spec.weekdays
    syms = list(bars_map)
    days_all, ret_all, sidx_all, i_all = [], [], [], []
    for si, sym in enumerate(syms):
        B = bars_map[sym]
        idx = signal_indices(B["ts"], F.bar_hour(B["ts"]), F.bar_day(B["ts"]),
                             lb, fb, sess, wds)
        if len(idx) == 0:
            continue
        days_all.append(F.bar_day(B["ts"])[idx])
        ret_all.append(_ret_at(B["close"], idx, lb))
        sidx_all.append(np.full(len(idx), si))
        i_all.append(idx)
    if not days_all:
        return []
    DAYS = np.concatenate(days_all)
    RETS = np.concatenate(ret_all)
    SIDX = np.concatenate(sidx_all)
    IS = np.concatenate(i_all)
    order = np.argsort(DAYS, kind="stable")
    DAYS, RETS, SIDX, IS = DAYS[order], RETS[order], SIDX[order], IS[order]
    entries = []  # (sym, entry_idx, side)
    day_bounds = np.flatnonzero(np.diff(DAYS)) + 1
    bounds = np.concatenate([[0], day_bounds, [len(DAYS)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        sub = slice(a, b)
        r = RETS[sub]
        o = np.argsort(r if spec.signal.pick == "n_worst" else -r, kind="stable")
        opened_today = set()
        cnt = 0
        for k in o:
            if cnt >= tn or SIDX[sub][k] in opened_today:
                continue
            sid = int(SIDX[sub][k])
            eidx = int(IS[sub][k]) + fb
            if eidx >= len(bars_map[syms[sid]]["close"]):
                continue
            opened_today.add(sid)
            entries.append((syms[sid], eidx, "BUY"))
            cnt += 1
    return entries
def _ret_at(close, idx, lb):
    """Engine _ret at a set of indices: (close[i]-close[i-lb])/close[i-lb]."""
    return (close[idx] - close[idx - lb]) / close[idx - lb]


def _scores(B, idx, spec: StrategySpec, symbol: str):
    """Vectorized port of engine signal_score over a symbol's signal indices.
    Returns the score array (NaN -> no signal). Supported rules mirror the
    triage/battery set; unknown rules raise (parity harness never hits them)."""
    rule = spec.signal.rule
    lb = spec.signal.lookback
    o, h, l, c, ts = B["open"], B["high"], B["low"], B["close"], B["ts"]
    r = _ret_at(c, idx, lb)
    if rule == "session_reversion":
        anchor = F.rolling_mean((h + l + c) / 3.0, lb)[idx]
        return np.where(anchor != 0, (anchor - c[idx]) / anchor * 100.0, 0.0)
    if rule == "big_move_fade":
        atr = F.trailing_atr(o, h, l, c)[idx]
        z = r / np.where(atr > 0, atr / c[idx], np.nan)
        thr = (lb / 144.0 + 0.5) * 2.0
        out = np.where(np.abs(z) > thr, -r * 10000.0, 0.0)
        return np.nan_to_num(out, nan=0.0)
    if rule == "cross_momentum":
        return r * 10000.0
    if rule == "range_reversion" or rule == "range_breakout":
        hi = F.rolling_max(h, lb)[idx]
        lo = F.rolling_min(l, lb)[idx]
        rng = np.where(hi - lo > 0, hi - lo, np.nan)
        if rule == "range_reversion":
            sc = np.where(c[idx] < lo, (lo - c[idx]) / rng + 1.0, 0.0)
            sc = np.where(c[idx] > hi, -(c[idx] - hi) / rng - 1.0, sc)
            return np.nan_to_num(sc, nan=0.0)
        sc = np.where(c[idx] > hi, (c[idx] - hi) / rng, 0.0)
        sc = np.where(c[idx] < lo, -(lo - c[idx]) / rng, sc)
        return np.nan_to_num(sc, nan=0.0)
    if rule == "liquidity_sweep":
        hi = F.rolling_max(h, lb)[idx]
        lo = F.rolling_min(l, lb)[idx]
        rng = np.where(hi - lo > 0, hi - lo, np.nan)
        sc = np.where((l[idx] < lo) & (c[idx] > lo), (lo - l[idx]) / rng, 0.0)
        sc = np.where((h[idx] > hi) & (c[idx] < hi), -(h[idx] - hi) / rng, sc)
        return np.nan_to_num(sc, nan=0.0)
    if rule == "session_open_breakout":
        # NOTE: engine uses hi0 = bars[i]["open"] (current bar open) despite the
        # comment; byte parity requires matching the CODE.
        hi0 = o[idx]
        sc = np.where(c[idx] != hi0, (c[idx] - hi0) / hi0 * 10000.0, 0.0)
        return sc
    if rule == "weekend_gap":
        g = F.gap_open(o, c, ts)[idx]
        return -g * 10000.0
    if rule == "fix_reversal":
        if symbol.startswith("USD"):
            return np.full(len(idx), -1.0)
        if symbol.endswith("USD"):
            return np.full(len(idx), 1.0)
        return np.zeros(len(idx))
    if rule == "domestic_hours":
        hh = F.bar_hour(ts)[idx]
        if "JPY" in symbol:
            return np.where(hh < 6, 1.0, 0.0)
        if symbol.startswith("EUR"):
            return np.where((hh >= 7) & (hh < 12), -1.0, 0.0)
        if symbol.endswith("USD"):
            return np.where((hh >= 12) & (hh < 21), 1.0, 0.0)
        return np.zeros(len(idx))
    if rule == "day_of_week_usd":
        wd = F.weekday(ts)[idx]
        if symbol.endswith("USD"):
            return np.where(np.isin(wd, (0, 1)), -1.0, 1.0)
        return np.zeros(len(idx))
    if rule == "carry_clock":
        if spec.signal.direction_map:
            return np.full(len(idx), float(spec.signal.direction_map.get(symbol, 0.0)))
        return np.ones(len(idx))
    if rule == "round_barrier_fade":
        is_jpy = "JPY" in symbol
        step = 0.50 if is_jpy else 0.0050
        lvl = np.round(c[idx] / step) * step
        eps = step * 0.4
        sc = np.zeros(len(idx))
        near = np.abs(c[idx] - lvl) < eps
        sc = np.where(near & (l[idx] < lvl) & (c[idx] > lvl), 1.0, sc)
        sc = np.where(near & (h[idx] > lvl) & (c[idx] < lvl), -1.0, sc)
        return sc
    raise NotImplementedError(f"nova rule not ported: {rule}")
def _run_signed_vec(bars_map, spec: StrategySpec):
    """Signed-score path: rank by |score|, top_n per day (or per day-hour when
    signal.per_hour), one entry per symbol per bucket, side from score sign."""
    lb, fb, tn = spec.signal.lookback, spec.signal.fill_bar, spec.signal.top_n
    sess, wds = spec.sessions, spec.weekdays
    per_hour = spec.signal.per_hour
    syms = list(bars_map)
    bk_all, sc_all, si_all, i_all, sd_all = [], [], [], [], []
    for si, sym in enumerate(syms):
        B = bars_map[sym]
        idx = signal_indices(B["ts"], F.bar_hour(B["ts"]), F.bar_day(B["ts"]),
                             lb, fb, sess, wds)
        if len(idx) == 0:
            continue
        sc = _scores(B, idx, spec, sym)
        side = np.where(sc >= 0, "BUY", "SELL")
        pref = spec.signal.side
        keep = (sc >= 0) & (pref in ("long", "both")) | (sc < 0) & (pref in ("short", "both"))
        keep &= ~np.isnan(sc)
        idx, sc, side = idx[keep], sc[keep], side[keep]
        if len(idx) == 0:
            continue
        d = F.bar_day(B["ts"])[idx]
        hh = F.bar_hour(B["ts"])[idx]
        bk = d * 100 + hh if per_hour else d
        bk_all.append(bk)
        sc_all.append(sc)
        si_all.append(np.full(len(idx), si))
        i_all.append(idx)
        sd_all.append(side)
    if not bk_all:
        return []
    BK = np.concatenate(bk_all)
    SC = np.concatenate(sc_all)
    SIDX = np.concatenate(si_all)
    IS = np.concatenate(i_all)
    SD = np.concatenate(sd_all)
    order = np.argsort(BK, kind="stable")
    BK, SC, SIDX, IS, SD = BK[order], SC[order], SIDX[order], IS[order], SD[order]
    entries = []
    bounds = np.flatnonzero(np.diff(BK)) + 1
    bounds = np.concatenate([[0], bounds, [len(BK)]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        sub = slice(a, b)
        o = np.argsort(-np.abs(SC[sub]), kind="stable")
        opened = set()
        cnt = 0
        for k in o:
            if cnt >= tn or SIDX[sub][k] in opened:
                continue
            sid = int(SIDX[sub][k])
            eidx = int(IS[sub][k]) + fb
            if eidx >= len(bars_map[syms[sid]]["close"]):
                continue
            opened.add(sid)
            entries.append((syms[sid], eidx, str(SD[sub][k])))
            cnt += 1
    return entries


def _simulate_exit_vec(B, entries, spec: StrategySpec):
    """Vectorized simulate_exit over (sym, entry_idx, side) entries.
    Window = [entry_idx, min(entry_idx+hold, len-1)] INCLUSIVE (engine range
    semantics); stop-first on same-bar SL+TP; HOLD exits at the last bar OPEN."""
    hold = spec.exit.hold_bars
    stop_first = spec.exit.stop_first
    out = []
    for sym, eidx, side in entries:
        sl_d, tp_d = (spec.exit.jpy_sl_tp if "JPY" in sym else spec.exit.non_jpy_sl_tp)
        t = _sim_exit_one(B[sym], eidx, side, sl_d, tp_d, hold, stop_first)
        t["symbol"] = sym
        out.append(t)
    return out


def _sim_exit_one(B, eidx, side, sl_d, tp_d, hold, stop_first):
    o, h, l, ts = B["open"], B["high"], B["low"], B["ts"]
    L = len(o)
    eidx = min(eidx, L - 1)
    entry = float(o[eidx])
    dirn = 1.0 if side == "BUY" else -1.0
    return _sim_exit_core(o, h, l, ts, eidx, entry, dirn, sl_d, tp_d, hold, stop_first)


def _sim_exit_core(o, h, l, ts, eidx, entry, dirn, sl_d, tp_d, hold, stop_first):
    L = len(o)
    sl = entry - dirn * sl_d
    tp = entry + dirn * tp_d
    last = min(eidx + hold, L - 1)
    j_sl = j_tp = None
    pnl = 0.0
    reason = "HOLD"
    exit_ts = ts[last]
    for k in range(eidx, last + 1):
        hi, lo = h[k], l[k]
        t_sl = (lo <= sl) if dirn == 1 else (hi >= sl)
        t_tp = (hi >= tp) if dirn == 1 else (lo <= tp)
        if t_sl and t_tp:
            rs = (sl - entry) * dirn
            rt = (tp - entry) * dirn
            pick = rs if stop_first else rt
            return {"entry": entry, "entry_ts": ts[eidx], "exit_ts": ts[k],
                    "reason": "SL-sto", "pnl_pts": pick, "side": "BUY" if dirn == 1 else "SELL"}
        if t_sl:
            return {"entry": entry, "entry_ts": ts[eidx], "exit_ts": ts[k],
                    "reason": "SL", "pnl_pts": (sl - entry) * dirn,
                    "side": "BUY" if dirn == 1 else "SELL"}
        if t_tp:
            return {"entry": entry, "entry_ts": ts[eidx], "exit_ts": ts[k],
                    "reason": "TP", "pnl_pts": (tp - entry) * dirn,
                    "side": "BUY" if dirn == 1 else "SELL"}
    return {"entry": entry, "entry_ts": ts[eidx], "exit_ts": exit_ts,
            "reason": "HOLD", "pnl_pts": (o[last] - entry) * dirn,
            "side": "BUY" if dirn == 1 else "SELL"}


def run_strategy(bars_map, spec, tick_value_map=FTMO_TICK_VALUES, volume=0.15,
                 raw=False, commission_per_lot=None, spread_pips_map=None):
    """NOVA entry point — same signature as the legacy engine. bars_map values
    are numpy dicts {ts, open, high, low, close}. Returns raw or USD trades."""
    if spec.signal.rule in LEGACY_RULES:
        entries = _run_legacy_vec(bars_map, spec)
    else:
        entries = _run_signed_vec(bars_map, spec)
    if len(entries) == 0:
        return []
    trades = _simulate_exit_vec(bars_map, entries, spec)
    if raw:
        return trades
    return [trade_to_usd(t, volume, tick_value_map, commission_per_lot,
                         spread_pips_map) for t in trades]
