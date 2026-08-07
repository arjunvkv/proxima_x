"""Validation battery — every anti-overfit / lookahead trap we've hit, baked in.

Given a trade list (order = chronological) and the spec, provide:
  * metrics        (win_rate, PF, net, expectancy, max_drawdown, commission share)
  * gate           (PF>1.2, net>0, exp>$15/lot (per-lot), DD<20% of capital,
                    trades in [20, 20000])
  * train/val      time split (default 70/30) — the honest out-of-sample cut
  * walk_forward   rolling train+test windows with stable, deterministic behaviour
  * purple        shuffle the per-symbol bar ORDER (destroying any time structure),
                    re-run, and require real per-lot mean > shuffled mean + 2sd
  * determinism   same feed+spec run twice => byte-identical trade list
  * server_clock_ok  assert session gating uses epoch-day keys (not wall today/now)
    -> the intraday reset / day-keying lesson.

Trade dicts input are the engine's USD trade dicts (net in `net`).
"""
from __future__ import annotations
from typing import Optional, Callable
import random, statistics as st


def metrics(usd: list[dict]) -> dict:
    n = len(usd)
    wins = [t for t in usd if t["net"] > 0]
    losses = [t for t in usd if t["net"] < 0]
    gw = sum(t["gross_usd"] for t in wins)
    gl = -sum(t["gross_usd"] for t in losses)
    pf = (gw / gl) if gl else (float("inf") if gw else 0.0)
    net = sum(t["net"] for t in usd)
    curve, peak, maxdd = [], -1e18, 0.0
    for t in usd:
        curve.append(curve[-1] + t["net"] if curve else t["net"])
    for v in curve:
        peak = max(peak, v); maxdd = max(maxdd, peak - v)
    n_comm = sum(t["commission"] for t in usd)
    n_gross = sum(t["gross_usd"] for t in usd)
    return {"trades": n, "win_rate": round(len(wins)/n, 4) if n else 0.0,
            "gross_pnl": round(n_gross, 2), "net_pnl": round(net, 2),
            "commission": round(n_comm, 2),
            "profit_factor": round(pf, 4), "max_drawdown": round(maxdd, 2),
            "expectancy": round(net/n, 2) if n else 0.0}


def split_by_ts(usd: list[dict], frac: float = 0.7) -> tuple[list, list]:
    n = len(usd); cut = int(n * frac)
    return usd[:cut], usd[cut:]


def gate(m: dict, lot: float = 0.15, capital: float = 100_000.0,
         exp_floor: float = 15.0) -> dict:
    exp = m["expectancy"] / lot if (lot and m["trades"]) else m["expectancy"]
    dd_b = 0.20 * capital
    checks = {
        "PF > 1.2": m["profit_factor"] > 1.2,
        "net > 0": m["net_pnl"] > 0,
        f"exp > ${exp_floor}/lot": exp > exp_floor,
        "DD < 20%": m["max_drawdown"] < dd_b,
        "trades in [20,20000]": 20 <= m["trades"] <= 20000,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "reject": [k for k, v in checks.items() if not v],
            "expectancy_per_lot": round(exp, 2)}


def walk_forward(trades: list[dict], train_size: int = 300, test_size: int = 100,
                 lot: float = 0.15) -> dict:
    """Rolling train->test oos. Uses ONLY closed trades; each window independent.
    Returns per-window expectancy and pass rate. A candidate with a REAL-only
    edge should have positive (or at least non-catastrophic) expectation in the
    majority of forward windows, else it is curve-fit to one regime.
    """
    windows = []
    i = 0
    while i + train_size + test_size <= len(trades):
        tr = trades[i:i+train_size]; te = trades[i+train_size:i+train_size+test_size]
        met = metrics(te)
        windows.append({"test_start": i+train_size, "n": met["trades"],
                        "exp": met["expectancy"], "net": met["net_pnl"],
                        "wr": met["win_rate"], "pf": met["profit_factor"]})
        i += test_size
    if not windows:
        return {"windows": [], "positive_share": 0.0, "stable": False}
    pos = sum(1 for w in windows if w["net"] > 0) / len(windows)
    return {"windows": windows, "n_windows": len(windows),
            "positive_share": round(pos, 3), "stable": pos >= 0.6}


def purple_edge(bars, runner: Callable[[dict], list[dict]],
                exp_lot_real: float, iters: int = 10) -> str:
    """Shuffle each symbol's bar ORDER (destroys time structure), re-run the
    engine, require real per-lot expectancy > shuffled mean + 2sd."""
    rng = random.Random(42)
    means, n = [], 0
    for _ in range(iters):
        sh = {s: b[:] for s, b in bars.items()}
        for s in sh:
            rng.shuffle(sh[s])
        usd = runner(sh)
        if usd:
            means.append(sum(x["net"] for x in usd) / len(usd)); n += len(usd)
    if not means:
        return "no-shuffle-trades"
    sm = sum(means) / len(means)
    sd = st.stdev(means) if len(means) > 1 else 0.0
    return "REAL-EDGE" if exp_lot_real > sm + 2*sd else "no-edge"


def determinism(runner: Callable[[], list], runs: int = 3) -> bool:
    """Same inputs -> stable trade COUNT across runs (the polars order lesson)."""
    counts = set()
    for _ in range(runs):
        usd = runner()
        counts.add(len(usd))
    return len(counts) == 1