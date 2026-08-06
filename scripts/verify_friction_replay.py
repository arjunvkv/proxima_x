"""verify_friction_replay.py — Phase 8: real-tape friction-aware replay.

GPT-recommended next step (2026-08-06): measure whether a strategy survives
realistic execution costs over enough market regimes, by comparing:

    naive mid-fill backtest   (fill at mid, zero friction)
        vs
    broker-realistic replay   (PaperBroker + ExecutionCost + FirmRisk)

Run from repo root (no MT5 needed — reads the ingested archive):
    unset PYTHONPATH && ./.venv/Scripts/python.exe scripts/verify_friction_replay.py

Produces comparison_report.json with:
    trades, win_rate, gross_pnl, commission, slippage, spread_cost,
    net_pnl, max_drawdown, profit_factor, FTMO daily violations
and the key metric: realistic_net_pnl / naive_net_pnl
  >0.8  excellent | 0.5-0.8 acceptable | <0.5 fragile edge | <0 unrealistic fills

The probe strategy is deliberately simple and deterministic (EMA cross on
canonical mid with a spread filter) — its purpose is to expose execution-cost
erosion on the REAL tape, not to claim alpha. Both executions see the SAME
signals; only the fill/friction model differs.
"""
import json
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from data.canonical_tick import normalize_tick
from replay.tick_archive import TickArchive
from replay.replay_feed import ReplayFeed
from replay.replay_clock import ReplayClock
from data.replay_tick_source import ReplayTickSource
from data.execution_cost import ExecutionCost
from core.adapters.broker import PaperBroker

SYMBOL = "EURJPY"
DAYS = 30
FAST = 20   # EMA fast period (ticks)
SLOW = 100  # EMA slow period
MAX_SPREAD_PTS = 30  # skip signals when spread blown out (3x base 15)
LOT = 0.1


def ema(series, period):
    out = []
    k = 2.0 / (period + 1)
    prev = None
    for v in series:
        prev = v if prev is None else v * k + prev * (1 - k)
        out.append(prev)
    return out


def run_naive(ticks):
    """Mid-fill, zero friction: the 'backtest fantasy' baseline."""
    mids = [t["mid"] for t in ticks]
    f = ema(mids, FAST)
    s = ema(mids, SLOW)
    trades = []
    pos = 0.0  # +lot long, -lot short, 0 flat
    entry = 0.0
    for i, t in enumerate(ticks):
        if i < SLOW:
            continue
        if t["spread_pts"] > MAX_SPREAD_PTS:
            continue
        if pos == 0:
            if f[i] > s[i]:
                pos = LOT
                entry = t["mid"]
            elif f[i] < s[i]:
                pos = -LOT
                entry = t["mid"]
        elif pos > 0 and f[i] < s[i]:
            trades.append((entry, t["mid"], LOT, "BUY"))
            pos = 0
        elif pos < 0 and f[i] > s[i]:
            trades.append((entry, t["mid"], LOT, "SELL"))
            pos = 0
    if pos != 0:
        trades.append((entry, ticks[-1]["mid"], abs(pos), "BUY" if pos > 0 else "SELL"))
    return trades, [t["mid"] for t in ticks]


def pip_value(symbol, price):
    s = symbol.upper()
    if "JPY" in s:
        return 1000.0 / price if price > 0 else 8.0
    return 10.0


def compute_metrics(trades_pnl, equity_curve, symbol):
    gross = sum(tp for _, tp in trades_pnl)
    wins = sum(1 for _, tp in trades_pnl if tp > 0)
    losses = sum(1 for _, tp in trades_pnl if tp <= 0)
    win_rate = wins / len(trades_pnl) if trades_pnl else 0.0
    gross_win = sum(tp for _, tp in trades_pnl if tp > 0)
    gross_loss = -sum(tp for _, tp in trades_pnl if tp < 0)
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    peak = equity_curve[0]
    mdd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak else 0
        mdd = max(mdd, dd)
    return {
        "trades": len(trades_pnl),
        "win_rate": round(win_rate, 4),
        "gross_pnl": round(gross, 2),
        "max_drawdown": round(mdd, 4),
        "profit_factor": round(pf, 3) if pf != float("inf") else None,
    }


def main():
    archive = TickArchive()
    start = datetime.now() - timedelta(days=DAYS)
    end = datetime.now()
    df = archive.load_range(SYMBOL, start, end)
    if df is None or len(df.collect()) == 0:
        print("NO REAL TAPE FOUND — ingest first via FTMOTickIngester.ingest_days")
        sys.exit(2)
    n_ticks = len(df.collect())
    print(f"tape: {SYMBOL} {n_ticks} ticks ({start.date()}..{end.date()})")

    # --- canonical replay ---
    feed = ReplayFeed(clock=ReplayClock())
    feed.load_symbol(SYMBOL, df)
    src = ReplayTickSource(feed)
    ticks = []
    while True:
        t = src.next_tick(SYMBOL)
        if t is None:
            break
        ticks.append(t)
    print(f"replayed canonical: {len(ticks)} ticks, first={ticks[0]['mid']:.5f}")

    # --- NAIVE: mid-fill, zero friction ---
    trades, _ = run_naive(ticks)
    naive_trades, curve_naive = [], []
    balance = 100000.0
    for entry, exit_, lot, side in trades:
        p = (exit_ - entry) * lot * pip_value(SYMBOL, entry) if side == "BUY" \
            else (entry - exit_) * lot * pip_value(SYMBOL, entry)
        naive_trades.append((side, p))
        balance += p
        curve_naive.append(balance)
    naive_metrics = compute_metrics(naive_trades, curve_naive, SYMBOL)

    # --- REALISTIC: PaperBroker + ExecutionCost on the same signals ---
    ec = ExecutionCost(commission_per_lot=3.5, min_commission=0.0,
                       slippage_bps_range=(0.0, 3.0), enabled=True)

    class SeekableSource:
        """Serves the canonical tick at the current signal index, so fills
        happen on the SAME tick the signal fired (fair vs naive path)."""

        def __init__(self, tape):
            self._tape = tape
            self._i = 0

        def seek(self, i):
            self._i = i

        def get_tick(self, symbol):
            return self._tape[self._i] if self._tape else None

    class TapeClock:
        """Returns the CURRENT signal tick's time_sec so broker closed_at
        maps to the real tape day (FirmRisk day snapshots)."""

        def __init__(self):
            self._src = None

        def bind(self, src, ticks):
            self._src = src
            self._ticks = ticks

        def time(self):
            if self._src is not None:
                t = self._ticks[min(self._src._i, len(self._ticks) - 1)]
                return float(t.get("time_sec", 0))
            return 0.0

        def sleep(self, _s):
            pass

    src2 = SeekableSource(ticks)
    tclock = TapeClock()
    pb = PaperBroker(tick_source=src2, clock=tclock, execution_cost=ec,
                     initial_balance=100000.0)
    tclock.bind(src2, ticks)
    mids = [t["mid"] for t in ticks]
    f = ema(mids, FAST)
    s = ema(mids, SLOW)
    pos = 0
    for i, t in enumerate(ticks):
        if i < SLOW or t["spread_pts"] > MAX_SPREAD_PTS:
            continue
        src2.seek(i)
        if pos == 0 and f[i] != s[i]:
            side = "BUY" if f[i] > s[i] else "SELL"
            pb.place_order(SYMBOL, side, LOT, t["ask"] if side == "BUY" else t["bid"])
            pos = LOT if side == "BUY" else -LOT
        elif pos > 0 and f[i] < s[i]:
            open_ticks = [k for k, p in pb._positions.items() if p["side"] == "BUY"]
            pb.close_order(open_ticks[-1])
            pos = 0
        elif pos < 0 and f[i] > s[i]:
            open_ticks = [k for k, p in pb._positions.items() if p["side"] == "SELL"]
            pb.close_order(open_ticks[-1])
            pos = 0

    # FirmRisk on the realized path — snapshots keyed by REAL tape days via
    # the broker's closed_at (returned by TapeClock as tick time_sec).
    from proxima_ops.risk.firm_risk import FirmRiskEvaluator, FirmRiskConfig
    F = FirmRiskEvaluator(FirmRiskConfig())
    day_pnl: dict = {}
    for h in pb.history:
        day = (datetime.fromtimestamp(h["closed_at"]).date()
               if h.get("closed_at", 0) > 0 else datetime.now().date())
        day_pnl[day] = day_pnl.get(day, 0.0) + h["profit"]
    snapshots = []
    cum = 100000.0
    for day in sorted(day_pnl):
        cum += day_pnl[day]
        snapshots.append((day, cum, 0.5))
    if not snapshots:
        snapshots = [(datetime.now().date(), 100000.0 + pb.total_pnl, 0.5)]
    verdict = F.evaluate(snapshots)
    realistic_trades = [(h["side"], h["profit"]) for h in pb.history]
    curve_real = [100000.0]
    for h in pb.history:
        curve_real.append(curve_real[-1] + h["profit"])
    real_metrics = compute_metrics(realistic_trades, curve_real, SYMBOL)
    # commission lives on closed positions (both legs), not in history
    closed_pos = [p for p in pb._positions.values() if p.get("status") == "CLOSED"]
    real_metrics["commission"] = round(sum(p.get("commission", 0.0) for p in closed_pos), 2)
    real_metrics["net_pnl"] = round(pb.total_pnl, 2)
    real_metrics["firm_verdict"] = verdict.survived
    real_metrics["firm_reason"] = verdict.reason

    naive_metrics["net_pnl"] = round(balance - 100000.0, 2)
    naive_metrics["commission"] = 0.0
    n_naive = balance - 100000.0
    n_real = pb.total_pnl
    if abs(n_naive) < 1.0:
        ratio = None
        interpretation = ("naive baseline ~0; edge destroyed by costs" if n_real < -100.0
                          else "naive baseline ~0; absolute realistic PnL small")
    elif n_naive > 0.0:
        ratio = round(n_real / n_naive, 4)
        interpretation = ("excellent" if ratio > 0.8 else
                          "acceptable" if ratio > 0.5 else
                          "fragile edge" if ratio > 0 else
                          "edge destroyed by costs")
    else:
        # naive itself loses money -> no edge to preserve; ratio not meaningful
        ratio = None
        interpretation = "no naive edge (naive PnL<0); costs only amplify the loss"

    report = {
        "symbol": SYMBOL,
        "ticks": n_ticks,
        "naive": naive_metrics,
        "realistic": real_metrics,
        "realistic_over_naive": round(ratio, 4) if ratio is not None else None,
        "interpretation": interpretation,
    }
    with open("comparison_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))
    print("comparison_report.json written")


if __name__ == "__main__":
    main()