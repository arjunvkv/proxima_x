"""NautilusTrader independent validation of the Tokyo H0 session-hour decision.

Replicates the honest Tokyo H0 strategy (lb / hold / top_n config) inside a
NautilusTrader 1.221 backtest engine with CASH multi-currency accounting and
BestPriceFillModel bar fills. Runs the same strategy at session_hour=0 (UTC
midnight, the claimed edge) and session_hour=3 (the claimed losing hour) and
compares win rate / event count against the honest backtest's canonical
lb=6 hold=12 n=5 result (~95.3% WR, 212 trades, +$3,519.60 on Exness).

Fill semantics (empirically confirmed): a market order submitted inside
`on_bar` fills at the CURRENT bar's CLOSE for both default and BestPrice fill
models. So entries (honest: session-bar OPEN) are reproduced as session-bar
CLOSE fills — a ~5 minute shift with the same 60-min reversion window; exits
(honest: hold-expiry bar CLOSE) match exactly.

Run:
    python nautilus_tokyo_h0.py                      # hours 0 and 3
    python nautilus_tokyo_h0.py --hours 0            # hour 0 only
    python nautilus_tokyo_h0.py --limit 1500         # quick smoke (last 1500 bars/pair)
"""
import argparse
import math
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import BestPriceFillModel
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.currencies import AUD, CAD, CHF, EUR, GBP, JPY, NZD, USD
from nautilus_trader.model.data import Bar, BarSpecification, BarType
from nautilus_trader.model.enums import (
    AccountType,
    BarAggregation,
    OmsType,
    OrderSide,
    PriceType,
    TimeInForce,
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

ALL_PAIRS = [
    "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY", "GBPJPY",
    "EURAUD", "EURNZD", "GBPAUD", "GBPNZD", "GBPCAD", "AUDNZD",
    "USDCAD", "NZDUSD", "EURGBP", "EURCHF", "USDCHF", "AUDJPY",
]

CURRENCIES = {
    "EUR": EUR, "USD": USD, "JPY": JPY, "GBP": GBP,
    "AUD": AUD, "NZD": NZD, "CAD": CAD, "CHF": CHF,
}

_VENUE = Venue("SIM")
_BAR_SPEC = BarSpecification(5, BarAggregation.MINUTE, PriceType.LAST)
_DATA_DIR = Path(__file__).parent.parent / "data" / "m5"


def _base_quote(pair: str) -> Tuple[str, str]:
    for q in ("JPY", "USD", "EUR", "GBP", "AUD", "NZD", "CAD", "CHF"):
        if pair.endswith(q):
            return pair[: -len(q)], q
    raise ValueError(pair)


def _quote_currency(pair: str) -> str:
    return _base_quote(pair)[1]


def _pnl_to_usd(pnl: float, pair: str, price: float) -> float:
    """Mirror proxima_honest_backtest.execution.models._pnl_to_usd."""
    quote = _quote_currency(pair)
    if quote == "USD":
        return pnl
    if quote == "JPY":
        return pnl / price if price > 0 else pnl
    if quote == "EUR":
        return pnl * 1.10
    if quote == "GBP":
        return pnl * 1.28
    if quote == "AUD":
        return pnl * 0.65
    if quote == "NZD":
        return pnl * 0.60
    if quote == "CAD":
        return pnl * 0.73
    if quote == "CHF":
        return pnl * 1.12
    return pnl


class _RunningVol:
    """O(1) streaming standard deviation via Welford (mirrors strategy.py)."""

    __slots__ = ("_window", "_deque", "_sum", "_sum_sq", "count")

    def __init__(self, window: int) -> None:
        self._window = window
        self._deque: deque = deque(maxlen=window)
        self._sum = 0.0
        self._sum_sq = 0.0
        self.count = 0

    @property
    def std(self) -> float:
        n = min(self.count, self._window)
        if n < 2:
            return 0.0
        mean = self._sum / n
        var = max(self._sum_sq / n - mean * mean, 0.0)
        return math.sqrt(var) if var > 1e-24 else 0.0

    def update(self, value: float) -> float:
        if self.count >= self._window:
            old = self._deque[0]
            self._sum -= old
            self._sum_sq -= old * old
        self._deque.append(value)
        self._sum += value
        self._sum_sq += value * value
        self.count += 1
        return self.std


class TokyoH0Config(StrategyConfig, frozen=True):
    pairs: tuple = tuple(ALL_PAIRS)
    top_n: int = 5
    lookback_bars: int = 6
    lookback_confirm_bars: int = 3
    hold_bars: int = 12
    session_hour: int = 0
    gap_threshold_pct: float = 0.5
    min_pairs: int = 8
    min_confidence: float = 0.30
    require_decline_persistence: bool = True
    vol_window: int = 50
    quantity: int = 10000
    min_pairs_delivered: int = 18


class TokyoH0Naut(Strategy):
    """Nautilus port of the honest Tokyo H0 cross-pair mean-reversion strategy.

    Replicates the honest signal exactly by evaluating on the SAME aligned
    (ffilled) 5-minute grid that MultiPairBacktestEngine._align_bars produces:
    at a session row, curr = aligned close at the row, prev_lb = aligned close
    `lookback_bars` grid steps back, prev_short = aligned close
    `lookback_confirm_bars` steps back, entry at the aligned open. Exits are
    scheduled on the grid at +hold_bars rows (matching the honest bars_held
    counter which ticks once per grid row).
    """

    def __init__(self, config: TokyoH0Config) -> None:
        super().__init__(config)
        self._c = config
        self._vol: Dict[str, _RunningVol] = {}
        self._entered_day: str | None = None
        self._entry_orders: Dict = {}
        self._exit_orders: Dict = {}
        self._entry_px: Dict[str, float] = {}
        self._exit_at: Dict[str, int] = {}
        self._trades: List[dict] = []
        self._entries: List[dict] = []
        self._session_attempts = 0
        self._session_log: List[dict] = []
        self._qty = Quantity(config.quantity, precision=0)

        # Aligned grid (set via load_aligned_grid before add_strategy)
        self._grid_ts: List[int] = []
        self._grid_idx: Dict[int, int] = {}
        self._ac: Dict[str, List[float]] = {}
        self._ao: Dict[str, List[float]] = {}

    def load_aligned_grid(self, grid_ts: List[int], ac: Dict[str, List[float]],
                          ao: Dict[str, List[float]]) -> None:
        self._grid_ts = grid_ts
        self._grid_idx = {t: i for i, t in enumerate(grid_ts)}
        self._ac = ac
        self._ao = ao

    # ------------------------------------------------------------------
    def on_start(self) -> None:
        for pair in self._c.pairs:
            self.subscribe_bars(
                BarType(InstrumentId(Symbol(pair), _VENUE), _BAR_SPEC)
            )

    def on_bar(self, bar: Bar) -> None:
        self._check_exits(bar)
        self._maybe_enter(bar)

    # ------------------------------------------------------------------
    def _check_exits(self, bar: Bar) -> None:
        ts = bar.ts_event
        for pair in [p for p, t in self._exit_at.items() if t == ts]:
            instrument = self.cache.instrument(InstrumentId(Symbol(pair), _VENUE))
            if instrument is None:
                continue
            order = self.order_factory.market(
                instrument_id=instrument.id,
                order_side=OrderSide.SELL,
                quantity=self._qty,
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
            self._exit_orders[order.client_order_id] = pair
            del self._exit_at[pair]

    def _maybe_enter(self, bar: Bar) -> None:
        c = self._c
        dt = pd.Timestamp(bar.ts_event, unit="ns")
        if dt.hour != c.session_hour:
            return
        day = dt.strftime("%Y-%m-%d")
        if self._entered_day == day:
            return
        if bar.ts_event not in self._grid_idx:
            return
        self._run_entry(day, bar.ts_event)

    def _run_entry(self, day: str, ts: int) -> None:
        c = self._c
        self._entered_day = day
        self._session_attempts += 1

        p = self._grid_idx[ts]
        lb, lbc = c.lookback_bars, c.lookback_confirm_bars

        candidates: List[Tuple[str, float, float]] = []
        table: List[dict] = []

        for pair in c.pairs:
            ac = self._ac.get(pair)
            if ac is None or p < lb + 1:
                continue
            curr, prev_lb = ac[p], ac[p - lb]
            if not (curr > 0 and prev_lb > 0) or math.isnan(curr) or math.isnan(prev_lb):
                continue
            ret = math.log(curr / prev_lb)

            # §17: shorter lookback must also show decline
            persist_ok = True
            if c.require_decline_persistence and p >= lbc + 1:
                prev_short = ac[p - lbc]
                if prev_short > 0 and not math.isnan(prev_short):
                    ret_short = math.log(curr / prev_short)
                    if ret_short > 0:
                        persist_ok = False
            if not persist_ok:
                table.append({
                    "pair": pair, "ret": round(ret, 6), "persist_ok": False,
                    "vol": None, "margin": None, "conf": None,
                })
                continue

            # §19: decision margin relative to recent volatility
            vol = self._update_vol(pair, ret) or 0.001
            margin = abs(ret) / max(vol, 1e-10)
            conf = min(0.95, margin * 0.15)
            table.append({
                "pair": pair, "ret": round(ret, 6), "persist_ok": True,
                "vol": round(vol, 6), "margin": round(margin, 3),
                "conf": round(conf, 4),
            })
            candidates.append((pair, ret, margin))

        if len(candidates) < c.min_pairs:
            self._session_log.append({
                "day": day, "ts": ts, "n_candidates_raw": len(table),
                "table": table, "n_pass": len(candidates), "selected": [],
            })
            return

        candidates.sort(key=lambda x: x[1])  # most declined first

        selected = []
        exit_ts = self._grid_ts[p + c.hold_bars] if p + c.hold_bars < len(self._grid_ts) else None
        for pair, ret, margin in candidates[: c.top_n]:
            if ret >= 0:
                break  # no more declining pairs
            confidence = min(0.95, margin * 0.15)
            if confidence < c.min_confidence:
                continue
            instrument = self.cache.instrument(InstrumentId(Symbol(pair), _VENUE))
            if instrument is None:
                continue
            order = self.order_factory.market(
                instrument_id=instrument.id,
                order_side=OrderSide.BUY,
                quantity=self._qty,
                time_in_force=TimeInForce.GTC,
            )
            self.submit_order(order)
            self._entry_orders[order.client_order_id] = (pair, exit_ts)
            selected.append(pair)
        self._session_log.append({
            "day": day, "ts": ts, "n_candidates_raw": len(table),
            "table": table, "n_pass": len(candidates), "selected": selected,
        })

    def _update_vol(self, pair: str, ret: float) -> float:
        key = f"vol_{pair}"
        if key not in self._vol:
            self._vol[key] = _RunningVol(self._c.vol_window)
        return self._vol[key].update(ret)

    # ------------------------------------------------------------------
    def on_order_filled(self, event) -> None:
        oid = event.client_order_id
        px = event.last_px.as_double()
        qty = event.last_qty.as_double()

        if oid in self._entry_orders:
            pair, exit_ts = self._entry_orders.pop(oid)
            self._entry_px[pair] = px
            if exit_ts is not None:
                self._exit_at[pair] = exit_ts
            self._entries.append({"pair": pair, "ts": event.ts_event, "px": px})
        elif oid in self._exit_orders:
            pair = self._exit_orders.pop(oid)
            entry_px = self._entry_px.pop(pair, 0.0)
            raw = (px - entry_px) * qty
            pnl = _pnl_to_usd(raw, pair, (entry_px + px) / 2.0)
            self._trades.append({
                "pair": pair,
                "entry_px": entry_px,
                "exit_px": px,
                "qty": qty,
                "pnl": pnl,
                "ts": event.ts_event,
            })

    # ------------------------------------------------------------------
    def result(self) -> dict:
        trades = self._trades
        n = len(trades)
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        gross_profit = sum(t["pnl"] for t in wins)
        gross_loss = abs(sum(t["pnl"] for t in losses)) or 1.0
        per_pair: Dict[str, dict] = {}
        for t in trades:
            d = per_pair.setdefault(t["pair"], {"n": 0, "pnl": 0.0})
            d["n"] += 1
            d["pnl"] += t["pnl"]
        return {
            "session_hour": self._c.session_hour,
            "n_trades": n,
            "win_rate": len(wins) / n if n else 0.0,
            "total_pnl": sum(t["pnl"] for t in trades),
            "profit_factor": gross_profit / gross_loss if gross_loss else 0.0,
            "avg_win": sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0,
            "avg_loss": sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0,
            "session_attempts": self._session_attempts,
            "entries": len(self._entries),
            "wins": len(wins),
            "losses": len(losses),
            "per_pair": dict(per_pair),
            "trades": trades,
            "entries": self._entries,
        }


# ----------------------------------------------------------------------
# Engine construction
# ----------------------------------------------------------------------
def make_instrument(pair: str) -> CurrencyPair:
    base, quote = _base_quote(pair)
    jpy = pair.endswith("JPY")
    return CurrencyPair(
        instrument_id=InstrumentId(Symbol(pair), _VENUE),
        raw_symbol=Symbol(pair),
        base_currency=CURRENCIES[base],
        quote_currency=CURRENCIES[quote],
        price_precision=3 if jpy else 5,
        size_precision=0,
        price_increment=Price.from_str("0.001" if jpy else "0.00001"),
        size_increment=Quantity.from_str("1"),
        lot_size=Quantity.from_str("100000"),
        max_quantity=Quantity.from_str("10000000"),
        min_quantity=Quantity.from_str("1"),
        max_price=Price.from_str("10000"),
        min_price=Price.from_str("0.00001"),
        ts_event=0,
        ts_init=0,
    )


def load_bars(pairs: List[str], limit: int | None = None) -> List[Bar]:
    all_bars: List[Bar] = []
    for pair in pairs:
        bt = BarType(InstrumentId(Symbol(pair), _VENUE), _BAR_SPEC)
        files = sorted((_DATA_DIR / pair).glob("*.parquet"))
        frames = [pd.read_parquet(f) for f in files]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        if limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        prec = 3 if pair.endswith("JPY") else 5
        ns = df["time"].astype("int64").to_numpy()
        o = df["open"].to_numpy()
        h = df["high"].to_numpy()
        lo = df["low"].to_numpy()
        c = df["close"].to_numpy()
        v = df["tick_volume"].to_numpy()
        for i in range(len(df)):
            ts = int(ns[i])
            all_bars.append(Bar(
                bt,
                Price(float(o[i]), prec),
                Price(float(h[i]), prec),
                Price(float(lo[i]), prec),
                Price(float(c[i]), prec),
                Quantity(int(v[i]), precision=0),
                ts_event=ts,
                ts_init=ts,
            ))
    all_bars.sort(key=lambda b: b.ts_event)
    return all_bars


def build_aligned(pairs: List[str], limit: int | None = None):
    """Replicate MultiPairBacktestEngine._align_bars: concat closes/opens on the
    union 5-min grid, forward-fill, return grid ts + per-pair aligned arrays."""
    frames = {}
    for pair in pairs:
        files = sorted((_DATA_DIR / pair).glob("*.parquet"))
        dfs = [pd.read_parquet(f) for f in files]
        if not dfs:
            continue
        df = pd.concat(dfs, ignore_index=True)
        df.sort_values("time", inplace=True)
        df.reset_index(drop=True, inplace=True)
        if limit:
            df = df.iloc[-limit:].reset_index(drop=True)
        frames[pair] = df
    pieces = []
    for pair, df in frames.items():
        sub = df.set_index("time")[["close", "open", "high", "low", "tick_volume", "spread"]]
        sub.columns = [
            pair, f"{pair}_open", f"{pair}_high",
            f"{pair}_low", f"{pair}_volume", f"{pair}_spread",
        ]
        pieces.append(sub)
    combined = pd.concat(pieces, axis=1, sort=True)
    combined.sort_index(inplace=True)
    combined.ffill(inplace=True)
    ts_ns = combined.index.values.astype("int64")
    grid_ts = ts_ns.tolist()
    ac = {pair: combined[pair].to_numpy(dtype="float64").tolist() for pair in frames}
    ao = {pair: combined[f"{pair}_open"].to_numpy(dtype="float64").tolist() for pair in frames}
    return grid_ts, ac, ao


def run_session(session_hour: int, pairs: List[str], limit: int | None = None,
                min_confidence: float | None = None,
                persist: bool | None = None) -> dict:
    bars = load_bars(pairs, limit=limit)
    grid_ts, ac, ao = build_aligned(pairs, limit=limit)
    engine = BacktestEngine(BacktestEngineConfig(trader_id=f"TOKYO-H0-{session_hour}"))
    engine.add_venue(
        venue=_VENUE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.CASH,
        starting_balances=[Money(10_000_000, c) for c in CURRENCIES.values()],
        fill_model=BestPriceFillModel(),
    )
    for pair in pairs:
        engine.add_instrument(make_instrument(pair))
    engine.add_data(bars)
    cfg = TokyoH0Config(
        session_hour=session_hour,
        min_confidence=min_confidence if min_confidence is not None else 0.30,
        require_decline_persistence=True if persist is None else persist,
    )
    strategy = TokyoH0Naut(cfg)
    strategy.load_aligned_grid(grid_ts, ac, ao)
    engine.add_strategy(strategy)
    engine.run()
    result = strategy.result()
    result["session_log"] = strategy._session_log
    engine.dispose()
    return result


def print_result(res: dict) -> None:
    h = res["session_hour"]
    print(f"  session_hour={h}: session_attempts={res['session_attempts']} "
          f"entries={res['entries']} trades={res['n_trades']}")
    print(f"  WR={res['win_rate']*100:5.1f}%  PnL=${res['total_pnl']:+9.2f}  "
          f"PF={res['profit_factor']:6.2f}  avg_win=${res['avg_win']:+6.2f}  "
          f"avg_loss=${res['avg_loss']:+6.2f}")
    if res["per_pair"]:
        print("  per-pair:")
        for pair, d in sorted(res["per_pair"].items(), key=lambda kv: kv[1]["n"], reverse=True):
            print(f"    {pair:8s} n={d['n']:>3d} pnl=${d['pnl']:+9.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", default="0,3", help="comma-separated session hours")
    ap.add_argument("--limit", type=int, default=None,
                    help="keep only last N bars per pair (smoke test)")
    ap.add_argument("--pairs", nargs="*", default=ALL_PAIRS)
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument("--no-persist", action="store_true")
    ap.add_argument("--debug-dir", default=None, help="dump per-session selections")
    args = ap.parse_args()

    hours = [int(x) for x in args.hours.split(",")]
    print(f"Tokyo H0 Nautilus validation — {len(args.pairs)} pairs, "
          f"hours {hours}" + (f", limit={args.limit}" if args.limit else ""))
    results = {}
    for h in hours:
        print(f"\n===== session_hour={h} =====")
        res = run_session(h, args.pairs, limit=args.limit,
                          min_confidence=args.min_confidence,
                          persist=None if args.no_persist else True)
        results[h] = res
        print_result(res)
        if args.debug_dir:
            Path(args.debug_dir).mkdir(parents=True, exist_ok=True)
            import json
            with open(Path(args.debug_dir) / f"naut_h{h}.json", "w") as f:
                json.dump(res["session_log"], f, indent=1)

    print("\n===== COMPARISON (vs honest backtest lb=6 hold=12 n=5) =====")
    print(f"{'hour':>4s} {'trades':>6s} {'WR':>7s} {'PnL USD':>10s} {'PF':>6s}  honest_ref")
    ref = {0: ("211 trades / 94.8% / +$3,520.48",),
           3: ("86 trades / 38.4% / -$349.83",)}
    for h in hours:
        r = results[h]
        refs = ref.get(h, ("no honest ref",))
        print(f"{h:>4d} {r['n_trades']:>6d} {r['win_rate']*100:6.1f}% "
              f"{r['total_pnl']:>+10.2f} {r['profit_factor']:>6.2f}  {refs[0]}")


if __name__ == "__main__":
    main()
