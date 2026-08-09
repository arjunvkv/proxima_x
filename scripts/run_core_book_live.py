"""run_core_book_live.py — COMBINED core-book live worker (Tokyo hour-0 fade +
post-Tokyo cascade @ UTC 2-4), sized for the $25k funded account with an
equity guard. FTMO is used as the demo/paper sandbox only.

Backtest<->live contract (proven live by the 54/54 micro-batch parity PASS):
  * SIGNAL bar = the first CLOSED M5 bar of each strategy's session hour(s),
    server bar stamp (never host wall clock — unreliable on this host).
  * RANK  = cross-sectional return over the strategy's lookback, ending at the
    signal bar's CLOSE, all 18 pairs.
  * ENTER = top_n most-negative (n_worst), filled at the OPEN of the bar AFTER
    the signal bar (never inside the signal bar — no lookahead).
  * EXIT  = SL/TP distances from fill, stop-first intrabar; else close at the
    open of entry + hold_bars.
  * Commission = $3.5/lot/side engine constant (live broker charges $3.0 ->
    $0.50/side built-in buffer).

Position sizing (tuned, 30% conservative risk tier on $25k):
  * Tokyo   : 0.35 lot  (2.0 x validated 0.15 x k=1.154)
  * Cascade : 0.09 lot  (0.5 x validated 0.15 x k=1.154)
  => worst historical day ~1.5% of account; 3x margin under 5% daily limit.

Equity guard (FTMO-style, distinct rules, keyed to SERVER time):
  * daily-loss: today's realized+floating loss >= 5% of day-open equity -> halt.
  * drawdown:   loss from peak >= 10% of peak -> halt (no new entries).
  Both block NEW entries; open positions still get SL/TP + hold-managed exits.

Safety posture (same as run_tokyo_h0_live.py):
  * attach-only (MT5_PATH pinned to FTMO terminal, creds neutralized, no login).
  * never enters a (day,strategy) twice (JSON state files).
  * 300s late-fill skip: a fresh process must never fire into a past fill
    window (deviates from the validated fill bar).
  * default dry-run; --execute is the explicit live switch.
  * NO cron: run manually with --daemon; no auto-relaunch on crash/reboot.
"""
from __future__ import annotations
import os, sys, argparse, time, json
from datetime import datetime, timedelta, timezone

ROOT = os.environ.get("PROXIMA_ROOT", r"C:\Trading\Proxima_X")
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ---------------- attach-only guard: neutralize identity creds; pin FTMO path --
# MT5_PATH honored from env (Linux/Wine deploys set it); Windows default below.
os.environ.setdefault("MT5_PATH",
                      r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe")
os.environ["MT5_ACCOUNT"] = ""
os.environ["MT5_PASSWORD"] = ""
os.environ["MT5_SERVER"] = ""

import MetaTrader5 as mt5

UNIVERSE = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY", "GBPJPY",
            "EURAUD", "EURNZD", "GBPAUD", "GBPNZD", "GBPCAD", "AUDNZD",
            "USDCAD", "NZDUSD", "EURGBP", "EURCHF", "USDCHF", "AUDJPY"]

# ---- strategy configs (identical to the $25k live manifests) -------------
STRATS = {
    "tokyo": {
        "sessions": [0], "lookback": 6, "top_n": 3, "hold_bars": 12,
        "lot": 0.35, "comment": "CORE_TOKYO_25",
        "sl_tp": {"JPY": (0.50, 0.70), "else": (0.0050, 0.0070)},
    },
    "cascade": {
        "sessions": [2, 3, 4], "lookback": 1440, "top_n": 8, "hold_bars": 24,
        "lot": 0.09, "comment": "CORE_CASCADE_25",
        "sl_tp": {"JPY": (0.40, 0.60), "else": (0.0040, 0.0060)},
    },
    "london": {
        "sessions": [7, 8, 9], "lookback": 1440, "top_n": 5, "hold_bars": 12,
        "lot": 0.15, "comment": "CORE_LONDON_25",
        "sl_tp": {"JPY": (0.40, 0.60), "else": (0.0040, 0.0060)},
    },
    "usfade": {
        "sessions": [14, 15, 16, 17, 18, 19], "lookback": 50, "top_n": 5, "hold_bars": 24,
        "lot": 0.30, "comment": "CORE_USFADE_25",
        "sl_tp": {"JPY": (0.40, 0.60), "else": (0.0040, 0.0060)},
    },
}

# $25k funded risk rules
DAILY_LOSS_LIMIT_PCT = 0.05
MAX_DD_PCT = 0.10
PROFIT_TARGET_USD = 2500.0

POLL_S = 10
POST_FILL_TOL_S = 300      # the fill M5 bar lives 5 min; skip if past it
MAX_RUNTIME_S = 30 * 24 * 3600  # daemon: ~monthly recycle
STATE_DIR = os.path.join(ROOT, "proxima_ops", "state")


def server_now() -> int:
    """FTMO server clock (TimeCurrent equivalent) — reliable for gating."""
    for sym in ("EURUSD", "USDJPY"):
        t = mt5.symbol_info_tick(sym)
        if t is not None:
            return int(t.time)
    return int(time.time())


# ----------------------------------------------------------- live journal ----
# Per-fill evidence for the measured alignment score (_alignment_score.py).
# Appends one JSON line per real fill / close / hourly spread snapshot.
JOURNAL = os.path.join(ROOT, "logs", "core_book_trades.jsonl")
SPREAD_LOG = os.path.join(ROOT, "logs", "core_book_spreads.csv")


def pip_size(sym: str) -> float:
    """1 pip in price terms: 0.01 for JPY pairs, 0.0001 otherwise."""
    return 0.01 if "JPY" in sym else 0.0001


def spread_pips(symbol: str, tick) -> float | None:
    if tick is None:
        return None
    return round((tick.ask - tick.bid) / pip_size(symbol), 2)


def journal_write(rec: dict) -> None:
    try:
        os.makedirs(os.path.dirname(JOURNAL), exist_ok=True)
        with open(JOURNAL, "a") as f:
            f.write(json.dumps(rec) + "\n")
    except Exception as e:
        print(f"[journal] write failed: {e}")


def journal_entry(name: str, symbol: str, lot: float, side: str,
                  requested: float, actual: float, sl: float, tp: float,
                  model_open: float | None, ticket: int) -> None:
    journal_write({
        "kind": "entry", "strategy": name, "symbol": symbol, "lot": lot,
        "side": side, "requested": requested, "actual": actual,
        "sl": sl, "tp": tp, "model_open": model_open,
        "requested_pips": round(abs(actual - requested) / pip_size(symbol), 2),
        "spread_pips_at_fill": spread_pips(symbol, mt5.symbol_info_tick(symbol)),
        "ticket": int(ticket), "ts": server_now(),
        "utc": datetime.utcfromtimestamp(server_now()).isoformat() + "Z",
    })


def journal_spread() -> None:
    """One hourly snapshot of UNIVERSE bid/ask spreads → CSV (live envelope)."""
    try:
        os.makedirs(os.path.dirname(SPREAD_LOG), exist_ok=True)
        row = [str(int(server_now()))]
        for sym in UNIVERSE:
            t = mt5.symbol_info_tick(sym)
            row.append(str(round((t.ask - t.bid) / pip_size(sym), 2)) if t else "")
        with open(SPREAD_LOG, "a") as f:
            f.write(",".join(row) + "\n")
    except Exception as e:
        print(f"[journal-spread] failed: {e}")


# trades that closed since last scan, keyed by position ticket
_ticket_map: dict[int, dict] = {}


def journal_closed(now: int) -> None:
    """Reconcile positions that exited since the last poll (SL/TP/managed)."""
    try:
        deals = mt5.history_deals_get(
            datetime.utcfromtimestamp(now - 7200),
            datetime.utcfromtimestamp(now))
    except Exception:
        return
    if not deals:
        return
    for d in deals:
        pos = int(d.position_id)
        if pos not in _ticket_map:
            continue
        if d.entry != mt5.DEAL_ENTRY_OUT:
            continue
        pnl = float(d.profit or 0.0)
        comm = float(d.commission or 0.0)
        journal_write({
            "kind": "close", "ticket": pos, "exit": float(d.price),
            "exit_pips": None, "pnl": pnl, "commission": comm,
            "reason": getattr(d, "comment", ""), "ts": int(d.time),
        })
        _ticket_map.pop(pos, None)


def journal_spread_hour(now_hour: int, last_spread_hour: list) -> None:
    if now_hour != last_spread_hour[0]:
        last_spread_hour[0] = now_hour
        journal_spread()


def bar_hour(ts: int) -> int:
    return (ts // 3600) % 24


def fetch_bars(symbol: str, n: int) -> list[dict]:
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n)
    if rates is None:
        return []
    return [{"ts": int(r["time"]), "open": r["open"], "high": r["high"],
             "low": r["low"], "close": r["close"]} for r in rates]


def fetch_day_bars(symbol: str, day: datetime) -> list[dict]:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) - timedelta(hours=2)
    end = start + timedelta(days=1, hours=2)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start, end)
    if rates is None:
        return []
    return [{"ts": int(r["time"]), "open": r["open"], "high": r["high"],
             "low": r["low"], "close": r["close"]} for r in rates]


def load_cached_day(symbol: str, day: datetime) -> list[dict]:
    import polars as pl
    p = os.path.join(ROOT, "audit_7_eas", "market", f"{symbol}.pqt")
    if not os.path.exists(p):
        return []
    df = pl.read_parquet(p)
    lo = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()) - 7200
    hi = lo + 86400 + 7200
    out = []
    for r in df.iter_rows(named=True):
        ts = int(r["time"])
        if lo <= ts <= hi:
            out.append({"ts": ts, "open": r["open"], "high": r["high"],
                        "low": r["low"], "close": r["close"]})
    return out


# ------------------------------------------------------------- signal --------
def session_rank(bars_map: dict[str, list[dict]], day: int, cfg: dict) -> list[dict]:
    """Top_n most-negative `lookback`-bar M5 returns at the FIRST bar of each of
    the strategy's session hours, mirroring the validated engine contract
    (fill = signal bar + 1 = next bar open). One signal per (day, hour)."""
    candidates = []
    for sym, bars in bars_map.items():
        if len(bars) < cfg["lookback"] + 1:
            continue
        seen_hours: set = set()
        for i, b in enumerate(bars):
            if b["ts"] // 86400 != day:
                continue
            h = bar_hour(b["ts"])
            if h not in cfg["sessions"] or h in seen_hours:
                continue
            if i < cfg["lookback"]:
                continue
            seen_hours.add(h)
            ret = (b["close"] - bars[i - cfg["lookback"]]["close"]) / bars[i - cfg["lookback"]]["close"]
            candidates.append({"symbol": sym, "ret": ret, "signal_ts": b["ts"],
                               "fill_ts": bars[i + 1]["ts"] if i + 1 < len(bars) else None,
                               "model_open": bars[i + 1]["open"] if i + 1 < len(bars) else None})
    if not candidates:
        return []
    candidates.sort(key=lambda x: x["ret"])
    return candidates[: cfg["top_n"]]


# -------------------------------------------------------------- state --------
def load_state(name: str) -> dict:
    p = os.path.join(STATE_DIR, f"corebook_{name}_state.json")
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_entry_day": None}


def save_state(name: str, st: dict) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    p = os.path.join(STATE_DIR, f"corebook_{name}_state.json")
    with open(p, "w") as f:
        json.dump(st, f)


# ----------------------------------------------------------- equity guard ----
class EquityGuard:
    """FTMO-style live guard on $25k reference capital. Daily-loss and peak-DD
    caps are expressed in ABSOLUTE dollars anchored to the reference capital
    (initial), so a larger/demo account is still governed by the 25k book's
    risk ceilings ($1,250 daily / $2,500 peak-DD at 5%/10%). Blocks NEW entries
    only; open positions are SL/TP-managed."""

    def __init__(self, initial: float = 25000.0):
        self.peak = initial
        self.day_key = None
        self.day_open_eq = initial
        self.daily_cap = initial * DAILY_LOSS_LIMIT_PCT
        self.max_dd_cap = initial * MAX_DD_PCT

    def check(self) -> tuple[bool, str]:
        acct = mt5.account_info()
        if acct is None:
            return True, "no account_info"
        eq = acct.equity if acct.equity else acct.balance
        bal = acct.balance
        # account may already have float: track peak as real equity high-water,
        # but the block thresholds stay anchored to the 25k reference capital.
        if eq > self.peak:
            self.peak = eq
        day = server_now() // 86400
        if self.day_key != day:
            self.day_key = day
            self.day_open_eq = eq   # day-open equity (server-day boundary)
        day_loss = self.day_open_eq - eq
        if day_loss > 0 and day_loss >= self.daily_cap:
            return True, (f"daily loss {day_loss:.0f}$ >= cap {self.daily_cap:.0f}$ "
                          f"({DAILY_LOSS_LIMIT_PCT:.0%} of ref cap) eq={eq:.0f}")
        dd = self.peak - eq
        if dd > 0 and dd >= self.max_dd_cap:
            return True, (f"drawdown {dd:.0f}$ >= cap {self.max_dd_cap:.0f}$ "
                          f"({MAX_DD_PCT:.0%} of ref cap) peak={self.peak:.0f} eq={eq:.0f}")
        return False, "ok"


def sl_tp_abs(sym: str, fill: float, side: str, cfg: dict) -> tuple[float, float]:
    d_sl, d_tp = cfg["sl_tp"]["JPY"] if "JPY" in sym else cfg["sl_tp"]["else"]
    if side == "BUY":
        return fill - d_sl, fill + d_tp
    return fill + d_sl, fill - d_tp


# --------------------------------------------------------------- replay ------
def replay(day_str: str) -> None:
    day = datetime.strptime(day_str, "%Y-%m-%d")
    ds = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()) // 86400
    for name, cfg in STRATS.items():
        live, cached = {}, {}
        for sym in UNIVERSE:
            live[sym] = fetch_day_bars(sym, day)
            cached[sym] = load_cached_day(sym, day)
        rl = session_rank(live, ds, cfg)
        rc = session_rank(cached, ds, cfg)
        lset = {(x["symbol"], round(x["ret"], 6)) for x in rl}
        cset = {(x["symbol"], round(x["ret"], 6)) for x in rc}
        print(f"[replay {name}] live={len(rl)} cached={len(rc)} "
              f"match={len(lset & cset)}/{len(cset)}")
        for x in rl:
            print(f"   {x['symbol']:<8} ret={x['ret']:+.5%}")


# ----------------------------------------------------------------- main ------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replay", metavar="YYYY-MM-DD", default=None,
                    help="prove live-oracle vs cached-bars alignment (no orders)")
    ap.add_argument("--execute", action="store_true",
                    help="actually place orders (default: dry-run)")
    ap.add_argument("--manage", action="store_true",
                    help="close core-book positions older than their hold_bars on poll")
    ap.add_argument("--once", action="store_true",
                    help="one poll-and-decide pass, then exit (for tests)")
    ap.add_argument("--daemon", action="store_true",
                    help="run across many sessions (manual; NO cron)")
    ap.add_argument("--init-balance", type=float, default=25000.0,
                    help="$25k funded starting balance for the equity guard")
    args = ap.parse_args()

    if not mt5.initialize(path=os.environ["MT5_PATH"], timeout=4000):
        print("INIT FAILED:", mt5.last_error())
        sys.exit(1)
    acct = mt5.account_info()
    print(f"[attach] account={acct.login if acct else 'N/A'} "
          f"server={acct.server if acct else ''} balance={round(acct.balance, 2) if acct else 'N/A'}")
    now = server_now()
    print(f"[clock] server now={datetime.utcfromtimestamp(now):%Y-%m-%d %H:%M:%S}Z "
          f"hour={bar_hour(now)}")

    if args.replay:
        try:
            replay(args.replay)
        finally:
            mt5.shutdown()
        return

    guard = EquityGuard(args.init_balance)
    states = {n: load_state(n) for n in STRATS}
    last_actionable: dict[str, int] = {}
    last_spread_hour = [None]
    started = server_now()
    while True:
        now = server_now()
        if now - started > MAX_RUNTIME_S:
            print(f"[exit] max runtime {MAX_RUNTIME_S//3600}h reached. bye")
            break
        today = now // 86400
        hour = bar_hour(now)

        # ---- 0) live-journal: reconcile closes since last poll + hourly spread
        if args.execute:
            journal_closed(now)
            journal_spread_hour(hour, last_spread_hour)

        # ---- 1) hold-managed exits (broker SL/TP handles most; close stragglers)
        if args.manage:
            try:
                for pos in mt5.positions_get() or []:
                    for name, cfg in STRATS.items():
                        if pos.comment.startswith(cfg["comment"]):
                            age_bars = (now - pos.time) // 300
                            if age_bars >= cfg["hold_bars"]:
                                print(f"[manage {name}] closing {pos.symbol} "
                                      f"ticket={pos.ticket} age={age_bars} bars")
                                _close(pos.ticket)
            except Exception as e:
                print(f"[manage] error: {e}")

        # ---- 2) session-window entry gate per strategy (server hours)
        for name, cfg in STRATS.items():
            if hour not in cfg["sessions"]:
                continue
            st = states[name]
            if st.get("last_entry_day") == today:
                continue
            bars_map = {sym: fetch_bars(sym, cfg["lookback"] + 4) for sym in UNIVERSE}
            rank = session_rank(bars_map, today, cfg)
            if not rank:
                continue
            fill_ts = rank[0]["fill_ts"]
            if fill_ts is None or now < fill_ts:
                continue
            if now > fill_ts + POST_FILL_TOL_S:
                print(f"[skip {name}] now {datetime.utcfromtimestamp(now):%H:%M:%S}Z > "
                      f"fill {datetime.utcfromtimestamp(fill_ts):%H:%M:%S}Z + tolerance — "
                      f"skipping today to preserve the validated fill bar")
                st["last_entry_day"] = today
                save_state(name, st)
                continue
            if last_actionable.get(name) == today:
                continue
            last_actionable[name] = today
            print(f"[signal {name} {datetime.utcfromtimestamp(now):%Y-%m-%d %H:%M:%S}Z] "
                  f"top-{cfg['top_n']} @ lot {cfg['lot']}:")
            for x in rank:
                print(f"     {x['symbol']:<8} ret={x['ret']:+.5%} "
                      f"fill_ts={datetime.utcfromtimestamp(x['fill_ts'])}")
            if not args.execute:
                print(f"[dry-run {name}] no orders — rerun with --execute")
            else:
                blocked, why = guard.check()
                if blocked:
                    print(f"[FirmRisk BLOCK {name}] {why}")
                else:
                    acct_info = mt5.account_info()
                    bal = acct_info.balance if acct_info else 25000.0
                    opened = 0
                    for x in rank:
                        tick = mt5.symbol_info_tick(x["symbol"])
                        if tick is None:
                            continue
                        fill = tick.ask
                        sl, tp = sl_tp_abs(x["symbol"], fill, "BUY", cfg)
                        res = _order_buy(x["symbol"], fill, sl, tp, cfg, bal)
                        if res:
                            opened += 1
                            _ticket_map[res["ticket"]] = x
                            journal_entry(name, x["symbol"], cfg["lot"], "BUY",
                                          res["requested"], res["actual_fill"],
                                          sl, tp, x.get("model_open"), res["ticket"])
                            print(f"[FILL {name}] BUY {x['symbol']} {cfg['lot']} "
                                  f"@ {res['actual_fill']} (req {fill}) "
                                  f"sl={sl} tp={tp} slip={res['actual_fill']-fill:+.5f}")
                        else:
                            print(f"[REJECT {name}] BUY {x['symbol']}")
                        time.sleep(0.4)
                    st["last_entry_day"] = today
                    save_state(name, st)
                    print(f"[done {name}] opened {opened}/{cfg['top_n']} — day {today} recorded")
        if args.once:
            break
        time.sleep(POLL_S)
    mt5.shutdown()


def _order_buy(symbol: str, fill: float, sl: float, tp: float,
               cfg: dict, bal: float) -> dict | None:
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": cfg["lot"],
           "type": mt5.ORDER_TYPE_BUY, "price": fill, "sl": sl, "tp": tp,
           "deviation": 5, "magic": 777200, "comment": cfg["comment"],
           "type_time": 0, "type_filling": mt5.ORDER_FILLING_IOC}
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        return None
    deal = getattr(res, "deal", None)
    return {
        "order": getattr(res, "order", None),
        "ticket": int(getattr(deal, "position_id", 0) or 0) or int(getattr(res, "order", 0) or 0),
        "actual_fill": float(getattr(deal, "price", fill) or fill),
        "requested": fill,
    }


def _close(ticket: int) -> None:
    pos = next((p for p in (mt5.positions_get() or []) if p.ticket == ticket), None)
    if pos is None:
        return
    tick = mt5.symbol_info_tick(pos.symbol)
    if tick is None:
        return
    side = mt5.ORDER_TYPE_SELL if pos.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY
    price = tick.bid if pos.type == mt5.POSITION_TYPE_BUY else tick.ask
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": pos.volume,
           "type": side, "position": ticket, "price": price, "deviation": 5,
           "magic": 777300, "comment": "CORE_CLOSE", "type_time": 0,
           "type_filling": mt5.ORDER_FILLING_IOC}
    res = mt5.order_send(req)
    if res is None or res.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[close] {pos.symbol} ticket={ticket} retcode={getattr(res,'retcode',None)}")


if __name__ == "__main__":
    main()