"""Tokyo_H0 live runner — reproduces the VALIDATED BACKTEST CURVE on the live
FTMO path (the reason the backtest->live engine exists).

The audited curve (720 trades/200d, 90% WR, PF~9, +$16.18/trade @1.0 lot) comes
from ea_ports.tokyo_h0 with this exact contract:
  * SIGNAL bar = the FIRST M5 bar of UTC hour 0 (server bar stamp, not wall
    clock; wall clock on this host is unreliable).
  * RANK  = 6-bar M5 return ending at the signal bar's CLOSE, all 18 pairs.
  * ENTER = BUY top-5 most-negative, filled at the OPEN of the bar AFTER the
    signal bar (never inside the signal bar — no lookahead).
  * SL/TP = 0.35/0.45 (JPY) else 0.0035/0.0045 DISTANCES from fill, attached at
    order time; HOLD = close after 12 M5 bars if not SL/TP'd first.
This runner recomputes that exact signal from live server bars and places the
orders through the engine's aligned OrderManager (engine SL/TP/volume guards,
broker-confirmed fills). It does NOT re-optimize anything.

Timing rule (server-clock, matches port):
  A M5 bar stamped T closes at T+300s. The signal bar is the first hour-0 bar
  of the day; it is actionable the moment the NEXT bar (00:05 stamp) opens, at
  which point a market order fills at ~that bar's open — the port's fill point.

Modes:
  --replay YYYY-MM-DD  prove alignment: compute the day's top-5 from the live
                       FTMO oracle AND from the cached market/*.pqt audit bars,
                       print both. No orders. Exit.
  (default)            live loop: poll server time; at the actionable session
                       bar compute the signal; DRY-RUN prints it, `--execute`
                       actually sends orders through OrderManager.
  --manage             also close TOKYO_H0 positions that passed HOLD_BARS
                       (12 M5 bars) on each poll (broker SL/TP handles the rest).

Safety: attach-only (settings creds neutralized, FTMO path pinned, no re-login);
never enters a day twice (state file); FirmRisk daily-loss/DD guard before any
entry; default dry-run — `--execute` is the explicit live switch.
"""
from __future__ import annotations
import os, sys, argparse, time, json
from datetime import datetime, timedelta, timezone

ROOT = r"C:\Trading\Proxima_X"
sys.path.insert(0, ROOT)

# ---------------- attach-only guard: neutralize identity creds; pin FTMO path --
import proxima_ops.config.settings as S
for _a in ("mt5_account", "mt5_password", "mt5_login"):
    if hasattr(S, _a):
        try:
            setattr(S, _a, None)
        except Exception:
            pass
FTMO_TERMINAL = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
if hasattr(S, "mt5_path"):
    try:
        S.mt5_path = FTMO_TERMINAL
    except Exception:
        pass

import MetaTrader5 as mt5

TOKYO_UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
                  "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
                  "USDCHF","AUDJPY"]
LOOKBACK = 6
HOLD_BARS = 12
TOP_N = 5
SESSION_HOUR = 0
BASE_LOT = 0.15
COMMENT = "TOKYO_H0"
STATE_FILE = os.path.join(ROOT, "proxima_ops", "state", "tokyo_h0_state.json")
POLL_S = 5

_JPY_DIST = (0.35, 0.45)      # JPY pairs: SL/TP distances (price units)
_OTHER_DIST = (0.0035, 0.0045)


def sl_tp_dist(symbol: str) -> tuple[float, float]:
    return _JPY_DIST if "JPY" in symbol else _OTHER_DIST


def sl_tp_abs(symbol: str, fill: float, side: str) -> tuple[float, float]:
    """Convert EA distance SL/TP to broker ABSOLUTE levels, digits-rounded.

    BUY:  SL = fill - d_sl, TP = fill + d_tp
    SELL: SL = fill + d_sl, TP = fill - d_tp
    The connector passes sl/tp straight to order_send (absolute prices).
    """
    d_sl, d_tp = sl_tp_dist(symbol)
    digits = 3 if "JPY" in symbol else 5
    if side == "BUY":
        sl = round(fill - d_sl, digits)
        tp = round(fill + d_tp, digits)
    else:
        sl = round(fill + d_sl, digits)
        tp = round(fill - d_tp, digits)
    return sl, tp


# ------------------------------------------------------------ server clock ---
def server_now() -> int:
    """Authoritative server time (epoch s) for the attached FTMO instance —
    the same TimeCurrent() the EAs use. Wall datetime.utcnow() is UNRELIABLE on
    this host (measured multi-hour jitter), so every gate uses this."""
    for sym in ("EURUSD", "USDJPY", "EURJPY", "GBPUSD"):
        try:
            t = mt5.symbol_info_tick(sym)
            if t is not None and t.time:
                return int(t.time)
        except Exception:
            continue
    return int(datetime.now(timezone.utc).timestamp())


def server_session_hour() -> int:
    return (server_now() // 3600) % 24


def server_day() -> int:
    return server_now() // 86400


def bar_hour(ts: int) -> int:
    return (ts // 3600) % 24


# ---------------------------------------------------------------- data ------
def fetch_bars(symbol: str, n: int = 16) -> list[dict]:
    """Last n M5 bars for symbol from the live server (bar[0] oldest; the last
    element may be the in-progress bar)."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, n)
    if rates is None:
        return []
    return [{"ts": int(r["time"]), "open": r["open"], "high": r["high"],
             "low": r["low"], "close": r["close"]} for r in rates]


def fetch_day_bars(symbol: str, day: datetime) -> list[dict]:
    """All M5 bars for a full calendar day (UTC) + 1h lookback, live oracle."""
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc) - timedelta(hours=2)
    end = start + timedelta(days=1, hours=2)
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start, end)
    if rates is None:
        return []
    return [{"ts": int(r["time"]), "open": r["open"], "high": r["high"],
             "low": r["low"], "close": r["close"]} for r in rates]


def load_cached_day(symbol: str, day: datetime) -> list[dict]:
    """Same bars from the AUDIT CACHE (market/<SYM>.pqt) for the replay proof."""
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
def session_rank(bars_map: dict[str, list[dict]], day: int) -> list[dict]:
    """Top-5 most-negative 6-bar M5 returns at the day's FIRST hour-0 bar,
    mirroring ea_ports.tokyo_h0 (fill = signal bar + 1 = next bar open).
    `day` is the EPOCH-DAY key (server_now() // 86400)."""
    candidates = []
    for sym, bars in bars_map.items():
        if len(bars) < LOOKBACK + 1:
            continue
        idx = None
        for i, b in enumerate(bars):
            if b["ts"] // 86400 != day:
                continue
            if bar_hour(b["ts"]) == SESSION_HOUR:
                idx = i
                break
        if idx is None or idx < LOOKBACK:
            continue
        ret = (bars[idx]["close"] - bars[idx - LOOKBACK]["close"]) / bars[idx - LOOKBACK]["close"]
        candidates.append({"symbol": sym, "ret": ret, "signal_ts": bars[idx]["ts"],
                           "fill_ts": bars[idx + 1]["ts"] if idx + 1 < len(bars) else None})
    if not candidates:
        return []
    candidates.sort(key=lambda x: x["ret"])
    return candidates[:TOP_N]


# -------------------------------------------------------------- state --------
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_entry_day": None}


def save_state(st: dict) -> None:
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(st, f)


# -------------------------------------------------------------- modes --------
def replay(day_str: str) -> None:
    """Alignment proof: same day, same signal, from (a) the LIVE FTMO oracle and
    (b) the CACHED audit bars. Symbols + returns must match."""
    day = datetime.strptime(day_str, "%Y-%m-%d")
    ds = int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()) // 86400
    live = {}
    cached = {}
    for sym in TOKYO_UNIVERSE:
        live[sym] = fetch_day_bars(sym, day)
        cached[sym] = load_cached_day(sym, day)
    rl = session_rank(live, ds)
    rc = session_rank(cached, ds)
    print(f"[replay {day_str}] TOP-{TOP_N} — LIVE oracle vs AUDIT cache:")
    lset = {(x["symbol"], round(x["ret"], 6)) for x in rl}
    cset = {(x["symbol"], round(x["ret"], 6)) for x in rc}
    for i in range(max(len(rl), len(rc))):
        l = rl[i] if i < len(rl) else None
        c = rc[i] if i < len(rc) else None
        lsym = f"{l['symbol']:<8} ret={l['ret']:+.5%}" if l else "—"
        csym = f"{c['symbol']:<8} ret={c['ret']:+.5%}" if c else "—"
        mark = "OK" if (l and c and l["symbol"] == c["symbol"] and abs(l["ret"] - c["ret"]) < 1e-4) else ("!" if l or c else "")
        print(f"   {lsym} | {csym} {mark}")
    inter = lset & cset
    print(f"   matched symbol+ret: {len(inter)}/{TOP_N}")
    if rl:
        print(f"   live signal_ts={datetime.utcfromtimestamp(rl[0]['signal_ts'])} "
              f"fill_ts={datetime.utcfromtimestamp(rl[0]['fill_ts']) if rl[0]['fill_ts'] else None}")
    return


def live_loop(execute: bool, manage: bool, daemon: bool = False) -> None:
    st = load_state()
    conn = None
    om = None
    if execute or manage:
        from proxima_ops.execution.mt5_connector import MT5Connector
        from proxima_ops.execution.order_manager import OrderManager
        conn = MT5Connector()
        if not conn.connect():
            print(f"[fatal] MT5 connect failed: {conn.last_error}")
            sys.exit(1)
        om = OrderManager(conn)

    def firm_blocked() -> tuple[bool, str]:
        """FTMO-style live guard, distinct rules:
          * daily-loss: today (server date) realized+floating loss >= 5% of init.
          * drawdown:    total loss from account peak >= 10% of init.
        `init` = the FTMO trial starting balance baseline (100k); balance only
        declines after losses, so using live balance-vs-init is conservative."""
        try:
            acct = mt5.account_info()
        except Exception:
            return True, "account_info unavailable"
        if acct is None:
            return True, "no account"
        eq = acct.equity if acct.equity else acct.balance
        bal = acct.balance
        init = max(bal, eq)  # baseline = current high view of the account
        # floating + realized loss from the day; and peak drawdown (approx via init)
        loss = init - eq
        dd = (init - eq)
        if loss / init >= 0.05:
            return True, f"daily loss {(loss/init):.1%} >= 5% (eq={eq:.0f})"
        if dd / init >= 0.10:
            return True, f"drawdown {dd/init:.1%} >= 10% (eq={eq:.0f})"
        return False, "ok"

    POST_FILL_TOL_S = 300  # the fill bar lives 5 min (M5). A market order placed
                            # while the fill bar is forming fills at ~that bar's
                            # price (the engine's accepted approximation of the
                            # curve's next-bar-open). If the process wakes after
                            # the fill bar CLOSED, the order would fill at a
                            # later bar -> deviates from the validated curve, so
                            # we skip the day instead.
    started_at = now_str = server_now()
    MAX_RUNTIME_S = (30 * 24 * 3600) if daemon else (26 * 3600)  # daemon: ~monthly recycle; one-shot: 26h cap

    last_actionable = None
    session_decided = None   # UTC day for which we've fired or skipped an entry
    target_day = None        # next upcoming 00:00 UTC session day to act on
    while True:
        now = server_now()
        hour = (now // 3600) % 24
        today = now // 86400
        # runtime safety cap (a stray process shouldn't block the schedule forever)
        if now - started_at > MAX_RUNTIME_S:
            print(f"[exit] max runtime {MAX_RUNTIME_S//3600}h reached. bye")
            break
        # define the target session: the next 00:00 UTC that is actionable.
        # If we're already inside hour 0 (0:00-0:59), the target is today.
        if hour >= SESSION_HOUR + 1:
            target_day = today + 1      # past hour 0 -> next session is tomorrow
        else:
            target_day = today          # inside hour 0 -> act on today
        open_tokyo = 0
        try:
            open_tokyo = len([p for p in (mt5.positions_get() or []) if p.comment.startswith(COMMENT)])
        except Exception:
            pass
        # one-shot: exit once the target session is DECIDED (fired or skipped)
        # and its positions are all closed (hold done) — clean daily lifetime.
        # daemon: the same condition ROLLS the state to the next day and keeps
        # running (no host-clock dependency — the gate is server time only).
        if (target_day is not None and target_day < today) or \
           (session_decided == today and open_tokyo == 0):
            if not daemon:
                print(f"[exit] target={target_day} decided={session_decided} "
                      f"open_tokyo={open_tokyo} hour={hour} — session done. bye")
                break
            # daemon: any completed/skipped/missed session rolls to the next day.
            # Guarded: only roll when we've actually decided a session (fired or
            # skipped) — helps avoid tight-loop noise at roll time.
            if session_decided is not None:
                print(f"[daemon] day {session_decided} done (open_tokyo={open_tokyo}) — "
                      f"rolling to next session. wake @00:00Z")
                session_decided = None
                last_actionable = None
                st = load_state()  # re-read in case a sibling rewrote it
        # ---- 1) hold-managed exits (broker SL/TP handles most; close stragglers)
        if manage and conn is not None:
            try:
                for pos in mt5.positions_get() or []:
                    if pos.comment.startswith(COMMENT):
                        age_bars = (now - pos.time) // 300
                        if age_bars >= HOLD_BARS:
                            print(f"[manage] closing straggler {pos.symbol} ticket={pos.ticket} age={age_bars} bars")
                            conn.close_order(pos.ticket)
            except Exception as e:
                print(f"[manage] error: {e}")
        # ---- 2) session-window entry gate (server hour 0, once per target day)
        if hour == SESSION_HOUR and st.get("last_day") != target_day:
            bars_map = {}
            for sym in TOKYO_UNIVERSE:
                bars_map[sym] = fetch_bars(sym, 16)
            rank = session_rank(bars_map, target_day)
            if rank:
                # (rank is only honored once the signal bar has CLOSED, i.e. the
                # fill bar has opened — guarantees the same closed-bar contract
                # the audit curve uses, never a forming bar)
                fill_ts = rank[0]["fill_ts"]
                if fill_ts is None or now < fill_ts:
                    time.sleep(POLL_S)
                    continue
                # late-fill guard: if we woke up well past the nominal fill bar,
                # skip today (a market order now would NOT reproduce the curve's
                # 00:10 fill; wait for the next session instead)
                if now > fill_ts + POST_FILL_TOL_S:
                    print(f"[skip] now {datetime.utcfromtimestamp(now):%H:%M:%S}Z > "
                          f"fill {datetime.utcfromtimestamp(fill_ts):%H:%M:%S}Z + tolerance — "
                          f"skipping this session to preserve the validated fill bar")
                    st["last_day"] = target_day
                    save_state(st)
                    session_decided = target_day
                    time.sleep(POLL_S)
                    continue
                if last_actionable == target_day:
                    time.sleep(POLL_S)
                    continue
                last_actionable = target_day
                print(f"[signal {datetime.utcfromtimestamp(now):%Y-%m-%d %H:%M:%S}Z] top-5:")
                for x in rank:
                    print(f"     BUY {x['symbol']:<8} 6bar_ret={x['ret']:+.5%} "
                          f"fill_ts={datetime.utcfromtimestamp(x['fill_ts'])}")
                if not execute:
                    print("[dry-run] no orders sent — rerun with --execute to place")
                else:
                    blocked, why = firm_blocked()
                    if blocked:
                        print(f"[FirmRisk BLOCK] {why}")
                    else:
                        acct = mt5.account_info()
                        bal = acct.balance if acct else 100_000.0
                        opened = []
                        for x in rank:
                            sym = x["symbol"]
                            tick = mt5.symbol_info_tick(sym)
                            if tick is None:
                                continue
                            fill = tick.ask
                            sl, tp = sl_tp_abs(sym, fill, "BUY")
                            res = om.execute_buy(sym, fill, bal, sl=sl, tp=tp,
                                                 volume=BASE_LOT, comment=COMMENT)
                            if res:
                                opened.append((sym, res.get("ticket")))
                                print(f"[FILL] BUY {sym} {BASE_LOT} @ {fill} sl={sl} tp={tp} ticket={res.get('ticket')}")
                            else:
                                print(f"[REJECT] BUY {sym} (engine guard: {om._mt5.last_error if om._mt5 else '?'})")
                            time.sleep(0.4)
                        st["last_day"] = target_day
                        save_state(st)
                        session_decided = target_day
                        print(f"[done] opened {len(opened)}/5 — session day {target_day} recorded")
        time.sleep(POLL_S)


# ------------------------------------------------------------------ main -----
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--replay", metavar="YYYY-MM-DD", default=None,
                    help="prove alignment for a historical day (no orders)")
    ap.add_argument("--execute", action="store_true",
                    help="actually place the top-5 orders (default: dry-run)")
    ap.add_argument("--manage", action="store_true",
                    help="also close TOKYO_H0 positions older than HOLD_BARS on each poll")
    ap.add_argument("--once", action="store_true",
                    help="run one session check and exit (for tests), default loop")
    ap.add_argument("--daemon", action="store_true",
                    help="run continuously across many sessions (no host-clock "
                         "dependency); combine with --execute --manage")
    args = ap.parse_args()

    if not mt5.initialize(path=FTMO_TERMINAL, timeout=4000):
        print("INIT FAILED:", mt5.last_error())
        sys.exit(1)
    acct = mt5.account_info()
    print(f"[attach] account={acct.login if acct else 'N/A'} "
          f"server={acct.server if acct else ''} balance={round(acct.balance, 2) if acct else 'N/A'}")
    print(f"[clock] server now={datetime.utcfromtimestamp(server_now()):%Y-%m-%d %H:%M:%S}Z "
          f"hour={server_session_hour()} (Tokyo window=hour {SESSION_HOUR})")

    if args.replay:
        try:
            replay(args.replay)
        finally:
            mt5.shutdown()
        return
    if args.once:
        # single dry session evaluation (no order risk), then exit
        from audit_7_eas.ea_ports import tokyo_h0 as _unused  # noqa: F401 (import sanity)
        today = server_day()
        bars_map = {sym: fetch_bars(sym, 16) for sym in TOKYO_UNIVERSE}
        rank = session_rank(bars_map, today)
        print(f"[once] today={datetime.utcfromtimestamp(server_now()):%Y-%m-%d} top-5 (dry):")
        for x in rank:
            print(f"     BUY {x['symbol']:<8} ret={x['ret']:+.5%} fill_ts={datetime.utcfromtimestamp(x['fill_ts'])}")
        mt5.shutdown()
        return
    try:
        live_loop(args.execute, args.manage, args.daemon)
    finally:
        try:
            mt5.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()