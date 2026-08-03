"""Proxima V2 — Impulse Fade Deployment.
Detection: raw price impulse, deque sliding window. Fixed 30s hold.
Includes per-tick CSV logger + live stats vs backtest validation.
"""
import sys, os, time, json, threading, csv
from collections import deque
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from paper_trade.core.feed import Feed
from paper_trade.core.executor import Executor
from paper_trade.core.risk import Risk
from paper_trade.core import registry as acct_registry

STRATEGY_NAME = "impulse_fade_v2"
MAGIC = 202408

CONFIG = {
    "pairs": ["EURUSD"],
    "lot_size": 2.0,
    "magic": MAGIC,
    "max_concurrent": 1,
    "max_spread_mult": 2.0,
    "stop_loss_pips": 10,
    "session_start": 14,
    "session_end": 19,
}
FN_TERMINAL = r"C:\Program Files\FundedNext MT5 Terminal\terminal64.exe"
LOOP_INTERVAL = 0.1
HEARTBEAT_INTERVAL = 30
REPORT_INTERVAL = 300

# Backtest reference stats (FundedNext ticks Jun-Jul 2026, 14-19 UTC filter, 10p stop, 2.0 lots)
BT_REF = {
    "EURUSD": {
        "config": "5p/20s hold=30s stop=10p hours=14-19UTC",
        "trades_20d": 765, "trades_per_day": 38.2, "wr_pct": 61.3,
        "avg_pips": 1.07, "gross_pips": 819,
        "max_consec_loss": 0, "max_dd_pips": 0,
        "avg_spread": 0.00008,
    },
}

class TickLogger:
    """Per-tick CSV + periodic validation report."""
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        self.fh = open(os.path.join(log_dir, "ticks.csv"), "w", newline="")
        self.w = csv.writer(self.fh)
        self.w.writerow(["ts_msc","pair","bid","ask","mid","spread",
                         "ws_idx","min_q_front","max_q_front","hp","lp",
                         "event","ev_dir","ev_extreme"])
        self.report_fh = open(os.path.join(log_dir, "stats.json"), "a")
        self._flush_counter = 0

    def write_tick(self, ts_msc, pair, bid, ask, mid, spread,
                   ws_idx, min_v, max_v, hp, lp,
                   event, ev_dir, ev_extreme):
        row = [f"{ts_msc:.0f}", pair, f"{bid:.5f}", f"{ask:.5f}", f"{mid:.5f}",
               f"{spread:.6f}", ws_idx, f"{min_v:.5f}", f"{max_v:.5f}",
               f"{hp:.6f}", f"{lp:.6f}", 1 if event else 0, ev_dir or 0,
               f"{ev_extreme:.5f}" if ev_extreme else ""]
        self.w.writerow(row)
        self._flush_counter += 1
        if self._flush_counter % 500 == 0:
            self.fh.flush()

    def write_report(self, report_dict):
        self.report_fh.write(json.dumps({**report_dict, "ts": time.time()}) + "\n")
        self.report_fh.flush()

    def close(self):
        self.fh.flush(); self.fh.close()
        self.report_fh.close()

class LiveStats:
    """Running tick + trade statistics per pair."""
    def __init__(self):
        self.pairs = {}

    def ensure(self, pair):
        if pair not in self.pairs:
            self.pairs[pair] = {
                "ticks": 0, "events_raw": 0, "events_traded": 0,
                "trades": 0, "wins": 0, "losses": 0,
                "streak": 0, "best_streak": 0, "worst_streak": 0,
                "pnl_pips": 0.0, "pnl_usd": 0.0,
                "spreads": [], "hp_vals": [], "lp_vals": [],
                "start_ts": time.time(),
            }

    def incr(self, pair, field, val=1):
        self.ensure(pair); self.pairs[pair][field] += val

    def append(self, pair, field, val):
        self.ensure(pair)
        lst = self.pairs[pair].get(field)
        if lst is not None:
            lst.append(val)
            # Trim to last 10000
            if len(lst) > 10000:
                self.pairs[pair][field] = lst[-10000:]

    def record_trade_close(self, pair, pnl_pips, pnl_usd):
        s = self.pairs[pair]
        s["trades"] += 1; s["pnl_pips"] += pnl_pips; s["pnl_usd"] += pnl_usd
        if pnl_pips > 0:
            s["wins"] += 1; s["streak"] = max(s["streak"] + 1, 1)
            s["best_streak"] = max(s["best_streak"], s["streak"])
            s["worst_streak"] = 0
        else:
            s["losses"] += 1; s["streak"] = min(s["streak"] - 1, -1)
            s["worst_streak"] = min(s["worst_streak"], s["streak"])
            s["best_streak"] = 0

    def snapshot(self):
        return {p: {k: v for k, v in s.items() if k != "spreads"}
                for p, s in self.pairs.items()}

    def validate_against_backtest(self, pair):
        """Compare live stats to backtest reference."""
        s = self.pairs.get(pair); ref = BT_REF.get(pair)
        if not s or not ref: return {}
        elapsed_h = (time.time() - s["start_ts"]) / 3600
        if elapsed_h < 1: return {}
        trades_per_day = s["trades"] / elapsed_h * 24 if elapsed_h > 0 else 0
        wr = s["wins"] / max(s["trades"], 1) * 100
        return {
            "pair": pair,
            "live_trades_pd": round(trades_per_day, 1),
            "bt_trades_pd": ref["trades_per_day"],
            "trades_pct_of_bt": f"{trades_per_day/ref['trades_per_day']*100:.0f}%",
            "live_wr": f"{wr:.1f}%",
            "bt_wr": f"{ref['wr_pct']:.1f}%",
            "wr_delta": f"{wr - ref['wr_pct']:+.1f}pp",
            "live_avg_pip": round(s["pnl_pips"] / max(s["trades"], 1), 2),
            "bt_avg_pip": ref["avg_pips"],
        }

class LiveDetector:
    """Sliding window impulse detection on live tick stream.
    Returns dict with detection state info for logging."""
    def __init__(self, pair, pip, thresh_pips, window_sec=20, hold_sec=30):
        self.pair = pair; self.pip = pip
        self.thresh = thresh_pips * pip; self.window_sec = window_sec
        self.hold_sec = hold_sec
        self.ticks = []
        self.min_q = deque(); self.max_q = deque()
        self.ws = 0; self.last_ws_traded = -1; self.last_ts = 0.0

    def add_tick(self, bid, ask, ts_msc):
        ts = ts_msc / 1000.0
        if ts <= self.last_ts:
            return "dup", {}
        self.last_ts = ts
        mid = (bid + ask) / 2.0
        idx = len(self.ticks)
        self.ticks.append((ts, mid))
        v = float(mid)
        while self.min_q and self.min_q[-1][0] >= v: self.min_q.pop()
        while self.max_q and self.max_q[-1][0] <= v: self.max_q.pop()
        self.min_q.append((v, idx)); self.max_q.append((v, idx))
        while idx > self.ws and self.ticks[idx][0] - self.ticks[self.ws][0] > self.window_sec:
            if self.min_q and self.min_q[0][1] == self.ws: self.min_q.popleft()
            if self.max_q and self.max_q[0][1] == self.ws: self.max_q.popleft()
            self.ws += 1
        info = {"ws_idx": self.ws, "min_q_front": self.min_q[0][0] if self.min_q else v,
                "max_q_front": self.max_q[0][0] if self.max_q else v,
                "hp": 0.0, "lp": 0.0, "event": False, "ev_dir": None, "ev_extreme": None}
        if idx > self.ws and self.ticks[idx][0] - self.ticks[self.ws][0] <= self.window_sec:
            wp = self.ticks[self.ws][1]
            hp = float(self.max_q[0][0] - wp); lp = float(wp - self.min_q[0][0])
            info["hp"] = hp; info["lp"] = lp
            if hp >= self.thresh or lp >= self.thresh:
                if self.ws <= self.last_ws_traded:
                    return "dup_ws", info
                self.last_ws_traded = self.ws
                if hp >= lp:
                    ext_idx = self.max_q[0][1]
                    info.update({"event": True, "ev_dir": 1, "ev_extreme": self.ticks[ext_idx][1]})
                    return "buy", info
                else:
                    ext_idx = self.min_q[0][1]
                    info.update({"event": True, "ev_dir": -1, "ev_extreme": self.ticks[ext_idx][1]})
                    return "sell", info
        return "none", info

def _heartbeat_loop(login, name, stop_event):
    while not stop_event.is_set():
        time.sleep(HEARTBEAT_INTERVAL)
        acct_registry.heartbeat(login, name)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    lock_path = os.path.join(base_dir, "live", f"process_{STRATEGY_NAME}.lock")
    log_dir = os.path.join(base_dir, "live", f"logs_{STRATEGY_NAME}")
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    # Single-process lock
    if os.path.exists(lock_path):
        with open(lock_path) as f:
            pid = int(f.read().strip())
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            h = kernel32.OpenProcess(0x400, False, pid)
            if h:
                kernel32.CloseHandle(h)
                print(f"Process already running (pid={pid}). Exiting.", file=sys.stderr)
                sys.exit(1)
        else:
            try: os.kill(pid, 0)
            except OSError: pass
            else:
                print(f"Process already running (pid={pid}). Exiting.", file=sys.stderr)
                sys.exit(1)
    with open(lock_path, "w") as f:
        f.write(str(os.getpid()))

    cfg = CONFIG
    feed = Feed(mode="live", pairs=cfg["pairs"], mt5_path=FN_TERMINAL).connect()
    mt5_login = feed.mt5.account_info().login
    print(f"MT5 account: {mt5_login}", file=sys.stderr)

    ok, reason = acct_registry.claim(mt5_login, STRATEGY_NAME)
    if not ok:
        print(f"ACCOUNT CONFLICT: {reason}", file=sys.stderr)
        os.remove(lock_path); sys.exit(1)

    stop_hb = threading.Event()
    hb_thread = threading.Thread(target=_heartbeat_loop,
        args=(mt5_login, STRATEGY_NAME, stop_hb), daemon=True)
    hb_thread.start()

    risk = Risk(cfg)
    exec_ = Executor(feed, None, magic=cfg["magic"])
    tick_log = TickLogger(log_dir)
    live_stats = LiveStats()
    last_report_ts = 0

    STOP_PIPS = cfg.get("stop_loss_pips", 5)
    detectors = {
        "EURUSD": LiveDetector("EURUSD", 0.0001, 5, 20, 30),
    }
    last_tick = {}  # pair -> (bid, ask)
    positions = {}

    def close_position(pair, stop_hit=False):
        pos = positions.get(pair)
        if not pos: return
        data = feed.current_bar()
        if pair not in data: return
        if stop_hit:
            exit_price = pos["stop_price"]
        else:
            exit_price = data[pair]["bid"] if pos["direction"] == 1 else data[pair]["ask"]
        result = exec_.close_position(
            {"pair": pair, "direction": pos["direction"],
             "entry_price": pos["entry_price"], "lot_size": pos["lot"]},
            exit_price=exit_price, exit_time=int(time.time()))
        if result:
            pnl_usd = result.get("gross_pnl", 0)
            entry_px = pos["entry_price"]
            pip = 0.0001
            pnl_pips = (exit_price - entry_px) / pip * pos["direction"]
            live_stats.record_trade_close(pair, pnl_pips, pnl_usd)
            label = "STOP" if stop_hit else "CLOSE"
            print(f"{label} {pair} pips={pnl_pips:+.1f} ${pnl_usd:.2f}", file=sys.stderr)
        del positions[pair]

    try:
        while True:
            now = time.gmtime()
            current_ts = int(time.time())
            data = feed.current_bar()
            if not data:
                time.sleep(LOOP_INTERVAL); continue

            # Check expired positions + stop-loss
            for pair in list(positions.keys()):
                pos = positions[pair]
                if current_ts - pos["entry_time"] >= detectors[pair].hold_sec:
                    close_position(pair)
                elif STOP_PIPS > 0:
                    p = data.get(pair, {})
                    bid, ask = p.get("bid", 0), p.get("ask", 0)
                    if bid > 0 and ask > 0:
                        if pos["direction"] == 1 and bid <= pos["stop_price"]:
                            close_position(pair, stop_hit=True)
                        elif pos["direction"] == -1 and ask >= pos["stop_price"]:
                            close_position(pair, stop_hit=True)

            # Process ticks + detect
            for pair in cfg["pairs"]:
                if pair not in data: continue
                p = data[pair]
                bid, ask = p.get("bid", 0), p.get("ask", 0)
                if bid <= 0 or ask <= 0: continue
                dt = detectors[pair]; pip = dt.pip
                spread = ask - bid
                risk.update_spread_baseline(pair, spread)
                normal_spread = risk.normal_spread(pair) or 0.00008

                # Only feed detector on actual price change
                prev = last_tick.get(pair)
                if prev and prev[0] == bid and prev[1] == ask:
                    continue
                last_tick[pair] = (bid, ask)

                ts_msc = int(time.time() * 1000)
                live_stats.ensure(pair)
                live_stats.incr(pair, "ticks")
                live_stats.append(pair, "spreads", spread)

                # Run detection
                status, info = dt.add_tick(bid, ask, ts_msc)
                if info.get("hp", 0) > 0:
                    live_stats.append(pair, "hp_vals", info["hp"])
                    live_stats.append(pair, "lp_vals", info["lp"])

                # Log every tick
                tick_log.write_tick(ts_msc, pair, bid, ask, (bid+ask)/2, spread,
                    info.get("ws_idx", 0), info.get("min_q_front", 0), info.get("max_q_front", 0),
                    info.get("hp", 0), info.get("lp", 0),
                    info.get("event", False), info.get("ev_dir"), info.get("ev_extreme"))

                # Process event (if any)
                if info.get("event"):
                    live_stats.incr(pair, "events_raw")
                    direction = info["ev_dir"]

                    if spread > normal_spread * CONFIG["max_spread_mult"]: continue
                    if pair in positions: continue

                    # Submit trade
                    fill = exec_.submit_market(pair, direction, cfg["lot_size"],
                        timestamp=current_ts, signal_meta={"strategy": STRATEGY_NAME})
                    if fill:
                        live_stats.incr(pair, "events_traded")
                        entry_px = fill["entry_price"]
                        stop_px = entry_px - STOP_PIPS * pip if direction == 1 else entry_px + STOP_PIPS * pip
                        positions[pair] = {
                            "ticket": fill.get("ticket", 0),
                            "direction": direction,
                            "entry_time": current_ts,
                            "entry_price": entry_px,
                            "stop_price": stop_px,
                            "lot": cfg["lot_size"],
                        }
                        print(f"ENTRY {pair} {'BUY' if direction>0 else 'SELL'} "
                              f"@{entry_px:.5f} stop={stop_px:.5f} extreme={info['ev_extreme']/pip:.1f}p",
                              file=sys.stderr)

            # Risk: market hours
            ok_hours, reason_h = risk.check_market_hours(now)
            if not ok_hours:
                for pair in list(positions.keys()): close_position(pair)
                time.sleep(60); continue

            # Periodic validation report
            if current_ts - last_report_ts >= REPORT_INTERVAL:
                last_report_ts = current_ts
                pair = "EURUSD"
                s = live_stats.pairs.get(pair, {})
                elapsed_h = (time.time() - s.get("start_ts", time.time())) / 3600
                tpd = s.get("trades", 0) / max(elapsed_h, 0.1) * 24 if elapsed_h > 0 else 0
                wr = s.get("wins", 0) / max(s.get("trades", 0), 1) * 100
                v = live_stats.validate_against_backtest(pair)
                msg = (f"[{datetime.utcnow().strftime('%H:%M:%SZ')}] EURUSD: "
                       f"ticks={s.get('ticks',0):,} events={s.get('events_traded',0)} "
                       f"trades={s.get('trades',0)} WR={wr:.1f}% ({tpd:.1f}/d) "
                       f"PnL={s.get('pnl_usd',0):+.2f} (pips={s.get('pnl_pips',0):+.1f})")
                if v:
                    msg += f" | vs BT: {v['trades_pct_of_bt']} trades/day, WR delta {v['wr_delta']}, avg_pip {v['live_avg_pip']}"
                print(msg, file=sys.stderr)
                tick_log.write_report({"snapshot": live_stats.snapshot(),
                    "validation": live_stats.validate_against_backtest(pair)})

            time.sleep(LOOP_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        stop_hb.set()
        for pair in list(positions.keys()): close_position(pair)
        tick_log.close()
        feed.close()
        acct_registry.release(mt5_login, STRATEGY_NAME)
        if os.path.exists(lock_path): os.remove(lock_path)
        print("Shutdown complete.", file=sys.stderr)

if __name__ == "__main__":
    main()
