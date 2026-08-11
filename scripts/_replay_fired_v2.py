"""Replay book-v2 live fires on 2026-08-11 — verify the EXACT live trades
against the deployed model (run_core_book_live.session_rank as the daemon
evaluated): rank symbol-set vs daemon-printed rank + journal fills,
model_open vs actual fill, exit walk (stop-first SL/TP, else hold_bars
close) vs live exits, net P&L (engine trade_to_usd @ live commission 2.5 +
TYPICAL/MEASURED spreads) vs live journaled net.

V2 fixes:
- journal entries filtered to TODAY's server day (yesterday's usfade rows
  were polluting the live comparison)
- walk reports DATA-HOLE when the fetched terminal history misses a bar
  mid-hold (terminal history for some crosses has vendor gaps — e.g.
  14:35-14:40 on 2026-08-11) instead of silently skipping
- daemon-printed [signal] top lists are parsed from the daemon log and
  compared against the clean-window replay rank (ragged-pool divergence)
"""
import sys, os, datetime, json, re
sys.path.insert(0, "/home/ubuntu/proxima_x/scripts")
sys.path.insert(0, "/home/ubuntu/proxima_x/proxima_ops/backtest")
import MetaTrader5 as mt5
import run_core_book_live as R
from pnl import trade_to_usd

if not mt5.initialize(path=os.environ["MT5_PATH"], timeout=4000):
    print("MT5 init FAILED"); sys.exit(1)
print("mt5 ok, terminal:", (mt5.terminal_info().name if mt5.terminal_info() else "?"))

for s in ("XAUUSD", "XAGUSD"):
    mt5.symbol_select(s, True)

NB = 2200
SPREAD_TYP = {"EURUSD": 0.8, "USDJPY": 1.2, "GBPUSD": 1.5, "AUDUSD": 1.1,
              "EURJPY": 2.2, "GBPJPY": 3.0, "EURAUD": 2.8, "EURNZD": 3.4,
              "GBPAUD": 3.6, "GBPNZD": 4.4, "GBPCAD": 3.2, "AUDNZD": 2.6,
              "USDCAD": 1.8, "NZDUSD": 1.4, "EURGBP": 1.2, "EURCHF": 1.8,
              "USDCHF": 1.4, "AUDJPY": 1.8}

# ---- fetch bars (18 FX + gold) ----
syms = list(R.UNIVERSE) + ["XAUUSD", "XAGUSD"]
bars = {s: R.fetch_bars(s, NB) for s in syms}
latest = [b[-1]["ts"] for b in bars.values() if b]
day = max(latest) // 86400 if latest else None
DAY0 = day * 86400
print("feed latest bar ts:", datetime.datetime.utcfromtimestamp(max(latest)),
      "| day:", day)

# ---- live journal, filtered to today ----
journal = []
try:
    for line in open(R.JOURNAL):
        r = json.loads(line)
        if r.get("ts", 0) >= DAY0 - 300:   # today only (server day)
            journal.append(r)
except Exception as e:
    print("journal read fail:", e)


def j_entries(name):
    return [r for r in journal if r.get("kind") == "entry" and r.get("strategy") == name]


def j_close_for_ticket(t):
    for r in journal:
        if r.get("kind") == "close" and r.get("ticket") == t:
            return r
    return None


# ---- daemon-printed ranks (ragged-pool ground truth at eval instant) ----
daemon_rank = {}
try:
    cur = None
    for line in open(os.path.join(R.ROOT, "logs", "core_book_daemon.log"), errors="ignore"):
        m = re.search(r"\[signal (\w+) 2026-08-11", line)
        if m:
            cur = m.group(1); daemon_rank[cur] = []
            continue
        m2 = re.match(r"\s+([A-Z]{6})\s+ret=", line)
        if m2 and cur is not None:
            daemon_rank[cur].append(m2.group(1))
        if re.search(r"\[(FILL|done|skip) ", line):
            cur = None
except Exception as e:
    print("daemon log parse fail:", e)


def win(sym, ts_end, n):
    bs = [x for x in bars[sym] if x["ts"] <= ts_end + 300]
    return bs[-n:] if len(bs) >= n else bs


def rank_at(cfg, ts_end, uni):
    wm = {s: win(s, ts_end, cfg["lookback"] + 4) for s in uni}
    return R.session_rank(wm, day, cfg)


SPREAD_ROWS = []
try:
    for line in open(R.SPREAD_LOG):
        p = line.strip().split(",")
        if len(p) == 19:
            SPREAD_ROWS.append({"ts": int(p[0]),
                                "pips": dict(zip(R.UNIVERSE, map(float, p[1:])))})
except Exception as e:
    print("spread log read fail:", e)


def spread_at(sym, fill_ts):
    rows = [r for r in SPREAD_ROWS if r["ts"] <= fill_ts]
    if rows and rows[-1]["pips"].get(sym) is not None:
        return rows[-1]["pips"][sym]
    return None


def walk(sym, fill_ts, entry, sl, tp, hold_bars):
    by_ts = {b["ts"]: b for b in bars[sym]}
    ts = fill_ts
    for step in range(hold_bars + 2):
        b = by_ts.get(ts)
        if b is None:
            hole = datetime.datetime.utcfromtimestamp(ts).strftime("%H:%M")
            return None, f"DATA-HOLE@{hole}"
        if b["low"] <= sl:
            return sl, "SL"
        if b["high"] >= tp:
            return tp, "TP"
        if step >= hold_bars:
            return b["close"], "HOLD"
        ts += 300
    return None, "exhausted"


# tokyo cfg (disabled in STRATS — informational only)
TOKYO_CFG = {"sessions": [0], "lookback": 6, "top_n": 3, "hold_bars": 12,
             "lot": 0.52, "comment": "CORE_TOKYO_25",
             "sl_tp": {"JPY": (0.40, 0.60), "else": (0.0040, 0.0060)}}
EVAL = {
    "cascade": DAY0 + 2 * 3600 + 300,
    "london":  DAY0 + 7 * 3600 + 300,
    "usfade":  DAY0 + 14 * 3600 + 300,
    "gold_s3": DAY0 + 16 * 3600 + 300 + 300,   # log: signal 16:25, fill 16:30
    "tokyo":   DAY0 + 0 * 3600 + 300,
}


def replay_one(name, cfg, ts_end, uni, live_entries, informational=False):
    tag = "INFORMATIONAL" if informational else "VERIFIED"
    print(f"\n=== [{tag}] {name} (eval {datetime.datetime.utcfromtimestamp(ts_end)}) ===")
    rank = rank_at(cfg, ts_end, uni)
    if not rank:
        print(f"[{name}] NO RANK — no candidates")
        return 0.0
    ranked_syms = [x["symbol"] for x in rank[: cfg["top_n"]]]
    live_syms = [e["symbol"] for e in live_entries]
    dr = daemon_rank.get(name, [])
    print(f"replay rank top-{cfg['top_n']}: {ranked_syms}")
    print(f"daemon rank top-{cfg['top_n']}: {dr[:cfg['top_n']]}")
    print(f"LIVE filled:        {live_syms}")
    print(f"replay==LIVE set: {'YES' if sorted(ranked_syms) == sorted(live_syms) else '*** NO ***'}   "
          f"daemon==LIVE set: {'YES' if sorted(dr[:cfg['top_n']]) == sorted(live_syms) else 'NO'}")
    tot = 0.0
    for x in rank[: cfg["top_n"]]:
        sym = x["symbol"]
        entry = x["model_open"]
        sl, tp = R.sl_tp_abs(sym, entry, "BUY", cfg)
        exit_p, reason = walk(sym, x["fill_ts"], entry, sl, tp, cfg["hold_bars"])
        live_e = next((e for e in live_entries if e["symbol"] == sym), None)
        if exit_p is None:
            print(f"  {sym:<8} {reason} — walk impossible (terminal history gap), model fill {entry:.5f}")
            continue
        sp = spread_at(sym, x["fill_ts"])
        sp_t = SPREAD_TYP.get(sym)
        t = {"symbol": sym, "pnl_pts": exit_p - entry, "entry": entry,
             "entry_ts": x["fill_ts"],
             "exit_ts": x["fill_ts"] + cfg["hold_bars"] * 300,
             "side": "BUY", "reason": reason}
        u = trade_to_usd(t, cfg["lot"], None, 2.5, {sym: sp} if sp else None)
        u_t = trade_to_usd(t, cfg["lot"], None, 2.5, {sym: sp_t})
        u3 = trade_to_usd(t, cfg["lot"], None, 3.0, {sym: sp_t})  # engine default
        tot += u_t["net"]
        live_txt = "-"
        if live_e:
            lc = j_close_for_ticket(live_e.get("ticket"))
            live_net = None
            live_exit = "?"
            if lc is not None:
                live_net = lc["pnl"] + 2 * lc.get("commission", 0.0)
                live_exit = lc.get("exit")
            d = abs(live_e.get("actual", 0) - entry)
            tol = 0.015 if "JPY" in sym else 0.00015
            fill_ok = "OK" if d < tol else "DIFF"
            live_txt = (f"live fill {live_e.get('actual')} vs model {entry} [{fill_ok}] | "
                        f"live exit {live_exit} vs walk {exit_p} ({reason}) | "
                        f"live net {live_net if live_net is not None else '?'}")
        sp_txt = f"{sp}" if sp is not None else "TYP"
        print(f"  {sym:<8} {reason:<9} modelEntry={entry:.5f} modelExit={exit_p:.5f} "
              f"netTYP(2.5)={u_t['net']:+8.2f} netENG(3.0)={u3['net']:+8.2f} "
              f"[spreadMEAS={sp_txt} netMEAS={u['net']:+.2f}]")
        print(f"            {live_txt}")
    print(f"[{name}] model net TYP @2.5 comm total: {tot:+.2f}")
    return tot


grand = 0.0
for name, cfg in R.STRATS.items():
    uni = cfg.get("universe", R.UNIVERSE)
    ts_end = EVAL.get(name)
    if ts_end is None:
        print(f"[{name}] no EVAL entry — skipped (not fired today)")
        continue
    grand += replay_one(name, cfg, ts_end, uni, j_entries(name),
                        informational=(name == "gold_s3"))

# tokyo informational (disabled leg — confirm model also takes the SL hit)
grand += replay_one("tokyo", TOKYO_CFG, EVAL["tokyo"], R.UNIVERSE,
                    j_entries("tokyo"), informational=True)

print(f"\n=== MODEL NET TOTAL (TYPICAL spreads, comm 2.5): {grand:+.2f} ===")
