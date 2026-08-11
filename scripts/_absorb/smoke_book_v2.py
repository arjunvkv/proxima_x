"""Offline smoke test for the book v2 worker changes (no MT5 needed).

Validates: STRATS = {cascade, london, usfade, gold_s3} (no tokyo),
gold sl_tp branch, per-leg universe enforcement in session_rank.
"""
import sys, types, os

# ---- stub MetaTrader5 before importing the worker ----
mt5 = types.ModuleType("MetaTrader5")
mt5.symbol_info_tick = lambda sym: None
mt5.account_info = lambda: None
mt5.positions_get = lambda: []
mt5.history_deals_get = lambda *a, **k: []
mt5.order_send = lambda req: None
for n in ("TRADE_ACTION_DEAL", "ORDER_TYPE_BUY", "ORDER_FILLING_IOC",
          "TRADE_RETCODE_DONE"):
    setattr(mt5, n, 1)
sys.modules["MetaTrader5"] = mt5

os.environ["PROXIMA_ROOT"] = r"C:\Trading\Proxima_X"
sys.path.insert(0, r"C:\Trading\Proxima_X\scripts")
import run_core_book_live as R

ok = True
def check(label, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + label)
    ok = ok and cond

# 1. STRATS shape
names = list(R.STRATS)
check(f"STRATS = {names} (tokyo removed, gold_s3 added)",
      names == ["cascade", "london", "usfade", "gold_s3"])
g = R.STRATS["gold_s3"]
check(f"gold_s3 sessions=0..23 ({len(g['sessions'])}h)", g["sessions"] == list(range(24)))
check(f"gold_s3 lb50/top3/hold12/lot0.15 comment=CORE_GOLD_25",
      (g["lookback"], g["top_n"], g["hold_bars"], g["lot"], g["comment"]) == (50, 3, 12, 0.15, "CORE_GOLD_25"))
check(f"gold_s3 universe={g['universe']}", g["universe"] == ["XAUUSD", "XAGUSD"])

# 2. sl_tp_abs branches
sl, tp = R.sl_tp_abs("XAUUSD", 3300.0, "BUY", g)
check(f"gold sl_tp: XAUUSD BUY @3300 -> ({sl},{tp})", abs(sl-3200.0) < 1e-9 and abs(tp-3400.0) < 1e-9)
sl, tp = R.sl_tp_abs("XAGUSD", 30.0, "BUY", g)
check(f"gold sl_tp: XAGUSD -> ({sl},{tp})", abs(sl-(-70.0)) < 1e-9)
c = R.STRATS["cascade"]
sl, tp = R.sl_tp_abs("EURUSD", 1.10, "BUY", c)
check(f"cascade EURUSD unchanged ({sl},{tp})", abs(sl-1.096) < 1e-9 and abs(tp-1.106) < 1e-9)
sl, tp = R.sl_tp_abs("USDJPY", 150.0, "BUY", c)
check(f"cascade USDJPY unchanged ({sl},{tp})", abs(sl-149.6) < 1e-9 and abs(tp-150.6) < 1e-9)

# 3. session_rank honors per-leg universe (gold pool = XAU/XAG only)
import time as _t
def mk_bars(price0, n=60, step=0.001):
    out = []
    day0 = 20670 * 86400
    for i in range(n):
        out.append({"ts": day0 + i * 300, "open": price0 + i*step,
                    "close": price0 + i*step + 0.0002})
    return out
# gold exhaustion: XAU dipped then recovered (negative 50-bar ret), XAG flat-positive
bars_xau = mk_bars(3300.0, 60, -0.05)   # 50-bar ret strongly negative -> ranks
bars_xag = mk_bars(30.0, 60, 0.001)     # slightly positive ret
bars_eur = mk_bars(1.10, 60, 0.0001)    # would rank if wrongly included
pool = {"XAUUSD": bars_xau, "XAGUSD": bars_xag}
rank = R.session_rank(pool, 20670, g)
syms = sorted(x["symbol"] for x in rank)
check(f"gold rank pool only metals: {syms}", syms == ["XAGUSD", "XAUUSD"] and len(rank) == 2)
# FX leg must NOT see gold even if gold bars were passed
fxpool = dict(pool); fxpool["EURUSD"] = bars_eur
rank_fx = R.session_rank(fxpool, 20670, c)
check(f"cascade rank excludes gold: {sorted(x['symbol'] for x in rank_fx)}",
      all("XAU" not in x["symbol"] and "XAG" not in x["symbol"] for x in rank_fx))

print("\n" + ("ALL SMOKE TESTS PASS" if ok else "SMOKE TESTS FAILED"))
sys.exit(0 if ok else 1)
