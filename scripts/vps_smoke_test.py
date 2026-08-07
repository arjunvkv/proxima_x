"""VPS broker smoke test: fire 3 REAL one-minute market orders (EURUSD, USDJPY,
GBPUSD) on the live-attached FTMO-Demo terminal, hold ~60s, close all, and
verify execution + exit work end-to-end without blocking the corebook daemon.
Comment SMOKE_TEST (NOT CORE_*) so the daemon's hold-managed exits ignore them."""
import time, datetime, sys
import MetaTrader5 as mt5

MT5_PATH = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"

def log(*a):
    print(f"[{datetime.datetime.utcnow():%H:%M:%S}Z]", *a, flush=True)

ok = mt5.initialize(path=MT5_PATH)
if not ok:
    log("FATAL initialize failed:", mt5.last_error())
    sys.exit(2)
acct = mt5.account_info()
log(f"attached: login={acct.login} server={acct.server} balance={acct.balance:.2f} "
    f"equity={acct.equity:.2f}")

PAIRS = ["EURUSD", "USDJPY", "GBPUSD"]
VOL = 0.10
tickets = []
for s in PAIRS:
    sym = mt5.symbol_info(s)
    if sym is None:
        log(f"  no symbol {s}"); continue
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": s,
        "volume": VOL, "type": mt5.ORDER_TYPE_BUY,
        "price": mt5.symbol_info_tick(s).ask,
        "deviation": 20, "magic": 99001, "comment": "SMOKE_TEST",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"  OPEN FAIL {s}: retcode={res.retcode} {res.comment}")
        continue
    tickets.append(res.order)
    log(f"  OPEN OK {s}: ticket={res.order} vol={res.volume} fill={res.price} "
        f"bid={mt5.symbol_info_tick(s).bid}")

log(f"open tickets: {tickets} — holding ~60s")
time.sleep(60)

# verify positions live, then close all
pos = mt5.positions_get()
open_now = [p for p in pos if p.comment == "SMOKE_TEST"] if pos else []
log(f"positions after hold: {len(open_now)} open (tickets "
    f"{[p.ticket for p in open_now]})")
closed = []
for p in open_now:
    sym = mt5.symbol_info(p.symbol)
    tick = mt5.symbol_info_tick(p.symbol)
    req = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": p.symbol,
        "volume": p.volume, "type": mt5.ORDER_TYPE_SELL,
        "position": p.ticket, "price": tick.bid,
        "deviation": 20, "magic": 99001, "comment": "SMOKE_TEST_X",
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    res = mt5.order_send(req)
    if res.retcode == mt5.TRADE_RETCODE_DONE:
        closed.append((p.symbol, p.volume, p.profit))
        log(f"  CLOSE OK {p.symbol}: ticket={res.order} fill={res.price} "
            f"profit={p.profit:.2f}")
    else:
        log(f"  CLOSE FAIL {p.symbol}: retcode={res.retcode} {res.comment}")

time.sleep(3)
pos2 = mt5.positions_get()
left = [p for p in pos2 if p.comment == "SMOKE_TEST"] if pos2 else []
log(f"leftover SMOKE_TEST positions: {len(left)} -> {'CLEAN' if not left else 'LEAK!'}")
acct2 = mt5.account_info()
log(f"after: balance={acct2.balance:.2f} equity={acct2.equity:.2f} "
    f"(delta {acct2.balance - acct.balance:+.2f})")
log(f"SUMMARY: opened={len(tickets)} closed={len(closed)} leaked={len(left)}")
mt5.shutdown()
sys.exit(0 if (len(tickets) == 3 and len(closed) == 3 and not left) else 1)