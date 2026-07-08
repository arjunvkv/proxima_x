import MetaTrader5 as mt5
mt5.initialize()
acct = mt5.account_info()
print(f"Balance: {acct.balance:.2f}" if acct else "N/A")
ps = [p for p in (mt5.positions_get() or []) if 236000 <= p.magic < 236200]
print(f"CD_v1: {len(ps)} positions")
from collections import Counter
labels = [f"{p.symbol} {'BUY' if p.type==0 else 'SELL'}" for p in ps]
for k, v in sorted(Counter(labels).items()):
    sym, drn = k.split()
    plist = [p for p in ps if p.symbol == sym and ("BUY" if p.type==0 else "SELL") == drn]
    entries = ", ".join(f"{x.price_open:.5f}" for x in plist)
    pnl = sum(x.profit for x in plist)
    print(f"  {k}: x{v}  entries=[{entries}]  pnl={pnl:.2f}")
total = sum(p.profit for p in ps)
print(f"Total PnL: {total:.2f}")
mt5.shutdown()
