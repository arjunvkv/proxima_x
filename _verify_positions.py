"""Query MT5 positions and compare with dashboard swing overlay values."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'currency_decomposition'))
import MetaTrader5 as mt5
import config.settings as s
s.EXECUTION_MODE = 'live'

if not mt5.initialize():
    print("MT5 init failed:", mt5.last_error())
    sys.exit(1)

positions = mt5.positions_get()
if positions is None:
    print("No positions or error:", mt5.last_error())
else:
    cd_positions = [p for p in positions if 236000 <= p.magic < 236200]
    print(f"Found {len(cd_positions)} Proxima positions:")
    for p in cd_positions:
        direction = "BUY" if p.type == 0 else "SELL"
        pl = p.profit + p.swap
        try:
            pl += p.commission
        except AttributeError:
            pass

        # Get point from symbol info
        sym_info = mt5.symbol_info(p.symbol)
        point = sym_info.point if sym_info else 0.0001
        digits = sym_info.digits if sym_info else 5

        sl_pips = abs((p.price_open - (p.sl or 0)) / point / 10) if p.sl else 0
        tp_pips = abs(((p.tp or 0) - p.price_open) / point / 10) if p.tp else 0
        curr_pips = (p.price_current - p.price_open) / point / 10
        
        print(f"  {p.symbol}: {direction} entry={p.price_open:.{digits}f} current={p.price_current:.{digits}f}")
        print(f"    SL={p.sl or 0:.{digits}f} ({sl_pips:.0f}p) TP={p.tp or 0:.{digits}f} ({tp_pips:.0f}p) from_entry={curr_pips:+.1f}p")
        print(f"    pnl={p.profit:.2f} swap={p.swap:.2f} total={pl:.2f}")

mt5.shutdown()
