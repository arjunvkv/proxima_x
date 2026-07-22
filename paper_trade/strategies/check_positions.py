"""Check current MT5 positions."""
import MetaTrader5 as mt5

if mt5.initialize():
    positions = mt5.positions_get()
    if positions and len(positions) > 0:
        print(f"MT5 has {len(positions)} open positions:")
        total_pnl = 0
        for pos in positions:
            typ = "BUY" if pos.type == 0 else "SELL"
            age_sec = int(mt5.symbol_info_tick(pos.symbol).time) - pos.time
            print(f"  {pos.ticket} {pos.symbol} {typ} vol={pos.volume} open={pos.price_open} sl={pos.sl} tp={pos.tp} profit={pos.profit:.2f} age={age_sec}s magic={pos.magic}")
            total_pnl += pos.profit
        print(f"Total PnL: ${total_pnl:.2f}")
    else:
        print("No open positions on MT5")
    mt5.shutdown()
else:
    print("MT5 init failed")
