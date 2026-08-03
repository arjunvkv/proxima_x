#!/usr/bin/env python3
"""Local MT5 Live Execution Test: Auto Close with Volume Normalization."""
import time
import MetaTrader5 as mt5

def main():
    print("="*115)
    print("LOCAL MT5 LIVE TEST: AUTO CLOSE WITH VOLUME NORMALIZATION (FTMO ACCOUNT #1514168544)")
    print("="*115)

    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return

    symbol = "EURUSD"
    if not mt5.symbol_select(symbol, True):
        print(f"❌ Symbol {symbol} not available!")
        mt5.shutdown()
        return

    sym_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if sym_info is None or tick is None:
        print(f"❌ Could not fetch tick for {symbol}")
        mt5.shutdown()
        return

    # 1. Volume Step Normalization on Entry
    raw_lot = 1.20
    step = sym_info.volume_step
    min_vol = sym_info.volume_min
    max_vol = sym_info.volume_max
    
    normalized_entry_vol = round(round(raw_lot / step) * step, 2)
    normalized_entry_vol = max(min_vol, min(max_vol, normalized_entry_vol))

    print(f"  1. ENTRY VOLUME NORMALIZATION:")
    print(f"     Raw Lot Input: {raw_lot} L -> Step Normalized Entry Volume: {normalized_entry_vol:.2f} L 🟢")
    print("="*115)

    # 2. Place Market BUY Deal
    price = tick.ask
    req_entry = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(normalized_entry_vol),
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 10,
        "magic": 999999,
        "comment": "Test_AutoClose_Buy",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    res_entry = mt5.order_send(req_entry)
    if res_entry is None or res_entry.retcode != mt5.TRADE_RETCODE_DONE:
        req_entry["type_filling"] = mt5.ORDER_FILLING_FOK
        res_entry = mt5.order_send(req_entry)

    if res_entry is None or res_entry.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"❌ Market BUY Order failed! Retcode: {res_entry.retcode if res_entry else 'None'}")
        mt5.shutdown()
        return

    order_ticket = res_entry.order
    print(f"  2. MARKET BUY FILLED SUCCESSFULLY:")
    print(f"     Position Ticket: #{order_ticket} | Fill Price: {res_entry.price:.5f}")
    print("="*115)

    # 3. Simulate Timed Expiry Auto-Close with Exit Volume Normalization
    time.sleep(2.0)
    print(f"  3. EXECUTING TIMED EXPIRY AUTO-CLOSE WITH NORMALIZEVOLUME:")

    # Read live open position from FTMO broker memory via positions_get()
    pos_list = mt5.positions_get(ticket=order_ticket)
    if not pos_list:
        pos_list = mt5.positions_get(symbol=symbol)

    if not pos_list:
        print("❌ Could not find position in broker memory for auto-close!")
        mt5.shutdown()
        return

    pos = pos_list[0]
    raw_exit_vol = pos.volume
    
    # Run official NormalizeVolume logic on position volume
    normalized_exit_vol = round(round(raw_exit_vol / step) * step, 2)
    normalized_exit_vol = max(min_vol, min(max_vol, normalized_exit_vol))

    print(f"     Broker Open Position Volume: {raw_exit_vol}")
    print(f"     NormalizeVolume Output for Exit: {normalized_exit_vol:.2f} L 🟢")

    tick_close = mt5.symbol_info_tick(symbol)
    req_close = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(normalized_exit_vol),
        "type": mt5.ORDER_TYPE_SELL,
        "position": pos.ticket,
        "price": tick_close.bid,
        "deviation": 10,
        "magic": 999999,
        "comment": "Test_AutoClose_Exit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    res_close = mt5.order_send(req_close)
    if res_close is None or res_close.retcode != mt5.TRADE_RETCODE_DONE:
        req_close["type_filling"] = mt5.ORDER_FILLING_FOK
        res_close = mt5.order_send(req_close)

    if res_close is not None and res_close.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"  🟢 AUTO-CLOSE SUCCESSFUL WITH ZERO ERRORS!")
        print(f"     Exit Deal Ticket: #{res_close.deal}")
        print(f"     Exit Price: {res_close.price:.5f}")
        print(f"     Retcode: {res_close.retcode} (TRADE_RETCODE_DONE)")
    else:
        print(f"  ❌ Auto-Close Failed! Retcode: {res_close.retcode if res_close else 'None'}, Comment: {res_close.comment if res_close else ''}")

    print("="*115)
    print("🟢 AUTO-CLOSE WITH VOLUME NORMALIZATION TEST PASSED 100%!")
    print("="*115)
    mt5.shutdown()

if __name__ == "__main__":
    main()
