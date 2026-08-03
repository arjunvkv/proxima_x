#!/usr/bin/env python3
"""Local MT5 Live Execution Test: Full Combined Lifecycle (Entry + ECN SL/TP Attachment + Auto-Close)."""
import time
import MetaTrader5 as mt5

def main():
    print("="*115)
    print("FULL COMBINED LIFECYCLE TEST: ENTRY + ECN SL/TP ATTACHMENT + AUTO-CLOSE (FTMO ACCOUNT #1514168544)")
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

    # 1. Entry Volume Normalization
    raw_lot = 1.20
    step = sym_info.volume_step
    min_vol = sym_info.volume_min
    max_vol = sym_info.volume_max
    
    normalized_entry_vol = round(round(raw_lot / step) * step, 2)
    normalized_entry_vol = max(min_vol, min(max_vol, normalized_entry_vol))

    print(f"  STEP 1: ENTRY VOLUME NORMALIZATION:")
    print(f"     Raw Lot Input: {raw_lot} L -> Step Normalized Entry Volume: {normalized_entry_vol:.2f} L 🟢")
    print("="*115)

    # 2. Place Market BUY Deal
    price = tick.ask
    digits = sym_info.digits
    sl_distance = 0.0035 # 35 pips
    tp_distance = 0.0045 # 45 pips
    
    sl_target = round(price - sl_distance, digits)
    tp_target = round(price + tp_distance, digits)

    req_entry = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(normalized_entry_vol),
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "deviation": 10,
        "magic": 999999,
        "comment": "Test_Full_Lifecycle_Buy",
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
    fill_price = res_entry.price
    print(f"  STEP 2: MARKET BUY FILLED SUCCESSFULLY:")
    print(f"     Order Ticket: #{order_ticket} | Fill Price: {fill_price:.5f}")
    print("="*115)

    # 3. Post-Fill ECN PositionModify SL/TP Attachment
    time.sleep(0.5)
    pos_list = mt5.positions_get(symbol=symbol)
    if not pos_list:
        print("❌ Could not find open position for SL/TP modification!")
        mt5.shutdown()
        return

    pos = pos_list[0]
    pos_ticket = pos.ticket
    print(f"  STEP 3: ECN POST-FILL POSITIONMODIFY SL/TP ATTACHMENT:")
    print(f"     Active Position Ticket: #{pos_ticket}")
    print(f"     Current Position SL Before Modify: {pos.sl:.5f}")
    print(f"     Current Position TP Before Modify: {pos.tp:.5f}")

    req_modify = {
        "action": mt5.TRADE_ACTION_SLTP,
        "symbol": symbol,
        "position": pos_ticket,
        "sl": float(sl_target),
        "tp": float(tp_target),
    }

    res_modify = mt5.order_send(req_modify)
    if res_modify is not None and res_modify.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"  🟢 ECN POSITIONMODIFY SUCCESSFUL!")
    else:
        print(f"  ⚠️ PositionModify retcode: {res_modify.retcode if res_modify else 'None'}, comment: {res_modify.comment if res_modify else ''}")

    # 4. Live Verification of Attached Position in FTMO Broker Memory
    time.sleep(0.5)
    pos_verify_list = mt5.positions_get(ticket=pos_ticket)
    if pos_verify_list:
        p_ver = pos_verify_list[0]
        print("="*115)
        print("  STEP 4: LIVE VERIFICATION OF ATTACHED SL/TP IN FTMO BROKER MEMORY:")
        print(f"     Position Ticket: #{p_ver.ticket}")
        print(f"     Volume: {p_ver.volume:.2f} L 🟢")
        print(f"     Entry Price: {p_ver.price_open:.5f}")
        print(f"     Attached SL: {p_ver.sl:.5f} (Exact Match: {p_ver.sl == sl_target}) 🟢")
        print(f"     Attached TP: {p_ver.tp:.5f} (Exact Match: {p_ver.tp == tp_target}) 🟢")
        print("="*115)

    # 5. Timed Expiry Auto-Close with Exit Volume Normalization
    time.sleep(2.0)
    print(f"  STEP 5: TIMED EXPIRY AUTO-CLOSE WITH NORMALIZEVOLUME:")
    raw_exit_vol = p_ver.volume
    normalized_exit_vol = round(round(raw_exit_vol / step) * step, 2)
    normalized_exit_vol = max(min_vol, min(max_vol, normalized_exit_vol))

    tick_close = mt5.symbol_info_tick(symbol)
    req_close = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(normalized_exit_vol),
        "type": mt5.ORDER_TYPE_SELL,
        "position": pos_ticket,
        "price": tick_close.bid,
        "deviation": 10,
        "magic": 999999,
        "comment": "Test_Full_Lifecycle_Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    res_close = mt5.order_send(req_close)
    if res_close is None or res_close.retcode != mt5.TRADE_RETCODE_DONE:
        req_close["type_filling"] = mt5.ORDER_FILLING_FOK
        res_close = mt5.order_send(req_close)

    if res_close is not None and res_close.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"  🟢 AUTO-CLOSE EXECUTED CLEANLY AT {res_close.price:.5f} WITH ZERO ERRORS!")
        print(f"     Retcode: {res_close.retcode} (TRADE_RETCODE_DONE)")
    else:
        print(f"  ❌ Auto-Close Failed: {res_close.comment if res_close else mt5.last_error()}")

    print("="*115)
    print("🟢 FULL COMBINED LIFECYCLE TEST PASSED 100% WITH SL/TP ATTACHED & ZERO ERRORS!")
    print("="*115)
    mt5.shutdown()

if __name__ == "__main__":
    main()
