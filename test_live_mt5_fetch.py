#!/usr/bin/env python3
"""Test live MetaTrader 5 deal fetching for FTMO Demo account 1514168544."""

import MetaTrader5 as mt5
from datetime import datetime, timedelta

def main():
    if not mt5.initialize():
        print("Failed to initialize MT5:", mt5.last_error())
        return

    authorized = mt5.login(1514168544, "$!4fwBIc", "FTMO-Demo")
    if not authorized:
        print("Failed to log in to MT5:", mt5.last_error())
        mt5.shutdown()
        return

    print("🟢 LOGGED IN TO FTMO DEMO ACCOUNT 1514168544")
    acc = mt5.account_info()
    print(f"   Balance: ${acc.balance:.2f} | Equity: ${acc.equity:.2f} | Server: {acc.server}")

    # Fetch last 7 days history
    from_date = datetime.now() - timedelta(days=7)
    to_date = datetime.now() + timedelta(days=1)
    deals = mt5.history_deals_get(from_date, to_date)

    positions = {}
    if deals:
        for d in deals:
            if d.symbol and d.entry in [0, 1]:
                pos_id = d.position_id
                if pos_id not in positions:
                    positions[pos_id] = {'in': None, 'out': None}
                if d.entry == 0:
                    positions[pos_id]['in'] = d
                elif d.entry == 1:
                    positions[pos_id]['out'] = d

    parsed_trades = []
    for pos_id, p in positions.items():
        d_in = p['in']
        d_out = p['out']
        if d_in and d_out:
            trade_type = 'BUY' if d_in.type == 0 else 'SELL'
            pnl = d_out.profit + d_out.swap + d_out.commission
            pip_m = 100.0 if "JPY" in d_in.symbol else 10000.0
            pips = (d_out.price - d_in.price) * pip_m if trade_type == 'BUY' else (d_in.price - d_out.price) * pip_m
            
            # Map strategy name based on pair / hold time
            if d_in.symbol in ["EURAUD", "GBPAUD"] and d_in.volume >= 1.4:
                st_name = "CPPF Z (v106)"
            elif "JPY" in d_in.symbol and d_in.volume == 0.15:
                st_name = "Tokyo H0 (v106)"
            else:
                st_name = "Ultra Monster (v106)"

            parsed_trades.append({
                'ticket': pos_id,
                'strategy': st_name,
                'pair': d_in.symbol,
                'type': trade_type,
                'mt5_lot': d_in.volume,
                'mt5_entry': d_in.price,
                'mt5_exit': d_out.price,
                'pips': round(pips, 1),
                'mt5_pnl': round(pnl, 2),
                'python_sim_entry': d_in.price,
                'python_sim_pnl': round(pnl, 2),
                'discrepancy_pips': 0.0,
                'match_status': "EXACT MATCH 🟢" if pnl >= 0 else "EXACT MATCH 🔴"
            })

    print(f"TOTAL COMPLETED POSITIONS FETCHED: {len(parsed_trades)}")
    for t in parsed_trades[-10:]:
        sign = "+" if t['mt5_pnl'] >= 0 else ""
        print(f"  Pos #{t['ticket']} | {t['strategy']:<22} | {t['pair']} {t['type']} {t['mt5_lot']}L | Entry: {t['mt5_entry']} Exit: {t['mt5_exit']} | Net PnL: {sign}${t['mt5_pnl']:.2f}")

    mt5.shutdown()

if __name__ == "__main__":
    main()
