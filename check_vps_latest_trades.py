#!/usr/bin/env python3
"""Check latest 2 trades from VPS MT5 terminal demo account #1514168544."""

import MetaTrader5 as mt5
from datetime import datetime, timedelta

if mt5.initialize():
    if mt5.login(1514168544, "$!4fwBIc", "FTMO-Demo"):
        deals = mt5.history_deals_get(datetime(2026, 8, 3, 0, 0), datetime.now() + timedelta(days=1))
        positions = {}
        if deals:
            for d in deals:
                if d.symbol and d.entry in [0, 1]:
                    pid = d.position_id
                    if pid not in positions: positions[pid] = {'in': None, 'out': None}
                    if d.entry == 0: positions[pid]['in'] = d
                    elif d.entry == 1: positions[pid]['out'] = d
                    
        parsed = []
        for pid in sorted(positions.keys(), reverse=True):
            p = positions[pid]
            d_in, d_out = p['in'], p['out']
            if d_in and d_out:
                net_pnl = d_out.profit + d_out.swap + d_out.commission
                parsed.append({
                    "ticket": pid,
                    "symbol": d_in.symbol,
                    "type": "BUY" if d_in.type == 0 else "SELL",
                    "volume": d_in.volume,
                    "entry_price": d_in.price,
                    "exit_price": d_out.price,
                    "profit": d_out.profit,
                    "commission": d_out.commission,
                    "swap": d_out.swap,
                    "net_pnl": round(net_pnl, 2),
                    "comment": d_in.comment or ""
                })
        print("=" * 115)
        print("LATEST 5 EXECUTED TRADES ON MT5 DEMO ACCOUNT #1514168544:")
        print("=" * 115)
        for tr in parsed[:5]:
            sign = "+" if tr['net_pnl'] >= 0 else ""
            print(f"  • Ticket #{tr['ticket']} | {tr['comment']:<22} | {tr['symbol']} {tr['type']} {tr['volume']}L | Entry: {tr['entry_price']} -> Exit: {tr['exit_price']} | Profit: ${tr['profit']:.2f} | Comm: ${tr['commission']:.2f} | Net PnL: {sign}${tr['net_pnl']:.2f}")
        print("=" * 115)
        mt5.shutdown()
