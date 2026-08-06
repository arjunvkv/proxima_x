import MetaTrader5 as mt5
from datetime import datetime, timezone

ACCOUNT = {"login": 1514168544, "password": "$!4fwBIc", "server": "FTMO-Demo"}

def main():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        return
    mt5.login(int(ACCOUNT["login"]), password=ACCOUNT["password"], server=ACCOUNT["server"])

    acc_info = mt5.account_info()
    positions = mt5.positions_get()

    print("=" * 90)
    print("MT5 LIVE ACCOUNT STATUS RIGHT NOW")
    print("=" * 90)
    if acc_info:
        print(f"  • Balance     : ${acc_info.balance:,.2f}")
        print(f"  • Equity      : ${acc_info.equity:,.2f}")
        print(f"  • Floating PnL: ${acc_info.profit:,.2f}")
        print(f"  • Margin      : ${acc_info.margin:,.2f}")
        print(f"  • Free Margin : ${acc_info.margin_free:,.2f}")

    print("\nCURRENTLY OPEN POSITIONS:")
    if positions is None or len(positions) == 0:
        print("  🟢 NO ACTIVE OPEN POSITIONS ON MT5 ACCOUNT RIGHT NOW.")
    else:
        for p in positions:
            print(f"  Ticket #{p.ticket} | {p.symbol} | {'BUY' if p.type==0 else 'SELL'} | Lot: {p.volume} | Entry: {p.price_open} | Current: {p.price_current} | PnL: ${p.profit:.2f}")

    mt5.shutdown()

if __name__ == "__main__":
    main()
