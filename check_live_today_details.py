import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone, timedelta

ACCOUNT = {"login": 1514168544, "password": "$!4fwBIc", "server": "FTMO-Demo"}

def main():
    print("=" * 110)
    print("PROXIMA X — LIVE MT5 DEALS AUDIT FOR TODAY (2026-08-04 UTC)")
    print("=" * 110)

    if not mt5.initialize():
        print("Failed to initialize MT5")
        return

    mt5.login(int(ACCOUNT["login"]), password=ACCOUNT["password"], server=ACCOUNT["server"])

    start_utc = datetime(2026, 8, 4, 0, 0, 0, tzinfo=timezone.utc)
    now_utc   = datetime.now(timezone.utc)

    print(f"Server Time Window : {start_utc.strftime('%Y-%m-%d %H:%M:%S UTC')} -> {now_utc.strftime('%H:%M:%S UTC')}")
    print(f"Local IST Time     : {(now_utc + timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d %H:%M:%S IST')}")
    print("=" * 110)

    deals = mt5.history_deals_get(start_utc, now_utc)
    if deals is None or len(deals) == 0:
        print("\n🟢 NO LIVE DEALS EXECUTED OR CLOSED ON MT5 TODAY (2026-08-04 UTC).")
        print("Note: The 4 deals seen earlier in the raw terminal query belonged to 21:30 & 23:30 UTC on 2026-08-03.")
        mt5.shutdown()
        return

    rows = []
    for d in deals:
        if d.type in (0, 1): # BUY=0, SELL=1
            utc_time = datetime.utcfromtimestamp(d.time - 10800) # EET -> UTC offset correction
            ist_time = utc_time + timedelta(hours=5, minutes=30)
            rows.append({
                "Ticket": d.order,
                "UTC Time": utc_time.strftime("%H:%M:%S"),
                "IST Time": ist_time.strftime("%H:%M:%S"),
                "Symbol": d.symbol,
                "Type": "BUY" if d.type == 0 else "SELL",
                "Volume": d.volume,
                "Price": round(d.price, 5),
                "Profit": round(d.profit, 2),
                "Comment": d.comment
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\n🟢 NO LIVE BUY/SELL DEALS FOR TODAY.")
    else:
        print(f"\nTotal Live Deals Today: {len(df)}")
        print(df.to_string(index=False))

    mt5.shutdown()

if __name__ == "__main__":
    main()
