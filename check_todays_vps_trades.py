import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
import pandas as pd

ACCOUNT = {"login": 1514168544, "password": "$!4fwBIc", "server": "FTMO-Demo"}
mt5.initialize()
mt5.login(int(ACCOUNT["login"]), password=ACCOUNT["password"], server=ACCOUNT["server"])

today_start = datetime(2026, 8, 4, 0, 0, 0, tzinfo=timezone.utc)
now_utc     = datetime.now(timezone.utc)
print(f"Fetching deals from {today_start.strftime('%Y-%m-%d %H:%M')} UTC to {now_utc.strftime('%H:%M')} UTC")
print(f"Current IST: {(now_utc + timedelta(hours=5, minutes=30)).strftime('%Y-%m-%d %H:%M')}")
print("=" * 100)

deals = mt5.history_deals_get(today_start, now_utc)
if deals is None or len(deals) == 0:
    print("No deals found today.")
    mt5.shutdown()
    exit()

rows = []
for d in deals:
    if d.type in (0, 1):  # BUY=0, SELL=1
        utc_time = datetime.utcfromtimestamp(d.time - 10800)
        ist_time = utc_time + timedelta(hours=5, minutes=30)
        rows.append({
            "UTC Time":  utc_time.strftime("%H:%M:%S"),
            "IST Time":  ist_time.strftime("%H:%M:%S"),
            "Ticket":    d.order,
            "Symbol":    d.symbol,
            "Type":      "BUY" if d.type == 0 else "SELL",
            "Lot":       d.volume,
            "Price":     round(d.price, 5),
            "Profit":    round(d.profit, 2),
            "Comment":   d.comment[:30] if d.comment else "",
        })

df = pd.DataFrame(rows)
if df.empty:
    print("No BUY/SELL deals today.")
    mt5.shutdown()
    exit()

df = df.sort_values("UTC Time")
print(df.to_string(index=False))
print("=" * 100)

wins   = df[df["Profit"] > 0]
losses = df[df["Profit"] <= 0]
net    = df["Profit"].sum()
wr     = len(wins) / len(df) * 100 if len(df) > 0 else 0

print(f"Total Trades : {len(df)}")
print(f"Win Rate     : {wr:.1f}%  ({len(wins)}W / {len(losses)}L)")
print(f"Net PnL      : ${net:.2f}")
print(f"Gross Wins   : +${wins['Profit'].sum():.2f}")
print(f"Gross Losses : ${losses['Profit'].sum():.2f}")

mt5.shutdown()
