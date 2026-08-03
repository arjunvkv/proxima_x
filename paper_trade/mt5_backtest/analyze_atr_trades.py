"""Analyze AUDUSD ATR test trades vs non-ATR trades from tester log."""
import re

log_path = r"C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\81A933A9AFC5DE3C23B15CAB19C63850\Tester\logs\20260727.log"

with open(log_path, encoding="utf-16") as f:
    lines = f.readlines()

# Parse into runs (a run ends with "final balance")
runs = []
current = {"trades": [], "final_balance": None, "final_time": None, "symbol": "UNKNOWN"}
for line in lines:
    fb = re.search(r"final balance ([\d.]+) USD", line)
    if fb:
        current["final_balance"] = float(fb.group(1))
        tm = re.search(r"(\d+:\d+:\d+\.\d+)", line)
        if tm: current["final_time"] = tm.group(1)
        if current["trades"]:
            runs.append(current)
        current = {"trades": [], "final_balance": None, "final_time": None, "symbol": "UNKNOWN"}
        continue
    if "AUDUSD" in line and ("OPEN" in line or "CLOSE" in line or "LOST" in line):
        current["symbol"] = "AUDUSD"
        current["trades"].append(line.strip())
    if "GBPUSD" in line and ("OPEN" in line or "CLOSE" in line or "LOST" in line):
        current["symbol"] = "GBPUSD"
        current["trades"].append(line.strip())

print(f"Found {len(runs)} AUDUSD/GBPUSD test runs")
for i, r in enumerate(runs):
    if r["trades"]:
        trade_count = len(r["trades"])
        opens = [t for t in r["trades"] if "OPEN" in t]
        closes = [t for t in r["trades"] if ("CLOSE" in t or "LOST" in t) and "OPEN" not in t]
        print(f"\n=== Run {i}: {r['symbol']} final={r['final_balance']:+.2f} ({r['final_time']}) ===")
        print(f"  {trade_count} log lines ({len(opens)} opens, {len(closes)} closes)")

        # Parse individual trades
        trades = []
        for o_line in opens:
            m = re.search(r"(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2}).*entry=([\d.]+).*atr=([\d.e+-]+).*z=([\d.e+-]+).*sprd=([\d.]+)", o_line)
            if m:
                trades.append({
                    "date": m.group(1), "entry": float(m.group(2)),
                    "atr": float(m.group(3)), "z": float(m.group(4)),
                    "sprd": float(m.group(5)), "pnl": None, "result": None, "held": None
                })
        for c_line in closes:
            pm = re.search(r"pnl=([\d.e+-]+)", c_line)
            if pm:
                for t in reversed(trades):
                    if t["pnl"] is None:
                        t["pnl"] = float(pm.group(1))
                        t["result"] = "WIN" if t["pnl"] >= 0 else "LOSS"
                        hm = re.search(r"held=(\d+)", c_line)
                        if hm: t["held"] = int(hm.group(1))
                        break

        wins = [t for t in trades if t["result"] == "WIN"]
        losses = [t for t in trades if t["result"] == "LOSS"]
        print(f"  {len(trades)} parsed trades: {len(wins)}W / {len(losses)}L")
        if trades:
            print(f"  PnL: ${sum(t['pnl'] for t in trades):+.2f}")
            
            # ATR analysis
            atr_wins = [t["atr"] for t in wins]
            atr_losses = [t["atr"] for t in losses]
            if atr_wins: print(f"  Avg ATR (wins): {sum(atr_wins)/len(atr_wins):.6f}")
            if atr_losses: print(f"  Avg ATR (losses): {sum(atr_losses)/len(atr_losses):.6f}")
            
            # Low-ATR trades (< 0.00007)
            low_atr = [t for t in trades if t["atr"] < 0.00007]
            high_atr = [t for t in trades if t["atr"] >= 0.00007]
            print(f"  Low ATR (<0.00007): {len(low_atr)} trades "
                  f"({len([t for t in low_atr if t['result']=='WIN'])}W/"
                  f"{len([t for t in low_atr if t['result']=='LOSS'])}L) "
                  f"PnL=${sum(t['pnl'] for t in low_atr):+.2f}")
            if high_atr:
                high_wins = [t for t in high_atr if t['result']=='WIN']
                high_losses = [t for t in high_atr if t['result']=='LOSS']
                print(f"  High ATR (>=0.00007): {len(high_atr)} trades "
                      f"({len(high_wins)}W/{len(high_losses)}L) "
                      f"PnL=${sum(t['pnl'] for t in high_atr):+.2f}")

            # Spread analysis
            sprd5_wins = [t for t in wins if t["sprd"] <= 5]
            sprd5_losses = [t for t in losses if t["sprd"] <= 5]
            sprd6_wins = [t for t in wins if t["sprd"] > 5]
            sprd6_losses = [t for t in losses if t["sprd"] > 5]
            print(f"  sprd<=5: {len(sprd5_wins)}W/{len(sprd5_losses)}L, "
                  f"sprd>5: {len(sprd6_wins)}W/{len(sprd6_losses)}L")

            # Print individual trades
            for t in trades:
                print(f"    {t['date']} | ATR={t['atr']:.6f} | z={t['z']:.2f} | "
                      f"sprd={t['sprd']} | {t['result']} ${t['pnl']:+.1f} | held={t['held']}b")
