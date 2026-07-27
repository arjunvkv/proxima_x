import re

logpath = r'C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\Tester\logs\20260726.log'
with open(logpath, 'r') as f:
    log = f.read()

idx = log.find('V2z_Pro.ex5')
if idx < 0:
    print("V2z_Pro.ex5 not found")
    exit()

rest = log[idx:]
closes = re.findall(r'raw=([-\d.]+)\s+comm=([-\d.]+)\s+pnl=([-\d.]+)', rest)
print(f"Trades found: {len(closes)}")

raws = [float(c[0]) for c in closes]
coms = [float(c[1]) for c in closes]
nets = [float(c[2]) for c in closes]

wins = sum(1 for n in nets if n > 0)
losses = sum(1 for n in nets if n < 0)

print(f"Gross PnL: ${sum(raws):+.2f}")
print(f"Commission: ${sum(coms):+.2f}")
print(f"Net PnL:    ${sum(nets):+.2f}")
print(f"Wins: {wins}/{len(nets)} ({wins/len(nets)*100:.1f}%)")
print(f"Losses: {losses}")
print(f"Avg gross PnL per trade: ${sum(raws)/len(raws):+.2f}")
print(f"Avg net PnL per trade: ${sum(nets)/len(nets):+.2f}")

pos_ra = [r for r,n in zip(raws, nets) if n > 0]
neg_ra = [r for r,n in zip(raws, nets) if n < 0]
if pos_ra: print(f"Avg win: ${sum(pos_ra)/len(pos_ra):+.2f}")
if neg_ra: print(f"Avg loss: ${sum(neg_ra)/len(neg_ra):+.2f}")
