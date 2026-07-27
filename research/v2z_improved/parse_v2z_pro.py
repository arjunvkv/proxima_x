import re

p = r'C:\Users\Arjun Sasi\AppData\Roaming\MetaQuotes\Terminal\D0E8209F77C8CF37AD8BF550E51FF075\Tester\logs\20260726.log'
with open(p, 'r', encoding='utf-16') as f:
    d = f.read()

idx = d.find('V2z_Pro.ex5 from 2026.06.08')
if idx < 0:
    print("V2z_Pro section not found!")
    exit()

rest = d[idx:]
closes = re.findall(r'raw=([-\d.]+)\s+comm=([-\d.]+)\s+pnl=([-\d.]+)', rest)
print("Trades: %d" % len(closes))

raws = [float(c[0]) for c in closes]
coms = [float(c[1]) for c in closes]
nets = [float(c[2]) for c in closes]

wins = sum(1 for n in nets if n > 0)
losses = sum(1 for n in nets if n < 0)

print("Gross PnL: $%+.2f" % sum(raws))
print("Commission: $%.2f" % sum(coms))
print("Net PnL: $%+.2f" % sum(nets))
print("Wins: %d/%d (%.1f%%)" % (wins, len(nets), wins/len(nets)*100 if nets else 0))
print("Avg gross/trade: $%+.2f" % (sum(raws)/len(raws) if raws else 0))
print("Avg net/trade: $%+.2f" % (sum(nets)/len(nets) if nets else 0))

pos_r = [r for r,n in zip(raws,nets) if n > 0]
neg_r = [r for r,n in zip(raws,nets) if n < 0]
if pos_r: print("Avg win gross: $%+.2f" % (sum(pos_r)/len(pos_r)))
if neg_r: print("Avg loss gross: $%+.2f" % (sum(neg_r)/len(neg_r)))

# Count exit types
stp = sum(1 for c in rest.split('CLOSE') if 'stp' in c[:20] or 'stop' in c[:20])
exp = sum(1 for c in rest.split('CLOSE') if 'expiry' in c[:20])
print("Stop exits: %d, Expiry exits: %d" % (stp, exp))
