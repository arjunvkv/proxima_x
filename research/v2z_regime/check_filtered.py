"""Check filtered run result (get data for the LAST run in the log)."""
import re, numpy as np

path = "C:\\Users\\Arjun Sasi\\AppData\\Roaming\\MetaQuotes\\Tester\\D0E8209F77C8CF37AD8BF550E51FF075\\Agent-127.0.0.1-3000\\logs\\20260726.log"
with open(path, 'rb') as f:
    raw = f.read()
text = raw.decode('utf-16-le', errors='replace')
lines = text.split('\n')

# Last run (between second-to-last and last final balance)
fbs = [i for i,l in enumerate(lines) if 'final balance' in l]
start = fbs[-2] + 1 if len(fbs) >= 2 else 0
last = lines[start:fbs[-1]]

deals = []
for l in last:
    m = re.search(r'deal #(\d+) (buy|sell) [\d.]+ (\w+) at ([\d.]+) done', l)
    if m:
        deals.append({'tkt':int(m.group(1)),'side':m.group(2),'sym':m.group(3),'pr':float(m.group(4))})

pnls = []
i = 0
while i < len(deals) - 1:
    for j in range(i+1, len(deals)):
        if deals[j]['side'] != deals[i]['side'] and deals[j]['sym'] == deals[i]['sym']:
            if deals[i]['side'] == 'buy':
                p = (deals[j]['pr'] - deals[i]['pr']) * 0.75 * 100000
            else:
                p = (deals[i]['pr'] - deals[j]['pr']) * 0.75 * 100000
            pnls.append(p)
            i = j + 1
            break
    else:
        i += 1

print(f"Filtered: {len(pnls)} trades")
arr = np.array(pnls)
print(f"Total: ${arr.sum():+.2f}")
print(f"Mean: ${arr.mean():+.2f}/trade")
print(f"WR: {np.mean(arr>0)*100:.0f}%")
print(f"Best: ${arr.max():+.2f}, Worst: ${arr.min():+.2f}")
for idx, p in enumerate(pnls):
    print(f"  Trade {idx}: ${p:+.2f}")
