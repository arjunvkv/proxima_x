"""Parse Agent log: compare every market order quote vs actual fill price."""
import re
import numpy as np

path = "C:\\Users\\Arjun Sasi\\AppData\\Roaming\\MetaQuotes\\Tester\\D0E8209F77C8CF37AD8BF550E51FF075\\Agent-127.0.0.1-3000\\logs\\20260726.log"
with open(path, "rb") as f:
    raw = f.read()
text = raw.decode("utf-16-le", errors="replace")
lines = text.split("\n")

fbs = [i for i,l in enumerate(lines) if "final balance" in l]

# Analyze last run (hybrid M5 - most trades)
start = fbs[-2]+1 if len(fbs) > 1 else 0
run = lines[start:fbs[-1]]

symbol = "EURAUD"

# Parse ALL deals with their market order context
deals = []
current_market = None

for l in run:
    # Market order line: shows the current bid/ask
    m = re.search(rf"market (buy|sell) [\d.]+ {symbol}(?:, close #\d+)?[^(]*\(([\d.]+) / ([\d.]+) / ([\d.]+)\)", l)
    if m:
        current_market = {"side": m.group(1), "bid": float(m.group(2)), "ask": float(m.group(3)), "last": float(m.group(4))}
        continue

    # Deal line: actual fill price
    m = re.search(rf"deal #(\d+) (buy|sell) ([\d.]+) {symbol} at ([\d.]+) done", l)
    if m and current_market:
        tkt = int(m.group(1)); side = m.group(2); vol = float(m.group(3)); fill = float(m.group(4))
        # Expected price based on side
        expected = current_market["ask"] if side == "buy" else current_market["bid"]
        slip_pips = (fill - expected) * 10000 if side == "buy" else (expected - fill) * 10000
        deals.append({"tkt": tkt, "side": side, "vol": vol, "fill": fill, "expected": expected, "slip_pips": slip_pips, "bid": current_market["bid"], "ask": current_market["ask"]})
        current_market = None

print(f"Found {len(deals)} deal entries in the run")
print()

# Analyze entry slip
entry_slips = [d["slip_pips"] for d in deals]
print("=" * 65)
print("ENTRY SLIPPAGE: expected quote vs actual fill")
print("=" * 65)
arr = np.array(entry_slips)
print(f"  Trades: {len(arr)}")
print(f"  Mean slip:  {arr.mean():+.4f} pips")
print(f"  Median:     {np.median(arr):+.4f} pips")
print(f"  Std:        {arr.std():.4f} pips")
print(f"  Min:        {arr.min():+.4f} pips")
print(f"  Max:        {arr.max():+.4f} pips")
nonzero = (np.abs(arr) > 0.001).sum()
print(f"  Non-zero:   {nonzero}/{len(arr)} ({nonzero/len(arr)*100:.1f}%)")
if nonzero == 0:
    print()
    print("  *** CRITICAL FINDING: ZERO REAL SLIPPAGE ***")
    print("  The MT5 Strategy Tester fills every order at EXACTLY the quoted price.")
    print("  The EA's 1-pip artificial slip adjustment is DOUBLE-COUNTING:")
    print("    1. Spread cost is ALREADY in bid/ask prices (buy at ask, sell at bid)")
    print("    2. Then EA adds +1 pip on entry and -1 pip on exit")
    print("  This makes backtest PnL WORSE than reality.")
    print()

# Show some trades
print("=" * 65)
print("SAMPLE TRADES")
print("=" * 65)
print(f"{'Tkt':<5s} {'Side':<5s} {'Bid':<10s} {'Ask':<10s} {'Expected':<10s} {'Fill':<10s} {'Slip(p)':<9s}")
print("-" * 65)
for d in deals[:10]:
    print(f"{d['tkt']:<5d} {d['side']:<5s} {d['bid']:<10.6f} {d['ask']:<10.6f} {d['expected']:<10.6f} {d['fill']:<10.6f} {d['slip_pips']:<+9.4f}")

# Now calculate: what's the ACTUAL edge with NO artificial slip?
# Pair deals (entry + exit) based on ticket numbers
true_pnls = []
true_slips = []  # entry slip in dollars

for i in range(0, len(deals)-1, 2):
    if i+1 >= len(deals): break
    e = deals[i]; x = deals[i+1]
    if e["side"] == x["side"]: continue  # not a pair
    
    cs = 100000
    if e["side"] == "buy":
        pnl = (x["fill"] - e["fill"]) * e["vol"] * cs
        slip = (e["fill"] - e["expected"]) * e["vol"] * cs  # positive = bought high = bad
    else:
        pnl = (e["fill"] - x["fill"]) * e["vol"] * cs
        slip = (e["expected"] - e["fill"]) * e["vol"] * cs  # positive = sold for expected or better
    
    # commission: $5/lot/side = $5 * vol * 2
    comm = 5 * e["vol"] * 2
    true_pnls.append({"entry": e, "exit": x, "gross": pnl, "comm": comm, "net": pnl - comm, "entry_slip_dollars": slip})

print()
print("=" * 65)
print("TRUE PnL (from actual fills, $5/lot/side commission)")
print("=" * 65)
arr_p = np.array([t["gross"] for t in true_pnls])
arr_n = np.array([t["net"] for t in true_pnls])
print(f"  Count:    {len(arr_p)}")
print(f"  Gross:    ${arr_p.sum():+.2f} (${arr_p.mean():+.2f}/trade)")
print(f"  Net:      ${arr_n.sum():+.2f} (${arr_n.mean():+.2f}/trade)")
print(f"  WR (gross): {(arr_p > 0).mean()*100:.1f}%")
print(f"  WR (net):   {(arr_n > 0).mean()*100:.1f}%")
print(f"  Max win:  ${arr_p.max():+.2f}")
print(f"  Max loss: ${arr_p.min():+.2f}")

# Entry slip cost (total across all trades)
total_slip_cost = sum(t["entry_slip_dollars"] for t in true_pnls)
print(f"\n  Entry slip cost: ${total_slip_cost:+.2f} ({total_slip_cost/len(true_pnls):+.2f}/trade)")

# Now let's see what the EA's ARTIFICIAL slip costs us
# EA adds 1 pip (0.0001) to entry and subtracts 1 pip from exit for same-side trades
# That's 2 pips of artificial cost per round trip = 2 * 0.0001 * vol * cs
ea_slip_cost = len(true_pnls) * 2 * 0.0001 * 0.75 * 100000
print(f"\n===== THE BOTTOM LINE =====")
print(f"  EA artificial slip cost: ${ea_slip_cost:.2f}")
print(f"  True gross PnL:         ${arr_p.sum():+.2f}")
print(f"  True net PnL (with ${5*0.75*2:.2f} comm/trade): ${arr_n.sum():+.2f}")
print(f"  If we remove artificial slip: gross improves by ${ea_slip_cost:.2f}")

# Final comparison with tester balance
print()
balance_change = float(re.search(r"balance ([\d.]+)", lines[fbs[-1]]).group(1)) - 10000
print(f"  Tester raw balance chg: ${balance_change:+.2f}")
print(f"  Our parsed gross sum:   ${arr_p.sum():+.2f}")
print(f"  Difference:             ${arr_p.sum() - balance_change:+.2f}")
