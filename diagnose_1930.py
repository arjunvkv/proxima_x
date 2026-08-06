import sys
sys.path.insert(0, 'proxima_command_center')
from mt5_history_loader import get_side_by_side_trade_comparison

trades = get_side_by_side_trade_comparison()

# Pattern of live bars: 18:00, 19:00, 20:00, 20:30 -- skipped 19:30
# Let's understand the bar pattern
print("=== LIVE BAR FIRING PATTERN TODAY (sorted) ===")
today = [t for t in trades if '2026-08-03' in t.get('entry_time','')]
today.sort(key=lambda x: x.get('entry_time',''))

bar_times_seen = {}
for t in today:
    et = t.get('entry_time','')
    bar_times_seen.setdefault(et[:16], []).append(t['symbol'])

for bar_time, pairs in sorted(bar_times_seen.items()):
    print(f"  {bar_time} UTC | fired on: {', '.join(pairs)}")

print()
print("=== EXPECTED 30-MIN BARS vs ACTUAL ===")
print("Expected bars (Ultra Monster fires every 30min):")
print("  18:00 ✅ FIRED (USDJPY, GBPUSD)")
print("  18:30 ❓ checking...")
bar_1830 = [t for t in today if '18:30' in t.get('entry_time','')]
print(f"  18:30 {'✅ FIRED' if bar_1830 else '❌ SKIPPED'} {[t['symbol'] for t in bar_1830]}")
bar_1900 = [t for t in today if '19:00' in t.get('entry_time','')]
print(f"  19:00 {'✅ FIRED' if bar_1900 else '❌ SKIPPED'} {[t['symbol'] for t in bar_1900]}")
bar_1930 = [t for t in today if '19:30' in t.get('entry_time','')]
print(f"  19:30 {'✅ FIRED' if bar_1930 else '❌ SKIPPED -- THIS IS THE GAP'} {[t['symbol'] for t in bar_1930]}")
bar_2000 = [t for t in today if '20:00' in t.get('entry_time','')]
print(f"  20:00 {'✅ FIRED' if bar_2000 else '❌ SKIPPED'} {[t['symbol'] for t in bar_2000]}")
bar_2030 = [t for t in today if '20:30' in t.get('entry_time','')]
print(f"  20:30 {'✅ FIRED' if bar_2030 else '❌ SKIPPED'} {[t['symbol'] for t in bar_2030]}")

print()
print("=== WHAT WAS HAPPENING AT 19:00 BAR (just before 19:30) ===")
for t in bar_1900:
    print(f"  #{t['ticket']} {t['symbol']} {t['type']} {t['lot']}L | Entry={t['entry_price']} Exit={t['exit_price']} | PnL={t['net_pnl']} | hold={t.get('hold_min','?')}min")

print()
print("=== POSSIBLE REASONS FOR 19:30 SKIP ===")
# Check if the 19:00 bar trades were still open at 19:30
for t in bar_1900:
    hold = t.get('hold_min', 0)
    print(f"  {t['symbol']} opened at 19:00, hold={hold}min -> would close at {19*60+hold}min UTC = {'still open at 19:30!' if hold > 30 else 'closed before 19:30'}")
