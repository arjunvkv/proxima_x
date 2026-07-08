"""Synthetic tests for CurrencyRangeBucket."""
import sys, os
sys.path.insert(0, os.path.join("C:", os.sep, "Trading", "Agentic_Trading"))
sys.path.insert(0, os.path.join("C:", os.sep, "Trading", "Agentic_Trading", "proxima_x", "currency_decomposition"))

from currency.range_bucket import CurrencyRangeBucket

bucket = CurrencyRangeBucket(window=20)

def feed(b, currency, values):
    for v in values:
        b.update({currency: v})

def show(b, currency, label):
    s = b.get_state(currency)
    print(f"  {label}: curr={currency} pct={s['percentile']} state={s['state']} median={s['median']} drift={s['drift']} samples={s['sample_size']}")

print("=== Test A: Stable range ===")
feed(bucket, "EUR", [5,5,5,6,5,6,5,6,5,5,6,6,5,5,6,6,5,5,6,5])
show(bucket, "EUR", "Expect WITHIN_RANGE, drift≈0")

print("\n=== Test B: Sudden dip ===")
bucket2 = CurrencyRangeBucket(window=20)
feed(bucket2, "EUR", [5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,5,6,2])
show(bucket2, "EUR", "Expect STRETCHED_LOW or OUTSIDE_RANGE_LOW")

print("\n=== Test C: Slow degradation ===")
bucket3 = CurrencyRangeBucket(window=20)
feed(bucket3, "EUR", [6,6,5,5,4,4,3,3,2,2,1,1])
show(bucket3, "EUR", "Range normalizes but drift negative")

print("\n=== Test D: Recovery after dip ===")
bucket4 = CurrencyRangeBucket(window=20)
feed(bucket4, "EUR", [6,6,6,3,3,6,6,6,6,6,6,6])
show(bucket4, "EUR", "Expect WITHIN_RANGE or STRETCHED_HIGH")

print("\n=== Test E: Mid-rank percentile vs naive ===")
bucket5 = CurrencyRangeBucket(window=20)
feed(bucket5, "EUR", [5,5,5,6,7,8,9])
show(bucket5, "EUR", "curr=5: mid-rank should give ~21%, not 42%")

print("\n=== Test F: Insufficient data ===")
bucket6 = CurrencyRangeBucket(window=20)
feed(bucket6, "EUR", [1,2,3])
show(bucket6, "EUR", "Expect INSUFFICIENT_DATA")

print("\n=== Test G: Reset ===")
bucket7 = CurrencyRangeBucket(window=20)
feed(bucket7, "EUR", [1,2,3,4,5,6,7,8,9,10])
bucket7.reset()
print(f"  After reset, history empty: {len(bucket7.history)} currencies")

print("\n=== Test H: get_all_states ===")
bucket8 = CurrencyRangeBucket(window=20)
feed(bucket8, "EUR", [1,2,3,4,5,6,7,8,9,10])
feed(bucket8, "USD", [10,9,8,7,6,5,4,3,2,1])
all_s = bucket8.get_all_states()
print(f"  Currencies: {list(all_s.keys())}")
for c, s in all_s.items():
    print(f"    {c}: pct={s['percentile']} state={s['state']}")

print("\nAll tests done.")
