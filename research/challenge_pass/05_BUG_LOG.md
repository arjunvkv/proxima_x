# Bug Log

## Bug 1: Missing `direction[t]` Multiplier — Dark Consensus PnL Calculation

**Found**: Jul 28, 2026
**Impact**: FALSE NEGATIVE — strategy appeared to have 48.5% WR on FundedNext data

### The Bug

In `test_fundednext_dark_consensus.py`, the PnL calculation was:

```python
gross = (xp - ep) * 100000  # ALWAYS LONG PnL
```

This computed a LONG PnL for EVERY trade regardless of whether the strategy
signal was LONG or SHORT. For SHORT trades (which represent ~50% of all trades),
a positive PnL requires price DOWN, but the calculation gave positive PnL for
price UP. Half the trades had their sign inverted.

### The Fix

```python
raw_pnl = (xp - ep) * direction[t]  # direction = +1 LONG, -1 SHORT
gross = raw_pnl * 100000
```

### Impact

| Metric | Before (Broken) | After (Fixed) |
|--------|:---------------:|:-------------:|
| Gross WR | 48.5% | 93.1% |
| Gross Sharpe | 0.51 | 22.99 |
| Net WR | 41.1% | 83.1% |
| Net/day | −$345 | +$650 |

The strategy went from "no edge exists on FundedNext" to "extraordinary edge."

## Bug 2: `// 10**9` Timestamp Misconversion — V2+z Simulator

**Found**: Jul 27, 2026
**Impact**: ALL V2+z backtest results were INVALID (many trades missed), Python
sim showed 17x inflation vs MT5 tester

### The Bug

In `sim_recon.py` and `challenge_z.py`, timestamps from `.npy` files (saved via
`time` field of MT5's `copy_rates_range`, which returns Unix seconds) were
divided by 10^9:

```python
times = np.array(common, dtype=np.int64) // 10**9
```

The `time` field from `copy_rates_range` is already in seconds. Dividing by 10^9
produced epoch ≈ 1 (Jan 1, 1970), making EVERY bar appear to precede the start
date. The warmup condition `dt < start_dt` was always True, so:

- `sim_recon.py`: ALL bars stuck in warmup → 0 trades ever
- `challenge_z.py`: ALL bars stuck in warmup → 0 trades ever (but fallback
  logic ran 100% of data as test)
- Every backtest with warmup was silently returning zero trades

### The Fix

```python
if times[0] < 1e9:  # already seconds
    raw_times = times
else:  # nanoseconds
    raw_times = times // 10**9
```

## Bug 3: EURUSD Zero Spread on FundedNext Data

**Found**: Jul 28, 2026
**Impact**: 70.67% of EURUSD bars have spread=0, causing unrealistic cost
estimates

### The Bug

FundedNext Server 3 `copy_rates_range` returns `spread=0` for 35,243 out of
49,872 EURUSD M1 bars (71%). This is likely a data artifact — possibly the
terminal does not report spread when at minimum (0.0 pips) or during certain
market conditions.

### The Fix

Applied a floor equal to the median of non-zero spread bars:

```python
eurusd_nnz = spreads[:, 1][spreads[:, 1] > 0]
eurusd_med = np.median(eurusd_nnz)  # 8 points = 0.8 pips
spreads[:, 1] = np.maximum(spreads[:, 1], eurusd_med)
```

### Impact

| Before Floor | After Floor |
|:------------:|:-----------:|
| EURUSD spread med=0 | EURUSD spread med=8 (0.8 pips) |
| Cost/trade: ~$13 (unrealistic) | Cost/trade: ~$16 (realistic) |

## Bug 4: JPY Pair Pip Value Overestimate

**Found**: Initial Dark Consensus test (Jul 28, 2026)
**Impact**: Cost model overestimated JPY pair costs by ~60%

### The Bug

All pairs were assigned pip value = $10. For JPY pairs (EURJPY, GBPJPY), the
correct pip value depends on the USDJPY rate:

```
pip_value_JPY = 1000 / USDJPY
```

At USDJPY ≈ 162, this gives $6.17/pip, not $10/pip.

### The Fix

```python
def pip_value(pair_idx, usdjpy):
    if pair_idx == 1:  # EURUSD
        return 10.0
    else:  # JPY pairs
        return 1000.0 / usdjpy
```

### Impact

Cost/trade for EURJPY went from ~$15.50 to ~$11.65.
