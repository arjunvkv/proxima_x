# Phase Dislocation — Dark Research Notes

## Core Thesis

**The market reveals hidden information through cross-rate dislocations.**

When FX cross rates diverge from their mathematically implied synthetic rates, the convergence creates a predictable, structural trading opportunity. This is not a statistical pattern — it's a mathematical identity that must resolve.

### The Hidden Edge

Most traders watch pairs in isolation. If EUR/JPY drops, they ask "should I buy the dip or sell the rip?" They never ask: "Which *other* pair will move to restore the EUR/USD × USD/JPY triangle?"

The answer is counterintuitive: **the pair with the least recent momentum.** This is the path of least resistance — it can move to restore equilibrium without fighting strong directional flow.

## Why This Won't Be Crowded

1. **Not obvious:** Everyone trades the mover. We trade the "silent" pair — the one that hasn't moved yet but must move.

2. **Structural, not statistical:** This is not a backtest-mining discovery. The triangle identity is mathematical. When it dislocates, convergence is mechanical, not probabilistic.

3. **Low frequency:** Dislocations are rare (3-8 per day in liquid hours). Low frequency = low overfitting surface area. The market cannot easily adapt to a strategy that fires a few times per day.

4. **Counterintuitive execution:** We look at 3 pairs at once, compute synthetic crosses, monitor rolling z-scores, and select the path of least resistance. This is beyond the typical retail trader's toolkit.

5. **Cannot be tricked:** The market cannot permanently dislocate a triangle. Arbitrage constraints force convergence. The only failure mode is if the dislocation *persists* — which doesn't happen in liquid FX markets.

## Anti-Overfitting Defenses

### What we fixed (no free parameters):
- **Threshold: 2σ z-score** — standard statistical significance, not optimized
- **Lookback: 60 ticks** — captures ~1-2 minutes of price action
- **History window: 100 samples** — sufficient for stable rolling statistics
- **Hold time: 5 minutes** — standard for microstructure reversion
- **Lot size: 0.5** — conservative position sizing

### What we did NOT do:
- ❌ Grid-search over parameters (no P95 threshold hunting)
- ❌ Session optimization (broad 07-21 UTC window)
- ❌ Pair selection optimization (all major triangles)
- ❌ Volatility filtering (adds overfitting surface)
- ❌ Confidence threshold tuning (fixed at 0.4)

### What traps we avoided:

| Trap | How We Avoid It |
|------|-----------------|
| **Breakeven hell** — strategy looks good in sample, dies OOS | Structural edge (triangle identity) rather than statistical pattern |
| **Market adaptation** — crowd finds the edge, trades it away | Low frequency + counterintuitive execution = hard to detect |
| **Curve fitting** — too many parameters optimize to noise | Only 1 real parameter (z threshold), fixed at standard 2σ |
| **Lookahead bias** — leaky signals | Only uses past prices from deque, no future data |
| **Survivorship bias** — strategy only works on certain pairs | 4 triangles, 8 pairs, diverse geography |
| **Regime dependence** — only works in certain market conditions | No regime filter; dislocation detection adapts via rolling stats |

## The "Dark Researcher" Mindset

Every trading strategy faces an adversary: **the market itself.** The market learns, adapts, and exploits predictable behavior.

The Phase Dislocation strategy defeats this adversary because:

1. **It exploits mechanics, not patterns.** Mechanical edges persist. Statistical edges decay.

2. **It's invisible.** The market sees order flow in EUR/JPY. It doesn't know we're trading USD/JPY because of an EUR/JPY dislocation. Our footprint is disguised.

3. **It's humble.** We don't predict direction. We detect imbalance and let the market's own convergence mechanics generate the edge. We're not fighting the market — we're riding its cleanup.

## Triangles and Expected Behavior

### Triangle 1: EURUSD × USDJPY = EURJPY (mult)
- Most liquid triangle. EURJPY is the most actively traded cross.
- Dislocations are shortest-lived here (deep liquidity).
- Signal preference: pair C (EURJPY) has lowest momentum → short/long EURJPY directly.

### Triangle 2: EURUSD / EURGBP = GBPUSD (div)
- Division triangle. Tests the EUR/GBP relationship.
- GBPUSD is deeply liquid; EURGBP is thinner.
- Dislocations often resolve through EURGBP (path of least resistance).

### Triangle 3: GBPUSD × USDJPY = GBPJPY (mult)
- GBPJPY is a volatile cross. Dislocations are frequent.
- Best signal: when GBPUSD and USDJPY both move but GBPJPY lags.

### Triangle 4: AUDUSD × USDJPY = AUDJPY (mult)
- AUDJPY is flow-driven (carry trade proxy).
- Dislocations often coincide with risk sentiment shifts.

## Validation Path

1. **Synthetic data test** — verify signal logic is self-consistent
2. **Dukascopy M1 backtest** — 9+ months of historical data across all 4 triangles
3. **Walk-forward validation** — non-overlapping windows
4. **OOS tick data** — Exness or MT5 tick data not seen in training
5. **Reality stress** — latency, slippage, spread widening, portfolio overlap
6. **Paper trade** — live via MT5 demo (if validated)
7. **Deprecate or deploy** — based on live evidence

---

*"The market reveals its structure through its dislocations. We don't predict — we detect and ride the convergence."*
