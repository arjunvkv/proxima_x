# Universal Cross-Broker FX Strategy Specification

## Designing One Strategy to Produce Near-Identical Decisions Across MT4, MT5, Python and Different FX Brokers

**Version:** 1.0
**Objective:** Cross-broker strategy equivalence
**Target environments:** Python, MT4, MT5, prop-firm broker feeds, retail FX brokers and FIX-connected venues

---

# 1. Objective

The strategy must be designed from the beginning to operate consistently across different FX brokers and trading platforms.

The objective is NOT:

> Develop a profitable strategy on one broker and subsequently modify other broker feeds until the strategy works there.

The objective is:

> Develop a strategy whose inputs, calculations, state transitions, entries and exits are inherently reproducible across independent broker feeds.

The same market event should therefore produce approximately the same strategic decision on:

* MT4
* MT5
* Python
* FundedNext-supported environments
* FTMO-supported environments
* Other compatible FX brokers

without changing strategy parameters for individual brokers.

Profitability and cross-broker reproducibility are separate requirements.

A strategy passes only if it satisfies both.

---

# 2. Fundamental Requirement

The strategy must exploit **market structure**, not the accidental characteristics of a particular broker's quote stream.

This distinction is foundational.

## Broker-specific phenomenon

Examples:

* number of ticks received
* exact sequence of ticks
* quote update frequency
* broker spread behaviour
* broker-specific volume
* millisecond timing of individual quotes
* temporary bid/ask anomalies
* broker-specific price spikes

A strategy built around these may work extremely well on the development broker while disappearing elsewhere.

## Market phenomenon

Examples:

* EURJPY appreciated materially over five minutes
* EUR strengthened simultaneously against several currencies
* JPY weakened across several independent pairs
* price broke a previous market structure level
* volatility expanded across a sustained interval
* a move reversed relative to a prior time-window return
* multiple related instruments confirmed the same currency move

These events should be substantially more reproducible across brokers.

The strategy should operate primarily on the second category.

---

# 3. Cross-Broker Reproducibility Is a First-Class Metric

Traditional strategy research asks:

> Is the strategy profitable?

Universal strategy research asks two questions:

> Is it profitable?

and:

> Do independent brokers make the same strategic decision?

A strategy producing excellent results on Broker A but substantially different signals on Broker B must be rejected.

Cross-broker agreement is therefore an optimisation objective alongside:

* expectancy
* profit factor
* drawdown
* stability
* trade frequency

---

# 4. The Universal Strategy Contract

For a strategy to qualify, the following must remain identical across brokers:

**Strategy code**

**Parameters**

**Feature definitions**

**Lookback periods**

**Signal thresholds**

**Entry rules**

**Exit rules**

**State-machine behaviour**

**Risk model**

No configuration such as:

```text
if broker == "FTMO":
    threshold = 0.72

if broker == "FundedNext":
    threshold = 0.64
```

is permitted.

If broker-specific parameter changes are necessary for the signal to survive, the strategy has failed the portability requirement.

Broker-specific configuration is permitted only for execution mechanics such as symbol names, contract specifications and supported order types.

---

# 5. Inputs Allowed in the Strategy

The safest primitive information is:

```text
TIME
BID
ASK
SYMBOL
```

From these:

```text
MID = (BID + ASK) / 2
SPREAD = ASK - BID
```

The strategy should prefer transformations of price over time rather than transformations of quote arrival behaviour.

Recommended inputs include:

```text
mid price
bid
ask
spread
time-window return
range
high
low
open
close
realised volatility
cross-pair price movement
relative currency strength
distance from previous levels
session/time-of-day
```

Inputs should be admitted only after demonstrating high cross-broker agreement.

---

# 6. Inputs That Should Not Drive Core Signals

The following should be considered dangerous until independently proven portable:

```text
raw tick count
ticks per second
tick arrival intervals
tick sequence patterns
MT4 tick volume
MT5 tick volume
broker-reported quote volume
individual quote flags
single-feed bid/ask flickering
millisecond quote bursts
broker-specific spread spikes
broker-specific Last values
broker-specific depth
DOM imbalance from one venue
```

This does not mean these variables contain no information.

It means they cannot be assumed to represent the same phenomenon across brokers.

A feature may be promoted into the universal strategy only after empirical cross-broker validation.

---

# 7. Time Must Define the Observation Window

Avoid:

```python
last_100_ticks
last_500_ticks
last_1000_ticks
```

because 500 ticks do not necessarily represent the same market period across brokers.

Prefer:

```text
last 1 second
last 5 seconds
last 15 seconds
last 1 minute
last 5 minutes
last 15 minutes
last 1 hour
```

Example:

Instead of:

```text
72 of the previous 100 ticks moved upward
```

measure:

```text
EURJPY return during the previous 5 seconds
```

The brokers may have received different numbers of updates, but they observed approximately the same underlying EURJPY movement during those five seconds.

---

# 8. Prefer Robust Price Movement Over Exact Price Equality

Never design a strategic condition requiring:

```python
price == 185.420
```

or:

```python
high == previous_high
```

Independent feeds can legitimately differ slightly.

Use relationships.

For example:

```text
current price > previous high
```

or:

```text
5-minute return > threshold
```

or:

```text
distance from reference level > required movement
```

The strategy should care about the market state, not an exact decimal printed by one broker.

---

# 9. Build Signals With Margin

This is one of the most important design principles.

Suppose the signal is:

```text
BUY if strength > 0.500
```

and feeds calculate:

```text
Broker A = 0.501
Broker B = 0.498
Broker C = 0.503
```

The strategy is structurally unstable.

Tiny measurement differences flip the decision.

Instead, seek conditions where valid signals sit comfortably beyond the decision boundary.

Example:

```text
BUY threshold = 0.50

Typical valid event:

Broker A = 0.73
Broker B = 0.71
Broker C = 0.74
```

The exact measurements differ.

The strategic conclusion does not.

This property will be called **decision margin**.

---

# 10. Decision Margin

For every threshold-based feature, calculate:

```text
distance_to_threshold =
abs(feature - threshold)
```

Signals near the threshold are fragile.

Signals far beyond it are robust.

Research should therefore evaluate not just:

```text
signal = feature > threshold
```

but the distribution of decision margins.

A universal strategy should preferentially trade high-margin events.

---

# 11. Hysteresis

State transitions should not occur repeatedly around a single threshold.

Bad:

```text
strength > 0.50 → bullish
strength < 0.50 → not bullish
```

If feeds observe:

```text
0.499
0.501
0.498
0.502
```

state can diverge rapidly.

Better:

```text
enter bullish state above 0.60

remain bullish until below 0.40
```

The state has separate activation and deactivation boundaries.

This reduces instability caused by small feed differences.

---

# 12. Cross-Pair Confirmation

Single-symbol information is more vulnerable to broker-specific differences.

Currency-level movements can be confirmed through several pairs.

Suppose the hypothesis is:

> JPY is undergoing broad weakness.

Do not infer this solely from EURJPY.

Check independent manifestations such as:

```text
EURJPY ↑
GBPJPY ↑
USDJPY ↑
AUDJPY ↑
```

A broad simultaneous move is much harder to attribute to an anomaly in one symbol.

Likewise EUR strength can be evaluated through multiple EUR pairs.

This converts:

```text
one instrument moved
```

into:

```text
a currency-level market event is occurring
```

---

# 13. Currency Strength Instead of Isolated Pair Noise

A useful architecture is:

```text
EURUSD
EURGBP
EURJPY
EURAUD
    ↓
EUR state

USDJPY
GBPJPY
EURJPY
AUDJPY
    ↓
JPY state
```

Then:

```text
EUR strong
+
JPY weak
=
EURJPY long candidate
```

The strategy becomes dependent on a broader market relationship rather than one broker's EURJPY microstructure.

---

# 14. Consensus Features

Avoid allowing a single noisy feature to determine the trade.

Example:

```text
EUR strength      PASS
JPY weakness      PASS
EURJPY momentum   PASS
volatility        PASS
structure         PASS
spread            PASS
```

Then:

```text
6/6 → strong signal
5/6 → possible signal
3/6 → no trade
```

A better implementation may weight features according to robustness, but the principle remains:

**independent evidence should agree.**

---

# 15. Avoid Excessively Precise Entry Timing

A rule such as:

```text
BUY on the exact tick where X crosses Y
```

is vulnerable to feed ordering.

Broker A may observe:

```text
X
Y
Z
```

Broker B:

```text
X
Z
```

Instead define persistent states.

For example:

```text
Condition A active
AND
Condition B active
AND
Condition C active
for >= required persistence
```

Then enter.

The strategy should respond to a state that exists for a meaningful period, not a single transient quote.

---

# 16. Confirmation Persistence

Example:

```text
EUR strong for >= 3 seconds
JPY weak for >= 3 seconds
EURJPY momentum positive
spread acceptable
```

Rather than:

```text
one quote satisfies all conditions
```

Persistence makes the signal less sensitive to individual feed updates.

However, persistence itself should be measured by time, not number of ticks.

Use:

```text
condition held for 3 seconds
```

not:

```text
condition held for 50 ticks
```

---

# 17. Multi-Timescale Evidence

A robust signal should preferably exist at more than one temporal resolution.

Example:

```text
5 second     momentum positive
30 second    momentum positive
5 minute     structural direction positive
```

or for mean reversion:

```text
5 minute     extreme move
30 second    deceleration
5 second     reversal confirmation
```

This makes the signal less dependent on the exact sequence of individual quotes.

---

# 18. Stable Feature Families

Candidate features should preferentially come from:

## Returns

```text
R_1s
R_5s
R_30s
R_1m
R_5m
R_15m
```

## Range

```text
high(T) - low(T)
```

## Volatility

Price variability over fixed time windows.

## Relative movement

```text
current movement / recent typical movement
```

## Structure

```text
previous high
previous low
session high
session low
rolling high
rolling low
distance from structure
```

## Cross-pair state

```text
EUR strength
JPY strength
relative EUR-JPY strength
```

## Session state

```text
Tokyo
London
New York
session transition
```

All must still be cross-broker tested.

---

# 19. Use Relative Quantities Where Appropriate

Absolute values can be fragile.

Instead of:

```text
price moved 0.013
```

a strategy may use:

```text
movement / recent volatility
```

Instead of:

```text
spread < fixed raw price
```

consider:

```text
spread / normal spread
```

Instead of asking whether a move is large in absolute terms, ask whether it is large relative to the recent market regime.

The objective is to describe the market phenomenon rather than one feed's exact numerical representation.

---

# 20. OHLC Construction Must Be Defined Precisely

If the strategy uses candles, candle construction must be identical.

Specify:

```text
price source = MID
interval = fixed wall-clock interval
boundary = UTC
```

Example:

```text
12:30:00.000 ≤ t < 12:35:00.000
```

defines the 12:30 five-minute bar.

Do not allow platform defaults to silently determine candle behaviour.

Python and MQL implementations must share the same:

```text
timezone
boundaries
price source
bar-close semantics
missing-data behaviour
```

---

# 21. Time Standard

Internally use:

```text
UTC
```

Session logic should be defined explicitly.

Never define strategy logic as:

```text
broker hour == 8
```

because server time differs between brokers.

Instead:

```text
UTC timestamp
    ↓
explicit session definition
```

Daylight-saving behaviour must be explicitly defined where relevant.

---

# 22. Closed Information Only

If a strategy operates on completed intervals, it must never accidentally consume future information.

At:

```text
10:05:00
```

a completed 5-minute calculation may use:

```text
10:00:00 → 10:04:59.999...
```

but not information arriving after the decision time.

Python backtests and live MQL execution must obey identical information boundaries.

This prevents apparent platform disagreement that is actually a look-ahead error.

---

# 23. One Mathematical Specification

Every feature needs a formal definition independent of programming language.

Example:

```text
mid(t) = (bid(t) + ask(t)) / 2

return(T) =
(mid(now) / mid(now - T)) - 1
```

Then implement that specification separately in Python/MQL if necessary.

Never rely on:

> approximately the same indicator.

The mathematical operation must be identical.

---

# 24. Floating-Point Robustness

Never allow irrelevant floating-point differences to change strategic decisions.

Avoid equality tests:

```python
x == threshold
```

Prefer explicit inequalities and meaningful decision margins.

Where price-grid comparisons matter, compare using the instrument's defined tick size.

Strategic thresholds should be much larger than floating-point noise.

---

# 25. Indicator Reproducibility

Built-in platform indicators should not automatically be assumed equivalent.

For example, two implementations called:

```text
ATR(14)
```

may differ because of:

```text
initialisation
bar selection
price source
rounding
warm-up
missing bars
update timing
```

For every important indicator, specify the formula explicitly.

For example:

```text
input
window
initialisation
update rule
warm-up requirement
output timing
```

Python and MQL must produce matching outputs on identical input data before live testing.

---

# 26. State-Machine Reproducibility

The strategy must define state explicitly.

Example:

```text
IDLE
↓
CANDIDATE
↓
CONFIRMED
↓
POSITION_OPEN
↓
EXIT_PENDING
↓
IDLE
```

Transitions must have deterministic conditions.

Avoid hidden differences where one implementation evaluates conditions in another order.

Specify:

```text
event ordering
feature update
state update
signal evaluation
order creation
```

---

# 27. Separate Strategy From Execution

The strategy produces:

```text
TradeIntent
```

Example:

```text
symbol = EURJPY
direction = LONG
signal_time = ...
risk = ...
stop_distance = ...
target_distance = ...
expiry = ...
```

It does NOT produce broker-specific API instructions.

Then:

```text
TradeIntent
      ↓
Execution Adapter
      ↓
MT4 / MT5 / FIX / other API
```

This keeps broker mechanics out of strategy logic.

---

# 28. Broker Differences Belong in Execution

The execution layer may handle:

```text
symbol mapping
digits
point
tick size
contract size
minimum lot
maximum lot
lot step
minimum stop distance
market hours
supported filling mode
order type
commission
margin requirements
```

These differences should not determine whether the market signal exists.

They determine whether and how the signal can be executed.

---

# 29. Execution Eligibility

A universal signal does not imply every broker must execute under every condition.

Example:

```text
Universal signal = BUY
```

but Broker C currently has an abnormally large spread.

Then:

```text
Signal: BUY
Execution: REJECTED_BY_COST_FILTER
```

This is preferable to altering the signal.

The strategy conclusion remains identical.

Execution feasibility differs.

---

# 30. Entry Intent Should Not Depend on One Broker's Absolute Quote

Prefer:

```text
BUY market
SL distance = X
TP distance = Y
```

rather than defining all brokers using an absolute entry quote from the development broker.

Each execution adapter observes its own executable price.

This separates:

```text
market prediction
```

from:

```text
execution price
```

---

# 31. Cost Model

A strategy cannot be called universal merely because signals match.

It must survive realistic differences in:

```text
spread
commission
slippage
latency
swap where relevant
```

For each trade:

```text
gross edge
-
spread
-
commission
-
slippage
=
net edge
```

The expected edge must remain positive across the intended execution environments.

---

# 32. Minimum Edge-to-Cost Requirement

Very small expected edges are inherently difficult to transport.

Suppose:

```text
expected gross move = 2.0 pips
```

while broker-to-broker execution differences can consume a meaningful fraction of that.

The strategy may be signal-portable but not economically portable.

Research should therefore track:

```text
expected move / expected transaction cost
```

and reject signals whose economics are too dependent on exceptionally favourable execution.

---

# 33. Development Dataset Requirement

Never develop a universal strategy using only one broker.

Minimum recommended research setup:

```text
Broker A
Broker B
Broker C
```

Preferably with independent quote sources.

Data should cover the same:

```text
symbols
dates
sessions
market regimes
```

The feeds should be recorded simultaneously where possible.

---

# 34. Never Optimise on One Broker First

Avoid:

```text
optimise on Broker A
↓
discover strategy
↓
test Broker B
```

This encourages feed overfitting.

Instead:

```text
candidate parameter
       │
       ├→ Broker A
       ├→ Broker B
       └→ Broker C
              ↓
        joint evaluation
```

Parameters should be selected based on aggregate robustness.

---

# 35. Multi-Broker Objective Function

Do not maximise:

```text
PnL_A
```

Use an objective incorporating all feeds.

Conceptually:

```text
Score =
profitability
×
cross-broker agreement
×
stability
×
out-of-sample robustness
```

A candidate with:

```text
A = excellent
B = poor
C = poor
```

should lose to:

```text
A = good
B = good
C = good
```

---

# 36. Signal Agreement

For every candidate strategy calculate signal agreement.

Suppose Broker A generates a long event around:

```text
10:31:20
```

Broker B:

```text
10:31:21
```

Broker C:

```text
10:31:20
```

These may represent the same strategic event.

Define a matching tolerance appropriate to the strategy, such as:

```text
±1 second
```

or:

```text
±5 seconds
```

depending on horizon.

Then calculate:

```text
matched signals / total signals
```

---

# 37. Direction Agreement

Signal matching must also require direction.

Example:

```text
A LONG
B LONG
C LONG
```

= agreement.

But:

```text
A LONG
B SHORT
C LONG
```

is a severe failure even if timestamps align.

Track direction disagreement separately.

It should approach zero.

---

# 38. Feature Agreement

Before comparing trades, compare features.

For every feature:

```text
Broker A value
Broker B value
Broker C value
```

Measure:

```text
correlation
absolute difference
relative difference
rank agreement
threshold agreement
state agreement
```

A feature that varies substantially between brokers should be removed before strategy optimisation.

---

# 39. Feature Admission Test

A candidate feature enters the strategy only if:

**1. It has plausible economic/market meaning.**

**2. It demonstrates high agreement across independent feeds.**

**3. Small feed differences rarely change its categorical interpretation.**

**4. It remains stable across time periods.**

**5. It contributes out-of-sample predictive information.**

This should occur before combining dozens of features into a model.

---

# 40. Leave-One-Broker-Out Validation

Suppose data is available from:

```text
A
B
C
D
```

Develop using:

```text
A + B + C
```

Then test unchanged on:

```text
D
```

Rotate:

```text
train A+B+D → test C
train A+C+D → test B
train B+C+D → test A
```

This directly tests whether the strategy learned market behaviour or broker identity.

---

# 41. Broker-Blind Test

The research engine should ideally hide broker identity from the strategy.

The strategy receives:

```text
symbol
time
bid
ask
derived universal features
```

not:

```text
broker_name
```

If the strategy does not know which broker generated the input, it cannot intentionally adapt its prediction to a specific feed.

---

# 42. Parameter Plateau Requirement

Do not select a strategy merely because:

```text
threshold = 0.7317
```

produced exceptional results.

Test neighbouring parameters:

```text
0.65
0.67
0.69
0.71
0.73
0.75
0.77
```

Look for a broad region where performance remains acceptable across all brokers.

A sharp isolated optimum is suspicious.

A plateau is substantially more desirable.

---

# 43. Time-Window Plateau Requirement

Likewise, if:

```text
lookback = 17 seconds
```

works but:

```text
15 seconds
20 seconds
25 seconds
```

fail, investigate heavily.

Prefer phenomena surviving reasonable changes in horizon.

For example:

```text
15s → works
20s → works
30s → works
```

This suggests a market effect rather than a precise artefact.

---

# 44. Cross-Symbol Validation

If the economic mechanism should apply broadly, test related symbols.

A currency-strength mechanism involving JPY should be examined across:

```text
EURJPY
GBPJPY
USDJPY
AUDJPY
```

It need not have identical profitability everywhere.

But the underlying feature should behave coherently.

If an allegedly universal currency phenomenon exists only in one symbol on one broker, that is a warning sign.

---

# 45. Regime Validation

Test:

```text
Tokyo
London
New York

low volatility
normal volatility
high volatility

trend
range

news-heavy days
ordinary days

month beginning
month end
```

The strategy may intentionally operate only in one regime.

That is acceptable.

What is unacceptable is accidentally depending on a tiny historical regime without recognising it.

---

# 46. Session Strategies Are Allowed

A strategy may intentionally trade:

```text
Tokyo open
London open
New York overlap
fixing windows
session transitions
```

provided session time is defined independently of broker server time and the phenomenon survives across feeds.

A session effect is not automatically broker-specific.

It can represent genuine global market structure.

---

# 47. News Handling

News can produce:

```text
feed divergence
spread expansion
quote gaps
latency
slippage
```

A universal strategy must explicitly decide whether to:

```text
trade news
```

or:

```text
exclude specified event windows
```

Do not allow accidental behaviour.

If news is traded, portability tests must include realistic execution differences.

---

# 48. Missing Data

Define deterministic behaviour when:

```text
one symbol stops updating
connection drops
one cross-pair feed is stale
spread becomes invalid
timestamp jumps
market reopens
```

Recommended default:

```text
insufficient trustworthy input
→
NO NEW SIGNAL
```

Do not silently substitute stale information.

---

# 49. Warm-Up

Every feature must define required history.

Example:

```text
5-minute return requires sufficient 5-minute history
60-minute volatility requires sufficient 60-minute history
```

Until all mandatory features are ready:

```text
state = WARMUP
```

No trades.

This behaviour must match in backtest and live environments.

---

# 50. Deterministic Processing Order

For each incoming observation, define:

```text
1. validate timestamp
2. update market state
3. update time-window structures
4. calculate features
5. update strategy state
6. evaluate signal
7. create TradeIntent
8. pass to risk/execution layer
```

Python and MQL must follow the same logical ordering.

---

# 51. Backtest/Live Parity

The same strategy logic should operate in:

```text
historical replay
paper trading
live trading
```

Do not create:

```text
backtest_strategy.py
```

and independently recreate its logic in:

```text
live_strategy.mq5
```

unless strict parity testing exists.

Ideally the strategy specification is implemented once or generated from a common specification.

If multiple implementations are unavoidable, identical input sequences must produce identical strategic outputs.

---

# 52. Golden Replay Tests

Maintain a small reference dataset.

For example:

```text
30 minutes
several symbols
known market events
```

Record expected:

```text
features
states
signals
```

Every implementation must reproduce these outputs.

Example:

```text
Python PASS
MQL5 PASS
MQL4 PASS
```

before deployment.

This detects implementation differences independently of broker differences.

---

# 53. Separate Three Types of Failure

Every disagreement must be classified as:

## DATA FAILURE

The feeds observed materially different market information.

## LOGIC FAILURE

Python/MQL4/MQL5 calculated different results from equivalent information.

## EXECUTION FAILURE

The same signal occurred but execution differed.

Never treat all three as simply:

> strategy doesn't work.

They require different solutions.

---

# 54. Required Research Pipeline

Every candidate strategy should pass:

```text
IDEA
 ↓
Feature Definition
 ↓
Cross-Broker Feature Test
 ↓
Reject unstable features
 ↓
Signal Construction
 ↓
Cross-Broker Signal Test
 ↓
Profitability Test
 ↓
Parameter Stability
 ↓
Leave-One-Broker-Out Test
 ↓
Out-of-Sample Test
 ↓
Execution Simulation
 ↓
Paper Trading
 ↓
Small Live Validation
 ↓
Deployment
```

A failure at an early portability stage should prevent expensive downstream optimisation.

---

# 55. Portability Gate

Before profitability optimisation, require:

```text
Feature agreement        HIGH
Direction agreement      VERY HIGH
Signal agreement         VERY HIGH
Timing agreement         HIGH
```

Exact targets depend on strategy horizon, but for the stated objective the research process should aim toward:

```text
direction agreement ≈ 100%
signal-event agreement ≈ 95–100%
```

These are research targets, not guarantees.

A strategy producing 70% signal agreement should not be labelled universal merely because both brokers happen to be profitable.

---

# 56. Profitability Gate

After portability passes:

```text
positive expectancy
acceptable drawdown
acceptable transaction costs
acceptable slippage sensitivity
out-of-sample profitability
multi-broker profitability
```

must also pass.

Signal agreement without economic edge is useless.

---

# 57. Robustness Gate

Perturb:

```text
thresholds
lookbacks
entry delays
spread assumptions
slippage
timestamps
execution price
```

slightly.

The strategy should degrade gradually.

Example:

```text
+100 ms → works
+250 ms → works
+500 ms → weaker
+1 sec → weaker
```

is healthier than:

```text
0 ms → highly profitable
+100 ms → loses money
```

unless the strategy is intentionally latency-sensitive, in which case retail cross-broker universality is probably the wrong deployment objective.

---

# 58. Delayed-Entry Test

Artificially delay signals during testing:

```text
0 ms
100 ms
250 ms
500 ms
1 second
2 seconds
```

Measure expectancy.

This reveals whether the strategy captures a persistent market phenomenon or relies on an execution opportunity too brief to transport between environments.

---

# 59. Price-Perturbation Test

Alter entry prices adversely:

```text
+0.1 pip
+0.2 pip
+0.5 pip
+1.0 pip
```

for longs, with equivalent adverse treatment for shorts.

If tiny perturbations destroy profitability, the strategy has little portability margin.

---

# 60. Spread Stress Test

Evaluate using:

```text
normal spread
1.25× spread
1.5× spread
2× spread
```

The strategy does not need to survive arbitrary costs.

But the expected edge should not depend on one unusually cheap broker unless that broker is explicitly the only deployment target.

---

# 61. Slippage Stress Test

Simulate realistic adverse slippage.

Evaluate both:

```text
average slippage
tail slippage
```

because execution tails can dominate short-horizon strategies.

---

# 62. Signal Timing Distribution

Do not report only:

```text
94% matched
```

Report timing differences:

```text
median
P90
P95
P99
maximum
```

Example:

```text
median = 120 ms
P95    = 480 ms
```

This tells us whether brokers agree on the event but detect it at slightly different times.

---

# 63. Cross-Broker Trade Identity

Assign a strategic event ID independent of broker execution.

Example:

```text
event_id = EURJPY_LONG_20260728_103120
```

Then record:

```text
Broker A execution
Broker B execution
Broker C execution
```

against that same event.

This makes discrepancies directly measurable.

---

# 64. Universal Strategy Telemetry

For every signal log:

```text
UTC timestamp
symbol
direction

all feature values
all feature states
decision margins

strategy state
signal reason

local bid
local ask
local spread

execution decision
execution price
slippage
result
```

Without this, cross-broker debugging becomes guesswork.

---

# 65. Disagreement Logger

Whenever brokers disagree:

```text
A = BUY
B = NO TRADE
```

capture exactly why.

Example:

```text
EUR strength
A = 0.72
B = 0.71

JPY weakness
A = 0.64
B = 0.63

momentum
A = PASS
B = PASS

structure
A = PASS
B = FAIL
```

Now the research process can identify the unstable component.

---

# 66. Disagreement-Driven Research

The most valuable samples may be the cases where brokers disagree.

Create a dataset containing only:

```text
A signal / B no signal
A long / B short
A signal 3 seconds before B
```

Then determine:

```text
which feature caused divergence?
```

Features repeatedly responsible for disagreement should be redesigned or removed.

---

# 67. Broker Diversity

Using five accounts that ultimately receive extremely similar pricing is less informative than several genuinely independent sources.

Research should seek diversity in:

```text
broker
liquidity arrangement
platform
pricing infrastructure
```

The objective is to test whether the phenomenon exists in the market rather than merely across several copies of a similar feed.

---

# 68. Never Optimise for Broker Identity

Do not create features that allow a model to infer:

```text
this looks like Broker A
```

from:

```text
tick frequency
spread signature
quote formatting
timestamp behaviour
```

If machine learning is used, broker-identification tests should be performed.

If broker identity is easily predictable from the feature set, investigate whether those features belong in a universal strategy.

---

# 69. ML-Specific Requirement

For machine-learning strategies, training data should contain multiple brokers.

Avoid:

```text
train Broker A
test Broker A
deploy Broker B
```

Prefer:

```text
train A+B+C
validate D
```

and rotate.

The target should represent future market movement, not broker-specific quote behaviour.

---

# 70. Cross-Broker Label Construction

If supervised learning is used, labels should describe market outcomes robustly.

Example:

```text
did EURJPY move upward sufficiently during the next 5 minutes?
```

rather than:

```text
was Broker A's next tick upward?
```

The former attempts to predict market movement.

The latter may learn feed mechanics.

---

# 71. Confidence Should Represent Market Agreement

A useful signal confidence model could combine:

```text
cross-pair confirmation
multi-timescale confirmation
distance from thresholds
market structure
volatility regime
```

High confidence should mean:

> multiple independent descriptions of the market support the same conclusion.

It should not mean:

> one broker-specific feature reached an extreme value.

---

# 72. Example Universal Signal

Conceptual example only:

```text
EURJPY LONG CANDIDATE
```

Requirements:

```text
EUR strength          > strong threshold
JPY strength          < weak threshold

EURJPY 5s return      > threshold
EURJPY 30s return     > threshold

GBPJPY confirmation   bullish
USDJPY confirmation   bullish

market structure      bullish
volatility regime     allowed

conditions persist    >= required time
```

Then:

```text
LONG
```

The exact values should be discovered jointly across multiple broker datasets.

---

# 73. Example Mean-Reversion Signal

Conceptual structure:

```text
large time-window move
+
cross-pair evidence move is excessive
+
short-horizon deceleration
+
reversal state persists
+
execution cost acceptable
```

Rather than:

```text
exact tick pattern XYZ occurred
```

Again, the mechanism is expressed through persistent market state.

---

# 74. Architecture

```text
             BROKER A DATA
                   │
             BROKER B DATA
                   │
             BROKER C DATA
                   │
                   ▼

        UNIVERSAL MARKET FEATURES
                   │
                   ▼
          STRATEGY STATE MACHINE
                   │
                   ▼
              TRADE INTENT
                   │
                   ▼
               RISK ENGINE
                   │
                   ▼
           EXECUTION ADAPTER
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
       MT4        MT5        FIX
```

During production, each environment can calculate the same strategy from its local market observations.

During research, independent feeds are evaluated simultaneously to verify that those observations lead to the same strategic state.

---

# 75. Cross-Broker Test Matrix

Every release should produce something similar to:

| Metric                   |   Broker A |   Broker B |   Broker C |
| ------------------------ | ---------: | ---------: | ---------: |
| Strategic events         |      1,002 |        995 |      1,007 |
| Matched events           |          — |        982 |        987 |
| Direction agreement      |          — |      99.8% |      99.7% |
| Median timing difference |          — |     180 ms |     220 ms |
| Win rate                 |      68.2% |      67.7% |      68.0% |
| Expectancy               |   positive |   positive |   positive |
| Profit factor            |     stable |     stable |     stable |
| Drawdown                 | acceptable | acceptable | acceptable |

Additionally produce pairwise matrices for:

```text
signal agreement
direction agreement
feature correlation
PnL correlation
```

---

# 76. Acceptance Criteria

A strategy may be called **Cross-Broker Qualified** only when:

### Data/Feature

* core features show strong cross-broker agreement
* no critical signal depends on raw quote frequency
* no critical signal depends on broker identity
* time calculations are UTC-defined
* feature formulas are deterministic

### Signal

* very high event agreement
* near-perfect direction agreement
* timing differences are small relative to trade horizon
* disagreements are understood

### Strategy

* same parameters across brokers
* same state machine
* same mathematical rules
* no broker-specific signal exceptions

### Implementation

* Python/MQL4/MQL5 pass golden replay tests
* historical/live calculations match
* no hidden platform indicator differences

### Economics

* positive expectancy across target brokers
* survives realistic spread
* survives realistic slippage
* survives modest entry delay
* survives reasonable parameter perturbation

### Validation

* multi-broker development
* unseen broker test
* out-of-sample period
* regime tests
* paper/live forward validation

---

# 77. What "Near 100%" Should Mean

Do not require identical raw:

```text
ticks
prices
spreads
timestamps
```

That is unrealistic.

Require near-identical:

```text
market interpretation
strategy state
signal direction
strategic event
```

Example:

```text
Broker A:
10:31:20.4 LONG

Broker B:
10:31:20.8 LONG

Broker C:
10:31:21.0 LONG
```

This is effectively the same strategic decision for a strategy whose expected move unfolds over minutes.

The acceptable timing tolerance must be small relative to the strategy horizon.

---

# 78. What We Are Explicitly Not Doing

This project will NOT attempt to:

```text
make FTMO ticks look like Exness ticks

invent missing ticks

resample feeds merely to force agreement

modify prices to match another broker

learn broker-specific correction factors

use different thresholds per broker

optimise each broker separately and call it one strategy
```

Agreement must arise because the **underlying strategy is robust**, not because inputs were artificially forced to match.

---

# 79. Strategy Discovery Philosophy

The discovery process should ask:

> What market events are so clear that independent brokers observing the same FX market reach the same conclusion?

Not:

> What combination of features maximises the backtest on this dataset?

That shift changes the optimisation problem fundamentally.

Search for:

```text
persistent effects
large decision margins
multi-pair confirmation
multi-timescale confirmation
stable parameter regions
cross-broker feature agreement
cross-broker signal agreement
```

and reject:

```text
fragile thresholds
isolated parameter peaks
single-tick effects
broker-specific activity
exact-price coincidences
extreme timing sensitivity
```

---

# 80. Final Design Principle

The universal strategy should not attempt to make brokers identical.

They are not identical.

Instead, it should deliberately ignore differences that do not matter to the underlying market phenomenon.

The hierarchy is:

```text
RAW BROKER OBSERVATION
          ↓
ROBUST MARKET FEATURES
          ↓
MARKET STATE
          ↓
STRATEGIC EVENT
          ↓
TRADE INTENT
          ↓
BROKER-SPECIFIC EXECUTION
```

The higher we move through this hierarchy, the greater the expected agreement.

The research goal is therefore:

> **Find predictive market states whose existence does not depend materially on which legitimate FX broker observed them.**

If such a state produces:

```text
FTMO      LONG
FundedNext LONG
Exness    LONG
Broker X  LONG
```

from their independently observed market data, using the same formulas and parameters, then the strategy has achieved the intended form of universality.

The final proof is not theoretical.

It is:

```text
same period
+
independent feeds
+
same code/specification
+
same parameters
+
near-identical strategic events
+
positive expectancy on every target environment
+
successful unseen-broker validation
```

Only after all of those conditions pass should the strategy be considered deployment-ready.

---

# 81. Non-Negotiable Rule

**No feature enters the final strategy merely because it improves PnL.**

It must improve PnL **and survive cross-broker validation**.

If removing a broker-sensitive feature reduces the backtest from:

```text
80% WR → 72% WR
```

but raises cross-broker signal agreement from:

```text
65% → 97%
```

the 72% strategy is the stronger universal candidate.

A strategy we can reproduce is more valuable than an exceptional backtest that exists only on one feed.

---

# 82. Definition of Success

Success is not:

> We found settings that work on FTMO, FundedNext and Broker A.

Success is:

> We identified a market phenomenon, encoded it using broker-robust information, and demonstrated that independently observed feeds consistently reach the same trading decision without broker-specific strategy parameters.

That is the standard against which every future strategy candidate should be measured.
