# Proxima Market State Graph — Complete Architecture Proposal v2

## Prologue: Design Philosophy

**Core principle:** The engine should behave like a database — it restores its state, not wakes up empty. Historical market memory + current market stream = continuous market state.

**Architectural rule:** Pairs are observations. Currencies are latent variables. Timeframes are measurement scales. Direction is a derived hypothesis. DRS is capital allocation. Execution is the final actuator.

**Operational rule:** The engine never starts empty. At startup it reconstructs world state from memory (snapshot + archive), validates it, then attaches to live flow. Zero warm-up, deterministic recovery, instant trade-capable state.

**Session model:** Proxima is not a continuously running daemon. It is a persistent market-state machine that wakes up, reconstructs reality, operates, commits memory, and sleeps.

---

# 1. Complete System Overview

```
+================================================================================+
|                              PROXIMA ENGINE                                   |
|                         STARTUP / RUNTIME CONTROLLER                          |
+================================================================================+

                                      |
                                      v
                         +------------------------+
                         |   Bootstrap Controller  |
                         +------------------------+

                                      |
                    +-----------------+------------------+
                    |                                    |
                    v                                    v
          +--------------------+              +----------------------+
          | Snapshot Manager   |              | Historical Replay     |
          |--------------------|              | Engine                |
          | load / save /      |              | recent + deep replay  |
          | validate / checksum|              | deterministic         |
          +--------------------+              +----------------------+

                    |                                    |
                    +----------------+-------------------+
                                     v
                         +----------------------+
                         | State Reconstruction |
                         |----------------------|
                         | Market State Graph   |
                         | Currency States      |
                         | Temporal States      |
                         | Regime States        |
                         +----------------------+

                                     v
                         +----------------------+
                         | Live Handoff Manager |
                         +----------------------+

                                     v
                              LIVE RUNTIME


                         LIVE RUNTIME PIPELINE

                    MT5 Live Tick Feed
                         |
                         v
                    Tick Receiver
                         |
                         v
                    Tick Normalizer
                         |
                         v
                    Tick Buffer
                         |
                         v
                    Pair Feature Engine
                         |
                         v
                    Temporal State Engine
                         |
                         +----------------------------+
                                                      v
                                         +--------------------------+
                                         | Currency Attribution     |
                                         | Engine                   |
                                         +--------------------------+
                                                      v
                                         +--------------------------+
                                         | Currency State Matrix    |
                                         +--------------------------+

                                                      v
                         +---------------------------------------+
                         |          MARKET STATE GRAPH            |
                         | Currency Nodes / Pair Edges /          |
                         | Temporal States / Regimes / Residuals  |
                         +---------------------------------------+
                                                      v
                         +---------------------------------------+
                         | Direction Hypothesis Engine           |
                         +---------------------------------------+
                                                      v
                         +---------------------------------------+
                         | DRS (ranking / portfolio / displace)  |
                         +---------------------------------------+
                                                      v
                         +---------------------------------------+
                         | Risk Engine                           |
                         +---------------------------------------+
                                                      v
                         +---------------------------------------+
                         | Execution Router                      |
                         +---------------------------------------+
                                                      v
                                  MT5 Order Execution
```

---

# 2. Market State Graph

## 2.1 Mathematical Foundation

The core equation:

```
P = A * C + ε
```

Where:
- **P**: 28×1 pair movement vector (observed)
- **A**: 28×8 currency incidence matrix (fixed structural topology)
- **C**: 8×1 currency strength vector (latent variables to solve)
- **ε**: 28×1 residual vector (first-class state, NOT discarded noise)

### Incidence Matrix A

The FX universe:
```
0: USD
1: EUR
2: GBP
3: JPY
4: AUD
5: NZD
6: CAD
7: CHF
```

Each row maps a pair to its base/quote currencies:

```
EURUSD:  [-1,  1,  0,  0,  0,  0,  0,  0]   = EUR strength - USD strength
USDJPY:  [ 1,  0,  0, -1,  0,  0,  0,  0]   = USD strength - JPY strength
GBPUSD:  [-1,  0,  1,  0,  0,  0,  0,  0]   = GBP strength - USD strength
EURJPY:  [ 0,  1,  0, -1,  0,  0,  0,  0]   = EUR strength - JPY strength
... (28 rows total)
```

### Currency Solving

Solved using Constrained Weighted Least Squares:

```
C = (A^T * W * A)^-1 * A^T * W * P
```

Subject to constraint: SUM(C) = 0 (global currency strength averages to zero)

Where W = diagonal weight matrix based on pair quality:
```
W_i = liquidity_i × volatility_quality_i × spread_quality_i × independence_i
```

Implementation: `numpy.linalg.lstsq()` with constraint row added.

## 2.2 Residual State (ε_pair as First-Class Object)

Every pair edge stores:

```python
@dataclass
class PairResidualState:
    symbol: str
    timestamp: int
    observed_return: float
    currency_explained_return: float
    epsilon: float                      # residual = observed - explained
    epsilon_zscore: float                # normalized residual
    residual_velocity: float
    residual_persistence: float
    anomaly_type: str                    # NORMAL | CURRENCY_MISFIT | EVENT_SHOCK | LIQUIDITY_ANOMALY
    confidence: float
```

### Residual Classification Logic

```python
def classify_residual(epsilon_z, persistence, event_state):
    if event_state:
        return "EVENT_SHOCK"
    if abs(epsilon_z) < 2:
        return "NORMAL"
    if persistence > 0.7:
        return "PAIR_ANOMALY"
    return "CURRENCY_MISFIT"
```

**Decision rules**:
- If MANY pairs have high residual → currency model is wrong
- If ONLY ONE pair has high residual → pair-specific event/anomaly

## 2.3 Graph Internal Structure

```
                         MARKET STATE GRAPH

              +--------------------------------+
              |       Currency Nodes           |
              +--------------------------------+

                    |       |       |
             +------+   +------+   +------+
             | EUR  |   | USD  |   | JPY  |
             +------+   +------+   +------+

                 \          |          /
                  \         |         /
                   \        |        /
                    \       |       /

              +--------------------------------+
              |          Pair Edges             |
              |--------------------------------|
              EURUSD    EURJPY    EURGBP
              EURAUD    USDJPY    GBPUSD
              ...

Each node stores:                    Each edge stores:
+--------------------------+        +--------------------------+
| CurrencyState            |        | PairState                |
|--------------------------|        |--------------------------|
| market_strength          |        | direction                |
| macro_pressure           |        | confidence               |
| event_pressure           |        | volatility               |
| combined_strength        |        | energy                   |
| velocity                 |        | residual (epsilon)       |
| acceleration             |        | timeframe states         |
| confidence               |        | age                      |
| uncertainty              |        +--------------------------+
| regime                   |
| age                      |
+--------------------------+
```

### Root Data Structure

```python
@dataclass
class MarketStateGraph:
    currencies: dict[str, CurrencyNode]     # 8 nodes
    pairs: dict[str, PairNode]              # 28 edges
    currency_pair_matrix: np.ndarray        # 28x8 incidence matrix A
    currency_state_matrix: np.ndarray       # 8x1 current strengths
    pair_state_matrix: np.ndarray           # 28x1 current pair movements
    timestamp: float
    epoch_id: int
```

### Currency State with Macro Fusion

```python
@dataclass
class CurrencyState:
    market_strength: float        # from WLS solve of pair returns
    macro_pressure: float         # from yields, rates, commodities
    event_pressure: float         # from calendar events, news
    combined_strength: float      # weighted fusion
    confidence: float
    uncertainty: float
    velocity: float
    acceleration: float
    regime: str
    age_seconds: float
```

Fusion formula:

```
C_total = α * C_market + β * C_event + γ * C_macro

where α + β + γ = 1, weights depend on confidence and regime
```

---

# 3. Temporal State Engine

## 3.1 TemporalState Data Model

```python
@dataclass
class TemporalState:
    symbol: str
    timeframe: str                    # "5m" | "1h" | "4h"
    clock_type: str                   # "FIXED" | "EVENT"
    timestamp: float

    # direction
    direction: float                  # [-1, +1]
    confidence: float                 # [0, 1]

    # movement
    return_value: float
    volatility: float
    directional_energy: float
    velocity: float
    acceleration: float

    # persistence
    persistence_score: float
    state_age_seconds: float
    decay_factor: float

    # regime
    regime: str                       # TREND | RANGE | TRANSITION | CRISIS
    volatility_regime: str

    # uncertainty
    entropy: float
    conflict_score: float

    # provenance
    sample_count: int
    last_update: float
    bootstrap_confidence: float
```

The schema is identical across timeframes. Only configuration changes:

| Timeframe | Window Ticks | Half-Life |
|---|---|---|
| 5m | 500 | 30 min |
| 1h | 5000 | 8 hours |
| 4h | 20000 | 48 hours |

## 3.2 Multi-Timeframe Parallel Processing

```
                         Pair EURUSD
                             |
        +--------------------+--------------------+
        |                    |                    |
        v                    v                    v

+---------------+    +---------------+    +---------------+
|     5 MIN     |    |     1 HOUR    |    |     4 HOUR    |
+---------------+    +---------------+    +---------------+
direction           direction           direction
energy              energy              energy
confidence          confidence          confidence
regime              regime              regime

        |                    |                    |
        +--------------------+--------------------+
                             |
                             v
                 +------------------------+
                 | Temporal Fusion Layer |
                 |------------------------|
                 | short-term state       |
                 | medium-term state      |
                 | long-term state        |
                 +------------------------+
                             |
                             v
                  Pair State Final Output
```

## 3.3 State Decay Formula

```
decay = exp(-market_elapsed_time / half_life)
effective_signal = state_value × decay
```

Two clocks per state:
- `created_market_time`: when state was born
- `last_update_market_time`: when state was last confirmed

Decay uses `current_market_time - last_state_time`, NOT engine uptime.

---

# 4. Currency Attribution Engine

## 4.1 Speed Hierarchy

Three attribution vectors:

| Layer | Frequency | Purpose |
|---|---|---|
| Micro attribution | Every tick aggregated | Short-term imbalance |
| Fast (intraday) | Every 5 seconds | Current session dynamics |
| Slow (macro) | Every 1 hour / 4h bar | Regime-level strength |

## 4.2 Attribution Engine Architecture

```
                    Pair Return Vector
                         |
                         v
              +------------------------+
              | Currency Attribution    |
              | Engine                  |
              |-------------------------|
              | Input: 28 pair returns  |
              | EURUSD, GBPUSD, USDJPY |
              | EURJPY, EURGBP, ...    |
              +-------------------------+
                         |
                         v
              +-------------------------+
              | Linear Solver           |
              |-------------------------|
              | P = A*C + noise         |
              | Weighted Least Squares  |
              | Constraint: SUM(C)=0    |
              +-------------------------+
                         |
                         v
+--------------------------------------------------------------------+
|                    CURRENCY STATE VECTOR                            |
+--------------------------------------------------------------------+
 USD = +0.35    EUR = +0.52    GBP = -0.10    JPY = -0.60
 AUD = +0.20    NZD = +0.15    CAD = -0.05    CHF = -0.47
+--------------------------------------------------------------------+
                         |
                         v
              +-------------------------+
              | Currency State Store    |
              |-------------------------|
              | fast_strength           |
              | intraday_strength       |
              | macro_strength          |
              +-------------------------+
```

**Note**: WLS outputs point estimate AND uncertainty. Bayesian estimation / Kalman filter should replace pure WLS for probability distributions.

## 4.3 External Causal Variables (Breaks Self-Referential Loop)

To break the circular pipeline (pairs → currencies → explain pairs), inject independent anchors:

- **Interest Rate Differentials**: Central bank rates per currency
- **Yield Curves**: 2Y, 10Y, curve slope
- **Commodity Exposure**: Oil (CAD), Iron Ore (AUD), Gold (AUD/CHF)
- **Risk Regime**: VIX, equity index returns, credit spreads
- **Macro Events**: FOMC, ECB, NFP in structured calendar

Fusion:

```
CurrencyState = α * MarketEvidence + (1 - α) * MacroPrior
```

Where α depends on regime (normal: 0.8, event: 0.4).

---

# 5. Bootstrap & Replay Engine

## 5.1 Three-Level Bootstrap

```
LEVEL 1 — Snapshot Restore
           Milliseconds. State available immediately.

LEVEL 2 — Recent Replay (24-48h of ticks)
           Refresh: volatility, directional energy,
           currency power, temporal states, regime.

LEVEL 3 — Deep Background Replay (weeks/months)
           Async worker. Improves entropy baselines,
           regime models, statistics. (REMOVED for intermittent operation —
           used only in research environment)
```

## 5.2 Startup Sequence

```
                              Process Start
                                   |
                                   v
                     +--------------------------+
                     | Bootstrap Controller     |
                     +--------------------------+
                                   |
                                   v
                  +--------------------------------+
                  | Snapshot Manager               |
                  +--------------------------------+
                                   |
                  +----------------+----------------+
                  |                                 |
                  v                                 v
        Snapshot Exists                    No Snapshot
                  |                                 |
                  v                                 v
+--------------------------------+      +-----------------------------+
| Load Snapshot                  |      | Historical Replay Required  |
| Market State Graph             |      | Full reconstruction         |
| Currency Vector                |      | from archive                |
| Temporal States / Buffers      |      |                             |
+--------------------------------+      +-----------------------------+
                  |                                 |
                  v                                 |
        +-----------------------------+              |
        | Snapshot Validation         |              |
        | checksum / schema / ts      |              |
        +-----------------------------+              |
                  |                                 |
          +-------+--------+                        |
          v                v                        |
      VALID              INVALID                    |
          |                |                        |
          v                v                        |
     Load State        Reject Snapshot              |
          |                |                        |
          |                v                        |
          |        Full Historical Replay <---------+
          |
          +-------------------------+
                                    v
                         Market State Ready
                                    |
                                    v
                         Live Tick Handoff
```

## 5.3 Gap Replay Fidelity Tiers

| Gap Duration | Replay Mode | Method |
|---|---|---|
| 0 - 6h | Full Tick Replay | Replay all ticks through full pipeline |
| 6 - 48h | Hybrid | Old gap: M1 bars reconstruct state; Recent gap: tick replay |
| >48h | State Reconstruction | M1 OHLC + spread + volume + events → reconstruct vol/regime/currency |

**Estimated performance**: 24h replay of 28 pairs (~8M ticks) ≈ 80 seconds at 100k ticks/sec.

## 5.4 Gap Replay Decision Engine

```python
def choose_replay_strategy(gap_duration):
    if gap_duration <= 6_hours:
        return TICK_REPLAY
    elif gap_duration <= 48_hours:
        return HYBRID
    else:
        return STATE_RECONSTRUCTION
```

Confidence reduction for non-tick replay: `bootstrap_confidence = 0.7` until live confirmation.

## 5.5 Historical Archive Structure

```
archive/
  hot/                              # 48h ring buffer (full ticks)
    EURUSD/
      2026-07-07.parquet
    GBPUSD/
      2026-07-07.parquet
    USDJPY/
      2026-07-07.parquet
    ...
  warm/                             # 30-90 day state archive
    temporal_state/
      EURUSD/
        5m.parquet
    currency_state/
      currencies.parquet
  cold/                             # Years (M1/M5 bars for research)
    EURUSD/
      M1/
      M5/
```

## 5.6 Streaming Replay (No Full Memory Load)

```
Historical Archive (Parquet)
         |
         v
Historical Reader (streaming chunks, ~100K ticks)
         |
         v
ReplayScheduler (market-time aware, not wall-clock)
         |
         v
Same Pipeline as Live
```

### Replay Scheduler

During replay, use **market timestamp** (not wall clock) for periodic operations:

```
Every 5 seconds of market time → run currency solver
Every 5m bar boundary → update 5m temporal states
Every 1h bar boundary → update 1h temporal states
```

### Replay-Safe vs Replay-Unsafe Components

**Replay-Safe** (must produce identical output):
- Tick Normalizer
- Pair Feature Engine (if no wall-clock usage)
- Temporal State Engine (if uses market timestamp)
- Currency Attribution (pure math)
- Market State Graph (state transition: S_t = f(S_{t-1}, X_t))

**Replay-Unsafe** (disabled during replay):
- Execution Router (latency matters)
- Broker Interaction (slippage, fill delay)
- Order Management
- Live Risk Governor (account state changes)
- External Event Listeners (news websocket, broker heartbeat)

---

# 6. Tiered Storage Architecture

## 6.1 Three Storage Tiers

| Tier | Content | Retention | Purpose |
|---|---|---|---|
| HOT | Full ticks (Parquet) | 48 hours | Exact restart replay |
| WARM | 5m states, currency states, regimes | 30-90 days | Long gap reconstruction |
| COLD | M1/M5 bars + events | Years | Research only |

Key design: **Do not store the entire market forever. Store the level of memory required for each type of decision.**

## 6.2 Hot Tier — Ring Buffer

Fixed-size circular storage using 1-hour partitions:

```
hot_ticks/
  partition_0001.parquet
  partition_0002.parquet
  ...
  partition_0048.parquet
```

When new partition arrives, oldest is deleted. Storage bound is always fixed (48 hours).

## 6.3 Incremental Archive (No Duplicates)

```python
@dataclass
class ArchiveRecord:
    symbol: str
    timestamp: int
    bid: float
    ask: float
    volume: float
    sequence_id: int
```

Every tick checks `last_archived_timestamp` — only appends new ticks. No duplicates. Continuous incremental append (not shutdown-only — crashes lose data otherwise).

## 6.4 I/O Budget

- Raw tick rate (max): ~5000 ticks/sec across 28 pairs
- Raw throughput: ~160 KB/sec
- Daily raw: ~13.8 GB
- Compressed (Parquet, 3-5x): ~3-5 GB/day
- SSD wear: ~1.8 TB/year at 5 GB/day (negligible for 300-600 TBW NVMe)

**Write policy**: Batch writes every 1-5 seconds or 64MB buffer (not per-tick writes).

---

# 7. Macro Event Layer

## 7.1 Event Lifecycle State Machine

```
PRE_EVENT (T-24h) → EXPECTATION → RELEASE → SURPRISE_UPDATE → ABSORPTION → NORMAL
```

## 7.2 Market Event Mutation

```python
@dataclass
class MarketEventMutation:
    event_id: str
    timestamp: int
    event_type: str              # NFP | FOMC | CPI | ECB_RATE | GEOPOLITICAL
    affected_entities: list      # ["USD"]
    surprise_score: float
    importance: float
    duration_half_life: int
    propagation_profile: dict
```

**Surprise calculation**:

```
Surprise = (Actual - Forecast) / σ_historical
```

**Anticipation effect**: Event pressure begins before release. Before release, use `ExpectedPressure` based on forecast deviation, options positioning, yield movement.

### Graph Mutation on Event

```
NFP EVENT
     |
     v
Event Mutation Engine
     |
  +--+--+
  |     |
  v     v
USD   Volatility
node  node
  |     |
  +--+--+
     |
     v
Pair Edge Reweighting
```

The incidence matrix A does NOT change — the relationship "EURUSD = EUR - USD" is structural. Events change the currency node states, not the graph topology.

## 7.3 Market Event Calendar Integration

Engine loads economic calendar at startup (not just MT5):

```python
@dataclass
class EconomicEvent:
    id: str
    currency: str
    scheduled_time: int
    importance: int              # 1-3
    forecast: float
    previous: float
    actual: float | None
```

## 7.4 Cross-Asset Integration (Separate Latent Factor Graph)

```
    Cross Asset State

Bonds    Equities    Commodities
  |          |           |
  +----------+-----------+
             |
             v
      Currency Graph
```

```python
@dataclass
class CrossAssetState:
    asset: str          # "US10Y"
    value: float
    velocity: float
    shock: float
    currency_links: dict  # {USD: 0.8, JPY: -0.3}
```

Propagation: `CurrencyPressure = B × AssetState` (B = influence matrix of shape 8×n_assets).

## 7.5 Session State Node

```python
@dataclass
class SessionState:
    session: str                # ASIA | LONDON | NY
    volatility_multiplier: float
    liquidity_score: float
    historical_bias: float
```

Not hardcoded — learned via rolling statistics per session/weekday/month-phase.

## 7.6 State Circuit Breaker (Protects Graph During Extreme Events)

Detection signals:
- Tick sanity (reject impossible moves: EURCHF 1.20 → 0.80)
- Spread explosion (normal 1 pip → 200 pips)
- Residual explosion (epsilon 0.2 → 8.0)

Protocol:

```
FLASH EVENT → Freeze Graph → Quarantine Inputs → Build Emergency State → Slow Recovery
```

Recovery requires confirmation window (e.g., 500 normal ticks + stable spread + residual recovery).

---

# 8. Direction Hypothesis & Arbitration

## 8.1 Arbitration Flow

```
                  Pair State                     Currency State
                      |                               |
                      v                               v
             +------------------+           +------------------+
             | Pair Direction   |           | Currency Field   |
             |------------------|           |------------------|
             | EURUSD           |           | EUR +0.40       |
             | BUY +0.70        |           | USD -0.20       |
             | confidence .80   |           +------------------+
             +------------------+                    |
                      |                              |
                      +------------------------------+
                               |
                               v

              Currency Explanation
              EURUSD implied: EUR - USD = +0.40 - (-0.20) = +0.60
                               |
                               v

             +-------------------------------------------+
             | Agreement = pair_direction * currency_dir |
             | If agreement > 0 → alignment amplifies    |
             | If agreement < 0 → conflict detected      |
             +-------------------------------------------+
                               |
                               v
             +----------------------+
             | Regime Engine        |
             |----------------------|
             | trending | ranging   |
             | transition | chaotic |
             +----------------------+
                               |
                               v
             +--------------------------------+
             | Direction Hypothesis Object    |
             |--------------------------------|
             | symbol                         |
             | direction                      |
             | confidence                     |
             | currency_alignment             |
             | regime                         |
             | uncertainty                    |
             +--------------------------------+
```

## 8.2 Confidence Fusion (Replaces Naive Multiplication)

Old (REJECTED): `Confidence = Pair × Currency × Regime` → too punitive

New: **Uncertainty-aware weighted evidence accumulation**

```python
@dataclass
class ConfidenceSignal:
    value: float
    uncertainty: float
    reliability: float = 1 - uncertainty
    weight: float
```

Fusion formula:

```
Confidence = Σ(value_i × reliability_i × weight_i) / Σ(reliability_i × weight_i)
```

With conflict penalty:

```
Final_Confidence = Fusion_Confidence × (1 - conflict_level)
```

## 8.3 Direction Hypothesis Output

```python
@dataclass
class DirectionHypothesis:
    symbol: str
    direction: float              # [-1, +1]
    confidence: float             # [0, 1], after uncertainty fusion
    conflict: bool                # true if pair vs currency disagree
    explanation: str              # "USD weakness driven"
    currency_alignment: float
    regime: str
    uncertainty: float
    bootstrap_confidence: float   # accounts for replay freshness
```

---

# 9. Snapshot & Watermark System

## 9.1 State Epoch Transactions

Every graph update belongs to an `epoch_id`. All components (PairState, CurrencyState, TemporalState, RegimeState) must share the same epoch_id.

```python
@dataclass
class StateEnvelope:
    epoch_id: int
    market_timestamp: int
    pair_states: dict
    currency_states: dict
    temporal_states: dict
    graph_state: dict
```

## 9.2 Snapshot Protocol

```
1. Pause ingestion (engine.pause_ingestion())
2. Complete current event (all modules processed tick N)
3. Create immutable StateEnvelope
4. Validate consistency (assert all component.epoch_id == snapshot.epoch_id)
5. Serialize to temp file (snapshot.tmp)
6. Compute SHA256 checksum
7. Atomic rename (snapshot.tmp → snapshot.latest)
```

## 9.3 Watermark (Inside Snapshot)

Watermark is inside the same transaction — never separate:

```python
@dataclass
class Snapshot:
    state: StateEnvelope
    watermark:
        last_tick_timestamp: int
        last_tick_id: int
    metadata: SnapshotMetadata
    checksum: str
```

## 9.4 Snapshot Metadata

```python
@dataclass
class SnapshotMetadata:
    snapshot_id: str
    created_at_wallclock: datetime
    last_market_timestamp: datetime
    last_tick_id: int
    engine_version: str
    schema_version: str
    shutdown_reason: str       # CLEAN | CRASH | CORRUPT
    clean_shutdown: bool
    broker_identity: BrokerIdentity
    checksum: str
```

## 9.5 Shutdown Protocol (Clean Stop)

```
SIGTERM / CTRL+C
     |
     v
STOP ACCEPTING NEW TICKS
     |
     v
FLUSH INTERNAL BUFFERS
     |
     v
FREEZE MARKET STATE
     |
     v
WRITE SNAPSHOT TEMP FILE
     |
     v
VALIDATE CHECKSUM
     |
     v
ATOMIC RENAME
     |
     v
WRITE WATERMARK
     |
     v
PROCESS EXIT
```

## 9.6 Restart Paths

| Shutdown Type | Startup Method |
|---|---|
| Clean shutdown | Snapshot + gap replay (replay delta only) |
| Crash (no/last snapshot) | Last valid snapshot + extended replay |
| Corrupt state (checksum fail) | Discard state, full reconstruction from archive |

---

# 10. Intermittent Session Model

## 10.1 Session Lifecycle

```
                  START
                    |
                    v
          Load Snapshot
                    |
                    v
          Validate Snapshot
                    |
        +-----------+-----------+
        v                       v
     VALID                  INVALID
        |                       |
        v                       v
 Recent Replay          Full Replay
        |                       |
        +----------+------------+
                   v
         Market State Ready
                   |
                   v
         Live Tick Handoff
                   |
                   v
     Normal Trading Runtime
                   |
                   v
     Continuous Snapshotting
                   |
                   v
          Shutdown (SIGTERM)
                   |
                   v
        Atomic Snapshot Commit
                   |
                   v
             STOP
```

## 10.2 State Aging (Market Time vs Engine Time)

Never use `process uptime` for state aging. Every state uses two clocks:

```python
@dataclass
class StateClock:
    created_market_time: datetime
    last_update_market_time: datetime
    last_engine_update_time: datetime
```

Decay formula (correct):
```
decay = exp(-(current_market_time - last_state_time) / half_life)
```

When engine stops at Friday 18:00 (EUR trend age = 10h) and restarts Monday 08:00, actual age = 62h (not 10h).

---

# 11. Operational Dashboard & Health Metrics

## 11.1 Five Essential Metrics

```
=================================
PROXIMA HEALTH

Graph Health:      92/100
State Freshness:   3.1 sec lag
Currency Residual: 0.19
Opportunities:     BUY 3 / SELL 2
Bootstrap:         100%
=================================
```

### Graph Health Score

```
Graph Health = 0.35 × ForecastAccuracy
             + 0.25 × ResidualStability
             + 0.20 × ReplayIntegrity
             + 0.20 × StateFreshness
```

Range: 0-100.

### State Freshness
"How close are we to current reality?" — latest tick, latest currency solve, latest regime update.

### Currency Attribution Residual
Mean of |ε| across all pairs. Normal baseline ~0.15. Alert when >0.50.

### Direction Opportunity Density
Not number of trades — number of valid hypotheses. A flat system with 0 opportunities may be broken; a flat system with 100 rejected opportunities is working.

### Bootstrap Confidence
Replay completeness + state maturity after restart.

## 11.2 Silent Divergence Prevention

Four validation monitors:
1. **Currency Forecast Residual**: Does predicted currency move match realized basket return? Track 24h / 7d / 30d.
2. **Residual Explosion**: Track mean(|ε|) — if it rises from 0.15 to 0.70, currency explanation is failing.
3. **Replay Consistency Score**: Compare snapshot hash vs replay hash at every restart.
4. **State Age Health**: Every state object's `last_confirmed_market_time`.

## 11.3 Alert System

### CRITICAL
- Trading Disabled (snapshot invalid, graph health <60, data feed corrupt, broker mismatch, state epoch mismatch)
- Replay Failure (gap cannot reconstruct, missing archive, hash mismatch)
- Data Poisoning (impossible prices, bad ticks, archive corruption)

### HIGH
- Graph Degrading (residual rising, forecast accuracy falling, currency confidence collapsing)
- Bootstrap Weak (confidence <80%)

### MEDIUM
- State Aging (component stale > threshold)
- Archive Growth (storage >80%)

### LOW
- Opportunity Drought (no signals for X hours — not necessarily failure)

---

# 12. Performance Engineering & Resource Budget

## 12.1 Target Machine

Windows trading PC:
- Intel i7/i9, 8-16 cores, 16-24 threads
- 32-64 GB RAM
- NVMe Gen3/4 SSD
- 1 Gbps ethernet

Reserved for MT5 + Windows + browser: ~6 cores + 10GB RAM.

**Engine budget**: 4-8 cores, 16-32GB RAM.

## 12.2 Component Cost Ranking

| Component | Complexity | Actual Cost |
|---|---|---|
| Tick normalization | O(1) | Negligible |
| Feature extraction | O(features) | Low |
| Temporal updates | O(symbols × TF) | Low |
| WLS solver | O(8²×28) | Negligible |
| Residual calculation | O(28) | Negligible |
| Snapshot SHA256 | O(state size) | Low |
| Parquet writing | I/O | Medium |
| **Replay** | **O(ticks)** | **Highest** |

The WLS solver is trivially small (8×8 matrix on a laptop = hundreds of thousands/sec). The real bottleneck is tick replay (~25M ticks for 24h gap).

## 12.3 Tick Processing Capacity

| Component | Throughput (per core) |
|---|---|
| Tick normalizer | 500K - 2M ticks/sec |
| Currency solver (WLS) | 10K - 100K solves/sec (need only 1-10/sec) |
| Temporal state engine | 50K - 200K updates/sec |
| Parquet writer (NVMe) | 100-500 MB/sec |
| Gap replay (Python optimized) | 200K - 1M ticks/sec |

## 12.4 Memory Budget

| Component | Footprint |
|---|---|
| Raw tick buffer (10K ticks/pair) | ~9 MB |
| Market State Graph | <5 MB |
| Rolling features (28×3×20×1000) | ~13.4 MB |
| Parquet buffer | 64-256 MB |
| Full snapshot (serialized) | 5-20 MB |
| Python runtime | 1-2 GB |
| **Total runtime** | **<2 GB** |

## 12.5 Latency Budget

| Stage | Target |
|---|---|
| MT5 receive | 1-5 ms |
| Normalize | <1 ms |
| Feature update | 1-3 ms |
| Graph update | <5 ms |
| Currency solve | <1 ms |
| Direction | <5 ms |
| DRS | <5 ms |
| **Total** | **<25 ms** |

Maximum acceptable latency: 100ms. Beyond that, skip stale ticks (batch or drop).

## 12.6 Windows-Specific Solutions

1. **Python GIL**: Don't use 28 threads. Single event loop + vectorized NumPy processing.
2. **MT5 Python API (synchronous)**: Separate IPC process. Never call MT5 inside the market processing loop.
3. **Timer resolution**: Use `time.perf_counter_ns()` for measurement, `asyncio` loop for scheduling.
4. **NTFS + Antivirus**: Exclude `archive/`, `snapshot/`, `logs/` from Windows Defender.
5. **CPU starvation**: MT5 → HIGH priority, Engine → NORMAL, Replay → BELOW NORMAL.

## 12.7 Queue Design (Backpressure)

```python
Queue(maxsize=10000)   # ~2 seconds at 5000 ticks/sec
```

Policy:
- Normal: Process every tick
- Overload: Batch (process `batch[100]` instead of individual ticks)
- Severe: Drop redundant ticks (keep first, last, extreme spread)
- Never drop: volatility spikes, spread shocks, event ticks

---

# 13. Testing & Validation

## 13.1 Validation Hierarchy

```
Layer 0: Data integrity
Layer 1: Deterministic state reconstruction
Layer 2: Synthetic truth recovery
Layer 3: Historical behavioral validation
Layer 4: Execution simulation
Layer 5: Small capital live validation
```

## 13.2 Synthetic Market Generator

```python
@dataclass
class SyntheticCurrency:
    name: str
    strength: float
    volatility: float
    regime: str

# Pair generation:
# Return(XXYZ) = strength(X) - strength(Y) + noise
# With correlated noise, pair-specific noise, regime switching
```

**Critical tests**:
1. Perfect currency recovery (EUR +0.5, USD -0.4 → recovered within ±0.05)
2. Noise robustness (30% noise → recovered vs true correlation >0.8)
3. Missing pair (remove EURUSD → recovery still works)
4. Currency shock (inject USD event → USD detected)

## 13.3 Regression Test Suite (Minimum 10)

| # | Test | Expected |
|---|---|---|
| 1 | Replay determinism | same input = same state hash |
| 2 | Snapshot round trip | graph_before == graph_after |
| 3 | Watermark integrity | gap between snapshot and archive = GapMissingError |
| 4 | Corrupt snapshot detection | 1 byte modified = ChecksumFailure |
| 5 | Currency synthetic recovery | error < threshold |
| 6 | Residual explosion detection | corrupted pair = PairResidualAlarm |
| 7 | Replay vs live equivalence | same ticks = same state hashes |
| 8 | Broker mismatch | Snapshot Broker A, Runtime Broker B = StartupBlocked |
| 9 | State epoch consistency | PairState epoch 10, CurrencyState epoch 11 = SnapshotRejected |
| 10 | Degraded replay confidence | Remove ticks, use M1 → state valid but confidence lower |

## 13.4 Leave-One-Pair-Out Test (Strongest Validation)

Remove EURUSD, solve using remaining 27 pairs, then predict EURUSD. This prevents circular self-validation.

## 13.5 Monte Carlo Stress Test

10,000 randomized scenarios over:
- Market variables: volatility, trend strength, correlation, spread, noise, regime duration
- Operational variables: shutdown time, gap duration, missing ticks, archive corruption, snapshot age
- Data variables: tick order, duplicates, missing ticks, bad prices

## 13.6 First Live MT5 Procedure

1. Connect + symbol audit (28 pairs, digits, contract size, spread)
2. Download 48h ticks
3. Historical bootstrap (expected confidence >90%)
4. Graph sanity checks (currency values in [-1,+1], residuals stable, SUM(C)≈0)
5. Live observe (4 hours, no execution)
6. Shadow execution (compare hypothesis vs future outcome)

---

# 14. Migration Strategy (Existing Proxima → New Architecture)

## 14.1 Migration Phases

```
PHASE 0: Observation Infrastructure (Week 1-2)
         Tick Bus, Archive, Snapshot system.
         Old Proxima executes, New Engine observes only.

PHASE 1: State Replacement (Week 3-4)
         Currency engine, Temporal states, Residual monitoring.
         Graph replaces internal analytics.

PHASE 2: Decision Shadow (Week 5-8)
         Both systems generate decisions, compared via DecisionComparator.
         Build disagreement database.

PHASE 3: Hybrid Authority (Week 9+)
         New can veto old. Old executes. Requires 30+ trading days of stability.

PHASE 4: Execution Authority Transfer
         New DRS controls entries. Old retired to kill switch only.
```

Minimum: 4-6 weeks. Safe: 8-12 weeks.

## 14.2 Authority Modes

```python
class AuthorityMode(Enum):
    OLD_ONLY    = 1   # Production unchanged
    SHADOW      = 2   # New observes only
    HYBRID      = 3   # New can veto (if old says BUY, new says BLOCK → BLOCK)
    NEW_ONLY    = 4   # New controls
```

## 14.3 Tick Bus (Shared Data Source)

```python
class TickBus:
    subscribers = []   # Both old and new engines
```

Both engines consume same ticks, same ordering, same timestamps.

## 14.4 Position Handover

Old positions get `source="OLD_PROXIMA"` tag. New DRS sees them as "external exposure" — MONITOR ONLY until closed naturally. New system can trade a symbol only after old position is closed.

## 14.5 Risk Governor Migration

During migration, **old risk layer always wins**:
```
New Signal → New Risk Proposal → Legacy Risk Governor → Execution
```

## 14.6 Rollback Plan

Prerequisites:
- Feature flag: `EXECUTION_ENGINE = "OLD"`
- Independent processes (proxima_old.exe, proxima_graph.exe)
- Shared TickBus
- Separate state directories (old_state/, new_state/)

Rollback time: minutes. Disable new execution flag, restore old authority, restart old process.

---

# 15. Learning Architecture

## 15.1 What Learns vs What's Fixed

| Component | Classification |
|---|---|
| Currency incidence matrix A | **FIXED** — structural topology |
| Pair weights W | **ADAPTIVE** — based on spread, liquidity, residual history |
| Temporal state half-lives | **ADAPTIVE** — regime-based buckets (FAST/NORMAL/SLOW) |
| Event mutation half-lives | **LEARNED** — per event type (NFP: 18h, FOMC: 48h) |
| Residual anomaly thresholds | **ADAPTIVE** — rolling z-score with clamped bounds |
| Session volatility multipliers | **LEARNED** — EMA of realized volatility per session |
| Market/Macro fusion α | **LEARNED** — optimized for forecast calibration, not PnL |
| Confidence calibration | **LEARNED** — map predicted confidence → realized outcome |

## 15.2 Three-Level Feedback

| Level | Signal | Timeframe |
|---|---|---|
| 1 | Prediction Error (currency basket forecast vs realized) | Fast |
| 2 | Decision Quality (hypothesis confidence vs actual outcome) | Medium |
| 3 | Trade Outcome (PnL, drawdown, execution quality) | Slow |

## 15.3 Objective Function (Not PnL Alone)

```
Objective = 0.4 × PredictionAccuracy
          + 0.3 × Calibration
          + 0.2 × RiskAdjustedReturn
          + 0.1 × Stability
```

## 15.4 Learning Governor (Guardrails)

- Parameter bounds (half-life min 5min, max 7 days; α in [0,1])
- Minimum samples before learning (500 observations, not 5 trades)
- Out-of-sample validation required
- Shadow deployment before activation
- Maximum daily parameter change (10%)
- Anti-overfit rules (no lone PnL optimization)

## 15.5 Parameter Registry with Provenance

```python
@dataclass
class LearnedParameter:
    name: str
    value: float
    created_at: timestamp
    last_updated: timestamp
    source: str               # DEFAULT | OFFLINE | ONLINE
    training_window: (start, end)
    regime_context: str       # TREND | RANGE | CRISIS
    sample_count: int
    confidence: float
    validation_score: float
    previous_value: float
```

---

# 16. Implementation Phases

## Phase 1 — Foundation (Week 1-2)
- Historical Archive (Parquet storage, streaming reader, incremental append)
- Replay Engine (deterministic replay, replay scheduler, market-time aware)
- Tick Pipeline (normalizer, buffer, bus)

## Phase 2 — Persistence (Week 3-4)
- Snapshot Manager (save/load/validate, state epoch transactions, atomic commits)
- Watermark System (gap detection, last tick tracking)
- Broker Identity (versioned snapshots, compatibility checks)

## Phase 3 — State Engine (Week 5-6)
- Currency Attribution (WLS solver, constrained least squares, pair weights)
- Temporal State Engine (5m/1h/4h, decay, age tracking)
- Residual Engine (epsilon computation, anomaly classification)
- Market State Graph (currency nodes, pair edges, fusion)

## Phase 4 — Live Handoff (Week 7-8)
- Gap Replay (fidelity tiers, tick/hybrid/reconstruction modes)
- Bootstrap Controller (3-level bootstrap, confidence graduation)
- Runtime Controller (session lifecycle, shutdown coordinator)

## Phase 5 — Direction & DRS (Week 9-10)
- Direction Hypothesis (arbitration, uncertainty fusion, conflict detection)
- DRS Integration (ranking, displacement, portfolio constraints with new signals)

## Phase 6 — Macro Layer (Week 11-12)
- Economic Calendar (event lifecycle, surprise calculation)
- Cross-Asset (yields, equities, commodities → propagation matrix)
- Session Model (learned volatility multipliers)
- State Circuit Breaker (freeze/quarantine/recovery)

## Phase 7 — Learning & Evolution (Ongoing)
- Prediction Recorder (feedback loop infrastructure)
- Parameter Registry (provenance tracking)
- Learning Governor (bounds, validation, shadow deployment)

---

# 17. Module Directory Structure

```
proxima_market_state/
├── bootstrap/
│   ├── bootstrap_controller.py
│   ├── historical_loader.py
│   ├── replay_engine.py
│   ├── state_validator.py
│   └── gap_detector.py
│
├── graph/
│   ├── market_state_graph.py
│   ├── currency_node.py
│   ├── pair_edge.py
│   └── residual_state.py
│
├── currency/
│   ├── currency_attribution.py
│   ├── currency_state.py
│   └── currency_fusion.py
│
├── temporal/
│   ├── temporal_state.py
│   ├── temporal_engine.py
│   └── decay_scheduler.py
│
├── pairs/
│   └── pair_engine.py
│
├── direction/
│   ├── hypothesis_engine.py
│   ├── confidence_fusion.py
│   └── arbitration.py
│
├── macro/
│   ├── event_calendar.py
│   ├── event_mutation_engine.py
│   ├── surprise_calculator.py
│   ├── cross_asset_engine.py
│   ├── session_state.py
│   └── state_circuit_breaker.py
│
├── persistence/
│   ├── snapshot_manager.py
│   ├── watermark_store.py
│   ├── archive_writer.py
│   └── schema_migration.py
│
├── storage/
│   ├── hot_tier.py       # ring buffer
│   ├── warm_tier.py      # state archive
│   ├── cold_tier.py      # bar archive
│   └── archive_reader.py
│
├── runtime/
│   ├── session_manager.py
│   ├── shutdown_coordinator.py
│   ├── tick_bus.py
│   └── tick_normalizer.py
│
├── risk/
│   └── risk_adapter.py
│
├── learning/
│   ├── learning_engine.py
│   ├── parameter_registry.py
│   ├── learning_governor.py
│   └── feedback_collector.py
│
├── ops/
│   ├── health_engine.py
│   ├── reality_monitor.py
│   ├── alert_manager.py
│   └── dashboard_metrics.py
│
├── migration/
│   ├── decision_comparator.py
│   ├── authority_controller.py
│   └── position_handover.py
│
└── tests/
    ├── synthetic_market.py
    ├── regression_tests.py
    ├── monte_carlo_harness.py
    └── replay_determinism_test.py
```

---

# 18. Runtime Scheduling Table

| Component | Trigger | Frequency | Replay Mode | Live Mode |
|---|---|---|---|---|
| Tick Receiver | incoming tick | every tick | historical reader | MT5 stream |
| Tick Normalizer | tick | every tick | yes | yes |
| Tick Buffer | tick | every tick | yes | yes |
| Pair Feature Engine | tick/window | every tick + rolling | yes | yes |
| Spread State | tick | every tick | yes | yes |
| Directional Energy | rolling window | every N ticks | yes | yes |
| Temporal State 5m | bar boundary | every 5 min | accelerated | every 5m |
| Temporal State 1h | bar boundary | every hour | accelerated | every hour |
| Temporal State 4h | bar boundary | every 4 hours | accelerated | every 4h |
| Currency Attribution Fast | timer | every 5 sec | market-time sched | every 5 sec |
| Currency Attribution Med | bar close | every 5 min | accelerated | every 5m |
| Currency Attribution Slow | bar close | every hour | accelerated | every hour |
| Regime Engine | state update | every minute | accelerated | every minute |
| Market Graph Update | state mutation | event driven | yes | yes |
| Direction Hypothesis | decision cycle | configurable | yes | live cycle |
| DRS | portfolio cycle | configurable | yes | live cycle |
| Risk Engine | trade candidate | event | yes | live |
| Snapshot Save | timer | every 5-15 min | optional | periodic |
| Reality Monitor | periodic | every minute | yes | yes |

---

# 19. Runtime State Machine (Final)

```
                  START
                    |
                    v
          Load Snapshot
                    |
                    v
          Validate Snapshot
                    |
        +-----------+-----------+
        v                       v
     VALID                  INVALID
        |                       |
        v                       v
 Recent Replay          Full Replay
        |                       |
        +----------+------------+
                   v
         Market State Ready
                   |
                   v
         Live Tick Handoff
                   |
                   v
     Normal Trading Runtime
                   |
             +-----+-----+
             |           |
             v           v
     Continuous      Background
     Snapshotting    Learning
             |           |
             +-----+-----+
                   |
                   v
            Shutdown (SIGTERM)
                   |
                   v
         Atomic Snapshot Commit
                   |
                   v
             STOP
```

---

# 20. Final Design Principles

1. **Ticks are observations. Currencies are latent variables. Timeframes are measurement scales. Direction is derived. DRS is capital allocation. Execution is the final actuator.**

2. **The engine should behave like a database** — it restores its state, not wakes up empty. Historical market memory + current market stream = continuous market state.

3. **Pairs are observations. Currencies are latent variables. Timeframes are measurement scales. Direction is a derived hypothesis. DRS is capital allocation. Execution is the final actuator.**

4. **Do not store the entire market forever. Store the level of memory required for each type of decision.**

5. **The system is session-based, not daemon-based.** A persistent market-state machine that wakes up, reconstructs reality, operates, commits memory, and sleeps.

6. **The system should not change itself arbitrarily.** It should measure when its assumptions stop working and propose controlled replacements.

7. **Do not ask "is the engine running?" Ask "does the engine's internal representation continue to agree with observable market reality?"**

8. **Information → state mutation → price observation → correction.** Price is one sensor. Not the entire reality model.

9. **Pairs are observations. Currencies are latent variables. Timeframes are measurement scales. Direction is a derived hypothesis.**

10. **Replace point estimates with probability distributions. Replace multiplicative confidence with uncertainty-aware fusion. Replace veto gates with explanation engines.**

---

# 21. Gap Integrity Failure Mode Analysis

## 21.1 State Sensitivity Classification

Different components have different replay sensitivity:

### Class 1 — Deterministic Tick States
Can be reconstructed exactly if ticks exist. Examples: directional energy, tick velocity, spread statistics, microstructure entropy.
**Requirement**: Same ticks + same order + same timestamps = same output.

### Class 2 — Time Aggregated States
Depends on boundary alignment. Examples: 5m state, 1h state, 4h state. Problem: partial candles at snapshot boundary.

### Class 3 — Learned / Adaptive States
Most dangerous. Examples: regime probability, currency weights, calibration, learned parameters. Depend on update timing and learning events. Replay may NOT reproduce them exactly.

## 21.2 Scenario-by-Scenario Analysis

### Scenario A — Intraday Gap (1-6 hours)

| Failure | Classification | Metric | Fix |
|---|---|---|---|
| Half-candle corruption (snapshot at 10:02:30, misses ticks from 10:00-10:02:30) | **WRONG STATE** | TemporalState hash mismatch | Snapshot must contain unfinished candle buffers (`PartialBarState`) |
| 1h boundary crossing (gap crosses 11:00 bar completion) | **EXACT** (if ticks exist, otherwise APPROXIMATE) | Bar continuity | Event ordering preserved during replay |
| Direction energy from M1 bars (loses tick path) | **APPROXIMATE→WRONG** | Microstructure confidence | Only use tick replay for microstructure |

```python
@dataclass
class PartialBarState:
    timeframe: str
    open_time: int
    ticks_seen: int
    partial_features: dict
```

### Scenario B — Overnight Gap (12-24 hours)

| Failure | Classification | Metric | Fix |
|---|---|---|---|
| Wall-clock dependency in state aging | **WRONG STATE** | Replay determinism score | Every component must use `tick.timestamp` not `datetime.now()` |
| Currency attribution learning during replay | **WRONG STATE** | Replay hash divergence | Disable ALL learning during replay (`EngineMode.REPLAY`) |
| Missing macro event (RBA decision during gap) | **APPROXIMATE** | Event coverage check | Inject `EconomicEvent` markers during replay |
| M1 hybrid pretending to be exact | **WRONG STATE** | Microstructure confidence | Capability masking: `microstructure_confidence *= 0.3` if source != tick |

### Scenario C — Weekend Gap (48-72 hours)

| Failure | Classification | Metric | Fix |
|---|---|---|---|
| `bootstrap_confidence = 0.7` is arbitrary, not justified | **APPROXIMATE** | Information loss ratio | Replace fixed 0.7 with `BootstrapConfidence = 1 - InformationLoss` |
| Weekend gap opening (Friday close 1.0500 → Monday open 1.0450 = 50 pip gap) | **WRONG STATE** | Energy anomaly spike | `MarketOpenGapEvent` — classify discontinuity, do NOT feed gap as normal energy |

### Scenario D — Holiday Gap (Multiple Days)

Multiple approximation layers stack: cold bars → state interpolation → graph reconstruction → confidence reduction. Error compounds:

| Stage | Error Contribution |
|---|---|
| M1 bar approximation | ~10% |
| State interpolation | ~15% |
| Currency reconstruction | ~20% |
| **Combined** | **Potentially ~40% error** |

**Fix**: Track uncertainty propagation through every state. Each `TemporalState` has `uncertainty_score` that feeds into the currency solver as weighted evidence.

### Scenario E — Crash / Power Loss

Recovery protocol when last snapshot is 60h old but hot tier only holds 48h:

```
Load snapshot
    ↓
Warm state archive (30-90 day states)
    ↓
Cold bars
    ↓
Rebuild approximate state
    ↓
Max confidence cap (e.g., 0.55 if missing ticks)
```

**Never silently continue** — must enter `RecoveryLevel.PARTIAL` mode.

### Scenario F — Multiple Stops Same Day

If deterministic: no degradation. Same input → same output. **However**: if snapshot contains learned parameters that update during replay, each replay modifies state → **drift**.

**Fix**: Replay mode is immutable — no learning, no parameter updates.

**Validation**: Five-restart equivalence test: continuous 5h vs restart every hour → final hash must match.

### Scenario G — Clock Drift

Windows system clock drift causes gap calculation errors (system thinks 2h gap, actual 26h).

```python
@dataclass
class ClockState:
    system_time: datetime
    market_time: datetime    # from last tick
    ntp_time: datetime       # from NTP sync

# Alert if system_time - market_time > 2 seconds
```

**Recovery**: Before replay, validate `archive_latest_timestamp >= requested_replay_end`. If impossible → abort.

## 21.3 Failure Classification Matrix

| Scenario | Main Failure | Type | First Broken Metric | Recovery |
|---|---|---|---|---|
| Intraday | Half candle | **Wrong** | State hash | Save partial bars |
| Overnight | Learning during replay | **Wrong** | Replay hash | Disable learning |
| Weekend | Gap open | **Wrong** | Energy anomaly | Gap event |
| Holiday | Approximation stack | **Approximate→Wrong** | Uncertainty | Confidence cap |
| Crash | Missing archive | **Approximate** | Bootstrap score | Degraded mode |
| Multiple restarts | Replay drift | **Wrong** | Hash comparison | Immutable replay |
| Clock drift | Wrong gap | **Wrong** | Clock monitor | Abort bootstrap |

## 21.4 Replay Fidelity Level (Critical New Concept)

Every restart must output:

```python
@dataclass
class ReplayResult:
    state: MarketStateGraph
    fidelity: str        # EXACT_TICK | EXACT_STATE | APPROXIMATE | DEGRADED
    confidence: float
    missing_information: list
    uncertainty: float
```

**Core rule**: The biggest hidden failure is not "the state is approximate." The system can handle approximation. The dangerous failure is: **the system has approximate information but believes it has exact information.**

---

# 22. Runtime Safeguards — Preventing False State From Reaching Execution

## 22.1 State Integrity Gate (New Pipeline Layer)

```
Market Data → Market State Graph → STATE INTEGRITY GATE → Direction Hypothesis → DRS → Execution
```

### StateIntegrityMonitor (Continuous)

```python
@dataclass
class StateIntegrityScore:
    graph_consistency: float       # [0, 1]
    residual_health: float
    currency_confirmation: float
    temporal_alignment: float
    data_quality: float
    overall: float                 # min() of all, not avg
```

### Check 1 — Currency Basket Confirmation

Build synthetic EUR basket (EURUSD, EURJPY, EURGBP, EURAUD, EURCAD, EURNZD, EURCHF) and compare:

```python
currency_error = abs(graph_strength - basket_strength)
# < 0.25: PASS
# 0.25-0.40: WARNING
# > 0.40: FAIL
```

### Check 2 — Pair Reconstruction Error

For every pair: `error = abs(actual_return - (currency_base - currency_quote))`. Rolling window = last 1000 ticks. Failure: `median residual > 2 sigma`.

### Check 3 — Temporal Continuity

Detect impossible state jumps (volatility 0.20 → 5.0 without event explanation). If `state_change > threshold`, require `event_explanation()`.

### Check 4 — Fidelity Integrity

```
EXACT_TICK   → trading allowed
EXACT_STATE  → trading allowed
APPROXIMATE  → limited (reduced size)
DEGRADED     → BLOCKED
```

### Overall Integrity Formula

```
Integrity = min(graph, currency, temporal, data, fidelity)
```

One broken layer invalidates the entire state.

## 22.2 Pre-Trade State Validation Gate

Executed BEFORE DRS for every hypothesis:

```python
def pre_trade_gate(hypothesis, state):
    if state.fidelity == DEGRADED:
        return BLOCK
    if state.integrity < 0.85:
        return BLOCK
    if state.bootstrap_confidence < 0.75:
        return BLOCK
    for currency in hypothesis.currencies:
        if currency.uncertainty > 0.40:
            return BLOCK
    if state.stabilization_active:
        return BLOCK
    return ALLOW
```

### Bootstrap Graduation State Machine

```
DEGRADED → VALIDATING → STABLE → FULL

DEGRADED:   NO trading
VALIDATING: shadow only
STABLE:     50% size
FULL:       normal
```

Graduation criteria (ALL required):
- Minimum 500 live ticks confirmed
- Residual error < normal range
- Currency basket alignment error < 0.25
- Minimum 5 minutes elapsed

## 22.3 Fidelity Tag Propagation

Every data object carries provenance:

```python
@dataclass
class StateMetadata:
    fidelity: str          # EXACT | APPROXIMATE | DEGRADED
    source: str            # LIVE_TICK | TICK_REPLAY | STATE_REPLAY | BAR_REPLAY
    uncertainty: float
    timestamp: int
```

Confidence propagation: `adjusted_confidence = raw_confidence * (1 - state.uncertainty)`

Fidelity combination: worst input wins — `min(child.fidelity for child in children)`

## 22.4 Stabilization Window After Handoff

After replay catches up and live handoff occurs, the system must NOT generate trades immediately:

```
Required:
- 500 ticks minimum
- 5 minutes minimum
- integrity > 0.85
```

During stabilization: state updates and validation allowed; execution and parameter tuning FORBIDDEN.

Compare replay prediction vs live observation. If mismatch exceeds threshold → reset.

## 22.5 Two-Phase Replay Validation (Pre-Live)

### Phase 1 — Deterministic Validation
```
checksum OK
watermark continuous
replay hash matches
```

### Phase 2 — Market Consistency Validation
```
∑ currency_strength ≈ 0 (absolute sum < 0.01)
residual < 3 sigma
volatility within historical percentile
spread percentile < 99%
```

## 22.6 Emergency State Reset Escalation Ladder

| Level | Trigger | Action |
|---|---|---|
| **Level 0 — Warning** | integrity < 0.85 | Reduce sizing, increase monitoring |
| **Level 1 — Trading Pause** | integrity < 0.70 OR currency_error > 0.4 | Block new entries, manage existing positions |
| **Level 2 — Auto Recovery** | Level 1 persistent | Reload latest snapshot, replay hot tier |
| **Level 3 — Degraded Rebuild** | Recovery fails | Use warm states + cold bars, set DEGRADED, no trading |
| **Level 4 — Factory Reset** | Multiple failures OR hash corruption OR clock failure | Stop execution, archive evidence, discard derived state, rebuild graph, wait graduation |

## 22.7 Final Threshold Table

| Metric | Normal | Warning | Block |
|---|---|---|---|
| Fidelity | EXACT | APPROX | DEGRADED |
| Bootstrap confidence | >0.90 | 0.75-0.90 | <0.75 |
| Graph integrity | >0.90 | 0.85-0.90 | <0.85 |
| Currency uncertainty | <0.25 | 0.25-0.40 | >0.40 |
| Residual error | <2σ | 2-3σ | >3σ |
| Stabilization | complete | running | active |
| Clock drift | <1s | 1-2s | >2s |

---

# 23. Edge Case Survival — Boundary Conditions

## 23.1 Half-Written Snapshot (Mid-Write Crash)

### Problem
System crashes during `write snapshot.tmp` → atomic rename never happens. Filesystem has both `snapshot_latest.bin` (previous valid) and `snapshot_latest.tmp` (incomplete new).

### Naive (Wrong) Approach
Loading the newest file by timestamp could load the corrupt `.tmp` file.

### Correct Snapshot State Machine

Every snapshot has lifecycle metadata:

```
snapshots/
    snapshot_1000.bin
    snapshot_1000.meta    # {status: VERIFIED, hash: xxxx}
    snapshot.tmp          # ignored on restart
    manifest.json
```

```python
def load_snapshot():
    valid = []
    for snapshot in snapshots:
        meta = load_manifest(snapshot)
        if meta.status == "VERIFIED" and sha256(snapshot) == meta.hash:
            valid.append(snapshot)
    return newest(valid)
    # .tmp files are never verified → ignored
```

**Write Protocol**:
1. Freeze: acquire `State Write Lock`, freeze all components at logical timestamp T
2. Serialize: write `snapshot_T.tmp`
3. Hash: compute SHA256
4. Write manifest: `status=VERIFIED, hash=xxxx`
5. Atomic rename: `snapshot_T.tmp → snapshot_T.bin`

**On restart**: temp files are moved to `quarantine/` — never loaded.

## 23.2 Tick in Both Snapshot and Gap (State Commit Consistency)

### Problem
Watermark = "last seen tick timestamp" is WRONG. If tick 1000 updated PairState but NOT CurrencyState (crash between), watermark says 1000 but CurrencyState is at 999.

### Fix: Transactional State Watermark

```python
@dataclass
class StateCommit:
    commit_id: int
    tick_id: int
    market_timestamp: int
    components: dict   # {pair_state_hash, currency_state_hash, temporal_state_hash, drs_state_hash}
```

A tick is NOT committed until ALL components finish:

```
Tick → Pending Buffer → Pair Update → Temporal Update → Currency Solve → DRS Update → Commit Marker
```

**Snapshot stores**: `last_committed_tick = 1000` (NOT `last_seen_tick = 1000`)

**Replay starts from**: `commit_tick + 1` — tick 1000 is guaranteed to be fully committed.

**Every state object stores**: `last_commit_id`. On snapshot validation, all must match. If `PairState.commit=5001` and `CurrencyState.commit=5000` → reject snapshot.

## 23.3 MT5 History Boundary (Truncated History)

### Problem
MT5 provides limited history (1-2M ticks per symbol). For EURUSD, that's ~2-4 hours. Engine requests "ticks from 6h ago" → MT5 returns only 3h of data.

### Wrong behavior
Silently continue with shorter replay → false state.

### Correct: Replay Coverage Contract

```python
@dataclass
class ReplayRequest:
    start_timestamp: int
    end_timestamp: int

@dataclass
class ReplayCoverage:
    requested_start: int
    actual_start: int
    requested_end: int
    actual_end: int

    def validate(self):
        if self.actual_start > self.requested_start:
            raise MissingHistoryError
```

**Recovery ladder**:
```
Tick replay available?       YES → Exact replay
NO → Warm state archive?     YES → Approximate mode
NO → BLOCK TRADING
```

## 23.4 Midnight Rollover (Day Boundary File Handling)

### Problem
Engine stops at 23:59:50, restarts at 00:01:10. Two daily Parquet files exist: `2026-07-07.parquet` (OPEN status) and new `2026-07-08.parquet` (needs creation).

### Correct File State Machine

Each partition has lifecycle:
```
OPEN → CLOSED → VERIFIED
```

```python
def open_archive():
    current_day = market_date()
    existing = find_partition(current_day)
    if existing:
        verify_footer()
        append(existing)
    else:
        create_new(current_day)

    # Before creating new day, check previous day's meta:
    # If prev.status == OPEN → repair index, verify checksum, close partition
```

## 23.5 Thin Tick Problem (Low-Liquidity Pairs)

### Problem
EURTRY has ticks 10+ seconds apart. During 1000x replay, 10-second market gaps are microseconds of CPU. During LIVE, these gaps are real — no ticks exist.

### Wrong classification
System falsely flags "missing data."

### Fix: Market Silence Model

```python
@dataclass
class TickGap:
    symbol: str
    start: int
    end: int
    duration: int
    expected_tick_rate: float

GapScore = gap / expected_interval
# EURUSD expected: 0.1s, gap 15s → GapScore=150 (SUSPICIOUS)
# EURTRY expected: 12s, gap 15s → GapScore=1.25 (NORMAL)
```

Thresholds:
- `GapScore < 3`: normal silence
- `3-20`: warning
- `>20`: missing data candidate

## 23.6 Duplicate Tick Stream (Replay→Live Handoff Boundary)

### Problem
MT5 may send the same ticks on subscribe that were just replayed. Processing same tick twice → double energy, wrong state.

### Fix: Tick Identity Dedup

```python
@dataclass(frozen=True)
class TickIdentity:
    symbol: str
    timestamp: int
    bid: float
    ask: float
    volume: float

    def hash(self): return hash((symbol, timestamp, bid, ask, volume))
```

Maintain `recent_tick_cache` (last 10,000 hashes). If tick.hash in cache → ignore.

**Why timestamp alone fails**: Two ticks at 10:00:01 can have different bid prices (1.1000 vs 1.1002) — both valid.

**State protection requirement**: Every duplicate must be idempotent. Processing the same tick twice must produce the same state, not double energy.

## 23.7 Calendar Event During Replay (Anticipation Phase Missing)

### Problem
Live: NFP anticipation builds for hours before release. Replay at 1000x speed jumps from 08:00 to 13:30 in microseconds — the anticipation phase is lost.

### Fix: Event Phase Reconstruction

Event state machine during replay:

```
SCHEDULED → ANTICIPATION → RELEASE → ABSORPTION
```

Replay does NOT jump directly to RELEASE. It reconstructs the event timeline:

```
10:00: ANTICIPATION_START
12:30: ANTICIPATION_HIGH
13:30: RELEASE
```

```python
@dataclass
class EventReplayState:
    event: str
    reconstructed_phase: str
    anticipation_confidence: float   # capped at 0.5 if no options/orderflow data
```

**Limitation**: Replay can reconstruct event timing but not perfectly reconstruct participant expectation formation. Therefore macro event replay creates **Approximate Event State**, not exact.

## 23.8 The Five Non-Negotiable Rules

1. **A watermark is a commit ID, not a timestamp.** Timestamps alone cannot represent distributed state consistency.

2. **Replay must fail loudly when data coverage is incomplete.** Never silently degrade.

3. **All replay-time logic uses market timestamps, never CPU time.** `tick.timestamp` not `datetime.now()`.

4. **Every approximation carries uncertainty forward.** If a state is reconstructed from bars, every dependent state inherits reduced confidence.

5. **A state transition is atomic only when the entire graph reaches the same commit boundary.** Partial state commits are invalid state.

---

# 24. Updated Module Directory Structure

```
proxima_market_state/
├── bootstrap/
│   ├── bootstrap_controller.py
│   ├── historical_loader.py
│   ├── replay_engine.py
│   ├── replay_fidelity.py          # NEW: ReplayResult fidelity tracking
│   ├── gap_detector.py
│   └── clock_monitor.py            # NEW: three-clock drift detection
│
├── graph/
│   ├── market_state_graph.py
│   ├── currency_node.py
│   ├── pair_edge.py
│   └── residual_state.py
│
├── currency/
│   ├── currency_attribution.py
│   ├── currency_state.py
│   └── currency_fusion.py
│
├── temporal/
│   ├── temporal_state.py
│   ├── temporal_engine.py
│   ├── decay_scheduler.py
│   └── partial_bar_buffer.py       # NEW: unfinished candle state at snapshot
│
├── pairs/
│   └── pair_engine.py
│
├── direction/
│   ├── hypothesis_engine.py
│   ├── confidence_fusion.py
│   ├── arbitration.py
│   └── pre_trade_gate.py           # NEW: state validation before DRS
│
├── integrity/                       # NEW: State integrity subsystem
│   ├── state_integrity_monitor.py   # NEW: continuous integrity scoring
│   ├── state_validation_gate.py     # NEW: pre-trade validation
│   ├── fidelity_tag_propagator.py   # NEW: provenance through pipeline
│   └── stabilization_window.py      # NEW: post-handoff lockout
│
├── macro/
│   ├── event_calendar.py
│   ├── event_mutation_engine.py
│   ├── event_replay_state.py       # NEW: anticipation phase reconstruction
│   ├── surprise_calculator.py
│   ├── cross_asset_engine.py
│   ├── session_state.py
│   └── state_circuit_breaker.py
│
├── persistence/
│   ├── snapshot_manager.py
│   ├── snapshot_manifest.py         # NEW: lifecycle state machine for snapshots
│   ├── state_commit_tracker.py      # NEW: transactional commit IDs
│   ├── watermark_store.py
│   ├── archive_writer.py
│   └── schema_migration.py
│
├── storage/
│   ├── hot_tier.py                  # ring buffer
│   ├── warm_tier.py                 # state archive
│   ├── cold_tier.py                 # bar archive
│   ├── tick_gap_analyzer.py         # NEW: market silence vs missing data
│   └── archive_reader.py
│
├── runtime/
│   ├── session_manager.py
│   ├── shutdown_coordinator.py
│   ├── tick_bus.py
│   ├── tick_identity_cache.py       # NEW: dedup cache for handoff boundary
│   ├── recovery_ladder.py           # NEW: 5-level emergency escalation
│   └── tick_normalizer.py
│
├── risk/
│   └── risk_adapter.py
│
├── learning/
│   ├── learning_engine.py
│   ├── parameter_registry.py
│   ├── learning_governor.py
│   ├── feedback_collector.py
│   └── replay_safe_mode_controller.py  # NEW: disables learning during replay
│
├── ops/
│   ├── health_engine.py
│   ├── reality_monitor.py
│   ├── alert_manager.py
│   └── dashboard_metrics.py
│
├── migration/
│   ├── decision_comparator.py
│   ├── authority_controller.py
│   └── position_handover.py
│
└── tests/
    ├── synthetic_market.py
    ├── regression_tests.py
    ├── monte_carlo_harness.py
    ├── replay_determinism_test.py
    ├── gap_scenario_tests.py        # NEW: A-G scenario test cases
    ├── snapshot_crash_recovery.py   # NEW: half-write, corrupt snapshot tests
    ├── boundary_conditions_test.py  # NEW: midnight rollover, thin tick, dedup
    └── five_restart_equivalence.py  # NEW: multiple restart drift detection
```

---

# 25. Updated Implementation Phases

## Phase 1 — Foundation (Week 1-2)
- Historical Archive (Parquet storage, streaming reader, incremental append)
- Replay Engine (deterministic replay, replay scheduler, market-time aware)
- Tick Pipeline (normalizer, buffer, bus, identity cache for dedup)
- **Tick Gap Analyzer** (thin tick vs missing data detection)

## Phase 2 — Persistence (Week 3-4)
- Snapshot Manager (save/load/validate, state epoch transactions, atomic commits)
- **Snapshot Manifest** (lifecycle state machine, half-write detection)
- **State Commit Tracker** (transactional commit IDs, not timestamps)
- Watermark System (gap detection, last commit tracking)
- Broker Identity (versioned snapshots, compatibility checks)

## Phase 3 — State Engine (Week 5-6)
- Currency Attribution (WLS solver, constrained least squares, pair weights)
- Temporal State Engine (5m/1h/4h, decay, age tracking)
- **Partial Bar Buffer** (unfinished candle state at snapshot)
- Residual Engine (epsilon computation, anomaly classification)
- Market State Graph (currency nodes, pair edges, fusion)

## Phase 4 — Live Handoff (Week 7-8)
- Gap Replay (fidelity tiers, tick/hybrid/reconstruction modes)
- **Replay Fidelity** (ReplayResult with EXACT/APPROXIMATE/DEGRADED)
- Bootstrap Controller (3-level bootstrap, confidence graduation)
- **Stabilization Window** (post-handoff lockout, 500 ticks / 5 min)
- **Integrity Monitor** (continuous scoring, currency basket confirmation)
- Runtime Controller (session lifecycle, shutdown coordinator)

## Phase 5 — Integrity & Safeguards (Week 9-10)
- **State Validation Gate** (pre-trade checks, uncertainty thresholds)
- **Fidelity Tag Propagation** (provenance through pipeline)
- **Clock Monitor** (three-clock drift detection)
- **Recovery Ladder** (5-level emergency escalation)
- Direction Hypothesis (arbitration, uncertainty fusion, conflict detection)
- DRS Integration (ranking, displacement, portfolio constraints)

## Phase 6 — Macro Layer (Week 11-12)
- Economic Calendar (event lifecycle, surprise calculation)
- **Event Replay State** (anticipation phase reconstruction during replay)
- Cross-Asset (yields, equities, commodities → propagation matrix)
- Session Model (learned volatility multipliers)
- State Circuit Breaker (freeze/quarantine/recovery)

## Phase 7 — Learning & Evolution (Ongoing)
- **Replay-Safe Mode Controller** (disables learning during replay)
- Prediction Recorder (feedback loop infrastructure)
- Parameter Registry (provenance tracking)
- Learning Governor (bounds, validation, shadow deployment)

---

# 26. Complete Runtime Protection Pipeline (Final)

```
                    Restart

                       |
                       v
                Bootstrap Engine
                       |
                       v
             Fidelity Assignment
                       |
                       v
        ┌─────────────────────────────┐
        │  Snapshot Validation        │
        │  - checksum (temp/quarantine)│
        │  - commit ID consistency    │
        │  - watermark continuity      │
        └─────────────────────────────┘
                       |
                       v
        ┌─────────────────────────────┐
        │  Replay Coverage Check      │
        │  - actual_start <= request  │
        │  - hot/archive availability │
        └─────────────────────────────┘
                       |
                       v
        ┌─────────────────────────────┐
        │  Gap Replay                 │
        │  - fidelity tier selection  │
        │  - no learning, market-time │
        │  - event phase recon        │
        └─────────────────────────────┘
                       |
                       v
        ┌─────────────────────────────┐
        │  Pre-Live Validation (2-ph) │
        │  - deterministic hash match │
        │  - currency ∑ ≈ 0           │
        │  - residual < 3σ            │
        │  - volatility/spread sanity │
        └─────────────────────────────┘
                       |
                       v
        ┌─────────────────────────────┐
        │  Live Handoff               │
        │  - tick dedup (identity)    │
        │  - stabilization window     │
        │  - state confirmation       │
        └─────────────────────────────┘
                       |
                       v
        ┌─────────────────────────────┐
        │  State Integrity Monitor    │
        │  - continuous scoring       │
        │  - basket confirmation      │
        │  - pair reconstruction      │
        │  - temporal continuity      │
        └─────────────────────────────┘
                       |
                       v
        ┌─────────────────────────────┐
        │  Direction Hypothesis       │
        │  - uncertainty fusion       │
        │  - fidelity tag propagation │
        └─────────────────────────────┘
                       |
                       v
        ┌─────────────────────────────┐
        │  Pre-Trade State Gate       │
        │  - fidelity check           │
        │  - integrity >= 0.85        │
        │  - bootstrap >= 0.75        │
        │  - currency uncertainty     │
        │  - stabilization check      │
        └─────────────────────────────┘
                    |          |
                  PASS        FAIL
                    |          |
                    v          v
                  DRS       Block + Escalation
                    |
                    v
              Execution

                    ↑
        ┌───────────┴───────────┐
        │  Emergency Escalation │
        │  L0: warning          │
        │  L1: trading pause    │
        │  L2: auto recovery    │
        │  L3: degraded rebuild │
        │  L4: factory reset    │
        └───────────────────────┘
```

---

# 27. Final Design Principles (Expanded)

1. **Ticks are observations. Currencies are latent variables. Timeframes are measurement scales. Direction is derived. DRS is capital allocation. Execution is the final actuator.**

2. **The engine should behave like a database** — it restores its state, not wakes up empty.

3. **The system is session-based, not daemon-based.** A persistent market-state machine that wakes up, reconstructs reality, operates, commits memory, and sleeps.

4. **Do not store the entire market forever. Store the level of memory required for each type of decision.**

5. **The system should not change itself arbitrarily.** It should measure when its assumptions stop working and propose controlled replacements.

6. **A watermark is a commit ID, not a timestamp.** Timestamps alone cannot represent distributed state consistency.

7. **Replay must fail loudly when data coverage is incomplete.** Never silently degrade.

8. **All replay-time logic uses market timestamps, never CPU time.** Every component must be deterministic under replay.

9. **Every approximation carries uncertainty forward.** If a state is reconstructed, every dependent state inherits reduced confidence.

10. **A state transition is atomic only when the entire graph reaches the same commit boundary.** Partial state commits are invalid state.

11. **The system must never confuse "I reconstructed a state" with "I trust this state."** Replay produces state. Integrity gates decide whether that state influences capital.

12. **The five non-negotiable rules:**
    - A watermark is a commit ID, not a timestamp
    - Replay must fail loudly when data coverage is incomplete
    - All replay-time logic uses market timestamps, never CPU time
    - Every approximation carries uncertainty forward
    - A state transition is atomic only when the entire graph reaches the same commit boundary

---

# 28. Direction Finding Reality — Part 1: The Core Question

## 28.1 Can Currency Decomposition Find Direction?

The uncomfortable answer: **The architecture has a plausible mechanism for discovering hidden currency-level information, but it does not yet prove directional edge.** It solves the problem "can we represent FX movement as a combination of currency latent states?" which is NOT the same as "can we predict future price movement?"

The entire system lives or dies on one question:

```
P(Return[t+h] > 0 | DirectionHypothesis[t])
```

### Confidence Calibration

The system outputs `confidence=0.63` — but this confidence has NO empirical meaning until calibrated. A calibration function must map raw confidence to historical win rate:

```python
@dataclass
class PredictionOutcome:
    timestamp: int
    symbol: str
    direction: float
    confidence: float
    horizon: int
    future_return: float
    success: bool
```

Calibration buckets example:

| Confidence Range | Predictions | Win Rate |
|---|---|---|
| 0.50-0.55 | 5000 | 51.2% |
| 0.55-0.60 | 4000 | 53.7% |
| 0.60-0.65 | 3000 | 57.9% |
| 0.65-0.70 | 2000 | 61.5% |
| 0.70-0.80 | 1000 | 67.2% |

A confidence of 0.63 means historically ~58%, not 63%. Calibration uses isotonic regression or Platt scaling.

**Minimum requirement before live trading**: Reliability condition — predicted 60% bucket must actually win 55-65%, not 48%.

### Does Currency Decomposition Actually Improve Direction?

Three-model experimental protocol:

| Model | Input | Features |
|---|---|---|
| A — Pair only | EURUSD history | returns, volatility, momentum |
| B — Currency only | Currency states | EUR/USD strength, differential, confidence |
| C — Combined | Both | All above |

Dataset: 5 years, 28 pairs, 5-minute bars (~2.7M observations). Evaluation metrics:
- AUC (random=0.50, useful >0.55, strong >0.60)
- Information Coefficient (signal vs future return correlation, need >0.03)

If Model A AUC=0.56 and Model B AUC=0.51 → currency decomposition failed.
If Model C AUC > Model A AUC → it adds value.

**Expectation**: Currency layer probably improves regime understanding and risk attribution, but should NOT be assumed to improve short-term direction prediction.

### Lead-Lag: Does Currency Strength Lead Price?

Critical test: measure `corr(C[t], R[t+h])` for h = 5m, 15m, 1h, 4h, 1d — AND backward correlation `corr(C[t], R[t-h])`.

- If future correlation positive AND past correlation weak → currency leads (GOOD)
- If both equal → currency just explains current movement (BAD)

Also: impulse response analysis — after currency strength crosses +0.5, how many minutes until price responds? If median lead time is negative → signal is late, system is useless.

## 28.2 Multi-Timeframe Conflict Resolution

Different timeframes answer DIFFERENT questions — they should not vote equally:

| Timeframe | Role |
|---|---|
| 5m | Entry timing |
| 1h | Directional bias |
| 4h | Macro regime |

Concrete rule:

```python
def timeframe_decision():
    regime = tf4h.direction      # must not oppose
    bias = tf1h.direction        # sets direction
    trigger = tf5m.direction     # entry timing
    if regime conflicts: return BLOCK
    if bias != trigger: return WAIT
    return ENTER
```

### DRS Timeframe Conflict

DRS must NOT see EURUSD 5m BUY and EURUSD 1h SELL as two separate opportunities. Collapse into `SymbolDirectionField`:

```python
@dataclass
class SymbolDirectionField:
    symbol: str
    direction: float
    timeframe_components: dict    # {5m: +0.8, 1h: -0.6, 4h: 0}
    conflict_score: float
    final_score: float           # weighted: 0.2×5m + 0.5×1h + 0.3×4h
```

DRS ranks `SymbolDirectionField` objects — not individual timeframe signals.

## 28.3 Silent Failure Detection

When all integrity checks pass but every trade loses, the first metric to break is **Prediction Calibration Drift**:

```python
ModelHealthMonitor:
- Expected Calibration Error (ECE)
- Direction IC decay (rolling signal vs future return correlation)
- Regime-conditioned failure analysis
- Prediction entropy collapse
```

**Suspension triggers**: 30-day IC < 0 OR ECE > 0.15.

## 28.4 Minimum Viable Direction Test

Before building the full architecture, run this experiment:

1. Build ONLY: Currency WLS solver + currency strength time series + future return predictor
2. Test on EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, USDCHF
3. At every 5-minute timestamp, compute EUR/USD strength, generate signal = EUR-USD
4. Predict EURUSD return at 15m, 1h, 4h, 1d horizons
5. Required: AUC > 0.55, IC > 0.03, consistent across 4/5 years
6. Benchmark: must beat random (0.50 AUC) AND simple EMA crossover

**Abandon criteria**: Currency model AUC < 0.52 or IC < 0.01 or edge disappears after spread costs.

---

# 29. Direction Finding Reality — Part 2: The Uncomfortable Truth About FX Predictability

## 29.1 The Predictability Ceiling

FX direction predictability is NOT uniform — it depends on horizon and regime:

| Horizon | Typical AUC | Dominant Factors |
|---|---|---|
| Tick (ms-s) | 0.50-0.53 | Order flow, liquidity, market maker inventory |
| 5-minute | 0.51-0.56 | Noise dominates |
| 1-hour | 0.53-0.60 | Macro drift, session effects, carry, positioning |
| 4-hour/daily | 0.55-0.65 | Rate differentials, monetary policy, risk sentiment |

The real ceiling is ECONOMIC, not statistical. A model with AUC=0.58 can still lose money:
```
EV = P(win)×reward - (1-P(win))×loss - cost
55% wins × 5 pips - 45% losses × 5 pips - 1.3 pip spread = -0.8 pips
```

**Target should not be "AUC > 0.60 at 5m"** — that is probably unrealistic. Target: **Find conditional regimes where AUC temporarily rises** (e.g., normal market AUC=0.52, after macro shock AUC=0.62).

## 29.2 The Spread Tax

Break-even win rate for typical FX trade: Target 10 pips, Stop 10 pips, Cost 2 pips:
```
p_w(10) - (1-p_w)(10) - 2 = 0 → p_w = 60%
```

A 55% directional model LOSES money unless winners are larger or losers smaller. The system should therefore predict NOT direction alone but:
```
ExpectedEdge = P(direction) × ExpectedRange - Spread - Slippage
```

## 29.3 What Regime Does Currency Strength Exploit?

Three possible mechanisms:

| Mechanism | Description | Works In | Fails In |
|---|---|---|---|
| Trend persistence | Strong currencies remain strong | Monetary cycles, risk trends | Ranges, reversals |
| Relative value mean reversion | Currency divergence overshoots then normalizes | Ranges, liquidity shocks | Genuine regime shifts |
| Event repricing | Information changes currency state | Macro events | Normal periods |

The edge is likely in **Currency State Velocity** (ΔC/Δt), not absolute strength. The profitable variable is not "EUR = +0.6" but "EUR changed from -0.1 to +0.6 in 30 minutes."

## 29.4 Majors vs Exotics — Pair Dependency

A single model across all 28 pairs is probably wrong. Use pair profiles:

```python
EURUSD: liquidity=1.0, spread_cost=low, model=trend
EURTRY: liquidity=0.2, spread=high, model=event_only
```

Three tiers:
- **Tier 1** — Major liquid (EURUSD, USDJPY, GBPUSD, AUDUSD): clean attribution
- **Tier 2** — Commodity currencies (USDCAD, AUDJPY, NZDJPY): macro regimes
- **Tier 3** — Exotics: event-driven only

## 29.5 Horizon Mismatch Problem

The architecture processes ticks, solves currency every 5s, but makes decisions on 1h cycles — 720 attribution updates per decision. The correct architecture is a hierarchy, not one signal:

```
4h: "EUR bullish"          → Regime
1h: "EUR strengthening"     → State
5m: "entry timing"         → Timing
Decision = Regime × State × Timing
```

## 29.6 The Overfitting Risk

Bad architecture: 28 pairs × 20 unique features = 560 degrees of freedom — very dangerous.
Better: Shared currency model (8 currency parameters) + pair adjustments = 100-300 effective parameters.

With 5 years of data (millions of bars), the ratio is good. But adaptive systems create HIDDEN overfitting from research decisions (choosing horizons, filters, thresholds). Protection: pre-register hypothesis, test horizon, and success metric BEFORE tuning.

## 29.7 "This Already Exists" Problem

Currency strength meters and factor models exist at every bank. What is genuinely new?

| Potentially Novel | Not Novel |
|---|---|
| State representation with uncertainty + residual + regime + fidelity | Calculating currency strength |
| Separation of explanation vs prediction | Point estimate of EUR/USD strength |
| Residual intelligence — asking what REMAINS unexplained | Simple currency meters |
| Information quality awareness (replay confidence, fidelity) | Assuming all data is equal |

**Brutal verdict**: The edge probably does NOT come from "currency strength predicts price." That is too weak. The more promising hypothesis: **the edge comes from identifying when the market's internal state becomes asymmetric enough that normal FX randomness temporarily breaks.**

**Revised success criteria**: 80% of time do nothing; 20% of time operate where expected edge > cost.

**The minimum scientific question**: When currency latent strength diverges sharply from historical equilibrium, does future pair displacement increase enough to overcome transaction costs? If yes → architecture has a foundation. If no → elegant decomposition engine with no trading edge.

---

# 30. Direction Finding Reality — Part 3: The Decision Boundary

## 30.1 The Threshold Problem — Why Fixed Confidence Thresholds Fail

A fixed rule like `if confidence >= 0.63: trade` creates instability at the boundary (0.631 trades, 0.629 does not — same signal, tiny noise). Replace with **Expected Value Boundary**:

```
EV = p × Reward - (1-p) × Risk - Cost
Trade only when EV > 0
```

Where `p = Calibration(confidence)` maps raw confidence to historical win rate.

Example:
```
p = 0.57, expected move = 15 pips, stop = 10 pips, cost = 2 pips
EV = 0.57(15) - 0.43(10) - 2 = 8.55 - 4.3 - 2 = +2.25 → TRADE
```

Another example:
```
p = 0.65, expected move = 3 pips, cost = 2 pips
EV = 1.95 - 3.5 - 2 = NEGATIVE → NO TRADE (even though confidence is higher)
```

**Minimum entry rule**:
```python
trade_allowed = (
    expected_value > minimum_edge
    AND uncertainty < maximum_uncertainty
    AND state_integrity > threshold
)
```

Initial values: `minimum_edge = +0.5R`, `uncertainty < 0.30`, `integrity > 0.85`.

## 30.2 The Consecutive Confirmation Problem

Do NOT confirm direction — confirm **state persistence**. Confirmation state machine:

```
NEW_SIGNAL → OBSERVING → CONFIRMED → EXECUTABLE
```

```python
@dataclass
class SignalPersistence:
    first_seen: int
    age: int
    confidence_history: list
    direction_stability: float
    velocity_confirmation: float
```

Example rule: 3 observations within 15 minutes, direction sign unchanged across ALL observations, confidence decay < 15%. If direction flips at any observation → reset.

**Why N=3?** Probability of random same-sign for N observations = 0.5^N. N=3 → 12.5% false persistence. N=4 → 6.25%. Tradeoff between delay and false positives.

## 30.3 The Opposite Signal Problem

A new opposite signal does NOT automatically mean reverse. Three cases:

| Case | Current | New | Action |
|---|---|---|---|
| New signal weak | BUY 0.70 | SELL 0.55 | **HOLD** |
| New signal stronger | BUY 0.60 | SELL 0.75 | **EXIT, WAIT (no immediate reverse)** |
| Regime transition | BUY thesis invalid | — | **CLOSE ALL** |

Reverse requires INDEPENDENT confirmation: exit confirmation AND new entry confirmation in separate cycles. Never `close BUY; open SELL` in same cycle.

## 30.4 The Signal Decay Problem

Every DirectionHypothesis has a half-life:

```python
created_at: int
signal_age: int
signal_half_life: int        # 30 min for 5m system, 4h for 1h system
```

Decay: `D[t] = D[0] × exp(-λt)`. Signal expires when EV_after_cost ≤ 0 or age > max_age.

**Price already moved problem**: Track opportunity remaining:
```
RemainingEdge = ExpectedMove - PriceDisplacement - Cost
```
If remaining ≤ 0 → cancel.

## 30.5 The Zero-Position Idle Mode

The system must prove "I am inactive because there are no opportunities" not "I am broken." Every cycle, run the **Opportunity Scanner**:

```python
@dataclass
class MarketOpportunityReport:
    top_candidate: str
    max_EV: float
    rejection_reason: str
    signal_distribution: dict
```

**Health test**: System is healthy if every cycle has `hypotheses_generated > 0` AND `rejected_by_gate > 0`. Broken if zero hypotheses for 6+ hours.

## 30.6 The Position-Sizing Boundary

Direction model outputs **Opportunity Quality**, not allocation:

```python
@dataclass
class TradeOpportunity:
    symbol: str
    direction: float
    expected_value: float
    confidence: float
    uncertainty: float
    expected_move: float
    correlation_group: str
```

Risk engine receives this and produces allocation:
```
size = baseRisk × confidence × (1 - uncertainty) × volatilityAdjustment
```

Direction uncertainty MUST survive into position sizing. If confidence=0.75 but uncertainty=0.35, the allocation factor becomes 0.75 × 0.65 = 0.488 — a meaningful reduction.

## 30.7 The Flatten-All Signal (Signal-Based, Not Risk-Based)

Three conditions where the market representation itself becomes unreliable:

| Condition | Trigger | Action |
|---|---|---|
| **Currency Collapse** | σ(all currencies) < 0.05 for 30 min → no structure | Stop entries, evaluate positions |
| **Attribution Failure** | Median residual > 3σ → model not explaining price | Stop entries |
| **Regime Unknown** | Regime confidence < 0.2 | Stop entries |

**Close all only if**: state_invalid AND positions depend on invalidated thesis.

## 30.8 Complete Decision Pipeline

```
Market State Graph
        ↓
Direction Hypothesis
        ↓
Opportunity Evaluator (EV > threshold?)
        ↓
Confirmation Engine (persistence check)
        ↓
State Integrity Gate (fidelity, integrity, uncertainty)
        ↓
DRS Ranking (SymbolDirectionField, not raw signals)
        ↓
Portfolio Risk Engine (size, correlation, max loss)
        ↓
Execution
```

### Final Trigger Algorithm (Concrete)

```python
def should_enter(opportunity):
    if state.integrity < 0.85: return False
    if opportunity.expected_value < 0.5: return False
    if opportunity.uncertainty > 0.30: return False
    if opportunity.persistence_score < 0.7: return False
    if opportunity.age > opportunity.max_age: return False
    if portfolio.rejects(opportunity): return False
    return True
```

The decision boundary is not a confidence threshold. It is a **sequence of survival tests** where the signal earns the right to consume capital.

---

# 31. Direction Finding Reality — Part 4: The Competition

## 31.1 The Counterparty Problem

When the system buys EURUSD at 1.0850, the seller on the other side falls into four categories:

| Counterparty | First name | What They Want | Can The System Beat Them? |
|---|---|---|---|
| Retail noise | RSI, fear, stop loss | Exit position | Yes — systematic model outperforms emotions |
| Corporate hedger | Revenue exposure | Manage risk, not alpha | Can exploit predictable flows |
| Market maker | Inventory management | Neutralize risk, not direction | Yes — their flow is not directional conviction |
| **Institutional alpha** | **Better model, data, execution** | **Extract profit** | **No** — cannot compete on this battlefield |

**The system must target forced flows + delayed information + behavioral inefficiencies**, not informed institutional alpha.

### New Required Object: ParticipantPressure

```python
@dataclass
class ParticipantPressure:
    informed_flow_probability: float
    forced_flow_probability: float
    liquidity_stress: float
    crowding_score: float
```

EUR strength +0.6 with liquidity_stress=0.9 → possible forced liquidation, different interpretation than normal.

## 31.2 Market Maker's Advantage

The information hierarchy (highest to lowest):
1. Market maker order book / queue depth
2. Bank client flow
3. Institutional models
4. Tick-derived features ← THIS SYSTEM IS HERE
5. Retail indicators

**A tick model cannot beat a market maker's inventory model at short horizons.** However, useful order flow proxies exist:

```python
OrderFlowProxy:   # Treat as shadows, NOT actual order flow
- Tick direction persistence (consecutive bid lifts)
- Price velocity (ΔPrice/ΔTime)
- Spread compression
- Volume proxy (tick frequency)
```

## 31.3 The Bank Flow Problem — Is Currency Attribution Structurally Late?

Partially yes. The naive pipeline (EUR buying starts → price rises → WLS detects EUR strength → system buys) is late. The model is explaining, not predicting.

The transformation: use NOT `C[t]` but **`ΔC[t]` and `ΔC/Δt`** (currency state velocity). The useful signal is not "EUR is strong" but "EUR just became strong faster than expected."

### Residual Analysis — Three Cases

| Case | Currency Model | Actual Pair | Interpretation |
|---|---|---|---|
| 1 | +0.5 | +0.5 | Normal repricing |
| 2 | 0 | +1.0 | Residual explosion: news, institution flow, intervention |
| 3 | +0.6 | 0 | Absorption: someone selling into buying (MOST INTERESTING) |

## 31.4 Strategy Crowding Problem

If many systems detect EUR strong → they all buy EURUSD → signal self-destructs. Add a **Crowding Detector**:

```python
Crowding = w1×SpreadExpansion + w2×SlippageIncrease + w3×SignalDecay
```

Inputs: historical signal effectiveness decay, entry slippage (expected 0.2 vs actual 1.0 pips), price reaction speed (signal appears → price instantly jumps → info already absorbed), residual compression.

**Action**: if crowding > 0.7, reduce size.

## 31.5 HFT Competition — Can Python Windows Compete at Tick Horizons?

No. Not realistically.

Latency hierarchy: Exchange colocated HFT (μs) → Institutional OMS (ms) → Broker VPS (tens of ms) → Home MT5 Python (tens to hundreds of ms).

**The architecture must abandon**: tick prediction, microsecond response, scalping.

**Viable horizon**: 15 minutes to several days. A 4-hour currency repricing makes a 100ms disadvantage irrelevant. Explicit `minimum_signal_horizon` control.

## 31.6 Central Bank Problem

Central banks are not maximizing profit — they can destroy models (SNB EURCHF floor removal). Add an **Intervention Detector**:

Signature: multi-pair residual explosion (all CHF pairs collapse), velocity anomaly (instant, not gradual), spread anomaly (liquidity disappears).

```python
InterventionScore = f(currency_residual, velocity, spread, crosspair_coherence)
# CHF residual > 5σ AND 3 CHF pairs move AND spread > 99th percentile = intervention
```

Action: freeze currency model, do not trade.

## 31.7 Zero Sum and Adaptive Markets

**Edge half-life**: how long until alpha decays by 50%?

| Edge Type | Half-Life |
|---|---|
| Technical pattern | Days to months |
| Simple factor | Years |
| Execution edge | Milliseconds |
| Macro inefficiency | Years |
| Behavioral anomaly | Months |

The architecture must track `Edge(t) = future return conditional on signal` monthly. If IC drops from 0.08 to 0.02 over 12 months → edge decay → adapt or retire.

### Realistic Battlefield for This System

| WINNABLE | NOT WINNABLE |
|---|---|
| Corporate flow | HFT latency |
| Retail behavior | Market maker inventory |
| Slow institutional adjustment | Bank client flow |
| Regime transitions | Central bank surprise |
| Macro repricing | Institutional order flow |

**The alpha must come from moments where `Market_state ≠ Market_price` and the gap is large enough that slower participants have not yet corrected it. That is the only realistic reason this system should exist.**

---

# 32. Direction Finding Reality — Part 5: The Correlation Trap

## 32.1 The Hidden Contradiction

The Market State Graph thinks in **currencies** (8 latent variables). The DRS thinks in **pairs** (28 tradeable instruments). These are not the same risk unit.

```
BUY EURUSD
BUY EURJPY
BUY EURGBP
```

DRS sees: 3 trades.

Currency exposure: +3 EUR, -1 USD, -1 JPY, -1 GBP.

The system is holding one dominant thesis (EUR appreciation), disguised as three independent ideas.

**The DRS must stop ranking pairs independently. The ranking unit must become: currency thesis + pair expression.**

## 32.2 Currency Exposure Graph

Every pair is an 8-dimensional currency vector:

```python
# Order: [USD, EUR, GBP, JPY, AUD, NZD, CAD, CHF]
# EURUSD BUY:
[-1, +1,  0,  0,  0,  0,  0,  0]   # long EUR, short USD
# EURJPY BUY:
[ 0, +1,  0, -1,  0,  0,  0,  0]
# EURGBP BUY:
[ 0, +1, -1,  0,  0,  0,  0,  0]
```

Portfolio = vector sum → normalized → EUR concentration = 50%.

### Currency Concentration Score

```python
@dataclass
class CurrencyExposure:
    currency: str
    net_exposure: float
    concentration: float   # max(|Exposure_i|)
```

Thresholds: <35% OK, 35-50% WARNING, >50% BLOCK.

**DRS modification**: `DRS' = DRS × (1 - CurrencyPenalty)` where penalty rises with concentration.

## 32.3 Expression Quality — Choosing the Best Vehicle

When the thesis is EUR strong, which pair expresses it best? Not by confidence alone.

```python
ExpressionQuality = (CurrencyAlignment × Liquidity × ExpectedMove) / (ResidualRisk × Cost)
```

The system asks not "which pair predicts EUR best?" but "which pair is the cheapest, cleanest vehicle for this currency thesis?"

## 32.4 Contradiction Detection

EURUSD BUY implies EUR +0.6, USD -0.5. USDCHF BUY implies USD +0.4, CHF -0.3. USD cannot be both weak and strong.

### Currency Contradiction Engine

Every pair generates `CurrencyClaim(currency, implied_direction, confidence, source_pair)`.

```python
Conflict = |C₁ - C₂| × Confidence
thresholds: <0.3 normal, 0.3-0.6 uncertainty, >0.6 contradiction
```

Resolution: identify the weakest explanation by residual. If EURUSD residual=0.05 and USDCHF residual=0.8, USDCHF is the outlier — reduce USDCHF confidence.

```python
AdjustedConfidence = Confidence × (1 - Conflict) × (1 - Residual)
```

## 32.5 Currency-Level Risk Model

Replace pair-level exposure with currency-level limits. Instead of "max 3 positions":

```
max EUR exposure < 200k
max USD exposure < 300k
max JPY exposure < 150k
```

### Portfolio Selection Algorithm

```python
def select_positions(opportunities):
    ranked = []
    for opp in opportunities:
        projected = portfolio + opp.vector
        if projected.currency_concentration > limit:
            continue
        score = opp.DRS * diversification_factor
        ranked.append(score)
    return top3(ranked)
```

## 32.6 Pair Decision Formula

EUR strong (+0.7) does not mean all EUR pairs rise. EURUSD also depends on USD (+0.6): difference = +0.1 (weak). EURJPY depends on JPY (-0.5): difference = +1.2 (strong).

```python
PairDirection = BaseCurrencyStrength - QuoteCurrencyStrength + Residual
```

The graph is correct. Pairs differ because of counterpart currencies.

## 32.7 Residual-Based Position Management

When latent currency model says EUR is strong but EURUSD is falling:
- **Small residual**: HOLD (temporary noise)
- **Large, temporary residual**: WAIT (duration < timeout)
- **Large, persistent residual**: EXIT (duration > timeout → model failure)

## 32.8 Revised DRS Formula

```
DRS = Signal × Calibration × StateQuality × ExpressionQuality × DiversificationFactor × (1 - Crowding)
```

## 32.9 Brutal Conclusion

28 pairs are not 28 opportunities. They are **28 measurements of 8 underlying currency states**. The real object being traded is not EURUSD — it is a position in currency space expressed through a pair.

DRS changes from "pick the best three pairs" to **"allocate the best three expressions of independent currency theses."** That is the difference between a retail pair scanner and a portfolio-level FX system.

---

# 33. Direction Finding Reality — Part 6: The Wound That Doesn't Heal

## 33.1 The Missing Layer

The architecture has NO explicit post-trade learning loop. Trades open, close, and are logged. Parameters stay the same. The same losing trade pattern repeats. A losing trade contains a compressed diagnostic message: was the world misread? Timing wrong? Thesis correct but execution poor? Without extracting that information, the system repeats identical mistakes.

## 33.2 Trade Autopsy

```python
@dataclass
class TradeAutopsy:
    trade_id: str; symbol: str; direction: int
    entry_currency_state: dict; entry_pair_state: dict
    entry_regime: str; entry_confidence: float
    pnl_r: float; holding_time_minutes: float
    currency_error: float; residual_error: float
    timing_error: float; execution_error: float
    failure_type: str; blame_target: str
    blame_probability: float; should_update_model: bool
```

### Failure Classification Tree

```
                LOSS
                  |
            Is direction correct eventually?
            /                          \
          YES                          NO
           |                            |
     Timing Error              Direction Error
                                    |
                            currency_error > threshold?
                            /                        \
                          YES                         NO
                           |                           |
              Currency Attribution              residual > 3σ?
                    Failure                    /                \
                                              YES                NO
                                               |                  |
                                        Event/Residual      Risk/Execution
                                          Failure             Failure
```

## 33.3 Blame Attribution

Each subsystem receives a blame score: `Blame_i = Evidence_i × Impact_i × Confidence_i`. Subsystems: Currency Engine, Pair Model, Calibration, Decision Layer, Risk, Execution. Normalize and assign primary blame to exactly one.

## 33.4 Bayesian Currency Prior Update

Current: `C = WLS`. Revised: `C_t = α·C_WLS + (1-α)·C_prior`.

Each currency maintains: `CurrencyBelief(value, confidence, reliability)`. After a loss where currency prediction was wrong: `confidence_new = confidence_old - learningRate × error`. Not a catastrophic halving — a controlled 4-5% reduction per confirmed failure.

## 33.5 Learning Rate & Time Decay

- One trade → small update. Many trades → larger update (Bayesian evidence accumulation)
- Old mistakes decay: `weight = severity × exp(-age_days/90)`
- 30 days → 0.7 weight, 180 days → 0.1 weight

## 33.6 Anti-Confirmation Bias

After winning streak: increase decision threshold (harder to trade). After losing streak: decrease threshold slightly (easier) but reduce size. Never: "losing streak = trade more."

```
EV Threshold = BaseThreshold + WinningStreakPenalty
5 wins → threshold 0.5 → 0.7
3 losses → threshold 0.5 → 0.45 (but 50% size)
```

## 33.7 Statistical Loss vs Structural Failure

| Diagnosis | Pattern | Action |
|---|---|---|
| Statistical loss | Matches historical win-rate characteristics | No update |
| Regime shift | One regime's win rate dropped significantly | Reset regime model |
| Structural failure | ALL regimes deteriorating | Retire model |

## 33.8 Meta-Governor

- No parameter changes from <20 trades
- Maximum daily adjustment: 5%
- Never learn during unknown regime
- Require 3 independent failures before structural change

## 33.9 Brutal Conclusion

A trading system should not ask "Did I win or lose?" It should ask **"Which internal belief was falsified?"** A loss without diagnosis is a failure. Every losing trade must become a scientific experiment: prediction → observation → difference → hypothesis → controlled action.

---

# 34. Direction Finding Reality — Part 7: How Not to Die — Minimum Viable Architecture

## 34.1 The Danger of Beautiful Architecture

Six iterations produced an intellectually coherent system: Market State Graph → Currency Attribution → DRS → Portfolio Control → Learning Loop. A production system does not die from missing features. It dies from: (1) nobody can prove the first assumption, (2) too many components hide the source of failure, (3) feedback loop arrives too late.

**The first version should prove the core insight creates measurable information advantage. Everything else is optional.**

## 34.2 The MVP Cut

### Required

| Component | Why |
|---|---|
| Multi-pair data ingestion (10-12 pairs) | Currency is latent, pairs are observations |
| Currency Attribution Engine | Core thesis — without it the system collapses to a retail pair scanner |
| Simple Direction Model (currency diff + residual + momentum) | Not the full Market State Graph yet |
| Risk Engine (fixed fractional sizing, stop, daily loss, kill switch) | Capital preservation |
| Measurement Layer (forward return, hit rate, calibration, spread-adjusted expectancy) | Must answer "did the signal work?" not "did the trade make money?" |

### Deferred

| Component | Reason |
|---|---|
| Historical Archive | Useful for operations, not proof |
| Replay Engine | Cold start acceptable during research |
| Portfolio Factor Controller | Need multiple live trades first |
| DRS | One signal ranking problem first |
| Learning Loop | A bad learner makes a bad model worse |
| Correlation Engine | Need a portfolio first |
| Macro Layer | Need to prove price-derived state first |
| Market State Graph | Replace with simple direction model initially |

## 34.3 Minimum Pair Count

With EURUSD alone: 1 equation, 2 unknowns (EUR, USD) → infinite solutions. Need at least 7 independent pair equations for 8 currencies (effective unknowns = 7 because FX is relative). Minimum practical universe: 10-12 pairs:

```
EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD, NZDUSD, EURJPY, GBPJPY, EURGBP
```

## 34.4 Proof-of-Value Build Sequence

| Phase | Duration | Build | Success Criteria |
|---|---|---|---|
| **0: Data** | 1-2 days | MT5 collector → normalized tick/bar store | No missing timestamps, no corrupted symbols |
| **1: Currency Lab** | 3-5 days | Pair returns → WLS solver → currency strength vector | Does EUR rise during ECB hawkish? Does JPY react to BOJ? |
| **2: Predictive Validation** | 1 week | Test if `C_t` predicts `Return_{t+n}` | IC, AUC, calibration curve. Confidence bucket 0.7-0.8 → actual 58%+ |
| **3: Sim Trading** | 1-2 weeks | Entry, exit, spread, slippage, risk | Does edge survive after costs? |
| **4: Production Shell** | Ongoing | Snapshots, replay, DRS, portfolio, dashboards | Operational stability |

**Stop if Phase 1 fails: currency attribution makes no economic sense. Nothing else matters.**

## 34.5 The Non-Negotiable Core

**Currency attribution.** Remove it and you have: `price → indicator → signal` — every retail system. The core thesis is: **"The market is a system of interacting currency states, not independent pairs."** That is the entire reason to build this.

## 34.6 80/20 Component Priority

| Component | Value | Cost | Risk | Phase |
|---|---|---|---|---|
| Currency Attribution | Very High | Medium | Medium | 1 |
| Data Pipeline | Very High | Low | Low | 1 |
| Direction Validation | Very High | Low | Low | 1 |
| Simple Risk Engine | High | Low | Low | 1 |
| Residual Analysis | High | Medium | Medium | 2 |
| Confidence Calibration | High | Medium | Low | 2 |
| Regime Detection | Medium | High | High | 3 |
| Portfolio Exposure | High | Medium | Low | 3 |
| DRS | Medium | Medium | Medium | 3 |
| Replay Engine | Operational High | High | Low | 4 |
| Snapshot System | Operational High | Medium | Low | 4 |
| Macro Layer | Unknown | High | High | 5 |
| Learning Loop | High eventually | High | Very High | 5 |

## 34.7 Testing Sequence (Non-Negotiable)

| Stage | What | Evidence Required |
|---|---|---|
| Historical Reconstruction | 3-5 years: does currency state make economic sense? | Qualitative sanity |
| Walk-forward Simulation | Training 2019-2023, validation 2024, OOS 2025 | Positive expectancy |
| Forward Walk | Unseen recent data | Same Sharpe |
| Paper Trading | Live data, no real money | Execution matches simulation |
| Micro Live | 0.01 lots, 100 trades | Real slippage, real psychology |

**Before real money: 300+ historical trades, positive expectancy after costs, stable across 3+ regimes, out-of-sample confirmation.**

## 34.8 The Three Death Scenarios

1. **Currency attribution is elegant but late**: WLS detects the move after it happened; costs consume edge. Early warning: signal returns decay with holding horizon. Kill switch: if confidence bucket expectancy < spread + slippage, disable trading.

2. **Overfitting to one regime**: works in 2024, fails in 2025. Early warning: training Sharpe = 1.8, live = 0.2. Kill switch: regime-specific performance collapse.

3. **Architecture complexity kills development**: 4000-line design → 20,000-line code → nobody understands failures. Kill switch: every subsystem must have input, output, metric, and test before addition.

## 34.9 The 90-Day Build Plan

- **Month 1**: Prove currencies can be extracted (Phase 0 + Phase 1)
- **Month 2**: Prove currency state predicts future returns (Phase 2)
- **Month 3**: Prove costs can be overcome (Phase 3)

Only then add: Market State Graph, Replay, Portfolio, Learning, Macro.

## 34.10 Final Verdict

The architecture's survival depends on one experiment:

> **Does a latent currency state extracted from multiple FX pairs contain forward information about future pair returns after realistic costs?**

If **no**: the entire architecture is a beautiful machine built around a false premise.

If **yes**: every later layer becomes justified — replay improves reliability, DRS improves allocation, learning improves adaptation, macro improves context.

**The first mission is not building a trading organism. The first mission is finding out whether the organism has a heartbeat.**

---

# 35. Integration Surgery — Part 1: System Inventory & Mapping (Proxima Demo → Proposal)

## 35.1 Executive Diagnosis

The current Proxima system is not a failed architecture — it is a **late-stage execution platform attached to an immature intelligence layer**. The proposal architecture is not replacing Proxima. It is replacing the missing middle: a unified market state representation that explains all signals.

The migration should not delete Proxima. It should **demote many current signals from decision makers into observation sensors**.

| Current System | Proposal System |
|---|---|
| "Which signal should win?" | "What hidden market state generated all these signals?" |

## 35.2 Layer-by-Layer Mapping

### Layer 0 — Market Data Foundation (~85% complete)

| Current Component | Status | Proposal Destination |
|---|---|---|
| MT5 connector | EXISTS_MATCH | Tick ingestion layer |
| MT5 tick poll 0.2s | EXISTS_MATCH | Tick stream adapter |
| TickBuffer | EXISTS_MATCH | Market Data Buffer |
| ECDF transform | EXISTS_NEEDS_WORK | Feature normalization |
| 28-pair symbol universe | EXISTS_MATCH | Currency graph input |
| ReplayEnvironment | EXISTS_MATCH | Bootstrap / replay engine |

### Layer 1 — Market State Graph (~0% complete — MISSING)

No equivalent exists. Closest: MVS Observer, ECDF, Shadow — but these are pair-space objects, not currency-space objects. Must build: `MarketStateGraph(CurrencyNode[], PairEdge[])`.

### Layer 2 — Currency Attribution Engine (~5% complete — MISSING)

No component solves `P = AC + ε`. Closest assets (ECDF, OSS drift, Shadow score) do not infer latent currencies. This is the largest architectural gap.

### Layer 3 — Direction Engine (~20% complete — CONFLICTS)

Five-direction competition (OSS, TPI, Shadow, Alpha, MVS) is the largest philosophical conflict. Fix: demote from generators to **sensors**:

| Current | New Role |
|---|---|
| OSS | Temporal persistence feature |
| TPI | Microstructure pressure feature |
| Shadow | Residual/anomaly feature |
| Alpha strategies | Market behavior features |
| MVS | State observation feature |

### Layer 4 — Temporal Intelligence (~45% complete — EXISTS_NEEDS_WORK)

Partial assets: OSS, MVS, TPI persistence, curvature. Missing: unified temporal state, market-time aging, fidelity tagging.

### Layer 5 — Portfolio Intelligence (~70% complete — EXISTS_NEEDS_WORK)

Actually one of the strongest areas. DRS with displacement field, slot inertia, and decay is more advanced than the proposal's DRS description. Problem: pair ranking instead of currency exposure ranking.

### Layer 6 — Risk (~90% complete — EXISTS_MATCH)

Very strong. RiskManager, catastrophic stop, kill switch, spread normalizer, drawdown manager, H20 cap. Proposal risk layer is weaker — do NOT rewrite.

### Layer 7 — Execution (~90% complete — EXISTS_MATCH)

MT5 connector, OrderManager, PositionManager, ExecutionRouter, ExecutionResult. Already production-oriented.

### Layer 8 — Persistence (~10% complete — MISSING)

No snapshots, no watermark, no state serialization, no archive. Major gap.

### Layer 9 — Learning (~50% complete — EXISTS_NEEDS_WORK)

Current learning (AFL, FWO, RSL, TCA, CWF, DRL) learns execution/system behavior, not "which market belief failed." Need TradeAutopsy layer.

### Layer 10 — Safeguards (~85% complete)

Already strong: 14-gate pipeline, gate trace/audit, funnel monitoring.

## 35.3 Overall Completeness Score

| Layer | Completion |
|---|---|
| Data Layer | 85% |
| Currency Attribution | 5% |
| Market State Graph | 0% |
| Direction Engine | 20% |
| Temporal State | 45% |
| DRS & Portfolio | 70% |
| Risk | 90% |
| Execution | 90% |
| Persistence/Snapshot | 10% |
| Learning | 50% |
| Safeguards/Integrity | 85% |
| **Overall** | **~45-50%** |

The missing 50% contains the most important intellectual component (currency attribution and market state graph).

## 35.4 Hidden Assets

| Asset | Value | Proposal Gap |
|---|---|---|
| DRS Displacement Field | Extremely high — memory, inertia, replacement pressure | Maps directly to Portfolio Allocation Engine |
| SDL | Symbol-level lock, regime-conditioned decay | Must upgrade to CurrencyThesisLock |
| Replay Environment | Burst mode, latency mode, clock patching | Already 80% of what proposal Phase 1 builds |
| Risk Pipeline | Institutional quality | Leave untouched |
| Gate Trace / Audit | Explainability infrastructure | Becomes State Explanation Trace |
| ExecutionResult | Correct abstraction | Reuse directly |

## 35.5 Dangerous Components (Corruption Risks)

1. **Synthetic exploration direction**: fake direction injected when no signal exists. Must be removed — exploration may explore parameters, states, or thresholds, NEVER fake market direction.
2. **Multiple direction arbitration**: OSS vs Shadow vs TPI vs Alpha creates two brains (Currency Graph Direction + Old Arbitration). Kill arbitration. Keep sensors.
3. **Wall clock aging**: `time.time()` instead of market timestamp. Breaks state after restart — system thinks state is fresh when actually stale.
4. **SDL symbol isolation**: EURUSD lock and GBPUSD lock don't understand EUR thesis. Must become currency-level.
5. **TOP3 pair ranking**: Can select EURUSD + EURJPY + EURAUD — same bet three times.

## 35.6 Three Quick Wins (Smallest Proof-of-Value Changes)

1. **Shadow Currency Attribution Prototype** (2-3 days): Create `currency_strength.py` using existing ECDF pair returns to solve WLS. Run beside system. Question answered: "Does latent currency state exist?"
2. **Currency-aware DRS** (1-2 days): Replace pair-level DRS score with `EUR strength - USD strength` alignment.
3. **Currency Exposure Report** (1 day): Print portfolio-level currency exposure before trading. No execution changes.

## 35.7 Rewrite Ratio

| Category | Percentage | Content |
|---|---|---|
| Reusable as-is | 35% | MT5, execution, risk, replay, DRS mechanics |
| Modifiable | 40% | OSS, TPI, Shadow, SDL, gates, learning |
| Must rewrite | 20% | Direction architecture, currency graph, attribution, confidence fusion |
| Delete | 5% | Synthetic exploration direction, duplicate arbitration logic, obsolete experiments |

## 35.8 Final Migration Strategy

Do NOT rewrite Proxima. Strategy:

1. Freeze execution
2. Build Currency Attribution beside it
3. Convert existing signals into sensors
4. Feed new DirectionHypothesis into existing DRS
5. Gradually retire pair-level arbitration

The fastest path is turning Proxima from a committee of competing opinions into a hierarchy:

```
Market State → Currency Reality → Pair Expression → Portfolio Decision → Execution
```

---

# 36. Integration Surgery — Part 2: Migration Path & Incremental Transformation

## 36.1 Executive Principle

The migration mistake would be "replace the old brain with the new brain" — a flag-day rewrite. The correct operation: **keep the body alive, implant a new brain, compare outputs, disconnect old reflexes one by one.**

**Only the signal authority is being replaced. Everything else — risk, execution, MT5 handling, position management — remains stable.**

## 36.2 TickCache → TickBus Adapter

**Minimum change**: wrap existing TickCache with subscriber pattern, not replace it.

```python
class TickCache:
    def __init__(self):
        self.cache = {}
        self.subscribers = []

    def subscribe(self, consumer):
        self.subscribers.append(consumer)

    def update(self, symbol, tick):
        self.cache[symbol] = tick
        for consumer in self.subscribers:
            consumer.on_tick(symbol, tick)
```

Create `TickEvent` envelope with symbol, bid, ask, timestamp, source, sequence. Old system unchanged. New CurrencyGraph receives a copy.

## 36.3 DirectionAuthority Layer

The most important migration component. Not a signal generator — it decides WHICH signal source influences execution.

### Three Modes

| Mode | Old System | New System | Execution Uses |
|---|---|---|---|
| SHADOW | Full authority | Observes only | Old system (100%) |
| HYBRID | Generates direction | Can veto | Old with new veto power |
| NEW_ONLY | Becomes sensor | Full authority | New system |

### Promotion Criteria (Not Time-Based)

```
promote if:
  new_IC > old_IC × 1.5        (over minimum 300 signals)
  AND new_EV > old_EV
  AND calibration_error < 0.1
```

### Architecture

```
Old Direction (OSS/TPI/Shadow/Alpha) → OldHypothesis
New Direction (CurrencyGraph) → NewHypothesis
                                   ↓
                            DirectionAuthority
                                   ↓
                                DRS / Execution
```

## 36.4 Position Handover via Magic Number Encoding

No position tagging exists. Solution: encode origin in magic number digits.

```
Digits: [system][signal_source][strategy][version]
Example: 24101 = system=2(new), source=4(Graph), strategy=1, version=01
```

Old positions tagged `origin="LEGACY"`. New DRS monitors but does not auto-close legacy positions. Exit only if risk violation OR new system opposite confidence > 0.85 with residual invalidation.

## 36.5 Gate Migration

Do NOT create 14+5=19 gates. New gates are a **State Validity Layer** that feeds INTO existing gates:

```
State Validation → DirectionAuthority → Existing Proxima Gates → Risk → Execution
```

| New Check | Status | Existing Replacement |
|---|---|---|
| Fidelity | NEW | — |
| Integrity | EXISTS | AuditLogger + Replay |
| Bootstrap | NEW | — |
| Currency uncertainty | NEW | — |
| Stabilization | NEW | — |

Keep: SDL, KillSwitch, Risk, H20, Spread controls. Retire later: CF gate, Reality vector, duplicate direction checks.

## 36.6 Minimum Crash Recovery (Before Full Snapshot)

Every 60 seconds, atomic-write crash state:

```json
{"timestamp": ..., "positions": [...], "last_ticks": {}, "DRS_state": {}, "mode": "LIVE"}
```

On restart: load state, compare MT5 positions, fetch missing ticks, resume. Does not need full Market Graph snapshot — needs **execution continuity**.

## 36.7 Retirement Schedule

| Component | Action | Trigger |
|---|---|---|
| OSS | ADAPT → sensor | CurrencyGraph IC > OSS IC × 1.5 for 300 events |
| TPI | KEEP permanently | Entry timing sensor — never retire |
| Alpha strategies | ADAPT → market feature | If graph+alpha doesn't improve EV over 500 trades |
| Shadow | ADAPT → residual detector | When residual correlation > shadow correlation |
| MVS Observer | KEEP | Market State Health Monitor — never retire |
| TOP3 rotation | ADAPT → currency thesis allocator | Live currency exposure engine |
| H20 Cap Engine | KEEP permanently | Risk survives architecture changes |

## 36.8 Final Migration Blueprint

```
                  MT5 → TickCache → TickBus Adapter
                                      |
                      +---------------+---------------+
                      |                               |
              Legacy Proxima                    CurrencyGraph
         OSS/TPI/Shadow/Alpha/MVS           Currency State
                      |                               |
                      +---------------+---------------+
                                      |
                              DirectionAuthority
                                      |
                                Unified DRS
                                      |
                            Existing Risk Stack
                                      |
                              MT5 Execution
```

## 36.9 First 30-Day Surgery Plan

| Week | Action | State |
|---|---|---|
| 1 | TickBus adapter, crash state file, CurrencyGraph prototype | No execution change |
| 2 | Both systems run: old trades, graph observes. Collect IC, calibration, EV | SHADOW |
| 3 | DirectionAuthority SHADOW mode | SHADOW |
| 4 | If metrics pass → HYBRID mode | HYBRID |

**Core rule**: execution remains boring. The only thing changing is who is allowed to suggest direction.

---

# 37. Integration Surgery — Part 3: Code Reuse Strategy (File-Level Decisions)

## 37.1 Executive Decision

Do NOT create `proxima_market_state` as a replacement project. That creates a second system and guarantees divergence. The correct structure lives inside `proxima_x/`:

```
proxima_x/
├── market_state/         # New brain (currency graph, WLS, state)
├── adapters/             # Old ↔ new translators (TickBus, signal adapters)
├── migration/            # DirectionAuthority, position handover
├── persistence/          # Snapshots, runtime state
├── currency/             # WLS solver, currency state
├── graph/                # MarketStateGraph
├── temporal/             # Temporal state engine
├── direction/            # DirectionHypothesis, confidence fusion
└── existing/             # Everything that already works
```

**Never** `from run_proxima_demo import *`. New code imports only from new modules.

## 37.2 File-Level Reuse Table

### Execution Layer — All KEEP_AS_IS

| File | Decision | Reason |
|---|---|---|
| `mt5_connector.py` | KEEP_AS_IS | Production-critical MT5 interface |
| `order_manager.py` | KEEP_AS_IS | Independent from signal architecture |
| `position_manager.py` | ADAPT | Add position provenance/source tags |
| `execution_router.py` | KEEP_AS_IS | Direction source should not affect execution |
| `execution_mapper.py` | KEEP_AS_IS | LONG/SHORT conversion remains valid |

### Risk Layer — All KEEP_AS_IS

| File | Decision | Reason |
|---|---|---|
| `risk_manager.py` | KEEP_AS_IS | Already architecture independent |
| `catastrophic_stop.py` | KEEP_AS_IS | Must survive migration |
| `spread_normalizer.py` | KEEP_AS_IS | Execution reality layer |
| H20 cap engine | KEEP_AS_IS | Portfolio safety |

### Signal Layer — Mostly ADAPT

| File | Decision | New Role |
|---|---|---|
| `outcome_surface_signal.py` | ADAPT | Signal → `TemporalPersistenceFeature(persistence, drift_velocity, stability)` |
| `get_tpi_signal.py` | ADAPT | Signal → `MicrostructureState(flow_pressure, imbalance, velocity)` |
| `shadow_mirror.py` | ADAPT | Direction → `ResidualState(expected_move, actual_move, residual, anomaly_score)` |
| Alpha strategies | ADAPT | Independent traders → feature generators |
| `market_graph.py` | SCAFFOLD | Name matches, architecture does not. Rewrite internals. |
| `tick_thermodynamics.py` | ADAPT | Signal → Market Energy State in TemporalState |
| `meta_state.py` | ADAPT | Becomes `MarketRegimeState` |
| `observer_collapse.py` | RETIRE | Proposal already has uncertainty/confidence/integrity |
| `rejection_engine.py` | ADAPT | Signal rejection → state quality rejection |
| `entropy_compression.py` | ADAPT | Signal → market disorder metric |

### MVS

| File | Decision | New Role |
|---|---|---|
| `observer_features.py` | KEEP + ADAPT | Becomes `MarketStateHealthFeatures` |
| `observer_decay.py` | ADAPT | CPU time aging → market timestamp aging |
| `weak_day_detector.py` | KEEP | Market activity regime detector |

### DRS / Portfolio

| File | Decision | Change |
|---|---|---|
| `ranking_engine.py` | ADAPT | Pair ranking → currency thesis expression ranking |
| `topk_rotation_engine.py` | ADAPT | Symbol score input → currency-adjusted opportunity score |

### Learning

| File | Decision | Change |
|---|---|---|
| `afl_engine.py` | ADAPT | Becomes graph state parameter learning |
| `fwo_engine.py` | ADAPT | Becomes graph state parameter learning |
| `rsl_engine.py` | KEEP | Regime-segmented learning still valid |
| `tca_engine.py` | KEEP | Temporal credit assignment still valid |
| `cwf_engine.py` | ADAPT | Causal weight fusion for graph parameters |

### Monitoring

| File | Decision |
|---|---|
| `gate_audit_logger.py` | KEEP_AS_IS |
| `funnel_dashboard.py` | ADAPT (add currency state metrics) |
| `signal_statistics.py` | ADAPT (track sensor features, not signals) |
| `execution_statistics.py` | KEEP_AS_IS |

## 37.3 Retirement Candidates

| Module | Decision | Reason |
|---|---|---|
| `research/reality_convergence/` | ARCHIVE | Useful research, not runtime. Move to `research/archive/` |
| `research/deployment_reality/` | KEEP | Already discovered starvation, gate problems — becomes architecture validation suite |
| `observer_collapse.py` | RETIRE | Duplicated by proposal's uncertainty/confidence/integrity |
| `entropy_compression.py` | ADAPT (not retire) | Keep as market disorder feature |

## 37.4 30-Line WLS Prototype

Create `research/currency_test.py` immediately. This is the minimum proof that latent currency structure exists:

```python
import numpy as np
pairs = [("EUR","USD",0.004), ("GBP","USD",0.003),
         ("USD","JPY",-0.002), ("EUR","JPY",0.001)]
currencies = ["EUR","USD","GBP","JPY"]
A, P = [], []
for base, quote, r in pairs:
    row = [0]*len(currencies)
    row[currencies.index(base)] = 1
    row[currencies.index(quote)] = -1
    A.append(row); P.append(r)
C = np.linalg.lstsq(np.array(A), np.array(P), rcond=None)[0]
for c, v in zip(currencies, C): print(c, v)
```

Output: `EUR +0.002 USD -0.001 GBP +0.001 JPY -0.003`. This is enough to ask: "Does the 28-pair WLS solution produce meaningful currency states?"

## 37.5 Calibration Reuse

`TPICalibrationLayer` is partially reusable. Refactor to `ProbabilityCalibrator`:

```python
class ProbabilityCalibrator:
    # Reuse: Platt scaling, isotonic regression, bins, reliability curves
    # Discard: TPI-specific thresholds
    def fit(self, predictions: list, outcomes: list) -> None: ...
    def predict(self, score: float, metadata: dict) -> float: ...
```

## 37.6 Shadow Repurposing

Shadow becomes Residual State:

```python
@dataclass
class ResidualState:
    symbol: str
    expected_move: float    # from currency graph
    actual_move: float      # from market
    residual: float         # actual - expected
    anomaly_score: float
```

Remove `shadow_direction`. Add `expected_return` and `observed_return`.

## 37.7 Highest ROI Changes (Sorted by Impact per Line)

| # | Change | Cost (lines) | Impact | Description |
|---|---|---|---|---|
| 1 | Currency WLS shadow engine | ~200 | ★★★★★ | Tests entire thesis |
| 2 | Currency strength in DRS | 5-10 | ★★★★★ | Instantly makes DRS currency-aware |
| 3 | DirectionAuthority | ~150 | ★★★★★ | Enables migration without breaking execution |
| 4 | TickBus adapter | ~20 | ★★★★☆ | Enables entire future architecture |
| 5 | Snapshot Lite | ~100 | ★★★★☆ | Stops silent death |

## 37.8 Build Order

```
1. TickBus adapter         → feeds both systems
2. Currency WLS shadow     → tests core thesis
3. Currency-aware DRS      → quick alignment win
4. DirectionAuthority      → safe migration layer
5. Snapshot persistence    → crash survival
6. Convert signals→sensors → retire old direction system
7. Full Market State Graph → final architecture
```

## 37.9 Final Surgery Verdict

The missing intelligence layer is approximately **3000-5000 lines** of new code — not a 20,000-line rewrite. The first proof point: **"Can a WLS currency state extracted from the existing 28-pair tick stream improve DRS ranking?"** If yes, the architecture has a heart. If no, building the rest is wasted effort.

---

# 38. Integration Surgery — Part 4: Quality & Purity Threats

## 38.1 Executive Diagnosis

The most dangerous migration mistake is feeding mathematically clean inputs from a contaminated execution environment into a mathematically elegant model. The currency graph is fragile because it assumes `Observation → Truth → State`, while the current pipeline often does `Observation → heuristics → filters → arbitration → synthetic recovery → state`.

**The new architecture must create a purity boundary: Old Proxima may consume Market State. Old Proxima may NOT modify Market State.**

## 38.2 Contamination Channels

### Channel A — Tick Data Quality (CRITICAL)

The WLS engine assumes `Pᵢ` is a clean observation. It is not. Missing ticks, stale ticks, and timestamp mismatch between pairs create phantom currency strength.

**Required**: `TickQualityFilter` before every WLS solve:

```python
@dataclass
class TickQuality:
    symbol: str; timestamp: int; age_ms: int
    duplicate: bool; sequence_gap: int
    spread_ratio: float; quality_score: float
```

Reject if: age > 5s, duplicate, spread > 5x normal, sequence gap > threshold.

WLS input becomes `WeightedObservation(return, weight, quality)`.

### Channel B — Five Direction Gods (EXTREME)

If both old and new direction systems exist, the old one will bias the new one. The fix is absolute separation:

```
Tick → Market State Graph → DirectionHypothesis → DirectionAuthority → Legacy Sensors (features only)
```

Legacy signals may write `FeatureVector` but NEVER `DirectionHypothesis`. Forbidden: `graph.confidence -= shadow_conflict`. Allowed: `hypothesis.features.shadow_residual = value`.

### Channel C — State Persistence Contamination (HIGH)

Process dies → memory disappears → graph believes stale state. Every state object needs `StateHeader(created_market_time, last_tick_time, snapshot_id, fidelity, source_hash)`. Before every solve: if `current_time - last_tick_time > threshold`, set `state.status = "STALE"`. No execution on stale state.

### Channel D — Learning Pollution (HIGH)

Current learning engines (AFL, FWO, CWF) optimize for OSS/TPI/Shadow. If left active during migration, they will pull the graph's parameters toward old-system behavior. **Freeze all learning during currency graph validation period** (minimum 300-500 shadow decisions). Then replace objective: old = signal accuracy, new = DirectionHypothesis EV.

### Channel E — Gate Trace Contamination (EXTREME)

The graph output should not pass through 14 direction-modifying gates. **Separate into Intelligence gates (must be retired: Reality Vector, CF Gate, old arbitration, synthetic exploration) and Safety gates (remain: KillSwitch, RiskManager, SDL converted, spread checks)**.

## 38.3 Mathematical Assumptions That Break

### Fixed Incidence Matrix A

Monitor `A_integrity_score` via median residual. If `residual > 3σ` for N consecutive solves → `INCIDENCE_DEGRADATION` → no trading.

### Linear WLS

During crisis, relationships become nonlinear. Track `R² = 1 - Σε²/Σ(P-P̄)²`. If `R² < 0.5` → `NON_LINEAR_REGIME` → increase uncertainty, don't force bad solve.

### Closed Currency System (ΣC = 0)

External drivers (commodities, yields, equities) push ΣC away from zero. Track constraint drift `D_c = |ΣC|`. If large → add macro node, don't force currency values.

## 38.4 Time Contamination Map

| Current Clock | Severity | Replace With |
|---|---|---|
| `_now_ts()` (wall clock) | CRITICAL | `market_clock.now()` |
| `age_cycles` (loop count) | HIGH | `market_seconds_elapsed` |
| SDL decay (cycle-based) | HIGH | `current_tick_time - last_update_time` |
| TPI persistence (tick count, no time window) | MEDIUM | `tick_time` window, not last 200 ticks |

200 ticks can mean 2 seconds or 20 minutes → must use timestamp windows.

## 38.5 Microstructure Contamination

TPI operates at milliseconds, currency graph at seconds. Direct injection creates aliasing. Solution: **Flow Aggregation Layer** — aggregate ticks into 5-second `FlowState(buy_pressure, sell_pressure, imbalance, decay)`.

## 38.6 Pre-WLS Purity Gate

```python
def pre_wls_quality_gate(pair_data):
    check_symbol_coverage()     # min 26/28 pairs
    check_timestamp_alignment() # max 2s cross-pair skew
    check_spread_quality()      # reject >5x median
    check_duplicate_rate()      # reject >1%
    check_missing_ticks()       # no gaps >5s
    check_residual_health()     # reject median residual >3σ
    check_symbol_registry()     # 100% mapped
    return aggregate(checks)
```

## 38.7 Execution Blocking Threshold

```python
quality_score = 0.3×tick_quality + 0.25×time_alignment + 0.2×spread_quality + 0.15×coverage + 0.1×residual_health
# ≥0.95: normal execution
# 0.85-0.95: reduced confidence
# <0.85: block
```

## 38.8 Final Contamination Map

| Threat | Severity | Protection |
|---|---|---|
| Tick inconsistency | CRITICAL | TickQualityGate |
| Old direction arbitration | EXTREME | DirectionAuthority isolation |
| Missing snapshots | HIGH | StateHeader + persistence |
| Old learning engines | HIGH | Freeze/retrain with new objective |
| 14-gate contamination | EXTREME | Separate intelligence/safety gates |
| Fixed incidence matrix failure | HIGH | Residual monitoring |
| Linear model failure | MEDIUM | R² regime detector |
| Clock contamination | CRITICAL | MarketClock |
| TPI aliasing | MEDIUM | Flow aggregation |
| Duplicate ticks | MEDIUM | Fingerprint dedup |
| Spread distortion | HIGH | WLS weights |
| Symbol mismatch | CRITICAL | SymbolRegistry |

## 38.9 The Migration Purity Rule

The new architecture survives only if this boundary is enforced:

```
Old Proxima (OSS, TPI, Shadow, Alpha, Gates) → features only → 
    Market State Graph → DirectionHypothesis → Existing Risk → Execution
```

**The old system may provide observations. It may not provide truth.** That is the single rule preventing the 6174-line system from recreating the exact problem the new architecture is designed to solve.

---

# 39. Integration Surgery — Part 5: Architecture Compatibility (DNA Analysis)

## 39.1 Executive Diagnosis

Proxima has a strong **control/execution nervous system**. The proposal has a stronger **perception/intelligence system**. The merge should be:

```
Market Reality → Market State Graph → Direction Hypothesis → Proxima Control Plane → Execution
```

The mistake would be deleting the control plane. Proxima's control architecture should survive. Proxima's **belief architecture** should not.

## 39.2 Gate vs Graph Philosophy

### Are They Incompatible?

No — but gate roles must change. The current mistake is gates doing **intelligence work** (deciding direction) when they should only do **safety work** (protecting capital).

### Gate Reclassification

**Intelligence gates** (must become features or disappear):

| Current Gate | Future Role |
|---|---|
| Reality Vector | Feature input to graph |
| CF Gate | Feature input to graph |
| Alpha arbitration | Feature input to graph |
| Trigger direction filter | Feature input to graph |
| OSS/Shadow agreement | Feature input to graph |

**Safety gates** (remain unchanged):

| Gate | Future Role |
|---|---|
| KillSwitch | KEEP |
| RiskManager | KEEP |
| Spread checks | KEEP |
| Margin checks | KEEP |
| Execution validation | KEEP |
| Position limits | KEEP |

The graph does not remove gates. It removes **gate opinions**.

## 39.3 Shadow as Validation Architecture

The proposal underestimated this asset. Current Shadow is not just an alternative trader — it's a **counterfactual world engine**. This maps perfectly to the proposal's integrity layer.

### New Architecture: Shadow as Continuous Model Examiner

```
Market State Graph → DirectionHypothesis → [Execution Path | Counterfactual Shadow]
                                                            ↓
                                                  Outcome Comparator
                                                            ↓
                                                  Model Integrity Score
```

```python
@dataclass
class IntegrityObservation:
    hypothesis_id: str
    predicted_direction: float
    predicted_confidence: float
    realized_return: float
    residual_error: float
    calibration_error: float
    counterfactual_delta: float
```

Shadow becomes not "alternative trader" but "continuous model examiner." This is superior to a simple pre-trade gate.

## 39.4 Replay Architecture

Our replay environment (ReplayConfig + clock patching + burst + seed) was built for "replay decisions at speed." The proposal's replay was designed for "reconstruct missing state." These are different objectives.

**Verdict**: Extend existing replay (70% reuse). Add:

```python
@dataclass
class ReplayRange:
    start_market_time: int
    end_market_time: int
    source: str = "archive"
```

Add fidelity tracking (EXACT/APPROXIMATE/DEGRADED). Add component replay contracts (currency solver=RUN, execution=DISABLED, learning=DISABLED).

## 39.5 Merged Mode State Machine

Three independent state machines, not one combined:

| Concern | Current | New |
|---|---|---|
| Execution mode | SystemMode (LIVE/PAPER/SIM) | KEEP_AS_IS |
| State fidelity | — | NEW: DEGRADED → VALIDATING → STABLE → FULL |
| Authority | — | NEW: OLD_ONLY → SHADOW → HYBRID → NEW_ONLY |

```python
@dataclass
class SystemState:
    execution_mode: ExecutionMode
    fidelity_mode: FidelityMode
    authority_mode: AuthorityMode
    bootstrap_state: BootstrapState
```

After restart: `(LIVE, VALIDATING, OLD_ONLY)`. After promotion: `(LIVE, FULL, NEW_ONLY)`.

## 39.6 Runtime Loop

The current tight cycle (tick → process → decide → execute → repeat) is not ideal for a continuous market graph. But full rewrite is unnecessary. Use **hybrid event architecture**:

```
MT5 → Tick Event → Event Bus → [Market Graph (continuous) | Proxima Loop (periodic)]
```

The graph updates on every tick. DRS executes periodically — not every tick.

| Activity | Frequency |
|---|---|
| Tick → update graph | Every tick |
| Currency solve | Every 5 seconds |
| Generate hypothesis | Every 30 seconds |
| DRS + execution | Every cycle (unchanged) |

## 39.7 Merged Monitoring Dashboard

Both dashboards answer different questions. Keep both.

**Dashboard 1 — System Health** (existing):

| Metric | Purpose |
|---|---|
| Submitted/accepted/rejected | Pipeline debugging |
| Gate trace | Which gate blocked |
| Shadow GT vs SY | Decision comparison |

**Dashboard 2 — Market Intelligence** (new):

| Metric | Question |
|---|---|
| Currency residual mean(\|ε\|) | Does graph explain reality? |
| Attribution confidence | How sure is the graph? |
| State freshness (last tick age) | Is state current? |
| Fidelity score | EXACT/APPROX/DEGRADED breakdown |
| Direction calibration | confidence 0.7 → actual win rate 68%? |
| Shadow divergence | Graph prediction vs reality gap |

## 39.8 Design Decisions That Must Unconditionally Change

| # | Decision | Action | Reason |
|---|---|---|---|
| 1 | Multiple direction authorities | DELETE | One DirectionHypothesis only |
| 2 | Synthetic exploration direction | DELETE | Graph must never observe fake reality |
| 3 | Pair-level intelligence | REPLACE | EUR thesis + USD thesis + expression quality |
| 4 | Direction-changing gates | REMOVE | No gate may change BUY→SELL |
| 5 | CPU-time state aging | REMOVE | Everything becomes market timestamp based |
| 6 | Learning from signal outcomes | REPLACE | Learn "which market hypothesis failed?" not "was TPI right?" |
| 7 | Replay as separate reality | REPLACE | Replay reconstructs live state, not alternative timeline |

## 39.9 Summary: Keep vs Transform vs Remove

| Keep Proxima DNA | Transform | Remove |
|---|---|---|
| Execution | OSS → persistence feature | Synthetic direction |
| Risk | TPI → flow pressure sensor | Signal arbitration |
| Kill Switch | Shadow → residual detector | Direction-changing gates |
| Position Management | SDL → currency thesis lock | Pair-only worldview |
| DRS Mechanics | Gates → safety-only | Wall-clock aging |
| Shadow Framework | Learning → graph-parameter-optimized | — |
| Replay Engine | Monitoring → split health/intelligence | — |
| Audit System | DRS ranking → currency-aware | — |

**Proxima's control architecture should survive. Proxima's belief architecture should not.** The proposal is not replacing Proxima — it is replacing the part that was never unified: its understanding of what the market actually is.

---

# 40. Integration Surgery — Part 6: Data Infrastructure Leverage

## 40.1 Executive Diagnosis

The proposal dramatically underestimates how much infrastructure already exists inside Proxima. The missing pieces are not MT5 connectivity, replay, execution, risk, or learning primitives — those all exist. The real missing layer is the unified market state model, currency attribution, persistent state graph, provenance/fidelity, and tick event distribution.

The migration is not "build a new trading system" — it is **"install a new perception layer above existing infrastructure."**

## 40.2 The Archive Gap

Current reality: there is NO historical tick archive. The system relies on MT5's terminal history buffer (1-2M ticks/symbol, ~2-4 hours for EURUSD). The currency graph is synchronization-sensitive — WLS requires same-timestamp data across all pairs.

### Minimum Archive for MVP: Tier 0

Not the full 3-tier production archive. Just 7 days, 28 symbols, raw ticks in Parquet:

```python
TickRecord: symbol, timestamp, bid, ask, volume, spread
```

Estimated: ~5.6 GB total for 140M ticks at 40 bytes. Manageable.

**Build `storage/tick_archive.py` first. Everything else depends on it.**

## 40.3 Snapshot Gap

The proposal's full atomic snapshot system (SHA256, commit IDs, 5-15 min periodic) is production-grade. For MVP, only `CurrencySnapshot` is needed:

```python
@dataclass
class CurrencySnapshot:
    timestamp: int
    currency_strengths: dict
    pair_residuals: dict
    solver_weights: dict
    confidence: float
    schema_version: int
```

Save every 5 minutes. Expand later.

## 40.4 Tick Bus Reality

The MVP does NOT need TickBus. The existing TickCache (latest tick per symbol at 0.2s poll) supports a 5-second WLS solve. The solver needs current market state, not every micro tick.

Later, replace TickCache with TickBus when microstructure attribution and exact reconstruction are needed.

```
Now:   TickCache → CurrencyCollector → WLS Solver
Later: TickBus → (same interface)
```

Implement `TickProvider` interface that both can satisfy.

## 40.5 Symbol Universe Minimum

The graph does not need 28 pairs initially. Graph theory minimum: 8 currencies, need connected graph. Minimum viable: **10 pairs**:

```
EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD, EURJPY, GBPJPY, EURGBP
```

Better: 15-20 pairs. **Danger**: a clean 12-pair graph beats a dirty 28-pair graph. Implement `CurrencyUniverseSelector` with quality_score = freshness × spread_quality × tick_density.

## 40.6 Non-Intellectual Infrastructure Reuse

### Extract Immediately to `core_utils/`

| Current Component | New Location | Used By |
|---|---|---|
| `_get_broker_symbol()` | `core_utils/symbol_map.py` | MT5, archive, graph |
| `_atomic_write_json()` | `core_utils/storage.py` | Future snapshot foundation |
| `SpreadModel` | Direct reuse | WLS pair weights |
| `SymbolTrustModel` | ADAPT → `CurrencyBelief` | Bayesian EMA confidence |
| `compute_micro_volatility()` | `core_utils/volatility.py` | Pair weighting |
| `_bars_elapsed()` | REWRITE | Must use market timestamp, not CPU elapsed |
| MT5 rates provider | Direct reuse | — |

## 40.7 Windows/MT5 Reality

### Advantages (Proposal ignored)
- MT5 provides broker-normalized execution (symbol mapping, order handling, position state)
- Local replay environment with clock patching, burst, latency sim, seed control
- Existing safety infrastructure (kill switch, risk manager, spread checks)

### Limitations (Proposal ignored)
- MT5 Python API is synchronous and blocking → WLS solve / archive write must NOT occur inside MT5 polling thread. Architecture: `MT5 Thread → Queue → [Graph | Archive | Monitor]`
- Laptop sleep is catastrophic → need `EnvironmentMonitor` to detect wall-clock jumps and enter DEGRADED mode
- Antivirus: exclude `archive/`, `snapshots/`, `parquet/`

## 40.8 Minimum File Structure (10 Files, Not 40)

```
proxima_x/market_state/
├── currency_graph.py       # Central object: update_pair(), solve(), get_currency_strength()
├── wls_solver.py           # Pure math: solve_currency_strength(pair_returns, weights)
├── pair_state.py           # PairState(symbol, return_5s, volatility, spread, freshness)
├── tick_collector.py       # Adapter: TickCache → Graph
├── tick_archive.py         # Parquet writer
├── currency_snapshot.py    # Minimal persistence
├── direction_hypothesis.py # CurrencyState + PairState → Trade candidate
├── graph_quality.py        # Purity protection: freshness, spread, missing pairs, residual
└── tests/
    ├── test_wls.py
    └── test_replay.py
```

## 40.9 True MVP Build Sequence

| Phase | Build | Goal |
|---|---|---|
| **1: Currency Proof** | TickCache → CurrencyCollector → WLS Solver → Currency Dashboard | Does EUR strength explain EUR pairs? |
| **2: Persistence** | Snapshot, archive | State survives crash |
| **3: Direction** | CurrencyState + PairState → DirectionHypothesis | Generate tradable signal |
| **4: Integration** | Replace OSS/Shadow arbitration with DirectionHypothesis | Full pipeline migration |

## 40.10 Current Infrastructure Assessment

| Layer | Status |
|---|---|
| MT5 connectivity | 95% |
| Tick acquisition | 70% |
| Replay | 80% |
| Execution | 95% |
| Risk | 90% |
| Monitoring | 80% |
| Archive | 10% |
| Snapshot | 5% |
| Currency graph | 0% |
| Direction ontology | 20% |

The true missing system is much smaller than the original proposal suggests. The first experiment: **"Can a WLS currency latent-state model extracted from our existing MT5 stream explain future FX movement better than our current pair-level signals?"** Everything else waits for that answer.

---

# 41. Integration Surgery — Part 7: Minimum Viable Transition (The Monday Morning Plan)

## 41.1 Executive Verdict

After all 7 iterations: **Do NOT build the Market State Graph yet.** The first mission is: add a latent currency attribution sensor into the existing Proxima brain and measure whether it contains forward information that current OSS/TPI/Shadow do not contain.

The smallest proof:

```
MT5 ticks → Existing TickCache → WLS Currency Extractor (~150 lines new)
    → Currency Strength Vector
        → [DRS feature (existing) | Shadow comparison (parallel)]
    → IC measurement
```

If this fails, the entire Market State Graph architecture dies. If this works, build the architecture around it.

## 41.2 The 5-Day Experiment

| Day | Task | Goal |
|---|---|---|
| 1 | Collect 12 clean pairs (EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, NZDUSD, USDCAD, EURJPY, GBPJPY, EURGBP, AUDJPY, CADJPY) to Parquet | 5+ trading days of synchronized ticks |
| 2 | Implement WLS solver: `numpy.linalg.lstsq()` with ΣC=0 constraint, solve every 5 seconds | Produce currency strength time series |
| 3 | Predictive test: `C[t] → R[t+n]` for n=5m, 15m, 1h, 4h. Measure IC, AUC, EV | Does currency strength predict? |
| 4 | Compare against existing signals (OSS, TPI, Shadow, DRS). Calculate correlation orthogonality | Does currency add NEW information? |
| 5 | Decision: continue if IC > 0.03, AUC > 0.53, positive EV, AND correlation with existing signals < 0.7 | Kill or proceed |

### Abandon Criteria

If currency strength does NOT outperform simple EURUSD momentum → stop. The graph is unnecessary.

## 41.3 Minimum Code Changes (~175 Lines Total)

| File | Change | Lines |
|---|---|---|
| `run_proxima_demo.py` | Add `PAIR_CURRENCY_MAP` (base/quote for all 28 pairs), currency strength cache dict | 35 |
| `run_proxima_demo.py` | Inject WLS solve after ECDF before DRS: `currency_strength = solve_currency_state(pair_returns)` | 50 |
| `run_proxima_demo.py` | Modify `_compute_drs`: add `currency_alignment = base_strength - quote_strength` weighted at 0.15 | 10 |
| `signals/statistics` | Add IC tracking: store (timestamp, signal, future_return) for rolling IC | 40 |
| Dashboard module | Currency state panel: colored bars for EUR, USD, JPY, GBP, CHF, AUD, NZD, CAD | 40 |

**Total: ~175 lines added or modified, not thousands.**

### The One-Hour Hack

Inject directly inside `_compute_drs`:

```python
import numpy as np
A = np.zeros((len(pairs), 8))
P = np.zeros(len(pairs))
for i, (sym, base, quote) in enumerate(pairs):
    A[i, idx[base]] = 1
    A[i, idx[quote]] = -1
    P[i] = pair_return[sym]
C = np.linalg.lstsq(A, P, rcond=None)[0]
eur_usd_strength = C[idx["EUR"]] - C[idx["USD"]]
```

That is enough to prove concept.

## 41.4 Shadow Currency Test (Highest-Value Shortcut)

Redirect ShadowCore to consume currency strength instead of its own direction model. Shadow becomes two parallel systems:
- **GT Shadow**: existing direction model (alternative trader)
- **CG Shadow**: currency graph hypothesis (new sensor)

Compare outcomes. Do NOT remove old Shadow.

## 41.5 Funnel Audit IC Tracking

Extend `SignalStatistics` with three fields:

```python
currency_signal: float       # current signal value
currency_future_return: float # realized return at horizon
currency_ic: float           # rolling IC = corr(signals, returns)
```

No new analytics system — reuse existing infrastructure.

## 41.6 Dashboard Addition (<50 Lines)

Add Currency State panel to existing dashboard:

```
=====================
Currency Strength     EUR  +0.42 ████████
                      USD  -0.31 █████
                      JPY  -0.12 ██
                      GBP  +0.18 ███
```

Data source: existing JSON state. No new UI architecture.

## 41.7 Monday Morning Plan

| Day | Morning | Afternoon |
|---|---|---|
| Mon | currency map + WLS prototype (~80 lines) | Integrate DRS + dashboard (~50 lines) |
| Tue-Fri | Collect live observations, measure IC at 5m/15m/1h/4h | — |
| Fri | Decision: continue or abandon | — |

### Decision Tree

```
Currency IC > 0.03? → YES → Does it beat OSS/TPI? → YES → Build Graph
                    → NO  → Kill hypothesis
                    ───────────────────────────────→ NO  → Stop architecture
```

## 41.8 Final Verdict

The minimum transition is NOT `Proxima → rewrite → Market State Graph`. It is:

```
Proxima → add one new sensor (Currency Attribution) → measure truth
```

The entire future architecture depends on one empirical fact: **Does the latent currency state contain predictive information after costs?**

If **yes**: build Currency Graph, Temporal State, Bootstrap, Portfolio Controller, Learning Loop.

If **no**: the correct action is not to optimize the architecture. It is to remove the hypothesis.

**The Monday implementation target: Add WLS currency strength as a non-executing sensor, feed it into DRS shadow scoring, and measure IC for one week.**

---

# 42. Clean Build — Part 1: System Extraction (What to Bring from Proxima)

## 42.1 Executive Decision

Building a clean system alongside `run_proxima_demo.py` is the correct move. The mistake would be copying Proxima into a smaller folder — that recreates the contamination. The correct approach: **extract only the infrastructure that already solved hard engineering problems. Rebuild the intelligence layer cleanly.**

## 42.2 Target New Project Structure

First production paper version (~4,500 lines, no file > 300 lines):

```
proxima_clean/
├── main.py                         (~200)
├── config/
│   ├── symbols.py                  (~100)
│   └── settings.py                 (~100)
├── data/
│   ├── mt5_adapter.py              (~300)
│   ├── tick_store.py               (~200)
│   └── market_clock.py             (~100)
├── currency/
│   ├── graph.py                    (~300)
│   ├── wls_solver.py               (~150)
│   └── attribution.py              (~200)
├── direction/
│   ├── hypothesis.py               (~200)
│   └── confidence.py               (~150)
├── portfolio/
│   ├── drs.py                      (~300)
│   └── exposure.py                 (~200)
├── risk/
│   ├── risk_engine.py              (~250)
│   └── stops.py                    (~150)
├── execution/
│   ├── paper_executor.py           (~200)
│   ├── mt5_executor.py             (~250)
│   └── models.py                   (~100)
├── persistence/
│   ├── snapshot.py                 (~250)
│   └── storage.py                  (~150)
├── monitoring/
│   ├── dashboard_state.py          (~200)
│   └── metrics.py                  (~200)
└── tests/
```

## 42.3 Component Extraction Matrix

| Component | Decision | New File | Target Lines |
|---|---|---|---|
| MT5 connector | REWRITE_LIGHT | `data/mt5_adapter.py` | 300 |
| Execution router | REWRITE_LIGHT | `execution/mt5_executor.py` | 250 |
| Execution mapper | IMPORT | `execution/models.py` | 100 |
| Order manager | REWRITE_LIGHT | `execution/mt5_executor.py` | (merged) |
| Position manager | REWRITE_LIGHT | `execution/mt5_executor.py` | (merged) |
| Magic resolver | IMPORT | `config/symbols.py` | 100 |
| Risk manager | REWRITE_LIGHT | `risk/risk_engine.py` | 250 |
| Catastrophic stop | REWRITE_LIGHT | `risk/stops.py` | 150 |
| Spread normalizer | IMPORT | `risk/risk_engine.py` | (merged) |
| H20 cap | REWRITE_LIGHT | `risk/risk_engine.py` | (merged) |
| TickCache | REWRITE_LIGHT | `data/tick_store.py` | 200 |
| TickBuffer | WRAPPER | `data/tick_store.py` | (merged) |
| ECDF | REFERENCE | `data/normalizer.py` | 150 |
| Currency map | IMPORT | `config/symbols.py` | 100 |
| Atomic writer | IMPORT | `persistence/storage.py` | 50 |
| Broker symbol mapper | IMPORT | `data/mt5_adapter.py` | (merged) |
| Micro volatility | IMPORT | `data/tick_store.py` | (merged) |
| SymbolTrustModel | REWRITE_LIGHT | `currency/attribution.py` | 200 |
| SpreadModel | IMPORT | `risk/risk_engine.py` | (merged) |
| ReplayEnvironment | REFERENCE | `persistence/replay.py` | 250 |

## 42.4 Components That Stay in Proxima (Not Brought)

| Component | Reason |
|---|---|
| OSS | Old ontology: many signals compete. New ontology: one market state, many observations |
| Shadow direction generator | Replaced by currency graph as truth source |
| Alpha strategies | Replaced by currency features |
| Phase A arbitration | New system has no arbitration — one DirectionHypothesis |
| SDL | Replaced by currency thesis persistence |
| TOP3 rotation | Replaced by currency-aware DRS |
| Reality Vector, CF gate | Old intelligence gates removed |
| Exploration controller | No fake directions |
| Observer collapse, Entropy compression | Not needed for clean architecture |

## 42.5 Minimum Paper Trading Pipeline

The smallest working live system:

```
MT5 Adapter → Tick Store → Currency WLS → Currency Strength
→ Direction Hypothesis → DRS → Risk → Paper Executor
```

**No learning, no macro, no replay, no portfolio optimizer, no full snapshot yet.**

## 42.6 Live-Alive Criteria

The new system is considered alive when it produces:

```python
DirectionHypothesis(
    symbol="EURUSD", direction=0.62, confidence=0.71,
    reason={"EUR_strength": 0.42, "USD_strength": -0.20, "residual": 0.08}
)
```

every decision cycle.

## 42.7 Build Order

| Day | Focus | Files |
|---|---|---|---|
| 1-2 | Infrastructure | `mt5_adapter.py`, `tick_store.py`, `symbols.py`, `settings.py` |
| 3 | Mathematics | `wls_solver.py`, `graph.py`, `attribution.py` |
| 4 | Decision | `hypothesis.py`, `confidence.py`, `drs.py`, `exposure.py` |
| 5 | Safety & Live | `risk_engine.py`, `stops.py`, `paper_executor.py`, `mt5_executor.py`, `dashboard_state.py`, `main.py` |

---

# 43. Clean Build — Part 2: Folder Structure & Clean Architecture

## 43.1 Dependency Graph

Correct layering with DRS in `portfolio/` (not `engine/`):

```
main.py → runtime/
    ├── data/ (mt5_adapter, tick_store) → config/
    ├── engine/ (currency/graph + wls_solver, direction/hypothesis) → config/, data/models
    ├── portfolio/ (drs) → direction/, currency/, config/
    ├── risk/ (safety) → portfolio/, execution/, config/
    ├── execution/ (paper) → data/, config/
    ├── persistence/ (snapshot) → nothing
    └── monitoring/ (dashboard) → nothing
```

**Import rules:**
- `config/` imports nothing from project
- `data/` imports only `config/`
- `currency/` imports only `config/` and `data/models`
- `direction/` imports only `currency/` and `config/`
- `portfolio/` imports only `direction/`, `currency/`, `config/`
- `risk/` imports only `portfolio/`, `execution/`, `config/`
- `execution/` imports only `data/`, `config/`
- No circular imports. No engine submodule imports another engine submodule.

## 43.2 Interface Contracts

### `data/tick_store.py`
```python
class TickStore:
    def add_tick(self, tick: Tick) -> None
    def latest(self, symbol: str) -> Tick | None
    def get_window(self, symbol: str, seconds: int) -> list[Tick]
    def calculate_returns(self, symbols: list[str]) -> dict[str, float]
    def freshness(self, symbol: str) -> float
```

### `currency/wls_solver.py` (pure math, no state)
```python
class WLSSolver:
    def solve(self, pair_returns: dict[str, float], weights: dict[str, float]) -> dict[str, float]
```

### `currency/graph.py` (stateful, owns currency state)
```python
class CurrencyGraph:
    def update(self, returns: dict[str, float]) -> None
    def strength(self, currency: str) -> float
    def strengths(self) -> dict[str, float]
    def residual(self, symbol: str) -> float
    def quality(self) -> float
```

### `direction/hypothesis.py`
```python
@dataclass
class DirectionHypothesis:
    symbol: str; direction: float; confidence: float
    base_strength: float; quote_strength: float; residual: float

class HypothesisGenerator:
    def generate(self, graph: CurrencyGraph, symbol: str) -> DirectionHypothesis
```

### `portfolio/drs.py`
```python
class DRS:
    def rank(self, hypotheses: list[DirectionHypothesis]) -> list[DirectionHypothesis]
    def select(self, ranked: list) -> list
```

### `risk/safety.py`
```python
class RiskEngine:
    def approve(self, hypothesis, portfolio) -> bool
    def size(self, hypothesis) -> float
```

### `execution/paper.py`
```python
class PaperExecutor:
    def execute(self, hypothesis) -> ExecutionResult
    def positions(self) -> list
```

### `persistence/snapshot.py`
```python
class SnapshotManager:
    def save(self, state: dict) -> None
    def load(self) -> dict
```

## 43.3 Runtime Loop Design

| Component | Frequency |
|---|---|
| MT5 polling | 200ms |
| Tick storage | every tick |
| Freshness check | 1s |
| Currency solve | 5s |
| Hypothesis generation | 30s |
| DRS ranking | 30s |
| Risk check | on trade |
| Snapshot | 5 min |
| Dashboard | 2s |

```python
while running:
    ticks = mt5.poll()
    for tick in ticks:
        store.add_tick(tick)
    if clock.every(5):         # currency solve
        returns = store.calculate_returns()
        graph.update(returns)
    if clock.every(30):        # decision cycle
        hypotheses = [generator.generate(graph, s) for s in SYMBOLS]
        selected = drs.select(hypotheses)
        for trade in selected:
            if risk.approve(trade):
                executor.execute(trade)
    if clock.every(300):       # snapshot
        snapshot.save(runtime.state())
    if clock.every(2):         # dashboard
        dashboard.update()
    health.check()
    sleep(0.1)
```

Missing from basic example: **health watchdog**, **exception boundary** (try/except around loop), **market clock** (never `time.time()` for state).

## 43.4 Threading Model

**Two threads** (not asyncio, not single-thread). MT5 Python API is synchronous/blocking.

```
Tick Thread (200ms): MT5 API → queue.put(ticks)
Decision Thread: queue.get() → process(ticks)
```

Queue maxsize=10000. If >80%, drop oldest ticks.

## 43.5 Config Structure

```python
# config/settings.py
SYMBOLS = ["EURUSD", "GBPUSD", ...]
BASE_CURRENCY_MAP = {"EURUSD": ("EUR", "USD"), ...}
CURRENCY_LIST = ["EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"]

SOLVE_INTERVAL = 5          # seconds
DECISION_INTERVAL = 30      # seconds
SNAPSHOT_INTERVAL = 300     # seconds

WLS_REGULARIZATION = 0.01
MIN_SOLVE_PAIRS = 20

MAX_POSITIONS = 3
MAX_EXPOSURE_PER_CURRENCY = 200_000
STOP_LOSS_PIPS = 15
TAKE_PROFIT_PIPS = 30
MIN_CONFIDENCE = 0.60
SIGNAL_EXPIRY_SECONDS = 300

MAX_TICK_AGE_SECONDS = 5
MIN_SYMBOL_COVERAGE = 0.90
MAX_SPREAD_MULTIPLE = 5

INITIAL_CAPITAL = 10_000
LOT_SIZE = 0.01
```

## 43.6 Error Handling

| What Happens | Action |
|---|---|
| MT5 disconnects | Retry 5x, then snapshot + shutdown |
| WLS fails | Log, use previous state, reduce confidence |
| Bad symbol data | Remove pair, reduce graph quality, continue |
| Unexpected exception | Catch in main, snapshot state, graceful shutdown |

Logging levels: DEBUG (ticks), INFO (solves), WARNING (stale data), ERROR (connection), CRITICAL (corruption).

## 43.7 Minimum File Count (14 Files)

```
proxima_clean/
├── main.py
├── config/
│   └── settings.py
├── data/
│   ├── mt5_adapter.py
│   ├── tick_store.py
│   └── models.py
├── currency/
│   ├── wls_solver.py
│   └── graph.py
├── direction/
│   └── hypothesis.py
├── portfolio/
│   └── drs.py
├── risk/
│   └── safety.py
├── execution/
│   ├── models.py
│   └── paper.py
├── persistence/
│   └── snapshot.py
└── monitoring/
    └── dashboard.py
```

These boundaries matter: WLS stays pure, MT5 stays replaceable, paper execution becomes live execution later, risk never mixes with signal generation.

## 43.8 First Proof Milestone

**For 28 FX pairs, every 5 seconds, produce a stable currency vector and rank the top 3 pair opportunities in paper mode.** If this works, then add snapshot, archive, fidelity system, macro layer, learning in order.

---

# 44. Clean Build — Part 3: Live Paper Deployment Architecture & Runtime

## 44.1 Runtime Architecture

```
main.py → RuntimeManager
    ├── TickWorker (thread 1) → MT5 API → Queue
    ├── DecisionWorker (thread 2) → Queue → CurrencyGraph → Direction → DRS → Risk → Executor
    └── SnapshotManager (periodic save/load)
```

`main.py` is <100 lines — composes the system, does NOT own runtime logic.

## 44.2 RuntimeManager Lifecycle

```python
class RuntimeManager:
    def __init__(self, config): ...
    def start(self): initialize() → connect_MT5() → load_snapshot() → gap_detect() → start_workers()
    def shutdown(self): stop_workers() → save_snapshot() → disconnect()
```

## 44.3 Startup Sequence (Deterministic)

| Step | Timeout | Failure Mode |
|---|---|---|
| Load config | 100ms | crash |
| Initialize logger | 100ms | crash |
| Connect MT5 | 15s (3 retries: 2s, 5s, 10s) | WAITING_FOR_MT5 state, retry loop |
| Load snapshot | 500ms | fallback to previous snapshot, then factory boot |
| Validate snapshot | 200ms | factory boot (empty graph, confidence=0, paper only) |
| Gap detection | 100ms | treat as gap >24h |
| Gap fill | depends on gap size | see gap strategy |
| Data quality check | 5s | flag stale symbols, degrade confidence |
| Start workers | 100ms | — |

**Factory boot**: empty graph, zero confidence, paper-only mode, no live execution allowed.

## 44.4 Shutdown Sequence

```
Ctrl+C → shutdown_event.set() → stop tick intake → finish current calculation (≤5s)
→ snapshot (atomic: freeze → serialize → write temp → fsync → rename)
→ disconnect MT5 → exit
```

**Double Ctrl+C**: first triggers graceful shutdown, second within 5s calls `os._exit()`.

## 44.5 Queue Design

`Queue(maxsize=1000)` — not 10000 (at 200ms polling, 1000 = 200s backlog, too much).

```python
@dataclass
class TickBatch:
    ticks: list[Tick]
    market_timestamp: float
    sequence: int
    received_timestamp: float
```

**Full queue**: drop oldest (`queue.get()` before `queue.put()`) — never block MT5 thread.

**Decision worker**: drain ALL batches from queue, not one at a time:
```python
while True:
    batch = queue.get()
    batches = [batch]
    while not queue.empty():
        batches.append(queue.get_nowait())
    process(batches)
```

**Stale detection**: if `age = market_timestamp - now > 5s`, mark degraded.

## 44.6 Gap Filler (Hybrid Strategy)

| Gap | Strategy |
|---|---|
| ≤5 min | Use tick history from MT5 (calculate returns from snapshot price → current price) |
| 5 min – 24h | Use M1 bars from MT5, solve WLS per minute, update graph incrementally |
| >24h | Degraded bootstrap — restore snapshot, lower confidence, block live execution |

Never generate fake ticks.

## 44.7 Windows-Specific Concerns

- MT5 must be running before Python script starts — `mt5.initialize()` with retry
- Use `pathlib.Path` for all paths — never hardcoded `\` separators
- Snapshot: atomic rename on NTFS (`state.tmp` → `state.json`)
- Ctrl+C via `signal.SIGINT` — do not rely on SIGTERM
- Launch via `python -m proxima_clean` (not `python main.py`)

## 44.8 Paper/Live Switch

One-line config change:
```python
EXECUTION_MODE = "paper"  # or "live"
```

Factory pattern:
```python
executor = ExecutorFactory.create(mode=config.EXECUTION_MODE)
```

Risk scales with mode: `MAX_RISK = 0.25` for live, `1.0` for paper. Dashboard always displays mode prominently.

## 44.9 Terminal Dashboard (10 Lines)

Fixed refresh with `\r` or ANSI clear screen. 10 lines:
```
PROXIMA CLEAN | MODE: PAPER | 12:34:56
MT5: CONNECTED | TICKS: 1,245/s | QUALITY: 0.94
EUR +0.42 | USD -0.18 | GBP +0.31 | JPY -0.55
POSITIONS: EURUSD BUY 0.63 [+$1.23]
HEALTH: OK
```

Optional ANSI colors (green=healthy, yellow=degraded, red=blocked). Never make logic depend on colors.

## 44.10 First Operational Milestone

**Start → stop → restart → restore state → recover gap → continue producing identical currency states reliably.** If lifecycle fails, trading intelligence does not matter.

---

# 45. Clean Build — Part 4: Core Pipeline

## 45.1 Tick → Return Calculation

### Mid Price
Use `mid = (bid + ask) / 2` — WLS measures currency movement, not execution cost. Bid/ask contains spread noise that would contaminate the intelligence layer. Execution cost is modeled later at the execution stage.

### Log Returns
```python
r = log(last_mid / first_mid)
```
Log returns are additive across time, compatible with linear WLS assumption, and handle multi-window aggregation cleanly.

### Boundary Median Sampling
Per 5-second window, take `opening_price = median(first 20% of ticks)` and `closing_price = median(last 20% of ticks)`. This removes MT5 quote spikes, single bad ticks, and spread widening artifacts.

### Pair Coverage Minimum
- `coverage ≥ 0.90` (≥25 pairs): normal operation
- `coverage 0.75–0.90` (21–25): degraded mode
- `coverage < 0.75` (<21): no solve

Minimum 15 pairs required for the graph to be connected. Below 20 is risky.

## 45.2 WLS Solver Exact Design

### Core Equation
```
r_ij = strength_i - strength_j + ε
```

### Matrix Form
Currencies: `x = [EUR, USD, GBP, JPY, CHF, AUD, CAD, NZD]^T` (8×1)

Example rows of A (28×8):
- EURUSD: `[1, -1, 0, 0, 0, 0, 0, 0]`
- GBPUSD: `[0, -1, 1, 0, 0, 0, 0, 0]`
- USDJPY: `[0, 1, 0, -1, 0, 0, 0, 0]`

### WLS Solution
```
min (Ax-b)^T W (Ax-b) + λ(x - x_prior)^T(x - x_prior)
x = (A^T W A + λI)^{-1} (A^T W b + λ x_prior)
```

### Code Implementation
```python
AtW = A.T @ W
x = np.linalg.solve(AtW @ A + lam * np.eye(8), AtW @ b + lam * x_prior)
```

### Prior-Weighted Ridge
Shrink toward previous solve state, not zero: `λ(C - C_prior)²`. "If uncertain, remain near previous known state."

### Weight Calculation
`w_i = tick_count_factor * freshness_factor * spread_factor`
- EURUSD example: 1.0 × 1.0 × 1.0 = 1.0
- EURTRY example: 0.3 × 0.5 × 0.8 = 0.12

### Full Solve Every 5 Seconds
Cost is O(8³) = microseconds. No incremental update needed — the expensive part is tick collection, not math.

## 45.3 Direction Hypothesis

### Direction Formula
```python
direction = tanh(base_strength - quote_strength)  # range [-1, +1]
```
Normalized: tanh keeps output bounded. No unbounded raw strength drift.

### Confidence Formula
```python
confidence = abs(direction) * (1 - residual_score) * graph_quality * stability
```
Example: 0.80 × 0.90 × 0.95 × 0.90 = 0.615

### Residual Interpretation
`ε = r - (C_base - C_quote)` — what the model failed to explain with currency factors.
- z-score < 1.5: normal, full confidence
- 1.5–3.0: reduce confidence
- > 3.0: block (news event, liquidity shock, pair-specific flow)

### Minimum Confidence Threshold
`confidence ≥ 0.60` for MVP. Below: observe only, no signal.

## 45.4 DRS Ranking

### Scoring Formula
```python
score = 0.45 * abs(h.direction) + 0.25 * h.confidence + 0.20 * h.graph_quality + 0.10 * h.diversification
```
- Currency conviction: 45% — the primary signal
- Model confidence: 25% — how sure we are
- Graph quality: 20% — overall data health
- Diversification: 10% — prefer pairs with different currency exposure than existing positions

### Position Replacement (3-Slot Model)
- Slot 1 (anchor): inertia = 1.0, hardest to displace
- Slot 2 (adaptive): inertia = 0.85
- Slot 3 (flux): inertia = 0.7, easiest to replace

**Replacement rule**: new score must exceed `DRS_held + 0.10` margin. Opposite signals require confirmation + 2-cycle buffer.

### Position Decay
```
DRS(t) = DRS_entry * exp(-λt) + DRS_current * (1 - exp(-λt))
```
λ = 0.05 per decision cycle (less aggressive than old system's 0.1).

## 45.5 Non-Trading Decision

Never silently idle. Store decision reason every cycle: `{"decision": "NO_TRADE", "reason": "EURUSD confidence 0.52 < threshold"}`.
- Dashboard: show top candidate with its score/rejection reason
- Logging: dashboard every cycle; log only state changes

## 45.6 Paper Execution Model

### Fill Price
- BUY: `ask + slippage`
- SELL: `bid - slippage`

### Slippage
`spread * 0.1` (spread-dependent, 0.2 pip for 2 pip spread)

### Commission
0 for MVP (spread already includes cost). Later, read from MT5 account settings.

### Swap
Use MT5 swap data:
```python
mt5.symbol_info(symbol).swap_long
```

### Partial Fills
Ignore for MVP. 0.01 lot always fills.

## 45.7 Confidence Calibration

**MVP**: no calibration. Use raw confidence for ranking.

Track calibration buckets: 0.60–0.65, 0.65–0.70, 0.70–0.75 — compare predicted probability vs actual win rate. Add isotonic regression or Platt scaling after ~500+ samples.

## 45.8 First Intelligence Milestone

**Does the currency vector contain forward information that survives transaction costs?** If IC < 0.01, AUC < 0.52, or doesn't beat pair-only momentum, kill the hypothesis before adding complexity.

## 45.9 Build Order

| Day | Focus | Files |
|---|---|---|
| 1 | Tick infrastructure | `tick_store.py`, `models.py`, `mt5_adapter.py` |
| 2 | Currency math | `wls_solver.py`, `graph.py` |
| 3 | Direction + selection | `hypothesis.py`, `drs.py` |
| 4 | Paper execution | `paper.py`, `safety.py` |
| 5 | Metrics & validation | IC, win rate, calibration tracking |

---

# 46. Clean Build — Part 5: State Management, Persistence & Crash Recovery

## 46.1 Guiding Principle

> Persist only what cannot be reconstructed deterministically. Everything else should be derived.

The snapshot exists only to preserve continuity of the market state machine. Restart target: <1 second with high enough state continuity that DirectionHypothesis does not reset to random.

## 46.2 State Classification

### CRITICAL (must survive)
| State | Reason |
|---|---|
| Currency strengths (8 floats) | Current latent market state |
| Currency covariance/confidence | Needed for uncertainty tracking |
| Last solved timestamp | Gap calculation |
| Open paper positions | Risk continuity, PnL, exit logic |
| DRS_entry score per position | Position displacement calculation |
| Position ID + entry price + direction + timestamp | Lifecycle tracking |
| Snapshot version | Migration |

### IMPORTANT (should survive, can reconstruct with penalty)
- Tick hot window (last 30 min): without it, first seconds after startup lack micro context
- Health metrics: quality score, solve count, IC history
- PnL history: can be rebuilt from trade ledger

### EPHEMERAL (reconstructed automatically)
- Configuration (changes take effect on restart)
- Logs (append-only artifacts)
- Runtime counters (ticks_processed, etc.)

## 46.3 Snapshot Format

**MVP**: JSON + Parquet hybrid. No pickle (dangerous, version-dependent, non-inspectable).

- `state/current.json` — currency strengths, covariance matrix, positions, timestamps (<10 KB)
- `state/previous.json` — fallback for recovery
- `state/hot/ticks.parquet` — rolling 30-min tick window (~2-50 MB compressed)
- `state/trades/ledger.jsonl` — append-only trade record
- `state/backup_001.json`, `backup_002.json` — up to 5 rotation snapshots

### Why JSON + Parquet
JSON: critical state is tiny (<10 KB), human-inspectable, debuggable. Parquet: ticks are tabular (timestamp, symbol, bid, ask, volume), column-prunable, appendable, compressed.

## 46.4 Atomic Save Protocol

### Write Protocol
```python
write(tmp) → flush → fsync → checksum → rename(tmp, path)
```
NTFS rename is atomic on same filesystem. Snapshot directory: `%APPDATA%\ProximaClean\` (survives reinstall, proper permissions).

### Separate Save Frequencies
| Component | Frequency | Size |
|---|---|---|
| State snapshot | 5 min | <10 KB |
| Tick archive | Continuous append | 2-50 MB |
| Trade ledger | On every trade | <1 KB/trade |

## 46.5 Snapshot Versioning & Migration

```json
{
    "schema_version": 2,
    "created": "2026-07-07T12:00:00",
    "engine_version": "0.1.0"
}
```

Migration layer in `persistence/snapshot.py`:
1. Detect version
2. Run migration function per version jump
3. Return current object

If migration fails → fallback to previous snapshot → factory boot. Never crash.

### Version 1 → Version 2 Example
```python
# v1: {"EUR": 0.4}
# v2: {"currencies": {"EUR": {"strength": 0.4, "confidence": 0.8}}}
if version == 1:
    convert_old_currency()
```

## 46.6 Boot Sequence (Detailed)

1. **Load config** — from `config/settings.py` (never from snapshot)
2. **Initialize logging** — file + stdout
3. **Connect MT5** — retry 3x with backoff
4. **Load snapshot** — `current.json` → `previous.json` → factory boot
5. **Validate snapshot** — schema fields exist, timestamp ≤ current time, no NaN/inf strengths, position entry price > 0
6. **Restore state** — currency graph (strength vector, covariance, quality) → positions (objects) → risk (exposure, limits) → last_market_timestamp
7. **Detect gap** — compare `snapshot.last_market_timestamp` vs latest MT5 tick
8. **Recover gap** — M1 bars replayed through WLS minute by minute
9. **Quality check** — mark `gap_risk_unknown=True` if gap > 5 min
10. **LIVE LOOP** — system operational

The restored currency state is the **starting prior** — it is NOT trusted forever. After gap fill: old state + new observations = updated state.

## 46.7 Gap Fill Mechanics

### Gap Classification
| Duration | Strategy |
|---|---|
| <5 min | Tick reconstruction from persisted hot window |
| 5 min – 24h | M1 bars from MT5, WLS solve per minute, incremental graph update |
| >24h | Degraded bootstrap: use snapshot strengths as prior, lower confidence, block live execution |

### Gap Fill Sequence
1. Request M1 bars for all pairs from T₁ to T₂
2. Calculate per-minute returns
3. Run WLS solve for each minute (replaying through time)
4. Update currency graph incrementally
5. Apply position carry PnL (positions were open during gap)
6. Mark `gap_risk_unknown=True`
7. Log summary: "Gap of 4.2h — recovered via 252 M1 bars — quality: 0.87"

### Positions During Gap
**MVP**: assume position survives, check MT5 for current price to calculate floating PnL. Mark `gap_risk_unknown=True`. Do not invent historical stop execution.

## 46.8 Tick Persistence (Hot Window)

The tick store is a rolling cache, not a database. **30 minutes** of recent ticks (the system solves every 5s, needs current context only).

**Format**: Parquet — columns: `timestamp, symbol, bid, ask, volume`.
**Rotation**: `current.parquet` → when >50 MB, compress and start new.
**Implementation**: RAM ring buffer (`deque(maxlen=N)`) + 5-minute parquet flush.

**File size estimates**:
- 28 pairs, 100 ticks/sec total, 1800s = 180,000 ticks ≈ 7 MB raw, 2-5 MB parquet
- Peak: 500 ticks/sec ≈ 50 MB — still fine for consumer PC

## 46.9 Log Management

Rotate by size: max 20 MB each, keep last 5. Never allow unbounded append.

## 46.10 Hard Crash Recovery Scenarios

| Crash Scenario | Recovery |
|---|---|
| **Crash during snapshot write** | `current.tmp` exists, `current.json` is previous version. Boot ignores tmp, loads current. Safe — rename never atomized. |
| **Crash after rename but before next snapshot** | Load last consistent snapshot. Gap fill handles 2-minute discrepancy. Safe. |
| **Crash during MT5 disconnect** | No impact — connection state is ephemeral. Reconnect. Safe. |
| **Disk corruption / checksum mismatch** | `current.json` FAIL → `previous.json` PASS → restore. If both fail → FACTORY BOOT. Safe. |

### The One Rule That Prevents State Corruption

Every persisted object needs a timestamp envelope:
```python
@dataclass
class StateEnvelope:
    market_timestamp: float
    wall_timestamp: float
    schema_version: int
    checksum: str
    payload: dict
```

Restart recovery is NOT "loading variables." It is restoring a previous market belief, measuring how much reality moved away from it, and safely reducing confidence until the graph catches up.

---

# 47. Clean Build — Part 6: Safety, Risk & Monitoring for Live Paper Demo

## 47.1 Design Principle

The clean system should NOT recreate the old 14-gate risk maze. The safety layer answers only:
1. **Can we trade?** (state check)
2. **How much?** (position sizing)
3. **Should we stop?** (stops, drawdown, kill switch)
4. **Can we trust our own state?** (health monitoring)

Clean separation: **Market State Graph decides what is likely. Risk Engine decides what is allowed. Execution decides how it happens.**

## 47.2 Position Sizing

### MVP: Fixed 0.01 Lot
```python
LOT_SIZE = 0.01
```
No adaptive sizing before the model proves itself. Adding confidence-sizing introduces a confound — bad signal + large size = unclear failure.

### Phase Progression
| Stage | Method | When |
|---|---|---|
| MVP | Fixed 0.01 | Day 1 |
| 100 trades | Confidence-scaled: `lot = base * clamp((confidence - 0.5) * 2, 0.25, 2.0)` | After model validation |
| Mature | Volatility-adjusted: `target_risk / stop_distance` | After 500+ trades |
| Never initially | Kelly fraction | Requires true probability distribution |

## 47.3 Stop Loss / Take Profit

Currency model creates slower information — 5-second WLS signal is not a 5-minute scalp. 15-pip stop is too tight.

### MVP Parameters
```python
STOP_LOSS_PIPS = 30
TAKE_PROFIT_PIPS = 60
MAX_HOLD_HOURS = 12
```

### Stop Checking Frequency
Check **every tick batch** (not every decision cycle) — stops are risk, not intelligence.

### Code
```python
def check_stops(position, price):
    if position.side == "BUY" and price <= position.stop:
        return CLOSE
    if position.side == "SELL" and price >= position.stop:
        return CLOSE
```

## 47.4 Drawdown & Daily Loss Limits

### Limits
```python
MAX_DAILY_LOSS = 100    # $100: soft shutdown, block new trades, keep monitoring, reset next day
MAX_DRAWDOWN = 500      # $500: hard shutdown, close positions, save snapshot, halt
```

### Trading States
```python
TradingState: ACTIVE | DEGRADED | HALTED_DAILY | HALTED_RISK | MANUAL_STOP
```

## 47.5 Currency Exposure Limits

### Two Parallel Limits
```python
MAX_EXPOSURE_PER_CURRENCY = 200_000    # nominal (kept for future scaling, not binding at 0.01)
MAX_CURRENCY_POSITIONS = 2             # max positions sharing same base currency
```

Example: EURUSD BUY + EURJPY BUY = allowed (2 EUR positions). EURUSD BUY + EURJPY BUY + EURGBP BUY = blocked (3 EUR positions).

## 47.6 Correlation & Portfolio Scoring

Do NOT hard-block correlated pairs — the currency graph already knows overlap (EURUSD BUY = EUR+, USD-; GBPUSD BUY = GBP+, USD-). Instead, use **currency exposure scoring**:

```python
new_currency_vector + existing_currency_vector → portfolio_currency_heat
if currency_heat > threshold → reduce DRS score (diversification_factor)
```

Not BLOCK — just rank lower.

## 47.7 Health Monitoring

### HealthStatus Dataclass
```python
@dataclass
class HealthStatus:
    state: str                # OK / DEGRADED / FAILED
    mt5_ok: bool
    tick_quality: float       # 0-1
    graph_quality: float      # 0-1
    last_snapshot_ok: bool
    solve_latency_ms: float
    memory_mb: float
```

### Thresholds
| Check | OK | Warning | Failed |
|---|---|---|---|
| MT5 | Responding | — | No response >10s |
| Tick freshness | <5s | 5-30s | >60s |
| WLS solve rate | >80% success | 60-80% | <60% |
| Graph quality | >0.7 | 0.5-0.7 | <0.5 |
| Solve latency | <100ms | 100-500ms | >500ms |
| Total decision | <500ms | 500ms-2s | >2s |

### Response to Degraded/Failed
- **DEGRADED**: continue with `confidence *= 0.7` multiplier
- **FAILED**: block new trades, preserve state, print CRITICAL alert

## 47.8 Trade Audit Trail

### MVP Minimum Trade Record
```python
@dataclass
class TradeRecord:
    id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: float | None
    entry_time: float
    exit_time: float | None
    confidence: float
    drs_score: float
    currency_snapshot: dict     # strengths at entry
    exit_reason: str            # STOP_LOSS, TAKE_PROFIT, DRS_DISPLACED, MANUAL, GAP_CLOSE
    pnl: float | None
```

### Storage
`trades.jsonl` — append-only, crash-safe, not JSON array. Fields added later: residual_at_entry, graph_quality, fidelity, spread, slippage.

## 47.9 Alerts (Terminal-Only)

No Windows notifications for MVP — terminal output is sufficient. Avoid notification dependency.

| Level | Trigger | Display |
|---|---|---|
| INFO | Currency solve completed, trade opened/closed | No alert (logged) |
| WARNING | State change (EURUSD stale 12s, graph quality degraded) | Print once |
| ERROR | MT5 disconnected, retry | Persistent print each cycle |
| CRITICAL | Max drawdown reached, trading halted | Large banner, pauses output |

## 47.10 Kill Switch

### MVP Solution: STOP File
```python
# Check every 1 second
if Path("runtime/STOP").exists():
    risk.close_all()
    snapshot.save()
    shutdown()
```

Works remotely, no UI, survives terminal focus issues. Priority: STOP file > Ctrl+C > process kill.

### Shutdown Sequence
```
STOP detected → disable new trades → close paper positions → save snapshot → flush logs → exit
```

## 47.11 MVP Safety Checklist (Build Now / Skip)

| Component | Build |
|---|---|
| Fixed 0.01 lot sizing | YES |
| 30 pip stop / 60 pip target | YES |
| Time stop (12h) | YES |
| Currency exposure (count + nominal) | YES |
| Correlation scoring (portfolio heat) | YES |
| Trade audit ledger | YES |
| Health monitor | YES |
| STOP file kill switch | YES |
| Daily loss / drawdown limits | YES |
| Kelly sizing | NO |
| Trailing stops | NO |
| Windows notifications | NO |
| Web kill switch | NO |
| Full risk optimizer | NO |

---

# 48. Clean Build — Part 7: Build & Deploy Plan for Immediate Paper Demo

## 48.1 Build Strategy

**Do not build the full 14-file architecture first.** The risk is spending 2 weeks building infrastructure around an unproven hypothesis.

The correct strategy:
```
Phase 0: Prove currency decomposition works.
Phase 1: Wrap it in safe runtime.
Phase 2: Add persistence + production hardening.
```

First objective: "Can latent currency strength extracted from MT5 data predict future FX movement better than pair-only signals?" Everything else depends on this.

## 48.2 Build Order (7 Days)

| Day | Focus | Files | Validation |
|---|---|---|---|
| 1 | Mathematical core | `settings.py`, `models.py`, `wls_solver.py`, `graph.py` | Solver recovers known strengths from synthetic data |
| 2 | Data ingestion | `mt5_adapter.py`, `tick_store.py` | Dashboard shows real currency strengths from live MT5 |
| 3 | Signal generation | `hypothesis.py`, `drs.py` | Hypotheses generated from currency graph |
| 4 | Paper execution | `execution/models.py`, `paper.py`, `safety.py` | First simulated trade |
| 5 | Persistence | `snapshot.py` | Save/restore cycle works |
| 6 | Runtime hardening | `runtime/manager.py`, `dashboard.py`, `main.py` | Full lifecycle: start → run → stop → restart |
| 7 | Deployment | Integration test | 8-hour paper run |

### Parallel Streams
Risk, dashboard, and persistence can be built while math/data is being tested.

### Sequential Chain (must be in order)
WLS → Graph → Direction → DRS → Execution — each depends on previous output.

## 48.3 Minimal Tests per Module

| Module | Test |
|---|---|
| `wls_solver.py` | Synthetic recovery: create known strengths, generate returns, verify solver recovers them within 0.05 tolerance |
| `graph.py` | Update with returns, verify timestamp advances; residual < 1 after update |
| `tick_store.py` | Feed mock ticks, verify non-zero return, verify freshness detection |
| `hypothesis.py` | Input EUR=0.5, USD=-0.2 → expect direction=BUY, score>0 |
| `drs.py` | Input two hypotheses with different confidences → verify top-ranked is higher confidence |
| `risk/safety.py` | At position_count=3, new_trade=False; at drawdown=600, halt=True |
| `paper_executor.py` | BUY fills at ask, SELL at bid; PnL matches price movement |
| `snapshot.py` | Save → modify in-memory state → load → state restored; corrupted file falls back |
| `runtime/manager.py` | Mock tick source, clock, executor → verify tick → solve → decision cycle |

## 48.4 Replay Mode (Mandatory Before Live)

Minimum validation: **7 trading days** of historical data (not hours — needs trends, ranges, news, sessions).

Pipeline: `historical ticks/M1 bars → TickStore → WLS → Graph → Hypothesis → PaperExecutor → TradeLedger`

Compare signal timestamp vs future return. Use recorded ticks if available; M1 bars as MVP fallback.

## 48.5 Two-Day Blitz Build

If speed is critical, cut to 8 files:

```
proxima_clean/
    main.py      (single-thread loop, everything inline)
    config.py
    models.py
    wls.py
    graph.py
    mt5.py
    paper.py
    risk.py
```

~1500-2000 lines. Loses: clean runtime separation, snapshot recovery, dashboard, replay abstraction, DRS sophistication, proper monitoring. Still proves: MT5 data → currency state → direction → paper PnL.

## 48.6 Deployment Checklist

### Preflight
- [ ] MT5 terminal running and logged in
- [ ] ≥20 of 28 symbols available on MT5
- [ ] Python deps installed (numpy, MetaTrader5, pathlib, pyarrow/parquet)
- [ ] State directory created and writable
- [ ] No stale STOP file from previous run
- [ ] First run: "No snapshot found — FACTORY BOOT" (expected)

### First 5 Seconds
- Expected: `WLS SOLVE OK`, Coverage: 24/28, Graph quality: 0.82

### First 30 Seconds
- Expected: Hypotheses generated (e.g., EURUSD BUY 0.63, GBPJPY SELL 0.58)
- No trade is acceptable

### Shutdown
- Expected: Snapshot saved, shutdown complete

### Restart
- Expected: Snapshot loaded, gap detected, recovery complete, mode LIVE

## 48.7 First Run: What to Watch

### Healthy Indicators
- Currency strengths: EUR +0.35, USD -0.22, JPY +0.08, GBP -0.12 — smooth movement, range -2 to +2
- Coverage > 22 pairs per solve
- Graph quality > 0.8
- Dashboard updates smoothly

### Warning Signals (Stop and Investigate)
- All strengths near zero (no returns, stale ticks, symbol mismatch)
- Currency flipping sign every 5 seconds (window too short, no regularization)
- |strength| > 10 (bad normalization)
- Mean residual > return magnitude (WLS not explaining market)
- Memory growing unbounded
- MT5 disconnects repeatedly

## 48.8 First Paper Trade

### Conditions
1. ≥15 minutes of runtime (startup transient matters)
2. Currency strengths stable (not initial transient)
3. Graph quality > 0.7
4. Hypothesis with confidence ≥ 0.60
5. Risk approves
6. Slot available

Expected timeline: 15 minutes to several hours on a normal market day. No trades in flat market (Friday afternoon, between sessions) is also valid data.

### Dashboard Entry Display
```
TRADE OPENED | EURUSD BUY | 1.08523 | Conf: 0.71 | DRS: 0.78
EUR +0.42 | USD -0.21 | Lot: 0.01 | Stop: 30 | Target: 60
```

## 48.9 Kill Hypothesis Criteria

The system must measure its own Information Coefficient automatically:
```
For every signal → store (timestamp, signal_strength, future_return)
Correlate signals against future returns
Benchmark: random (IC=0) and 1h pair-only momentum
```

### Minimum Sample
Do not judge before **500 signals** (2000+ preferred).

### Survival Threshold
Currency model survives if:
- IC > 0.02
- AUC > 0.53
- Beats pair-only momentum

### Kill Criteria
After 2000 samples:
- IC < 0.01
- AUC < 0.52
- No improvement over momentum

## 48.10 Post-MVP Priority

### Week 2 (Mandatory)
1. Replay mode + 7-day historical validation
2. Trade analytics (why wins, why losses)
3. Confidence calibration (Platt scaling / isotonic)
4. Snapshot hardening for long runs

### Week 3-4 (Important)
5. Confidence-scaled position sizing
6. Multi-timeframe support
7. Macro event awareness

### Month 2+ (Later)
8. Live MT5 execution
9. Full fidelity system
10. Learning architecture
11. Portfolio factor controller

## 48.11 The Minimum System That Must Exist

10 files, ~1500-2000 lines:
```
config.py  models.py  mt5.py  tick_store.py
wls.py  graph.py  direction.py
paper.py  risk.py  main.py
```

Not 20,000 lines. The first milestone is not profitability. It is: **"Given current FX market data, can this system extract a stable latent currency state that contains forward information?"** If no, every additional layer is wasted. If yes, the rest of the architecture becomes worth building.
