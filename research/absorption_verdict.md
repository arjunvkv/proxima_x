# Absorption → Price-Impact Transition — research verdict (2026-08-10)

Research brief: institutional order-flow absorption hypothesis — can the
"high directional flow absorbed (low price impact) → absorption deteriorates →
disproportionate price discovery" transition predict short-horizon FX moves
well enough to survive FTMO/FundedNext?

**Verdict: REJECT — no tradable edge on the available real tape (§7, §31, §32
all invoked). The phenomenon is measurable but does not predict; every
parameterization is net-negative after real costs; the only apparent signal is
an episodic artifact of two outlier days.**

## Evidence base

| Item | Status |
|---|---|
| Data | Real FTMO MT5 quote ticks, EURJPY, 2,123,678 ticks / 28 days (archive `data/ticks/EURJPY/2026`), 100% carry genuine bid/ask, ~2 quotes/sec median |
| True volume | IMPOSSIBLE — all `volume`/`volume_real` = 0 on this broker feed. Flow measured as **quote-pressure proxy** (ask-up/bid-up vs ask-dn/bid-dn revision counts) — the brief's explicitly authorized fallback |
| Coverage | Only EURJPY archived at month depth; EURUSD 1h only. Other 5 universe pairs NOT tested (see §Extension) |
| Observable spread | EURJPY median 12 pts (price units 0.012, point 0.001) → $7.57/lot RT + $6.00 comm = **$13.57/lot round trip** |

## Measurement (causal, non-repainting, distribution-based)

- Window W=60s: signed quote pressure P = (ask_up+bid_up) − (ask_dn+bid_dn);
  mid move dM; impact-per-flow λ = |dM|/(|P|+1).
- **ABSORPTION state** (endpoint t): |P| ≥ q90 of trailing 4h **and**
  λ ≤ q25 of trailing 4h (trailing window only — no future info). 419 states /
  16,640 windows (2.52%).
- Hypothesis both directions: signed forward return = fwd(t→t+H)·sign(P);
  **+ = continuation (Hyp B), − = reversal (Hyp A)**.

## Results

| Config (W/trail/PQ/LQ) | H=60s | H=300s | H=900s | Net/lot (worst→best) |
|---|---|---|---|---|
| 60s/4h/0.90/0.25 (base) | −1.7 pts t−0.5 | −8.3 t−1.2 | −21.8 t−1.6 | **−$14.7 → −$27.4** |
| 30s/4h/0.90/0.25 | −0.7 t−0.4 | −14.2 t−2.8 | −14.6 t−1.8 | −$14.0 → −$22.8 |
| 120s/4h/0.90/0.25 | +1.0 t+0.5 | −2.1 t−0.3 | −3.5 t−0.3 | −$13.0 → −$15.8 |
| 60s/2h/0.90/0.25 | −1.5 t−0.5 | −11.0 t−1.8 | −15.6 t−1.3 | −$14.5 → −$23.4 |
| 60s/4h/0.95/0.10 (strict) | +0.9 t+0.2 | −35.6 t−1.8 | −62.1 t−1.5 | −$13.0 → −$52.7 |

Transition-gated variant (absorb@t, same-sign pressure continues, λ rises to
q75): 57 states, all horizons insignificant (t −0.2…+0.4).

**Temporal decomposition (§22) kills the only survivor**: the strict-config
900s "reversal" (−62 pts mean) is driven by TWO days — 20664 (−684 pts, n=7)
and 20668 (−465 pts, n=6). Leave-one-day-out: removing either collapses t to
~0 (H=300s flips to negative t on almost every day — i.e. the sign flips to
continuation when the outliers go). 18 of 20 days show small POSITIVE signed
returns = continuation drift, not the claimed reversal. Episode-concentrated
profit → reject per §22 regardless of t-looking-good-in-aggregate.

Baseline sanity: unconditional 60s signed return is −0.58 pts (t −2.4) — the
well-known quote-bounce. The probe detects it, so the apparatus is sound; the
absorption condition simply does not add predictive information.

## Cost gate (§14–15, §30) — decisive and uniform

Breakeven on EURJPY = 12 pts spread + $6 comm = $13.57/lot RT (median spread
only; p90 16 pts → $16.1; stress p99 93 pts → $64.7). The largest measured raw
signal (strict-config 300s, −35.6 pts gross ≈ $22/lot pre-cost) is (a) not
significant (t −1.79, n=75), (b) episodic (2 days), and (c) would still leave
~$8–9/lot before slippage/latency — which an absorption scalper of seconds-
scale holds cannot survive (250–1000 ms latency ≈ multiple quote revisions;
the edge would need to persist minutes while paying seconds-scale costs).

No config → positive expectancy at ANY horizon and ANY W/trail/threshold
neighborhood. Execution-fragility escalation (§15) is moot: expectancy starts
negative at baseline costs.

## Market-mechanism explanation (§34) — why it fails

1. **Phenomenon**: retail-visible "absorption" (price pinned while flow
   accumulates) is, on a quote tape, largely **market-maker quote management**:
   a dealer who widened/held quotes while netting inventory. The "transition"
   (subsequent dislocation) happens when the dealer's inventory limit is hit.
2. **Why it doesn't pay at these horizons**: the dislocation is (a) rare
   (~2.5% of windows), (b) already priced by the widening spread when it
   happens (the spread IS the absorption cost), and (c) directionally
   confounded by liquidity suppliers' simultaneous re-quoting. The measured
   conditional moves are 1–9 pts — smaller than the 12-pt round-trip toll.
3. **Who's who**: liquidity suppliers = dealers/latency-arbitrageur HFTs
   (immune to retail absence — passes §10's non-retail test conceptually);
   demand = macro flow. The phenomenon WOULD exist without retail — but
   existing at institutional scale means the edge is captured by quote
   management itself, not observable as a free public signal from tick data.
4. **Competition**: interdealer markets + HFTs arbitrage absorption gaps in
   microseconds; what remains after their activity is sub-cost at our
   execution layer.
5. **Proxy limitation (§3, §11)**: no real volume/trade direction on the
   broker feed — aggressive-vs-passive classification is impossible; quote
   pressure is a noisy stand-in. A genuine edge (if any) would require
   centralized FX flow data (EBS/Reuters) or CME futures volume — data NOT
   obtainable from MT5. Stated explicitly per §11.

## §33 mechanical spec

NOT PRODUCED — the research does not survive; the honest deliverable per §7
is this rejection. (Do not manufacture a spec by adding filters until a
backtest becomes profitable — explicitly forbidden.)

## Extension (only if user wants multi-symbol confirmation)

- Pull `copy_ticks_range` for the other universe pairs: EURUSD, GBPUSD,
  USDJPY, EURJPY(done), GBPJPY, AUDUSD, USDCAD × ~30 days via
  `data/ftmo_tick_ingester.py` (needs the FTMO terminal running on the local
  box or VPS; **pause the live core-book daemon before a second Wine MT5
  init to avoid the IPC wedge** — see proxima-codebase skill pitfall).
- Re-run `research/absorption_probe.py` per symbol; same battery.
- Expected: costs on majors are lower (spread 6–10 pts RT typical busy
  session, still ≥ $4/lot + $6 comm) — but the measured signal magnitudes
  (1–9 pts, insignificant) are below even that toll. Verdict is unlikely to
  flip; this is confirmation-style evidence only.

## Artifacts (committed `62eda68`)

- `research/absorption_probe.py` — causal absorption-state probe (env-tunable:
  PROBE_W/PROBE_TRAIL/PROBE_PQ/PROBE_LQ/PROBE_MINP).
- `research/absorption_robust.py` — per-day + leave-one-day-out decomposition.
- `research/absorption_ftmo_fundednext_rules.md` — verified official rules
  (FTMO 2-Step confirmed: 5%/10%/10%/4d; FundedNext: friendlier daily-loss
  math, +3%→1% risk cap, Quick-Strike <30s ban, 40% news-window credit).