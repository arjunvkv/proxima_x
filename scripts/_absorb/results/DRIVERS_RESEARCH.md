# DRIVERS_RESEARCH — causal factors that shift a pair (the apple tree)

Frame: the pair is the apple. Fertilizers = rate differentials, flows, terms of
trade, positioning, relative value. Climate = risk regime, volatility, liquidity,
event weather. Each idea below = a fertilizer/climate factor with a measurable
proxy. STATUS: TESTABLE-NOW = expressible with the existing M5 cache (22 FX + 13
assets, 7-8 months) with no new data; EXTERNAL = needs a data pull.

## A. Money & rates (soil chemistry) — EXTERNAL (rate data)
1. Real-rate differentials (10y TIPS-style) -> strongest documented gold driver,
   strong medium-term FX driver. Proxy: US10Y real vs DE/JP real. Needs bond data.
   MT5 angle: pull bond-future/IR CFDs if broker offers (US10Y, DE10Y, JGB10Y) ->
   makes this TESTABLE.
2. Policy-rate path divergence (1y OIS minus 3m per country, momentum of it).
   Documented: shocks to rate-spread expectations dominate medium-term FX.
3. Short-end yield momentum (3m rate change 3m-ago). Classic carry-adjacent.
4. CIP deviations / cross-currency basis (F/S vs rate diff). Documented edge,
   institutional access needed; but basis WIDENING = USD funding stress = risk-off
   filter for USDJPY/EM. Needs swap/fwd data.
5. Carry (rate diff level) as FILTER not signal: long high-yielder only in risk-on
   regimes (combine with C12). Pure carry tested dead (R2 carry_clock, hour-0 trap).

## B. Balances & flows (water) — mostly TESTABLE-NOW via asset proxies
6. TERMS-OF-TRADE MOMENTUM for commodity currencies: AUD/NZD/CAD follow their
   export basket. Proxy from existing cache: XAU & USOIL & XAG momentum -> long
   AUDUSD/NZDCAD when commodity basket up. TESTABLE-NOW. Documented link.
7. EQUITY-DIVERGENCE FLOWS: relative stock-market strength => capital flows =>
   currency. Proxy: US500 vs GER40/UK100 momentum -> EURUSD/GBPUSD direction.
   TESTABLE-NOW (indices in cache).
8. Bond-flow differentials: needs yield data (see A1).
9. Official/reserve flows: not observable retail; skip.
10. MONTH-END REBALANCING: calendar day-of-month bias in USDJPY/USD pairs around
    month-end windows. TESTABLE-NOW (weak sample: ~7 month-ends; context filter).
11. WMR 4PM LONDON FIX pressure/reversal: pre-fix run -> post-fix fade, ~0.7%/yr
    documented in GBPUSD. Proxy: 15:45-16:15 server-window behavior. TESTABLE-NOW
    (verify server-UTC mapping; 16:00 London = 15:00 UTC = 15:00 server if UTC).

## C. Risk, positioning & volatility (climate) — mostly TESTABLE-NOW
12. RISK-ON/OFF REGIME: proxy = US500 momentum + DXY + XAU sign (all in cache) ->
    regime filter for carry (A5) and haven flows. TESTABLE-NOW.
13. COT positioning extremes: weekly CFTC (external pull, 1-wk lag). Conditioning
    variable: extreme net positioning -> mean-reversion bias. EXTERNAL, cheap.
14. VOLATILITY REGIME: realized vol from index/gold bars (VIX proxy) ->
    compression/expansion regime filter; session strategies only in low-vol +
    London-NY overlap. TESTABLE-NOW.
15. SAFE-HAVEN CORRELATION: XAU & USDJPY & USDCHF co-movement; regime detection
    for haven flows (correlation state is a state variable). TESTABLE-NOW.
16. Funding stress: EURUSD/USDJPY basis widening = stress (see A4). EXTERNAL.

## D. Relative value (the graft) — TESTABLE-NOW
17. TRIANGULATION RESIDUALS: EURJPY vs EURUSD x USDJPY (also GBPJPY, EURGBP,
    AUDJPY). Daily residual z-score -> mean reversion 1-5d. Documented as HFT
    domain at ms scale, but SLOW residual reversion at daily scale is untested by
    most; slow residual = flow/market-maker inventory signal. TESTABLE-NOW.
18. Synthetic cross chains: all 22 FX pairs -> build implied vs actual for every
    triangle; rank residuals. TESTABLE-NOW.
19. Gold/silver ratio: TESTED DEAD (PF 0.67-0.95, direction right, economics dead).
20. GOLD/OIL RATIO (XAU/USOIL) -> risk-regime indicator + commodity-currency bias.
    TESTABLE-NOW (both in cache).
21. DXY-IMPLIED DIVERGENCE: DXY is 57% EUR; implied EUR from DXY vs actual EURUSD
    divergence = non-EUR DXY-component flow signal (GBP/JPY/CHF/CAD moves inside
    DXY). We have DXY.cash! NOVEL, TESTABLE-NOW.
22. Equity-implied risk: US500/USOIL (risk appetite vs oil bill) -> USDJPY bias.
    TESTABLE-NOW.

## E. Macro events & surprises (weather) — EXTERNAL
23. NEWS-SURPRISE ASYMMETRY: documented ~50-70% of short-run FX variance from
    macro surprises. Strategy: avoid trading into events; fade the post-event
    overreaction. Needs MT5 economic calendar + event timestamps. EXTERNAL but
    MT5-native (calendar API) -> feasible.
24. Inflation-differential momentum: monthly CPI (external, slow). Long-horizon.
25. Growth/PMI divergence: monthly (external). Long-horizon; weak short-horizon.
26. Political/election regimes: flags (external). Rare events; regime overlay only.

## F. Microstructure (the pruning) — mostly TESTABLE-NOW
27. SESSION LIQUIDITY WINDOWS: London-NY overlap best execution/trends; thin
    windows worst. Confirmed by our Tokyo hour-0 spread study. Use as time filter.
    TESTABLE-NOW.
28. Fixing auctions: see B11.
29. Rollover timing: KNOWN TRAP (hour-0 spread blowout; live daemon has no spread
    filter). Do NOT trade hour 0-1. Lesson recorded.
30. Opening gaps: weekend_gap FX dead; index break-gap follow = S2 WINNER.
31. Liquidity-sweep false breaks: tested, weak (PF ~1.2 gate-outs).

## G. Calendar & behavior (the seasons)
32. Day-of-week: TESTED DEAD (day_of_week_usd).
33. Month-end: see B10.
34. Pre-holiday drift: calendar flags (external calendar; weak sample).

## Ranking for the next hunt (fresh + testable now)
T1 (TESTABLE-NOW, novel): #21 DXY-implied divergence; #17 triangulation residuals;
    #6 commodity-terms-of-trade momentum.
T2 (TESTABLE-NOW, documented): #11 fix reversal; #7 equity-divergence flows;
    #12 risk-regime filter; #14 vol-regime filter; #20 gold/oil ratio.
T3 (needs external data): #1 real yields (via MT5 bond CFDs if available);
    #13 COT; #23 news-surprise windows.
Dead/known: #19 gold-silver ratio, #32 day-of-week, #29 rollover trading, pure
carry as signal (#5).

---
## Validation results (2026-08-11, nova_tree_probes.py on NOVA factors, daily scale, triage-grade)

| Idea | n | win | exp/lot | PF | Verdict |
|---|---|---|---|---|---|
| #17 triangulation EURJPY fade (z±1.5, 20d) | 21 | 47.6

---
## Validation results (2026-08-11, nova_tree_probes.py on NOVA factors, daily scale, triage-grade)

| Idea | n | win | exp/lot | PF | Verdict |
|---|---|---|---|---|---|
| #17 triangulation EURJPY fade (z+-1.5, 20d) | 21 | 47.6% | -$195.9 | 0.52 | DEAD |
| #21 DXY-implied div EURUSD fade (z+-1.5) | 59 | 64.4% | +$193.6 | 3.04 | **PROMISING** |
| #6 commodity mom AUDUSD (XAU+USOIL 5d) | 83 | 54.2% | +$68.1 | 1.44 | borderline |
| #6 commodity mom NZDUSD | 83 | 56.6% | +$50.5 | 1.36 | borderline |
| #7 equity div GBPUSD (UK100-US500) | 64 | 56.2% | +$9.9 | 1.04 | DEAD |
| #11 London fix fade GBPUSD (enter pre-close) | 144 | 47.2% | -0.45bp/trade | - | DEAD (costs ~1bp) |

- **#21 robustness:** threshold sweep z+-1.0/1.5/2.0/2.5 -> exp/lot +$160/+$194/+$191/+$288, win 63->71%, both half-splits positive (decaying, not single-event). Daily reversion on the gap between DXY and its EURUSD component. **Next: deep battery** (hour-concentration check, plateau grid, FTMO sim).
- **#6:** weak but positive both AUD and NZD with identical signal - construction to revisit (NZDCAD direct, 2-day hold) before calling dead.
- **#11 structural note:** post-fix (server 18:00->20:00) GBPUSD drift is negative on BOTH sides (after rise -0.32bp, after fall -0.88bp); unconditional short earns +0.60bp gross < ~1bp costs -> dead as a trade, but the asymmetry (continuation after falls, fade after rises) is real microstructure.
