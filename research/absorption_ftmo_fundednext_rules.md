# Absorbtion → Price-Impact Transition probe — FTMO/FundedNext rules reference

Verified 2026-08-10 from official sources by delegated research agent
(47 API calls, official pages only). Used by `research/absorption_probe.py` /
`absorption_robust.py` survival analysis.

## FTMO 2-Step (flagship; matches Proxima engine's FirmRiskConfig)

- Profit target: 10% (Phase 1) / 5% (Phase 2) / none (funded)
- Max daily loss: **5%** — limit = *balance at 00:00 CE(S)T* − 5% of initial
  capital; intraday profit does NOT expand the day's allowance. Day reset
  00:00 CE(S)T (GMT+1/+2).
- Max loss: **10%, static** (initial-capital based), no trailing.
- Min trading days: 4 per phase (day = ≥1 position opened). No time limit.
- Leverage 1:100 standard (1:30 Swing). No % consistency rule.
- Commission: exact EURUSD $/lot shown only in-platform — **unverified on site**
  (engine's $3.0/lot/side from live broker remains authoritative).
- Swaps/weekend: overnight/weekend holds **restricted on funded FTMO Account
  only** (close before weekend; >2h rollover); free during evaluation.
- News: **allowed in evaluation (all phases)**; funded Standard account only —
  no open/close (incl. SL/TP triggers) 2 min before → 2 min after restricted
  high-impact events (central-bank rates, NFP, CPI, GDP). Swing exempt.
- Hedging: within one account OK; cross-account banned. Arbitrage/latency
  forbidden. Copy trading forbidden. EAs allowed; hyperactivity ban
  (>2,000 server requests/day).
- Payout: up to 90% (2-Step: 80% → 90% qualified; 1-Step: 90%); fee 100%
  refunded on first payout.

## FundedNext Stellar (1-Step / 2-Step / Lite)

| | 1-Step | 2-Step | Lite |
|---|---|---|---|
| Target | 10% | 8% / 5% | 8% / 4% |
| Max daily loss | **3%** | **5%** | **4%** |
| Max loss | **6%** static | **10%** static | **8%** static |
| Min days | 2 | 5 / phase | 5 / phase |
| Consistency | Best Day ≤ 50% of positive days' profit (Ch. + funded) | none | none |

- Daily loss: limit = *initial balance* × % **PLUS any intraday profit**
  (losing today's profit is allowed) — structurally easier than FTMO;
  swaps/commissions count. Reset 00:00 server time (GMT+3 summer).
- Trailing drawdown only on Stellar **Instant** (6% trailing MLL, ratchets up,
  never exceeds initial balance). Challenge plans are static.
- Leverage 1:100 FX challenge/funded. Commission: Instant $7/lot FX (opening
  only); Challenge FX schedule is image-based → **unverified**; crypto
  0.04%/lot, metals 0.0016%.
- Swaps/weekend: **allowed** on all plans (triple-swap Wed FX, Fri
  indices/crypto); swap-free option.
- News: allowed everywhere; funded-account trades ±5 min around high-impact
  news count only **40% of profit** (1-Step/2-Step/Lite/Instant); challenges
  unrestricted.
- Prohibited: arbitrage/latency/tick-scalping/grid/HFT/one-sided betting.
  Hedging within one account only. EAs allowed; **Quick-Strike: ≥30% of profit
  from <30s trades = violation**; hyperactivity ≥200 trades/day (3 strikes).
- Risk cap: **3% risk-cap on funded accounts (2nd breach → 1%)**.
- Payout: 14-day cycles, within 24h of cycle end; 1-Step/2-Step 80% (→90% at
  Scale-Up) + 15%-of-target challenge bonus; Lite 80%→90%; Instant 70%→80%.

## Key implications for Proxima survival analysis

1. Engine's FTMO config (5%/10%/10%/4 days) is current — but daily-loss
   math: FTMO anchors **midnight balance** (intraday profit does NOT buffer
   the day's allowance).
2. FundedNext daily loss is friendlier (intraday profit expands allowance),
   but funded adds 3%→1% risk cap + 40%-credit news-window rule + Quick-Strike
   (<30s trades) — an absorption scalper holding seconds-to-minutes must watch
   Quick-Strike on FN.
3. FTMO funded Standard: weekend close required, news ±2min ban — relevant
   if the strategy holds overnight or fires at news.

Sources (official): ftmo.com trading-objectives / comparison-table / FAQ /
forbidden-trading-practices; help.fundednext.com articles (daily-loss-limit,
Stellar 1/2-step & Lite rules, leverage, commission, overnight, news,
restricted-strategies, risk-limits, payout/cycles).