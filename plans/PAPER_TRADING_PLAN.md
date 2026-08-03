# Paper Trading Plan — V2z_CPPF_RECON

## Goal
Align **backtest data → paper trade → live trade** on the **same FTMO platform** so the EA swaps perfectly with zero changes. Validate real execution for 1 week, then pay for the challenge and swap the account.

## Why FTMO Free Trial (not Fusion Markets)

| | Fusion Markets Demo | FTMO Free Trial |
|:--|:------------------:|:---------------:|
| Cost | Free | Free |
| Data feed | Fusion tick data | **FTMO tick data** (same as challenge) |
| Commission model | $2.25/side | **$2.50/side** (matches FTMO Raw) |
| When funded | Fusion account needed | **Use same terminal, swap account** |
| Backtest alignment | ❌ Fusion data ≠ FTMO data | ✅ 100% aligned |

**The pipeline:**
1. FTMO Free Trial ($10k, $0 fee) — download their MT5, backtest with their data
2. Run RECON EA on FTMO trial for 1 week — validates execution
3. Buy FTMO $10k challenge (~$155) — same MT5 terminal, log into challenge account
4. Pass challenge → funded account — same charts, same EA, zero changes

## Step 1 — Open FTMO Free Trial

1. Go to ftmo.com → **Free Trial** (top nav)
2. Select **$10,000 Free Trial** (no payment, no card)
3. Choose **MT5** platform
4. Download and install **FTMO MetaTrader 5** terminal
5. Log in with credentials from FTMO (sent via email)
6. Account credited with **$10,000 virtual balance**

## Step 2 — Compile & Deploy RECON EA

Now using the **FTMO MT5 terminal**, run the MCP compile tool:

```
mcp-mt5_compile_and_deploy V2z_CPPF_RECON.mq5
```

This compiles the EA with `WarmUpBuffers()` and `RecoverPosition()` and copies the `.ex5` to FTMO's `MQL5/Experts/` folder.

**Recent changes (already in code):**
- `WarmUpBuffers()` — loads close price history on startup so z-score works immediately
- `RecoverPosition()` — finds existing positions by magic on restart, prevents double-entries

## Step 3 — Backtest on FTMO Data

Run 4 individual pair backtests on FTMO terminal (Jun 8-Jul 25, 0.5 lot):

| Test | Pair | Config |
|:----|:----|:------:|
| PORTFOLIO_FTMO_AUDNZD | AUDNZD | Z=2.5, S=4.0, T=1.5, G=0.08, LOT=0.5, M=202415 |
| PORTFOLIO_FTMO_EURAUD | EURAUD | same, M=202416 |
| PORTFOLIO_FTMO_EURNZD | EURNZD | same, M=202417 |
| PORTFOLIO_FTMO_GBPAUD | GBPAUD | same, M=202418 |

These backtests use **FTMO's actual tick data** — same data the EA will trade on during the challenge. Results will match live within expected variance.

## Step 4 — Deploy on 4 Charts in FTMO MT5

| Chart | Symbol | Magic | Lot |
|:----:|:------:|:-----:|:---:|
| #1 | AUDNZD M1 | 202415 | 0.5 |
| #2 | EURAUD M1 | 202416 | 0.5 |
| #3 | EURNZD M1 | 202417 | 0.5 |
| #4 | GBPAUD M1 | 202418 | 0.5 |

**EA Inputs (all 4):**
| Parameter | Value |
|-----------|-------|
| Z_THRESHOLD | 2.5 |
| STOP_A | 4.0 |
| TRIG_A | 1.5 |
| GAP_A | 0.08 |
| BASE_LOT | 0.5 |
| MAX_DAILY_LOSS | 500.0 |
| MAX_SPREAD_PIPS | 5.0 |
| TRADE_START_HOUR | 0 |
| TRADE_END_HOUR | 7 |
| MIN_GAP_PIPS | 0.5 |

**Per chart (Common tab):**
- ✅ Allow live trading
- ✅ Allow Automated Trading
- ✅ **Use AutoStart** — re-attaches EA after restart

## Step 5 — Runtime Schedule

| Timezone | Session Time |
|----------|-------------|
| Server (UTC+3) | 0:00-7:00 |
| India (IST, UTC+5:30) | 2:30-9:30 AM |

Run on local PC during Asia session (Sun 22:00 UTC — Fri 22:00 UTC).

**Settings:**
- Disable PC sleep during trading hours
- Keep FTMO MT5 open with all 4 charts visible
- Enable AutoTrading (green triangle, top-left)
- Stable internet connection

## Step 6 — Handling Interruptions

| Scenario | EA Behavior |
|----------|------------|
| PC restart | FTMO MT5 reopens → AutoStart re-attaches EA → `RecoverPosition()` finds existing positions |
| Internet drop < 5 min | MT5 auto-reconnects, EA resumes |
| Internet > 5 min | Missed trades — acceptable for trial |
| MT5 crash | Same as restart — recovery handles it |

**Missed trades estimate:** ~1-3 per 30-min downtime. Over 1 week: ~5-15 out of ~100 expected.

## Step 7 — Monitor for 1 Week

**Trades log:** View → Terminal → History tab — full MT5 audit trail
**EA log:** View → Terminal → Experts tab — entry reasons, stop moves, exits

| Metric | Expected | Where to check |
|--------|----------|----------------|
| Trades/day (4 pairs) | 15-20 total | History tab |
| Win rate | 55-70% (regime dependent) | History tab |
| Avg slippage | < 1 pip | Compare trigger price to fill |
| Comission per trade | ~$2.50/RT (0.5 lot) | Trade tab |
| Max DD | < $200 (2% of $10k) | Account History |
| Restart recovery | No double-entries | Experts tab after manual restart |

## Step 8 — Decision Gate

| Outcome | Action |
|---------|--------|
| Week 1 PnL > -$100 and no execution issues | ✅ Buy FTMO $10k challenge ($155) |
| Week 1 PnL -$100 to -$300 | Analyze, run another week |
| Double-entries or EA bugs | Fix and retest |
| Week 1 PnL < -$300 | Re-evaluate strategy |

## Step 9 — FTMO Challenge (After Decision)

1. Buy FTMO $10k Challenge at ftmo.com ($155)
2. FTMO sends credentials for the **challenge MT5 account** on the **same FTMO server**
3. In FTMO MT5 terminal: **File → Login to Trade Account** → enter challenge credentials
4. **Same 4 charts, same EA, same settings** — zero changes
5. Trade the challenge (14 days to reach 10% profit)
6. Pass → Verification phase → Funded account

## Step 10 — Verification & Funded

1. **Verification** — same terminal, same EA, reach 5% profit with no max daily loss breach
2. **Funded Account** — FTMO provides new login credentials
3. Login to funded account on same FTMO MT5 terminal
4. **Same EA, same 4 charts, same settings** — trade profit split 80/20

## Files

- `V2z_CPPF_RECON.mq5` — Production EA (WarmUpBuffers + RecoverPosition included)
- `ftmo_runs/PORTFOLIO_FTMO_*.ini` — FTMO backtest configs (to create)
- `evidence_recon_*.txt` — Previous evidence (MetaQuotes-Demo, for comparison only)
