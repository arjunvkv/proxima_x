# Deployment Freeze — V2.5

**Date:** 2026-06-16 13:35 UTC
**Status:** FROZEN — no logic changes permitted during paper trading phase.

---

## 1. Trigger Logic

| Parameter | Value | Source |
|-----------|-------|--------|
| Signal threshold | 0.80 (80.0%) | `SETTINGS.threshold` |
| ES rank | percentile across asset universe | `GlobalRankEngine.es_percentile()` |
| AT rank | percentile across asset universe | `GlobalRankEngine.at_percentile()` |
| Combined trigger | ES >= threshold AND no block conditions | `run_proxima_demo.py:980-1010` |
| Deployment mode | `GLOBAL_ALL_QUALIFIED` | `SETTINGS.deployment_mode` |
| Global rank threshold | 80.0 | `SETTINGS.global_rank_threshold` |

Trigger condition: signal qualifies when ES rank >= 80.0 in GLOBAL_ALL_QUALIFIED mode.

---

## 2. Energy Storage (ES) Logic

| Parameter | Value | Source |
|-----------|-------|--------|
| ES source | AAE output | `proxima_ops/monitoring/global_rank_engine.py` |
| ES percentile | ECDF cross-section across all assets | `es_percentile()` |
| Frozen | YES — no changes | N/A |

---

## 3. Adaptive Time (AT) Logic

| Parameter | Value | Source |
|-----------|-------|--------|
| AT source | AAE output | `proxima_ops/monitoring/global_rank_engine.py` |
| AT percentile | ECDF cross-section across all assets | `at_percentile()` |
| Frozen | YES — no changes | N/A |

---

## 4. Ranking Logic

| Parameter | Value | Source |
|-----------|-------|--------|
| Mode | GLOBAL_ALL_QUALIFIED | `run_proxima_demo.py:911` |
| Engine | `GlobalRankEngine` | `proxima_ops/monitoring/global_rank_engine.py` |
| Method | ECDF cross-section, all qualified assets | `compute()` |
| Frozen | YES — no changes | N/A |

---

## 5. Risk Logic

| Parameter | Value | Source |
|-----------|-------|--------|
| Risk per trade | 0.25% | `SETTINGS.risk_per_trade` |
| Position sizing | `risk_amount / (stop_pips * pip_value)` | `order_manager.py:calculate_volume()` |
| Pip value (JPY) | `1000.0 / price` | `order_manager.py:25` |
| Pip value (non-JPY) | `10.0` | `order_manager.py:27` |
| Max volume cap | 1.0 lots | `order_manager.py:32` |
| Volume rounding | `round(lots, 2)` | `order_manager.py:31` |
| Min volume floor | 0.01 lots | `order_manager.py:31` |
| Frozen | YES | N/A |

---

## 6. Stop Logic

| Parameter | Value | Source |
|-----------|-------|--------|
| Source of truth | `get_risk_stop_distance()` | `catastrophic_stop.py:14` |
| Stop pips (EURUSD) | 50 | `CATASTROPHIC_STOP_PIPS` |
| Stop pips (EURJPY) | 50 | `CATASTROPHIC_STOP_PIPS` |
| Stop pips (USDJPY) | 50 | `CATASTROPHIC_STOP_PIPS` |
| Stop pips (GBPJPY) | 70 | `CATASTROPHIC_STOP_PIPS` |
| Stop pips (XAUUSD) | 500 | `CATASTROPHIC_STOP_PIPS` |
| Verifier tolerance | 1.05x budget | `trade_risk_verifier.py:34` |
| Frozen | YES | N/A |

---

## 7. Exit Logic

| Parameter | Value | Source |
|-----------|-------|--------|
| Exit mode | H20 (research-driven) | `run_proxima_demo.py` |
| Take profit | 0.0 (disabled) | `catastrophic_stop.py:catastrophic_tp()` |
| Stop loss | Catastrophic SL | `catastrophic_stop.py:catastrophic_sl()` |
| Frozen | YES — no changes | N/A |

---

## 8. Asset Universe

| Symbol | Active | Notes |
|--------|--------|-------|
| EURUSD | YES | Standard forex |
| EURJPY | YES | JPY cross |
| USDJPY | YES | JPY major |
| GBPJPY | YES | JPY cross |
| XAUUSD | YES | Gold |
| NAS100 | **DISABLED** | Excluded from V2.4A+ scope |

Disabled assets remain in `SETTINGS.symbols` only if explicitly re-enabled.

---

## 9. Guards & Monitoring

| Guard | Status | Source |
|-------|--------|--------|
| SampleIntegrityGuard | ACTIVE | `sample_integrity_guard.py` |
| ResearchAlignmentMonitor | ACTIVE | `research_alignment_monitor.py` |
| ExceptionDashboard | ACTIVE | `exception_dashboard.py` |
| PositionWatchdog | ACTIVE | `risk_manager.py:watchdog_check()` |

---

## 10. Frozen Files (SHA-256)

| File | Hash |
|------|------|
| `proxima_ops/risk/catastrophic_stop.py` | `62FE50FBD132225C7A996F3D65A316E06C3AE6A6E1AA1FFF71D36F98A7402FD0` |
| `proxima_ops/execution/order_manager.py` | `B9EFC2478ACB4B272092BA85AD46F9074B405B0F180330761E762C42614588E3` |
| `proxima_ops/risk/trade_risk_verifier.py` | `CD8972B62EA1A512A2AED52FF4A1B223AB8A2F7DC227DC5F62C439EB9647E0DB` |
| `proxima_ops/risk/risk_manager.py` | `965F14D4518057DD891E0CC3D37FC931DBFDA627D593CF024694F68F25A762D3` |
| `proxima_ops/config/settings.py` | `D257708570868C1266160CC65F02818BC2108436BF95D7418C99C635CDDE3AD6` |
| `run_proxima_demo.py` | `6CC4834701C19A3590831BB43B606080BD1AEA437332C12F48D7AA0BC0CCE8AF` |
| `proxima_ops/monitoring/global_rank_engine.py` | `F7A44B22FD5BD1476250B5BA725C3367AB7BBB674D10D65D6D6A2DD30A39E16F` |
| `proxima_ops/monitoring/sample_integrity_guard.py` | `C2C87DB01D1986E8E08A0DCC00334FF89DDD518333322A21EF4EAB5B479B8C1D` |
| `proxima_ops/monitoring/research_alignment_monitor.py` | `A82CF6416876EA27E2214F003AC7D7C9FB44F1970A4F3F1A2207FE106DE7DDDF` |
| `proxima_ops/monitoring/exception_dashboard.py` | `03AE55972ADEF7AD68178F9A29DF4918F1929DCED528A806A618F03838A6F770` |
| `proxima_ops/execution/mt5_connector.py` | `8FC2B51BFC6FF58DADE6242100685AD4E9A3B2420D6213BA1920669B53DFF7DC` |

---

## Freeze Rules

1. **No file modifications** to any frozen file without lifting the freeze
2. **No threshold changes** — ES threshold (0.80) is frozen
3. **No risk changes** — 0.25% risk per trade is frozen
4. **No stop changes** — catastrophic stop pips are frozen
5. **No ranking changes** — GLOBAL_ALL_QUALIFIED mode is frozen
6. **No frequency filter changes** — observe only
7. **No asset universe changes** — 5 FX+Gold only

To lift the freeze: minimum 300 trades collected + quantitative evidence + explicit authorization.
