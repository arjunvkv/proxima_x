"""
PROXIMA ASSET BIAS AUDIT — RQ1 through RQ10.

Determines why EURUSD generates nearly all live trades.
No trading logic modified — audit only.
"""

import sys
import os
import json
import time
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from proxima_ops.config.settings import SETTINGS
from research.mechanism_discovery.energy_dynamics import EnergyDynamics


# ============================================================
# DATA LOADERS
# ============================================================

def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _safe_db_access(fn, default=None):
    try:
        return fn()
    except Exception as e:
        return default


def _load_signal_ledger():
    from proxima_ops.ledger.signal_ledger import SignalLedger
    sl = SignalLedger()
    try:
        rows = sl._ensure_db()
    except Exception:
        pass
    try:
        sl._ensure_db()
        r = sl._conn.execute("SELECT * FROM signals ORDER BY signal_id DESC").fetchall()
        return [dict(zip([desc[0] for desc in sl._conn.description], row)) for row in r]
    except Exception as e:
        print(f"  Signal ledger read error: {e}")
        return []


def _load_trade_ledger():
    from proxima_ops.ledger.trade_ledger import TradeLedger
    tl = TradeLedger()
    try:
        r = tl._conn.execute("SELECT * FROM trades WHERE status = 'CLOSED' ORDER BY trade_id ASC").fetchall()
        return [dict(zip([desc[0] for desc in tl._conn.description], row)) for row in r]
    except Exception as e:
        print(f"  Trade ledger read error: {e}")
        return []


# ============================================================
# RQ1: Count Signals by Asset
# ============================================================

def rq1_count_signals(signals, stats_json):
    lines = []
    lines.append("=" * 62)
    lines.append("RQ1: SIGNAL COUNT BY ASSET")
    lines.append("=" * 62)

    assets = SETTINGS.symbols
    sym_stats = stats_json.get("symbol_stats", {})

    # Count from DuckDB signals table
    counts = {}
    for s in signals:
        sym = s.get("symbol", "?")
        if sym not in counts:
            counts[sym] = {"evaluations": 0, "es_gt_90": 0, "triggered": 0, "executed": 0}
        counts[sym]["evaluations"] += 1
        ep = s.get("es_percentile", 0)
        if isinstance(ep, (int, float)) and ep > 0.90:
            counts[sym]["es_gt_90"] += 1
        if s.get("signal_state") == "LONG":
            counts[sym]["triggered"] += 1
        if s.get("executed") == 1 or s.get("executed") is True:
            counts[sym]["executed"] += 1

    lines.append("")
    lines.append(f"{'Asset':<10s} {'Evaluations':<14s} {'ES>90%':<10s} {'Triggers':<12s} {'Executed':<10s}")
    lines.append("-" * 56)
    for a in assets:
        db = counts.get(a, {"evaluations": 0, "es_gt_90": 0, "triggered": 0, "executed": 0})
        js = sym_stats.get(a, {})
        db_ev = db["evaluations"]
        js_ev = js.get("evaluated", 0)
        trig = js.get("triggered", 0)
        exe = js.get("executed", 0)
        # Use the larger of DB or JSON count (JSON is more frequently written)
        ev = max(db_ev, js_ev)
        es90 = db["es_gt_90"]
        lines.append(f"{a:<10s} {ev:<14d} {es90:<10d} {trig:<12d} {exe:<10d}")

    # Separate EURUSD line
    eu = counts.get("EURUSD", {})
    total = sum(counts.get(a, {}).get("evaluations", 0) for a in assets)
    eu_ev = max(eu.get("evaluations", 0), sym_stats.get("EURUSD", {}).get("evaluated", 0))
    lines.append("")
    lines.append(f"EURUSD share of evaluations: {eu_ev}/{max(total, 1)} ({eu_ev/max(total,1)*100:.1f}%)" if total > 0 else "EURUSD share: N/A (no data)")

    lines.append("")
    return "\n".join(lines), counts


# ============================================================
# RQ2: ES Rank Distributions
# ============================================================

def rq2_es_distributions(signals):
    lines = []
    lines.append("=" * 62)
    lines.append("RQ2: ES RANK DISTRIBUTIONS")
    lines.append("=" * 62)

    assets = SETTINGS.symbols
    by_asset = {a: [] for a in assets}

    for s in signals:
        sym = s.get("symbol", "?")
        if sym in by_asset:
            ep = s.get("es_percentile")
            if isinstance(ep, (int, float)):
                by_asset[sym].append(ep)

    lines.append("")
    lines.append(f"{'Asset':<10s} {'Count':<8s} {'p50':<10s} {'p75':<10s} {'p90':<10s} {'p95':<10s} {'p99':<10s} {'Mean':<10s} {'Std':<10s}")
    lines.append("-" * 76)
    for a in assets:
        vals = np.array(by_asset.get(a, []))
        if len(vals) == 0:
            lines.append(f"{a:<10s} {'N/A':<8s} {'N/A':<10s} {'N/A':<10s} {'N/A':<10s} {'N/A':<10s} {'N/A':<10s} {'N/A':<10s} {'N/A':<10s}")
            continue
        p50 = np.percentile(vals, 50)
        p75 = np.percentile(vals, 75)
        p90 = np.percentile(vals, 90)
        p95 = np.percentile(vals, 95)
        p99 = np.percentile(vals, 99)
        mu = np.mean(vals)
        sd = np.std(vals)
        lines.append(f"{a:<10s} {len(vals):<8d} {p50:<10.4f} {p75:<10.4f} {p90:<10.4f} {p95:<10.4f} {p99:<10.4f} {mu:<10.4f} {sd:<10.4f}")

    lines.append("")
    return "\n".join(lines)


# ============================================================
# RQ3: Trigger Frequency Estimate
# ============================================================

def rq3_trigger_frequency(signals, stats_json):
    lines = []
    lines.append("=" * 62)
    lines.append("RQ3: TRIGGER FREQUENCY ESTIMATE")
    lines.append("=" * 62)

    assets = SETTINGS.symbols
    sym_stats = stats_json.get("symbol_stats", {})

    lines.append("")
    lines.append(f"{'Asset':<10s} {'Triggers':<12s} {'Hours':<10s} {'Sig/Mo':<10s} {'Executed':<12s} {'Exec/Mo':<10s}")
    lines.append("-" * 64)

    timestamps = [s.get("timestamp", 0) for s in signals if isinstance(s.get("timestamp"), (int, float))]
    hours_span = 1
    if timestamps:
        span = max(timestamps) - min(timestamps)
        hours_span = max(1, span / 3600)

    for a in assets:
        js = sym_stats.get(a, {})
        trig = js.get("triggered", 0)
        exe = js.get("executed", 0)
        sig_mo = (trig / hours_span) * 730 if hours_span > 0 else 0
        exe_mo = (exe / hours_span) * 730 if hours_span > 0 else 0
        lines.append(f"{a:<10s} {trig:<12d} {hours_span:<10.0f} {sig_mo:<10.1f} {exe:<12d} {exe_mo:<10.1f}")

    lines.append(f"\n  Data span: {hours_span:.0f} hours ({hours_span/24:.1f} days)")
    lines.append("  Monthly estimate: 730 hours/month")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# RQ4: Lookback Integrity
# ============================================================

def rq4_lookback_integrity():
    lines = []
    lines.append("=" * 62)
    lines.append("RQ4: LOOKBACK INTEGRITY")
    lines.append("=" * 62)

    from proxima_ops.execution.mt5_connector import MT5Connector
    mt5 = MT5Connector()
    if not mt5.connect():
        lines.append("\n  Could not connect to MT5. Skipping live lookback check.")
        lines.append("")
        return "\n".join(lines)

    assets = SETTINGS.symbols
    lines.append("")
    lines.append(f"{'Asset':<10s} {'Bars':<8s} {'Warmup':<10s} {'NaN_ES':<10s} {'Missing':<10s} {'Status':<12s}")
    lines.append("-" * 60)

    for a in assets:
        rates = mt5.get_rates(a, count=550, timeframe="H1")
        if rates is None or len(rates) == 0:
            lines.append(f"{a:<10s} {'0':<8s} {'FAIL':<10s} {'N/A':<10s} {'N/A':<10s} {'NO_DATA':<12s}")
            continue

        bars = len(rates)
        warmup = bars >= 524
        warmup_str = "OK" if warmup else "SHORT"

        # Compute ES, check for NaN
        try:
            prices = np.array([r["close"] for r in rates], dtype=np.float64)
            highs = np.array([r["high"] for r in rates], dtype=np.float64)
            lows = np.array([r["low"] for r in rates], dtype=np.float64)
            volumes = np.array([r["volume"] for r in rates], dtype=np.float64)
            returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
            dd = {"price": prices, "returns": returns, "volume": volumes, "high": highs, "low": lows}
            ed = EnergyDynamics()
            res = ed.compute(dd)
            es_arr = res.get("energy_storage", np.zeros(len(prices)))
            nan_count = int(np.sum(np.isnan(es_arr)))
            missing = int(np.sum(es_arr == 0.0))
        except Exception as e:
            nan_count = -1
            missing = -1

        # Check the 504-bar window requirement
        es_window = 504
        has_504 = bars >= es_window + 20
        status_str = "READY" if (warmup and has_504) else "WARMING"

        lines.append(f"{a:<10s} {bars:<8d} {warmup_str:<10s} {nan_count:<10d} {missing:<10d} {status_str:<12s}")

    mt5.disconnect()
    lines.append("")
    return "\n".join(lines)


# ============================================================
# RQ5: Percentile Engine Verification
# ============================================================

def rq5_percentile_verify():
    lines = []
    lines.append("=" * 62)
    lines.append("RQ5: PERCENTILE ENGINE VERIFICATION")
    lines.append("=" * 62)

    lines.append("""
  Algorithm (live deployment):
    es_window = es_history[-504:]
    es_percentile = sum(es_window <= current_es) / 504

  This is an ECDF (empirical CDF) rank, NOT interpolation-based.
  Correctness: yes, standard rolling percentile rank.

  Verification with synthetic data:
""")

    # Test with known data
    test = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    for val, expected in [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9), (1.0, 1.0), (0.0, 0.0), (2.0, 1.0)]:
        rank = float(np.sum(test <= val)) / len(test)
        match = "OK" if abs(rank - expected) < 0.01 else "MISMATCH"
        lines.append(f"  value={val:.1f} -> rank={rank:.2f} (expected {expected:.2f}) [{match}]")

    lines.append("""
  Edge case: all identical values
""")
    test2 = np.full(100, 0.5)
    rank2 = float(np.sum(test2 <= 0.5)) / len(test2)
    lines.append(f"  all=0.5, query=0.5 -> rank={rank2:.2f} (expected 1.0) [OK]")

    lines.append("""
  Edge case: window < 504 (warmup)
""")
    lines.append("  Guard: `if len(rates) < 524: continue` — skips evaluation entirely.")
    lines.append("  No rank is computed for incomplete windows.")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# RQ6: Normalization Bias
# ============================================================

def rq6_normalization_bias():
    lines = []
    lines.append("=" * 62)
    lines.append("RQ6: NORMALIZATION BIAS CHECK")
    lines.append("=" * 62)

    lines.append("""
  The ES percentile is computed per-symbol using a 504-bar rolling window:
    es_percentile(sym) = rank(current_es(sym), window_es(sym)[-504:])

  Key insight: EACH SYMBOL HAS ITS OWN 504-BAR WINDOW.
  The rank is relative to the symbol's OWN history, not cross-asset.

  This means:
  - EURUSD's ES at the 95th percentile means EURUSD's current ES is
    higher than 95% of EURUSD's own recent history.
  - It does NOT mean EURUSD's ES is higher than other assets' ES.

  If EURUSD has a narrower ES distribution (low variance), it will
  reach extreme percentiles more easily than a wide-distribution asset.

  Implication: the 90th percentile threshold is NOT comparable across assets.
  EURUSD could trigger at 90th percentile while its raw ES is modest,
  while another asset might have a higher raw ES but lower rank.
""")

    # Try to compute actual raw ES values per symbol
    from proxima_ops.execution.mt5_connector import MT5Connector
    mt5 = MT5Connector()
    if mt5.connect():
        assets = SETTINGS.symbols
        ed = EnergyDynamics()
        lines.append(f"\n{'Asset':<10s} {'Raw_ES_mean':<14s} {'Raw_ES_std':<14s} {'Raw_ES_last':<14s} {'CV':<10s}")
        lines.append("-" * 62)
        for a in assets:
            rates = mt5.get_rates(a, count=550, timeframe="H1")
            if rates and len(rates) >= 100:
                prices = np.array([r["close"] for r in rates], dtype=np.float64)
                highs = np.array([r["high"] for r in rates], dtype=np.float64)
                lows = np.array([r["low"] for r in rates], dtype=np.float64)
                volumes = np.array([r["volume"] for r in rates], dtype=np.float64)
                returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
                dd = {"price": prices, "returns": returns, "volume": volumes, "high": highs, "low": lows}
                res = ed.compute(dd)
                es = np.nan_to_num(res.get("energy_storage", np.zeros(len(prices))), nan=0.0)
                mu = np.mean(es)
                sd = np.std(es)
                last = es[-1]
                cv = sd / max(mu, 1e-12)
                lines.append(f"{a:<10s} {mu:<14.6f} {sd:<14.6f} {last:<14.6f} {cv:<10.4f}")
        mt5.disconnect()
    else:
        lines.append("\n  Could not connect to MT5 for raw ES computation.")

    lines.append("""
  Interpretation:
  - If EURUSD has the LOWEST CV (least variance), its rank is most
    sensitive to small raw ES movements — it reaches 90th percentile
    more easily than other assets.
  - If EURUSD has the HIGHEST raw ES, that's market-driven.
  - If EURUSD has similar raw ES but higher rank, that's percentile bias.

""")
    return "\n".join(lines)


# ============================================================
# RQ7: Cross-Asset Opportunity
# ============================================================

def rq7_cross_asset(signals):
    lines = []
    lines.append("=" * 62)
    lines.append("RQ7: CROSS-ASSET OPPORTUNITY ANALYSIS")
    lines.append("=" * 62)

    assets = SETTINGS.symbols
    by_asset = {a: [] for a in assets}
    all_ranked = []

    for s in signals:
        sym = s.get("symbol", "?")
        ep = s.get("es_percentile")
        if sym in by_asset and isinstance(ep, (int, float)):
            by_asset[sym].append(ep)
            all_ranked.append((sym, ep))

    lines.append("")
    lines.append(f"  Would other assets ever reach the 90th percentile threshold?")
    lines.append("")

    lines.append(f"{'Asset':<10s} {'Total Eval':<14s} {'Count >=90%':<14s} {'Pct >=90%':<12s} {'Max Rank':<10s}")
    lines.append("-" * 60)
    for a in assets:
        vals = by_asset.get(a, [])
        if not vals:
            lines.append(f"{a:<10s} {'0':<14s} {'0':<14s} {'0.0%':<12s} {'N/A':<10s}")
            continue
        total = len(vals)
        above90 = sum(1 for v in vals if v > 0.90)
        pct = above90 / total * 100
        mx = max(vals)
        lines.append(f"{a:<10s} {total:<14d} {above90:<14d} {pct:<12.1f} {mx:<10.4f}")

    lines.append("")
    # Check if EURUSD dominance is about total evaluations or trigger rate
    lines.append("  EURUSD dominance breakdown:")
    eu_vals = by_asset.get("EURUSD", [])
    non_eu_vals = [v for a2 in assets if a2 != "EURUSD" for v in by_asset.get(a2, [])]
    eu_above = sum(1 for v in eu_vals if v > 0.90)
    ne_above = sum(1 for v in non_eu_vals if v > 0.90)
    eu_total = len(eu_vals)
    ne_total = len(non_eu_vals)
    eu_rate = eu_above / max(eu_total, 1) * 100
    ne_rate = ne_above / max(ne_total, 1) * 100
    lines.append(f"    EURUSD trigger rate:  {eu_above}/{eu_total} = {eu_rate:.1f}%")
    lines.append(f"    Other assets rate:     {ne_above}/{ne_total} = {ne_rate:.1f}%")
    lines.append("")
    if eu_rate > ne_rate * 1.5:
        lines.append("  CONCLUSION: EURUSD triggers at a significantly HIGHER RATE.")
        lines.append("  This suggests either market conditions favor EURUSD, or the")
        lines.append("  percentile engine is biased toward EURUSD.")
    elif eu_total > ne_total * 2:
        lines.append("  CONCLUSION: EURUSD dominates by TOTAL VOLUME of evaluations.")
        lines.append("  This suggests an evaluation loop imbalance (more EURUSD checks).")
    else:
        lines.append("  CONCLUSION: EURUSD and other assets trigger at similar rates.")
        lines.append("  Dominance is driven by higher signal volume.")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# RQ8: Frequency Filter Interaction
# ============================================================

def rq8_frequency_filter(blocked_tracker_json):
    lines = []
    lines.append("=" * 62)
    lines.append("RQ8: FREQUENCY FILTER INTERACTION")
    lines.append("=" * 62)

    lines.append("""
  The frequency filter blocks signals when market_closed=True (weekends/off-hours).
  It does NOT block per-asset — it's a global time filter.

  However, blocked signals can also come from:
  - SPREAD (spread > max per symbol)
  - POSITION_EXISTS (already in a trade for that symbol)
  - MAX_POSITIONS / RISK_LIMIT (too many positions)
  - FREQUENCY_FILTER (market closed)

""")

    # Read from observability_stats.json for per-symbol block breakdown
    stats_file = os.path.join(os.path.dirname(__file__), "..", "..", "proxima_ops", "data", "observability_stats.json")
    stats = _load_json(stats_file)
    sym_stats = stats.get("symbol_stats", {})

    assets = SETTINGS.symbols
    lines.append(f"{'Asset':<10s} {'Spread':<10s} {'PosExist':<12s} {'RiskLim':<10s} {'Freq':<10s} {'Total':<10s}")
    lines.append("-" * 62)
    total_s = total_p = total_r = total_f = 0
    for a in assets:
        js = sym_stats.get(a, {})
        sp = js.get("spread_blocks", 0)
        pe = js.get("position_blocks", 0)
        rl = js.get("risk_blocks", 0)
        fr = js.get("frequency_blocks", 0)
        tt = sp + pe + rl + fr
        total_s += sp
        total_p += pe
        total_r += rl
        total_f += fr
        lines.append(f"{a:<10s} {sp:<10d} {pe:<12d} {rl:<10d} {fr:<10d} {tt:<10d}")
    lines.append("-" * 62)
    lines.append(f"{'TOTAL':<10s} {total_s:<10d} {total_p:<12d} {total_r:<10d} {total_f:<10d} {total_s+total_p+total_r+total_f:<10d}")

    lines.append("")
    lines.append("  Leakage analysis:")
    exec_count = stats.get("executed_count", 0)
    lines.append(f"  Total executed: {exec_count}")
    if exec_count > 0 and total_s > 0:
        lines.append(f"  Blocked by spread (potentially profitable signals lost): {total_s}")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# RQ9: Asset Opportunity Leaderboard
# ============================================================

def rq9_leaderboard(signals, stats_json):
    lines = []
    lines.append("=" * 62)
    lines.append("RQ9: ASSET OPPORTUNITY LEADERBOARD")
    lines.append("=" * 62)

    assets = SETTINGS.symbols
    sym_stats = stats_json.get("symbol_stats", {})

    # 1. Signal frequency
    freq_scores = {}
    for a in assets:
        js = sym_stats.get(a, {})
        freq_scores[a] = js.get("triggered", 0)

    # 2. Alpha quality (ES rank distribution)
    by_asset = {a: [] for a in assets}
    for s in signals:
        sym = s.get("symbol", "?")
        ep = s.get("es_percentile")
        if sym in by_asset and isinstance(ep, (int, float)):
            by_asset[sym].append(ep)

    alpha_scores = {}
    for a in assets:
        vals = by_asset.get(a, [])
        if len(vals) >= 5:
            alpha_scores[a] = np.mean(vals)
        else:
            alpha_scores[a] = 0.0

    # 3. Leakage contribution
    leak_scores = {}
    for a in assets:
        js = sym_stats.get(a, {})
        block_total = sum([
            js.get("spread_blocks", 0),
            js.get("position_blocks", 0),
            js.get("risk_blocks", 0),
            js.get("frequency_blocks", 0)])
        executed = js.get("executed", 0)
        total_opp = block_total + executed
        if total_opp > 0:
            leak_scores[a] = executed / total_opp
        else:
            leak_scores[a] = 0.0

    lines.append("")
    lines.append(f"{'Rank':<6s} {'Asset':<10s} {'Freq':<12s} {'Alpha(mean ES)':<16s} {'Leakage(exec%)':<16s} {'Composite':<10s}")
    lines.append("-" * 70)

    sorted_freq = sorted(freq_scores.keys(), key=lambda a: -freq_scores[a])
    composite = {}
    for a in assets:
        # Normalize each dimension to 0-1
        max_f = max(freq_scores.values()) or 1
        max_a = max(alpha_scores.values()) or 1
        max_l = max(leak_scores.values()) or 1
        f_norm = freq_scores[a] / max_f
        a_norm = alpha_scores[a] / max_a
        l_norm = leak_scores[a] / max_l
        composite[a] = (f_norm * 0.4 + a_norm * 0.4 + l_norm * 0.2)

    for rank, a in enumerate(sorted(sorted_freq, key=lambda a: -composite[a]), 1):
        lines.append(f"{rank:<6d} {a:<10s} {freq_scores[a]:<12d} {alpha_scores[a]:<16.4f} {leak_scores[a]:<16.2%} {composite[a]:<10.4f}")

    lines.append("")
    return "\n".join(lines)


# ============================================================
# RQ10: Final Adjudication
# ============================================================

def rq10_adjudication(signals, stats_json, rq_results: dict):
    lines = []
    lines.append("=" * 62)
    lines.append("RQ10: FINAL ADJUDICATION")
    lines.append("=" * 62)

    assets = SETTINGS.symbols
    sym_stats = stats_json.get("symbol_stats", {})

    # Gather evidence
    eu_evals = sym_stats.get("EURUSD", {}).get("evaluated", 0)
    total_evals = sum(sym_stats.get(a, {}).get("evaluated", 0) for a in assets)
    eu_share = eu_evals / max(total_evals, 1)

    eu_trig = sym_stats.get("EURUSD", {}).get("triggered", 0)
    total_trig = sum(sym_stats.get(a, {}).get("triggered", 0) for a in assets)
    trig_share = eu_trig / max(total_trig, 1)

    eu_exe = sym_stats.get("EURUSD", {}).get("executed", 0)
    total_exe = sum(sym_stats.get(a, {}).get("executed", 0) for a in assets)
    exe_share = eu_exe / max(total_exe, 1)

    # Get raw ES CV from RQ6 if available
    # (we'll re-compute here)
    lines.append(f"""
  Evidence Summary
  {'='*40}
  EURUSD evaluation share:  {eu_share:.1%}
  EURUSD trigger share:     {trig_share:.1%}
  EURUSD execution share:   {exe_share:.1%}""")

    # Build classifications
    scores = {}
    for label, condition in [
        ("MARKET_DRIVEN", None),
        ("PERCENTILE_BIAS", None),
        ("DATA_BIAS", None),
        ("NORMALIZATION_BIAS", None)
    ]:
        scores[label] = 0.0

    # Test 1: Market-driven — EURUSD genuinely has the highest raw ES
    from proxima_ops.execution.mt5_connector import MT5Connector
    mt5 = MT5Connector()
    raw_es_values = {}
    if mt5.connect():
        ed = EnergyDynamics()
        for a in assets:
            rates = mt5.get_rates(a, count=550, timeframe="H1")
            if rates and len(rates) >= 100:
                prices = np.array([r["close"] for r in rates], dtype=np.float64)
                highs = np.array([r["high"] for r in rates], dtype=np.float64)
                lows = np.array([r["low"] for r in rates], dtype=np.float64)
                volumes = np.array([r["volume"] for r in rates], dtype=np.float64)
                returns = np.diff(np.log(prices), prepend=np.log(prices[0]))
                dd = {"price": prices, "returns": returns, "volume": volumes, "high": highs, "low": lows}
                res = ed.compute(dd)
                es = np.nan_to_num(res.get("energy_storage", np.zeros(len(prices))), nan=0.0)
                raw_es_values[a] = {
                    "current": float(es[-1]),
                    "mean": float(np.mean(es)),
                    "std": float(np.std(es)),
                    "cv": float(np.std(es) / max(np.mean(es), 1e-12))
                }
        mt5.disconnect()

    if raw_es_values:
        lines.append(f"""
  Raw ES Comparison (current bar)
  {'='*40}""")
        for a in assets:
            v = raw_es_values.get(a, {})
            lines.append(f"  {a:<10s} current={v.get('current', 0):.6f}  mean={v.get('mean', 0):.6f}  std={v.get('std', 0):.6f}  CV={v.get('cv', 0):.4f}")

        # Check if EURUSD has highest raw ES
        eu_current = raw_es_values.get("EURUSD", {}).get("current", 0)
        others_max = max(raw_es_values.get(a, {}).get("current", 0) for a in assets if a != "EURUSD")
        if eu_current >= others_max:
            scores["MARKET_DRIVEN"] += 0.6
            lines.append(f"\n  EURUSD current raw ES ({eu_current:.6f}) >= others max ({others_max:.6f})")
            lines.append(f"  -> EURUSD objectively has highest energy storage right now")
        else:
            lines.append(f"\n  EURUSD current raw ES ({eu_current:.6f}) < others max ({others_max:.6f})")
            lines.append(f"  -> EURUSD dominance is NOT driven by higher raw ES")

        # Check CV (lower CV = easier to reach extreme percentiles)
        eu_cv = raw_es_values.get("EURUSD", {}).get("cv", 0)
        others_cv = [raw_es_values.get(a, {}).get("cv", 0) for a in assets if a != "EURUSD"]
        if eu_cv < min(others_cv):
            scores["PERCENTILE_BIAS"] += 0.5
            lines.append(f"\n  EURUSD has LOWEST CV ({eu_cv:.4f}) -> rank is most sensitive to small changes")
            lines.append(f"  -> PERCENTILE BIAS: EURUSD reaches 90th percentile too easily")
        elif eu_cv <= np.mean(others_cv):
            scores["PERCENTILE_BIAS"] += 0.2
            lines.append(f"\n  EURUSD CV ({eu_cv:.4f}) below average -> mild percentile bias")
        else:
            lines.append(f"\n  EURUSD CV ({eu_cv:.4f}) is not notably low -> percentile bias unlikely")
    else:
        lines.append(f"\n  (Could not connect to MT5 for raw ES comparison)")

    # Test 2: Data bias — does EURUSD have better data quality?
    rates_counts = {}
    mt5 = MT5Connector()
    if mt5.connect():
        for a in assets:
            rates = mt5.get_rates(a, count=550, timeframe="H1")
            rates_counts[a] = len(rates) if rates else 0
        mt5.disconnect()

        eu_bars = rates_counts.get("EURUSD", 0)
        others_bars = [rates_counts.get(a, 0) for a in assets if a != "EURUSD"]
        if eu_bars > max(others_bars):
            scores["DATA_BIAS"] += 0.3
            lines.append(f"\n  EURUSD has more bars ({eu_bars}) than any other asset (max {max(others_bars)})")
            lines.append(f"  -> DATA BIAS: EURUSD gets more frequent evaluations")
        else:
            scores["DATA_BIAS"] += 0.0

    # Test 3: Normalization bias — check if other symbols also reach 90th
    by_asset = {a: [] for a in assets}
    for s in signals:
        sym = s.get("symbol", "?")
        ep = s.get("es_percentile")
        if sym in by_asset and isinstance(ep, (int, float)):
            by_asset[sym].append(ep)

    other_above90 = 0
    for a in assets:
        if a == "EURUSD":
            continue
        vals = by_asset.get(a, [])
        if any(v > 0.90 for v in vals):
            other_above90 += 1

    if other_above90 == len(assets) - 1:
        scores["NORMALIZATION_BIAS"] = 0.0
        lines.append(f"\n  All other assets DO reach 90th percentile -> no normalization bias")
    elif other_above90 > 0:
        scores["NORMALIZATION_BIAS"] += 0.2
        lines.append(f"\n  {other_above90}/{len(assets)-1} other assets reach 90th percentile -> partial normalization")
    else:
        scores["NORMALIZATION_BIAS"] += 0.5
        lines.append(f"\n  NO other asset reaches 90th percentile -> strong normalization bias")

    # Determine primary and secondary classifications
    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    primary = sorted_scores[0][0]
    secondary = sorted_scores[1][0] if len(sorted_scores) > 1 else "NONE"

    lines.append(f"""
  {'='*40}
  Classification Scores
  {'='*40}""")
    for label, score in sorted_scores:
        bar = "#" * int(score * 20)
        lines.append(f"  {label:<25s} {score:.2f} {bar}")

    lines.append(f"""
  PRIMARY:   {primary}
  SECONDARY: {secondary}

  {'='*40}
  VERDICT
  {'='*40}""")

    verdicts = {
        "MARKET_DRIVEN": "EURUSD is genuinely high-energy right now. Its raw ES is objectively highest.",
        "PERCENTILE_BIAS": "EURUSD has a narrower ES distribution (lower CV), so its rank reaches 90th percentile more easily than other assets. The engine isn't biased — the PERCENTILE RANKING is.",
        "DATA_BIAS": "EURUSD receives more frequent evaluations or has better data availability than other assets.",
        "NORMALIZATION_BIAS": "Other assets rarely or never reach the 90th percentile threshold, suggesting the per-symbol normalization window is not comparable across assets."
    }

    lines.append(f"\n  {verdicts.get(primary, 'Unknown classification.')}")
    if secondary != "NONE":
        lines.append(f"\n  Secondary factor: {verdicts.get(secondary, '')}")
    lines.append("""

  Is EURUSD dominating because the market favors it,
  or because the engine is structurally biased toward it?
""")
    if primary == "MARKET_DRIVEN":
        lines.append("  ANSWER: MARKET-DRIVEN. EURUSD has the highest raw energy storage.")
        lines.append("  The engine is correctly identifying EURUSD as the best opportunity.")
    elif primary == "PERCENTILE_BIAS":
        lines.append("  ANSWER: PERCENTILE BIAS. EURUSD's narrow ES distribution makes it")
        lines.append("  easier to reach the 90th percentile threshold. The raw ES difference")
        lines.append("  may not justify the observed dominance.")
    elif primary == "DATA_BIAS":
        lines.append("  ANSWER: DATA BIAS. EURUSD gets more or better data, leading to")
        lines.append("  more frequent evaluations and thus more triggers.")
    else:
        lines.append("  ANSWER: NORMALIZATION BIAS. The per-symbol 504-bar percentile")
        lines.append("  normalization creates incomparable thresholds across assets.")
    lines.append("")

    return "\n".join(lines), scores


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 62)
    print("PROXIMA ASSET BIAS AUDIT")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    # Load data
    print("\nLoading signal ledger from DuckDB...")
    signals = _safe_db_access(_load_signal_ledger, [])
    print(f"  Loaded {len(signals)} signal records")

    stats_file = os.path.join(os.path.dirname(__file__), "..", "..", "proxima_ops", "data", "observability_stats.json")
    stats_json = _load_json(stats_file)
    print(f"  Observability stats: {stats_json.get('evaluated_count', 0)} evaluations")

    print("")

    all_reports = {}
    rq_results = {}

    # RQ1
    r1, r1_counts = rq1_count_signals(signals, stats_json)
    print(r1)
    all_reports["RQ1"] = r1
    rq_results["counts"] = r1_counts

    # RQ2
    r2 = rq2_es_distributions(signals)
    print(r2)
    all_reports["RQ2"] = r2

    # RQ3
    r3 = rq3_trigger_frequency(signals, stats_json)
    print(r3)
    all_reports["RQ3"] = r3

    # RQ4
    r4 = rq4_lookback_integrity()
    print(r4)
    all_reports["RQ4"] = r4

    # RQ5
    r5 = rq5_percentile_verify()
    print(r5)
    all_reports["RQ5"] = r5

    # RQ6
    r6 = rq6_normalization_bias()
    print(r6)
    all_reports["RQ6"] = r6

    # RQ7
    r7 = rq7_cross_asset(signals)
    print(r7)
    all_reports["RQ7"] = r7

    # RQ8
    r8 = rq8_frequency_filter(stats_json)
    print(r8)
    all_reports["RQ8"] = r8

    # RQ9
    r9 = rq9_leaderboard(signals, stats_json)
    print(r9)
    all_reports["RQ9"] = r9

    # RQ10
    r10, r10_scores = rq10_adjudication(signals, stats_json, rq_results)
    print(r10.encode('utf-8', errors='replace').decode('utf-8'))
    all_reports["RQ10"] = r10

    # Generate ASSET_BIAS_REPORT.md
    lines = []
    lines.append("# PROXIMA ASSET BIAS REPORT")
    lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    for k, v in r10_scores.items():
        lines.append(f"- {k}: {v:.2f}")
    lines.append("")
    primary = max(r10_scores.keys(), key=lambda k: r10_scores[k])
    lines.append(f"**Primary Classification: {primary}**")
    lines.append("")
    lines.append("---")
    lines.append("")

    for rq_name in ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "RQ6", "RQ7", "RQ8", "RQ9", "RQ10"]:
        content = all_reports.get(rq_name, "")
        # Extract just the lines, removing the header decorations
        report_lines = content.split("\n")
        # Skip the "======" header lines, keep the substance
        lines.append(f"## {rq_name}")
        lines.append("")
        lines.append("```")
        for l in report_lines:
            lines.append(l)
        lines.append("```")
        lines.append("")

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "ASSET_BIAS_REPORT.md")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to: {out_path}")
    print("=" * 62)


if __name__ == "__main__":
    main()
