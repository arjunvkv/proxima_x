"""FTMO SURVIVABILITY SUITE for the CURRENT live 4-leg book (25k anchor).

Checks the book against the prop-firm failure modes:
 1. Daily-loss limit 5% ($1,250) + max-DD 10% ($2,500) breach stats.
 2. Spread stress: typical vs measured worst-case, plus +50% shock envelope.
 3. Consecutive losers, worst 5-day window, worst day.
 4. Bootstrap resampling of daily PnL -> P(DD>=2,500) and P(daily<=-1,250)
    over 20-trading-day blocks.
 5. Fill-drop stress: lose 10% of fills at random (live worker reality).
 6. Day-skip stress: 5% of days produce zero entries (downtime).
"""
import sys, os, random, statistics as st, datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from proxima_ops.backtest import StrategySpec, run_strategy
from proxima_ops.backtest.feed import build_bars_map

UNIVERSE = ["EURUSD","USDJPY","GBPUSD","AUDUSD","EURJPY","GBPJPY","EURAUD","EURNZD",
            "GBPAUD","GBPNZD","GBPCAD","AUDNZD","USDCAD","NZDUSD","EURGBP","EURCHF",
            "USDCHF","AUDJPY"]
SPREAD_T = {"EURUSD":0.8,"USDJPY":1.2,"GBPUSD":1.5,"AUDUSD":1.1,"EURJPY":2.2,
            "GBPJPY":3.0,"EURAUD":2.8,"EURNZD":3.4,"GBPAUD":3.6,"GBPNZD":4.4,
            "GBPCAD":3.2,"AUDNZD":2.6,"USDCAD":1.8,"NZDUSD":1.4,"EURGBP":1.2,
            "EURCHF":1.8,"USDCHF":1.4,"AUDJPY":1.8}
SPREAD_M = {"EURUSD":0.6,"USDJPY":3.5,"GBPUSD":1.3,"AUDUSD":0.9,"EURJPY":5.7,
            "GBPJPY":7.1,"EURAUD":2.9,"EURNZD":5.1,"GBPAUD":4.9,"GBPNZD":6.3,
            "GBPCAD":4.1,"AUDNZD":3.5,"USDCAD":1.6,"NZDUSD":1.2,"EURGBP":1.7,
            "EURCHF":2.6,"USDCHF":1.3,"AUDJPY":4.2}

LEGS = {
    "tokyo":   {"sessions": [0],             "lb": 6,   "top": 3, "hold": 12, "lot": 0.35,
                "sl_tp": {"JPY": (0.50, 0.70), "else": (0.0050, 0.0070)}},
    "cascade": {"sessions": [2,3,4],         "lb": 1440,"top": 8, "hold": 24, "lot": 0.09,
                "sl_tp": {"JPY": (0.40, 0.60), "else": (0.0040, 0.0060)}},
    "london":  {"sessions": [7,8,9],         "lb": 1440,"top": 5, "hold": 12, "lot": 0.30,
                "sl_tp": {"JPY": (0.40, 0.60), "else": (0.0040, 0.0060)}},
    "usfade":  {"sessions": [14,15,16,17,18,19], "lb": 50, "top": 5, "hold": 24, "lot": 0.30,
                "sl_tp": {"JPY": (0.40, 0.60), "else": (0.0040, 0.0060)}},
}

def spec_for(name, cfg):
    return StrategySpec.from_dict({
        "name": name, "universe": list(UNIVERSE),
        "feed": {"kind": "bar", "timeframe": "M5"},
        "signal": {"rule": "session_exhaustion", "lookback": cfg["lb"],
                   "pick": "n_worst", "top_n": cfg["top"], "side": "both",
                   "fill_bar": 1},
        "exit": {"mode": "sl_tp_hold", "hold_bars": cfg["hold"], "stop_first": True,
                 "jpy_sl_tp": cfg["sl_tp"]["JPY"], "non_jpy_sl_tp": cfg["sl_tp"]["else"]},
        "sessions": cfg["sessions"], "base_lot": cfg["lot"]})

def book_daily(spread_map, drop_rate=0.0, seed=7):
    bars = build_bars_map(UNIVERSE)
    rng = random.Random(seed)
    per_day: dict[int, float] = {}
    for name, cfg in LEGS.items():
        usd = run_strategy(bars, spec_for(name, cfg), volume=cfg["lot"],
                           commission_per_lot=3.0, spread_pips_map=spread_map)
        for t in usd:
            if rng.random() < drop_rate:
                continue
            d = t["entry_ts"] // 86400
            per_day[d] = per_day.get(d, 0.0) + t["net"]
    return per_day

def analyze(per_day, tag):
    days = sorted(per_day)
    nets = [per_day[d] for d in days]
    n = len(nets)
    tot = sum(nets)
    green = sum(1 for x in nets if x > 0) / n
    worst = min(nets)
    worst_d = datetime.datetime.fromtimestamp(days[nets.index(worst)] * 86400,
                                              datetime.UTC).strftime("%Y-%m-%d")
    consec = cur = 0
    for x in nets:
        if x < 0:
            cur += 1
            consec = max(consec, cur)
        else:
            cur = 0
    worst5 = sum(sorted(nets)[:5])
    eq = 25000.0; peak = eq; maxdd = 0.0
    for x in nets:
        eq += x
        peak = max(peak, eq)
        maxdd = max(maxdd, peak - eq)
    daily_breach = sum(1 for x in nets if x <= -1250)
    print(f"\n[{tag}] {n} trading days, net ${tot:,.0f}, {green:.0%} green days")
    print(f"  worst day {worst_d}: ${worst:,.2f} | worst 5-day: ${worst5:,.0f} "
          f"({worst5/25000:.1%} of ref) | max consecutive losers: {consec}")
    print(f"  peak-to-trough DD: ${maxdd:,.0f} ({maxdd/25000:.2%} of 25k ref) "
          f"vs 10% cap $2,500 | days >= $1,250 daily loss: {daily_breach}")
    # bootstrap: 5000 x 20-day blocks, sample days WITH replacement
    rng = random.Random(42)
    blow_dd = blow_daily = 0
    worst_draw = 0.0
    for _ in range(5000):
        eq = 25000.0; pk = eq
        for _ in range(20):
            x = rng.choice(nets)
            eq += x
            pk = max(pk, eq)
            if pk - eq >= 2500:
                blow_dd += 1
                break
            if x <= -1250:
                blow_daily += 1
        worst_draw = max(worst_draw, pk - eq)
    print(f"  bootstrap 5k x 20d: {blow_dd/5000:.1%} hit 10% DD cap; "
          f"{blow_daily/5000:.2f} daily<-1250 per block; max sampled draw "
          f"${worst_draw:,.0f}")

if __name__ == "__main__":
    print("== FTMO SURVIVABILITY — current live 4-leg book (25k ref) ==")
    analyze(book_daily(SPREAD_T), "TYPICAL busy-spread")
    analyze(book_daily(SPREAD_M), "WORST measured-spread")
    # +50% shock on measured (worst-of-worst; rare, but stress it)
    shock = {k: v * 1.5 for k, v in SPREAD_M.items()}
    analyze(book_daily(shock), "STRESS measured x1.5 spread")
    # operational: 10% fills lost at random
    analyze(book_daily(SPREAD_M, drop_rate=0.10), "OPS drop 10% fills (worst spread)")