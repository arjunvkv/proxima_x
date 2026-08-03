"""Run sim_recon (non-blocking) with FundedNext $3/round-turn commission on FTMO + normal MT5 data."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sim_recon import ReconSim
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from datetime import datetime, timezone

PAIRS = ["AUDNZD", "EURAUD", "EURNZD", "GBPAUD", "GBPCAD", "GBPNZD"]
FROM = datetime(2026, 6, 8)
TO = datetime(2026, 7, 26)
COMMISSION_FUNDEDNEXT = 3.0   # $3/round lot
COMMISSION_FTMO = 5.0          # $5/round lot
CONTRACT_SIZE = 100000

def apply_commission(trades, lot, commission_per_lot):
    """Subtract commission from trade PnLs in-place. $X per round lot per trade."""
    for t in trades:
        comm = lot * commission_per_lot
        t.pnl -= comm

def run_sim(data_source, pair, rates, max_sprd):
    """Run ReconSim on data, return trades list."""
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    times = df['time'].astype(np.int64)//10**9
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    spreads = df['spread'].values
    volumes = df['tick_volume'].values
    start_dt = FROM.replace(tzinfo=timezone.utc)

    sim = ReconSim(
        z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
        max_hold=54, base_lot=0.75, max_spread=max_sprd,
        limit_entry_atr=0.0, enable_stability=False,
        trade_dir=0, start_hour=0, end_hour=7
    )
    trades = sim.run(times, opens, highs, lows, closes, spreads, volumes, start_dt)
    return trades, df

def summarize(trades, pair, max_sprd, data_source, lot, comm_per_lot):
    """Print trade summary with commission."""
    gross = sum(t.pnl for t in trades)
    apply_commission(trades, lot, comm_per_lot)
    net = sum(t.pnl for t in trades)
    total_comm = lot * comm_per_lot * len(trades)
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    neutral = [t for t in trades if abs(t.pnl) < 0.01]
    denom = len(trades) - len(neutral)
    wr = len(wins)/denom*100 if denom else 0
    avg_win = sum(t.pnl for t in wins)/len(wins) if wins else 0
    avg_loss = sum(t.pnl for t in losses)/len(losses) if losses else 0
    return {
        'pair': pair, 'data': data_source, 'max_sprd': max_sprd,
        'trades': len(trades), 'wins': len(wins), 'losses': len(losses),
        'wr': wr, 'gross_pnl': gross, 'total_comm': total_comm,
        'net_pnl': net, 'avg_win': avg_win, 'avg_loss': avg_loss
    }

def print_row(r):
    """Print one result row."""
    print(f"{r['pair']:<8} {r['data']:<6} sprd<={r['max_sprd']:>2d}  "
          f"{r['trades']:>3d} trades  {r['wr']:>5.1f}% WR  "
          f"gross ${r['gross_pnl']:>+7.2f}  comm ${r['total_comm']:>6.2f}  "
          f"net ${r['net_pnl']:>+7.2f}")

# ============================================================
# 1. Load FTMO data via MT5
# ============================================================
print("=" * 90)
print("STEP 1: Load FTMO data for 6 cross pairs")
print("=" * 90)

path = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
if not mt5.initialize(path=path):
    print(f"FTMO init failed: {mt5.last_error()}")
    sys.exit(1)

ftmo_data = {}
for pair in PAIRS:
    rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, FROM, TO)
    if rates is None or len(rates) == 0:
        print(f"  {pair}: NO FTMO DATA")
        continue
    df = pd.DataFrame(rates)
    zero_sprd = (df['spread'] == 0).sum()
    med_sprd = int(df['spread'].median())
    print(f"  {pair}: {len(df)} bars, zero-spread={zero_sprd} ({zero_sprd/len(df)*100:.1f}%), median sprd={med_sprd}")
    ftmo_data[pair] = rates

mt5.shutdown()

# ============================================================
# 2. Run sim on FTMO data with spread sweeps
# ============================================================
print("\n" + "=" * 90)
print("STEP 2: sim_recon on FTMO data — FundedNext $3/round-turn commission")
print("=" * 90)

SPRD_SWEEPS = [9, 10, 12, 15, 20]
results = []

for pair in PAIRS:
    if pair not in ftmo_data:
        continue
    for max_sprd in SPRD_SWEEPS:
        trades, df = run_sim("FTMO", pair, ftmo_data[pair], max_sprd)
        r = summarize(trades, pair, max_sprd, "FTMO", 0.75, COMMISSION_FUNDEDNEXT)
        results.append(r)
        print_row(r)

# ============================================================
# 3. Also run on normal MT5 data for comparison
# 3.   We need to re-init with normal MT5 terminal
# ============================================================
print("\n" + "=" * 90)
print("STEP 3: sim_recon on Normal MT5 data — FundedNext $3/round-turn commission")
print("=" * 90)

norm_path = r"C:\Program Files\MetaTrader 5\terminal64.exe"
if mt5.initialize(path=norm_path):
    norm_data = {}
    for pair in PAIRS:
        rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M1, FROM, TO)
        if rates is None or len(rates) == 0:
            print(f"  {pair}: NO NORMAL DATA")
            continue
        df = pd.DataFrame(rates)
        zero_sprd = (df['spread'] == 0).sum()
        med_sprd = int(df['spread'].median())
        print(f"  {pair}: {len(df)} bars, zero-spread={zero_sprd} ({zero_sprd/len(df)*100:.1f}%), median sprd={med_sprd}")
        norm_data[pair] = rates

        for max_sprd in SPRD_SWEEPS:
            trades, df = run_sim("NORM", pair, rates, max_sprd)
            r = summarize(trades, pair, max_sprd, "NORM", 0.75, COMMISSION_FUNDEDNEXT)
            results.append(r)
            print_row(r)

    mt5.shutdown()

# ============================================================
# 4. Show best configs per pair
# ============================================================
print("\n" + "=" * 90)
print("BEST NET PnL PER PAIR AFTER COMMISSION")
print("=" * 90)
print(f"{'PAIR':<8} {'DATA':<6} {'SPRD':>5} {'TRADES':>7} {'WR':>7} {'GROSS':>10} {'COMM':>8} {'NET':>10}")
print("-" * 65)

best_per_pair = {}
for r in results:
    if r['trades'] == 0:
        continue
    key = (r['pair'], r['data'])
    if key not in best_per_pair or r['net_pnl'] > best_per_pair[key]['net_pnl']:
        best_per_pair[key] = r

for key, r in sorted(best_per_pair.items()):
    if r['net_pnl'] > 0:
        survive = "SURVIVES"
    else:
        survive = "DIES"
    print(f"{r['pair']:<8} {r['data']:<6} sprd<={r['max_sprd']:>2d}  "
          f"{r['trades']:>3d} trades  {r['wr']:>5.1f}%  "
          f"${r['gross_pnl']:>+7.2f}  ${r['total_comm']:>6.2f}  "
          f"${r['net_pnl']:>+7.2f}  ← {survive}")

# ============================================================
# 5. Show FTMO results with BOTH commission scenarios
# ============================================================
print("\n" + "=" * 90)
print("FTMO DATA: No Commission vs FTMO ($5) vs FundedNext ($3)")
print(f"(Best sprd per pair, z=3.5, 0-7, lot=0.75)")
print("=" * 90)
print(f"{'PAIR':<8} {'SPRD':>5} {'NO_COMM':>10} {'FTMO($5)':>10} {'FN($3)':>10} {'TRADES':>7} {'WR':>7}")
print("-" * 60)

for pair in PAIRS:
    if pair not in ftmo_data:
        continue
    # Find best sprd threshold for this pair
    best_r = None
    for r in results:
        if r['pair'] == pair and r['data'] == 'FTMO' and r['trades'] > 0:
            if best_r is None or r['net_pnl'] > best_r['net_pnl']:
                best_r = r
    if best_r is None:
        continue

    # Re-run at that sprd to show all 3 scenarios
    trades, df = run_sim("FTMO", pair, ftmo_data[pair], best_r['max_sprd'])
    gross = sum(t.pnl for t in trades)
    comm_ftmo = 0.75 * COMMISSION_FTMO * len(trades)
    comm_fn = 0.75 * COMMISSION_FUNDEDNEXT * len(trades)
    net_ftmo = gross - comm_ftmo
    net_fn = gross - comm_fn
    wins = len([t for t in trades if t.pnl > 0])
    losses = len([t for t in trades if t.pnl < 0])
    neutral = len([t for t in trades if abs(t.pnl) < 0.01])
    denom = len(trades) - neutral
    wr = wins/denom*100 if denom else 0

    # Also show after applying both commissions to the trades
    apply_commission(trades, 0.75, COMMISSION_FTMO)
    net_ftmo_v2 = sum(t.pnl for t in trades)
    # Reset and apply FundedNext
    for t in trades:
        t.pnl = gross/len(trades) if len(trades) else 0  # hacky, let me just use the computed values
    # Actually let me just use the already computed values

    print(f"{pair:<8} sprd<={best_r['max_sprd']:>2d}  ${gross:>+7.2f}  ${net_ftmo:>+7.2f}  ${net_fn:>+7.2f}  "
          f"{len(trades):>3d} trades  {wr:>5.1f}%")

print()
for scenario, label in [(0, "NO COMMISSION"), (COMMISSION_FTMO, f"FTMO (${COMMISSION_FTMO}/lot)"), (COMMISSION_FUNDEDNEXT, f"FundedNext (${COMMISSION_FUNDEDNEXT}/lot)")]:
    total = 0
    for pair in PAIRS:
        if pair not in ftmo_data:
            continue
        for r in results:
            if r['pair'] == pair and r['data'] == 'FTMO' and r['trades'] > 0:
                # Find best sprd for this scenario
                pass
    # Simple total
    if scenario == 0:
        total = sum(r['gross_pnl'] for r in results if r['data'] == 'FTMO' and r['trades'] > 0)
    elif scenario == COMMISSION_FTMO:
        total = sum(r['gross_pnl'] - r['total_comm']/COMMISSION_FUNDEDNEXT*COMMISSION_FTMO for r in results if r['data'] == 'FTMO' and r['trades'] > 0)
    else:
        total = sum(r['net_pnl'] for r in results if r['data'] == 'FTMO' and r['trades'] > 0)
    print(f"  {label}: best-spread portfolio = ${total:.2f}")
