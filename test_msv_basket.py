#!/usr/bin/env python3
"""MSV Asian FX Currency Network Dispersion — True Basket Level Backtest."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pandas as pd
import numpy as np
from proxima_honest_backtest.strategies.sunday_h22.sweep import load_and_align, BROKERS
from proxima_honest_backtest.execution.execution_simulator import ExecutionSimulator

ALL_PAIRS = [
    "AUDJPY", "AUDNZD", "AUDUSD", "EURAUD", "EURCHF", "EURGBP",
    "EURJPY", "EURNZD", "EURUSD", "GBPAUD", "GBPCAD", "GBPJPY",
    "GBPNZD", "GBPUSD", "NZDUSD", "USDCAD", "USDCHF", "USDJPY",
]

def main():
    t0 = time.time()
    print("Loading 18-pair M5 dataset for MSV True Basket Evaluation...")
    raw, pre_align = load_and_align()
    print(f"  Loaded {len(raw)} pairs, {len(pre_align):,} bars ({time.time()-t0:.1f}s)")

    pieces = [df.set_index("time")[["close","open"]] for df in raw.values()]
    for i, p in enumerate(raw.keys()):
        pieces[i].columns = [p, f"{p}_open"]
    df_all = pd.concat(pieces, axis=1, sort=True).ffill().bfill()
    times = pd.to_datetime(df_all.index)

    close_mat = df_all[[p for p in ALL_PAIRS]].values
    open_mat = df_all[[f"{p}_open" for p in ALL_PAIRS]].values
    hours = times.hour.values

    # Pre-compute 60-min basket returns (12 bars)
    ret_60m = np.zeros_like(close_mat)
    ret_60m[12:] = np.log(close_mat[12:] / close_mat[:-12])

    # Compute rolling 500-bar network dispersion & percentile
    disp_series = np.std(ret_60m, axis=1, ddof=0)
    s_disp = pd.Series(disp_series)
    disp_pct = s_disp.rolling(500).rank(pct=True).values

    print("\n" + "="*85)
    print("MSV ASIAN EXHAUSTION — TRUE BASKET-LEVEL EVALUATION")
    print("="*85)

    brokers = ["exness", "ftmo", "fundednext", "fusionmarkets", "dukascopy"]

    for b in brokers:
        sim = ExecutionSimulator(b)
        n_bars = len(df_all)
        in_basket = False
        exit_bar = 0
        entry_prices = np.zeros(len(ALL_PAIRS))
        basket_pairs = []
        
        basket_pnls = []

        for t in range(505, n_bars):
            # Check exit
            if in_basket and t >= exit_bar:
                gross_basket_pnl = 0.0
                total_comm = 0.0
                for p_i in basket_pairs:
                    c_exit = close_mat[t, p_i]
                    c_entry = entry_prices[p_i]
                    # 0.1 lot per pair in basket
                    pnl = (c_exit - c_entry) / c_entry * 10000.0
                    comm = sim.profile.commission_per_lot * 0.1
                    gross_basket_pnl += pnl
                    total_comm += comm

                net_basket_pnl = gross_basket_pnl - total_comm
                basket_pnls.append(net_basket_pnl)
                in_basket = False

            # Entry check (Asian hours 00:00 to 07:00 UTC)
            if not in_basket and 0 <= hours[t] <= 6:
                r_t = ret_60m[t]
                mean_r = np.mean(r_t)

                # Condition 1: Basket in decline <= -0.02%
                if mean_r <= -0.0002 and disp_pct[t] >= 0.95:
                    basket_pairs = [p_i for p_i in range(len(ALL_PAIRS)) if r_t[p_i] < mean_r]
                    if len(basket_pairs) >= 3:
                        in_basket = True
                        exit_bar = t + 3  # 15-min hold
                        for p_i in basket_pairs:
                            entry_prices[p_i] = open_mat[t, p_i]

        wins = [p for p in basket_pnls if p > 0]
        losses = [p for p in basket_pnls if p < 0]
        gw = sum(wins)
        gl = abs(sum(losses))
        net = sum(basket_pnls)
        wr = len(wins) / len(basket_pnls) * 100 if basket_pnls else 0.0
        pf = gw / gl if gl > 0 else 0.0
        avg_basket = net / len(basket_pnls) if basket_pnls else 0.0

        print(f"[{b.upper():<13}] Basket Events: {len(basket_pnls):<4} | Win Rate: {wr:5.1f}% | Net PnL: +${net:<8.2f} | PF: {pf:5.2f} | Avg Basket: +${avg_basket:.2f}")

if __name__ == "__main__":
    main()
