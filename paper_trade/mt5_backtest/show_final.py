"""Show final comparison."""
sim_pnl = [1654.94, 2070.51, 4062.06, 2129.52, 2397.74, 4175.41]
mt5_pnl = [-581, 313, -52, 499, -351, -792]
pairs = ['AUDNZD','EURAUD','EURNZD','GBPAUD','GBPCAD','GBPNZD']
s_trades = [109,103,109,114,126,101]
m_trades = [214,215,119,179,180,115]
s_wr = [77.1,81.6,81.7,78.1,84.1,81.2]
m_wr = [58.4,66.5,66.4,70.9,61.7,60.0]

print(f"{'PAIR':<10} {'SIM_TRD':>8} {'MT5_TRD':>8} {'SIM_WR':>7} {'MT5_WR':>7} {'SIM_PnL':>10} {'MT5_PnL':>10}")
print("-" * 62)
for i in range(6):
    print(f"{pairs[i]:<10} {s_trades[i]:>8d} {m_trades[i]:>8d} {s_wr[i]:>6.1f}% {m_wr[i]:>6.1f}% ${sim_pnl[i]:>+8.2f} ${mt5_pnl[i]:>+8.2f}")
print("-" * 62)
print(f"{'TOTAL':<10} {sum(s_trades):>8d} {sum(m_trades):>8d}                ${sum(sim_pnl):>+8.2f} ${sum(mt5_pnl):>+8.2f}")
print()
print(f"Sim PnL: +${sum(sim_pnl):,.2f}  MT5 PnL: ${sum(mt5_pnl):,.2f}")
print(f"Inflation: {sum(sim_pnl)/abs(sum(mt5_pnl)):.1f}x")
print(f"Trade count: sim {sum(s_trades)} vs MT5 {sum(m_trades)}")
print(f"Avg WR: sim {sum(s_wr)/6:.1f}% vs MT5 {sum(m_wr)/6:.1f}%")
print()
print("ROOT CAUSE: sim uses z.iloc[i] (close[i] - close[i-1]) and")
print("bar['close'] on the SAME bar i. EA computes z from PREVIOUS bar,")
print("enters at FIRST TICK of new bar. This is look-ahead bias.")
print()
print("EA BUG: g_bars_held = 0 on line 294 resets every new bar.")
print("EA never expires positions (only stops). Sim expires at 54 bars.")
