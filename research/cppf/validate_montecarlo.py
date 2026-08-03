"""Monte Carlo shuffle test: 10K sign-randomizations on OOS walk-forward trades."""
import numpy as np

# OOS trades from refined walk-forward (33 total)
# EURAUD z>=6 sprd<=50 LONG: 13 trades
eur_trades = [+56.88, +5.65, +19.76, +15.17, +12.27, -42.15, -61.20,
              +83.81, -54.00, +93.90, +40.68, +36.89, -60.67]
# GBPAUD z>=6 sprd<=50 LONG: 11 trades
gbp_trades = [+64.32, +10.82, +9.15, +111.49, -64.50, -50.63,
              +83.81, -0.27, -7.85, +72.51, +108.46]
# GBPCAD z>=6 sprd<=20 LONG: 7 trades  
gpc_trades = [+59.66, -59.55, +36.03, +15.74, -2.50, -22.38, -35.00]
# EURNZD z>=6 sprd<=20 SHORT: 2 trades
eur_nzd_trades = [+31.25, +39.73]

all_trades = np.array(eur_trades + gbp_trades + gpc_trades + eur_nzd_trades)
observed_pnl = all_trades.sum()

print("=" * 80)
print("MONTE CARLO SHUFFLE TEST (10K iterations)")
print("=" * 80)
print(f"\nTotal OOS trades: {len(all_trades)}")
print(f"Observed portfolio PnL: ${observed_pnl:.2f}")
print(f"Wins: {(all_trades > 0).sum()}/{len(all_trades)} ({(all_trades > 0).mean()*100:.1f}%)")
print(f"Losses: {(all_trades < 0).sum()}/{len(all_trades)}")
print(f"Avg win: ${all_trades[all_trades > 0].mean():.2f}" if (all_trades > 0).any() else "Avg win: N/A")
print(f"Avg loss: ${all_trades[all_trades < 0].mean():.2f}" if (all_trades < 0).any() else "Avg loss: N/A")

# Test 1: Sign-randomization test
# Null hypothesis: directional edge is zero (signs are random)
# Method: randomly flip each trade's sign, repeat 10,000 times
np.random.seed(42)
N = 10000
count_survive = 0
count_better = 0
null_pnls = []

for i in range(N):
    signs = np.random.choice([-1, 1], size=len(all_trades))
    shuffled = all_trades * signs
    null_pnl = shuffled.sum()
    null_pnls.append(null_pnl)
    if null_pnl >= observed_pnl:
        count_better += 1
    if null_pnl > 0:
        count_survive += 1

null_pnls = np.array(null_pnls)
p_value = count_better / N
survival_rate = count_survive / N

print(f"\n--- TEST 1: Sign-Randomization (directional edge) ---")
print(f"p-value (PnL >= ${observed_pnl:.2f} under null): {p_value:.4f}")
print(f"  => ", end="")
if p_value < 0.001:
    print("p < 0.001 — HIGHLY SIGNIFICANT edge (<<0.1% chance of random)")
elif p_value < 0.01:
    print("p < 0.01 — VERY SIGNIFICANT edge")
elif p_value < 0.05:
    print("p < 0.05 — SIGNIFICANT edge")
elif p_value < 0.10:
    print("p < 0.10 — MARGINALLY significant edge")
else:
    print("NOT significant — edge may be random")
print(f"Null distribution: mean=${null_pnls.mean():.2f} std=${null_pnls.std():.2f}")
print(f"95th percentile of null: ${np.percentile(null_pnls, 95):.2f}")
print(f"Observed PnL: ${observed_pnl:.2f}")
print(f"Probability of positive PnL under null: {survival_rate*100:.1f}%")
print(f"Probability of PnL >= observed under null: {p_value*100:.2f}%")

# Test 2: Bootstrap CI
# Sample with replacement, 10,000 times
boot_pnls = []
for i in range(N):
    sample = np.random.choice(all_trades, size=len(all_trades), replace=True)
    boot_pnls.append(sample.sum())
boot_pnls = np.array(boot_pnls)

ci_low = np.percentile(boot_pnls, 2.5)
ci_high = np.percentile(boot_pnls, 97.5)

print(f"\n--- TEST 2: Bootstrap 95% CI ---")
print(f"95% CI: [${ci_low:.2f}, ${ci_high:.2f}]")
print(f"Observed: ${observed_pnl:.2f} ", end="")
if ci_low > 0:
    print("— ENTIRE CI IS POSITIVE (robust edge)")
else:
    print("— CI includes negative values (edge may be fragile)")
print(f"% of bootstrap samples with positive PnL: {(boot_pnls > 0).mean()*100:.1f}%")

# Test 3: Binomial test — can a 50/50 coin flip produce this many wins?
from math import comb
wins = (all_trades > 0).sum()
prob_50 = 0.5
# P(X >= wins) under Binomial(n, 0.5)
p_binom = sum(comb(len(all_trades), k) * (prob_50**k) * ((1-prob_50)**(len(all_trades)-k))
              for k in range(wins, len(all_trades)+1))
print(f"\n--- TEST 3: Binomial Test (50/50 null) ---")
print(f"{wins} wins out of {len(all_trades)} trades")
print(f"P(>= {wins} wins | 50/50): {p_binom:.6f}")
print(f"  => ", end="")
if p_binom < 0.05:
    print("Win rate is significantly > 50%")
else:
    print("Win rate is NOT significantly > 50%")

# Final verdict
print(f"\n{'='*80}")
print("FINAL VERDICT")
print("=" * 80)
passed = 0
total = 3
if p_value < 0.05:
    print(f"[PASS] Test 1: Sign-randomization p={p_value:.4f} < 0.05")
    passed += 1
else:
    print(f"[FAIL] Test 1: Sign-randomization p={p_value:.4f} >= 0.05")

if ci_low > 0:
    print(f"[PASS] Test 2: Bootstrap CI entirely positive [${ci_low:.2f}, ${ci_high:.2f}]")
    passed += 1
else:
    print(f"[FAIL] Test 2: Bootstrap CI includes zero [${ci_low:.2f}, ${ci_high:.2f}]")

if p_binom < 0.05:
    print(f"[PASS] Test 3: Binomial p={p_binom:.6f} < 0.05")
    passed += 1
else:
    print(f"[FAIL] Test 3: Binomial p={p_binom:.6f} >= 0.05")

print(f"\nOverall: {passed}/{total} tests passed")
if passed >= 2:
    print("OVERALL: PASS — edge is statistically significant")
else:
    print("OVERALL: FAIL — edge is not statistically significant")
print("=" * 80)
