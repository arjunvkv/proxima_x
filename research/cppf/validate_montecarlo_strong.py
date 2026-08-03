"""Monte Carlo on just the strong pairs: EURAUD + GBPAUD."""
import numpy as np

# OOS trades from refined walk-forward — STRONG pairs only
eur = np.array([+56.88, +5.65, +19.76, +15.17, +12.27, -42.15, -61.20,
                +83.81, -54.00, +93.90, +40.68, +36.89, -60.67])
gbp = np.array([+64.32, +10.82, +9.15, +111.49, -64.50, -50.63,
                +83.81, -0.27, -7.85, +72.51, +108.46])
trades = np.concatenate([eur, gbp])
obs = trades.sum()

print(f"EURAUD + GBPAUD: {len(trades)} trades, ${obs:.2f}")
print(f"Wins: {(trades>0).sum()}/{len(trades)} ({(trades>0).mean()*100:.1f}%)")

# Sign-randomization
np.random.seed(42); N=10000
null = np.array([(trades * np.random.choice([-1,1],len(trades))).sum() for _ in range(N)])
p = (null >= obs).mean()
print(f"Sign-randomization p-value: {p:.4f} ({'PASS' if p<0.05 else 'FAIL'})")
print(f"Null 95th percentile: ${np.percentile(null,95):.2f}")

# Bootstrap
boot = np.array([np.random.choice(trades,len(trades),replace=True).sum() for _ in range(N)])
ci = (np.percentile(boot,2.5), np.percentile(boot,97.5))
print(f"Bootstrap 95% CI: [${ci[0]:.2f}, ${ci[1]:.2f}] ({'ALL POSITIVE' if ci[0]>0 else 'INCLUDES ZERO'})")

print(f"\n=== FINAL VERDICT (Strong Pair Portfolio) ===")
print(f"Walk-forward OOS: +${obs:.2f} on {len(trades)} trades")
print(f"Sign-randomization: p={p:.4f} {'PASS' if p<0.05 else 'FAIL'}")
