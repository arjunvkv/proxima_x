"""Model: 1.0 lot per trade, max 3 lots total open positions at any time."""
import numpy as np

COMM = 3.0  # $/round lot FundedNext Stellar 2-Step
LOT = 1.0   # per trade
MAX_LOTS = 3

# z>=2.5 data per pair from combined backtest
z25 = {
    "gbpnzd":  {"net_mp": 2.27, "tpd": 31, "wr": 0.798},
    "eurnzd":  {"net_mp": 1.79, "tpd": 30, "wr": 0.793},
    "gbpaud":  {"net_mp": 1.66, "tpd": 30, "wr": 0.815},
    "euraud":  {"net_mp": 1.45, "tpd": 30, "wr": 0.803},
    "gbpcad":  {"net_mp": 1.16, "tpd": 31, "wr": 0.796},
    "audnzd":  {"net_mp": 0.93, "tpd": 31, "wr": 0.806},
    "gbpjpy":  {"net_mp": 132.98, "tpd": 31, "wr": 0.795, "jpy": True},
    "chfjpy":  {"net_mp": 132.69, "tpd": 30, "wr": 0.762, "jpy": True},
    "usdjpy":  {"net_mp": 127.13, "tpd": 33, "wr": 0.773, "jpy": True},
    "eurjpy":  {"net_mp": 96.98, "tpd": 30, "wr": 0.792, "jpy": True},
    "gbpusd":  {"net_mp": 0.95, "tpd": 33, "wr": 0.754},
    "eurusd":  {"net_mp": 0.71, "tpd": 32, "wr": 0.763},
}

# z>=2.0 data
z20 = {
    "gbpnzd":  {"net_mp": 1.79, "tpd": 62, "wr": 0.790},
    "eurnzd":  {"net_mp": 1.43, "tpd": 62, "wr": 0.788},
    "gbpaud":  {"net_mp": 1.27, "tpd": 62, "wr": 0.792},
    "euraud":  {"net_mp": 1.09, "tpd": 62, "wr": 0.787},
    "gbpcad":  {"net_mp": 0.99, "tpd": 63, "wr": 0.795},
    "audnzd":  {"net_mp": 0.73, "tpd": 62, "wr": 0.804},
    "gbpjpy":  {"net_mp": 100.72, "tpd": 62, "wr": 0.786, "jpy": True},
    "chfjpy":  {"net_mp": 103.49, "tpd": 61, "wr": 0.768, "jpy": True},
    "usdjpy":  {"net_mp": 104.94, "tpd": 63, "wr": 0.759, "jpy": True},
    "gbpusd":  {"net_mp": 0.81, "tpd": 64, "wr": 0.752},
    "eurusd":  {"net_mp": 0.59, "tpd": 63, "wr": 0.747},
}

def mp2d(mp, jpy, lot):
    if jpy:
        return mp / 100 * 0.067 * (lot / 0.01)
    else:
        return mp * 0.10 * (lot / 0.01)

def analyze(label, pairs, zdata):
    tpd = sum(zdata[p]["tpd"] for p in pairs if p in zdata)
    wrs = [zdata[p]["wr"] for p in pairs if p in zdata]
    tpd_list = [zdata[p]["tpd"] for p in pairs if p in zdata]
    wr = np.average(wrs, weights=tpd_list)
    
    gross = 0
    comm = 0
    loss_ests = []
    
    for p in pairs:
        if p not in zdata:
            continue
        d = zdata[p]
        is_jpy = d.get("jpy", False)
        net_mp = d["net_mp"]
        
        net_dollar = mp2d(net_mp, is_jpy, LOT)
        comm_trade = COMM * LOT
        
        gross += net_dollar * d["tpd"]
        comm += comm_trade * d["tpd"]
        
        # Estimate loss from WR and net
        # net = wr*win + (1-wr)*loss, win = -4*loss (4:1 payoff)
        # net = wr*(-4*loss) + (1-wr)*loss = loss*(1 - 5*wr)
        denom = 1 - 5*wr
        if abs(denom) > 0.001:
            loss_est = net_dollar / denom
        else:
            loss_est = -net_dollar
        loss_ests.append(abs(loss_est))
    
    net = gross - comm
    avg_loss = np.mean(loss_ests)
    
    print(f"{label}:")
    print(f"  Pairs={len(pairs)}, TPD={tpd}, WR={wr:.0%}")
    print(f"  Per trade: gross=${mp2d(zdata[pairs[0]]['net_mp'], zdata[pairs[0]].get('jpy',False), LOT):.2f}, comm=${COMM*LOT:.2f}")
    print(f"  Gross: ${gross:>8,.0f}/d   Comm: ${comm:>8,.0f}/d   Net: ${net:>8,.0f}/d")
    print(f"  Avg loss/trade: ${avg_loss:.2f}")
    print(f"  10-consecutive loss: ${avg_loss*10:.0f}")
    print(f"  Worst day (20% losers): ${avg_loss*tpd*0.2:.0f}")
    print(f"  Monthly net: ${net*22:>10,.0f}")
    print(f"  Daily loss limit ($1,250): {net/1250*100:.0f}% of limit")
    print(f"  Commission as % of gross: {comm/gross*100:.0f}%")
    print(f"  Max 3 lots constraint: {tpd} trades/day at ~3min avg = {tpd*3/1440*LOT:.1f} lot-hrs/day needed, {MAX_LOTS*24:.0f} available -> {'' if tpd*3/1440*LOT <= MAX_LOTS*24 else 'OVER'}")
    print()

print(f"SCENARIO: {LOT} lot per trade, max {MAX_LOTS} lots total open positions")
print("=" * 60)

analyze("z>=2.5 top 6 non-JPY", ["gbpnzd","eurnzd","gbpaud","euraud","gbpcad","audnzd"], z25)
analyze("z>=2.5 top 8 mixed", ["gbpnzd","eurnzd","gbpaud","euraud","gbpjpy","chfjpy","usdjpy","audnzd"], z25)
analyze("z>=2.5 top 10", ["gbpnzd","eurnzd","gbpaud","euraud","gbpcad","gbpjpy","chfjpy","usdjpy","audnzd","gbpusd"], z25)
analyze("z>=2.0 top 6 non-JPY", ["gbpnzd","eurnzd","gbpaud","euraud","gbpcad","audnzd"], z20)
analyze("z>=2.0 top 8 mixed", ["gbpnzd","eurnzd","gbpaud","euraud","gbpjpy","chfjpy","usdjpy"], z20)
