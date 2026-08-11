"""nova_tree_probes.py — validate the DRIVERS_RESEARCH tree-factor ideas on
NOVA's factor engine (daily scale, triage-grade first look, NOT battery-grade).

T1: #17 triangulation residuals (EURJPY vs EURUSD x USDJPY)
    #21 DXY-implied divergence (DXY vs its EURUSD component)
    #6  commodity terms-of-trade momentum (XAU+USOIL -> AUD/NZD)
T2: #7  equity-divergence flows (UK100 vs US500 -> GBPUSD)
    #11 WMR London-fix window reversal (GBPUSD pre/post-fix)

Signals fill next day open, 1-day hold, costs via pnl.trade_to_usd with
corrected maps (identical math to the book).
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts", "_absorb"))

from costmaps_r3 import corrected_maps
from proxima_ops.backtest.feed import load_bars_cached
from proxima_ops.backtest.pnl import trade_to_usd
from proxima_ops.nova import feed as nfeed
from proxima_ops.nova import factors as F

VOL = 0.15
COMM = 3.0


def daily(sym):
    """(day, first_open, last_close) per server day from M5 bars."""
    b = nfeed.bars_list_to_arrays(load_bars_cached(sym))
    n = len(b["close"])
    day = F.bar_day(b["ts"])
    first = np.zeros(n, bool)
    first[np.searchsorted(day, day, side="left")] = True  # first idx per day
    last = np.zeros(n, bool)
    last[np.searchsorted(day, day, side="right") - 1] = True  # last idx per day
    d = np.unique(day)
    o = b["open"][np.searchsorted(day, d, side="left")]
    c = b["close"][np.searchsorted(day, d, side="right") - 1]
    return d, o, c


def probe(name, sym, sigs, d_sym, o_sym, tick, spr, vol=VOL):
    """sig[i] in {-1,0,+1} from day-i close; fill day i+1 open, exit i+2 open.
    d_sym/o_sym = daily day-stamps/opens of the TRADED symbol (aligned to sigs)."""
    sigs = np.asarray(sigs)
    n = len(sigs)
    assert len(d_sym) >= n
    i = np.where(sigs[:-2] != 0)[0]  # signal days (need i+2 to exist)
    if len(i) == 0:
        print(f"{name}: NO TRADES"); return
    dirn = sigs[i]
    entry = o_sym[-(n):][i + 1]
    exit_p = o_sym[-(n):][i + 2]
    pts = (exit_p - entry) * dirn / tick
    trades = [dict(symbol=sym, side="BUY" if dirn[k] > 0 else "SELL",
                   entry=float(entry[k]), entry_ts=int(d_sym[-(n):][i[k] + 1]),
                   exit_ts=int(d_sym[-(n):][i[k] + 2]), reason="HOLD", pnl_pts=float(pts[k]))
              for k in range(len(i))]
    net = [trade_to_usd(t, vol, {sym: tick}, COMM, {sym: spr})["net"] for t in trades]
    wins = [x for x in net if x > 0]
    losses = [x for x in net if x <= 0]
    pf = sum(wins) / abs(sum(losses)) if losses else float("inf")
    print(f"{name}: n={len(net):4d}  win={len(wins)/len(net)*100:4.1f}%  "
          f"net=${sum(net):9.0f}  exp/lot=${sum(net)/len(net)/vol:6.1f}  "
          f"PF={pf:5.2f}  gross/day=${sum(float(t['pnl_pts']) for t in trades)*tick*vol/len(net):6.1f}")


def main():
    TICK, SPR = corrected_maps()
    # ---- 1. triangulation residual: log(EURJPY) - log(EURUSD) - log(USDJPY) ----
    _, _, ec = daily("EURUSD")
    _, _, jc = daily("USDJPY")
    dj, oj, jpc = daily("EURJPY")
    n = min(len(ec), len(jc), len(jpc))
    resid = np.log(jpc[-n:]) - np.log(ec[-n:]) - np.log(jc[-n:])
    z = F.rolling_z(resid, 20)
    sig = np.where(z < -1.5, 1.0, np.where(z > 1.5, -1.0, 0.0))
    probe("T1 triangulation EURJPY fade", "EURJPY", sig, dj[-n:], oj[-n:],
          TICK["EURJPY"], SPR.get("EURJPY", 0.6))

    # ---- 2. DXY-implied divergence: log(DXY) vs 0.576*log(EURUSD) ----
    # DXY = k * EURUSD^0.576 * GBP^0.119 * JPY^0.119 * CAD^0.091 * CHF^0.036
    # resid = log(DXY) - log(EURUSD^0.576) reveals non-EUR components.
    dd, od, dc = daily("DXY.cash")
    de, oe, ec2 = daily("EURUSD")
    _, _, gc = daily("GBPUSD")
    _, _, cadc = daily("USDCAD")
    _, _, chfc = daily("USDCHF")
    n2 = min(len(dc), len(ec2), len(gc), len(cadc), len(chfc))
    implied = np.log(dc[-n2:]) - 0.576 * np.log(ec2[-n2:])
    z2 = F.rolling_z(implied, 20)
    sig2 = np.where(z2 < -1.5, 1.0, np.where(z2 > 1.5, -1.0, 0.0))  # fade on EURUSD
    probe("T2 DXY-implied div EURUSD fade", "EURUSD", sig2, de[-n2:], oe[-n2:],
          TICK["EURUSD"], SPR.get("EURUSD", 0.6))

    # ---- 3. commodity momentum -> AUD/NZD (buy both, same signal) ----
    _, _, auc = daily("XAUUSD")
    _, _, oic = daily("USOIL.cash")
    ad, oa, audc = daily("AUDUSD")
    nd, onz, nzdc = daily("NZDUSD")
    n3 = min(len(auc), len(oic), len(audc), len(nzdc))
    rau = np.log(auc[-n3:] / np.roll(auc, 5)[-n3:])
    roi = np.log(oic[-n3:] / np.roll(oic, 5)[-n3:])
    basket = (F.rolling_z(rau, 5) + F.rolling_z(roi, 5)) / 2.0
    sig3 = np.where(basket > 0.5, 1.0, np.where(basket < -0.5, -1.0, 0.0))
    probe("T1 commodity mom AUDUSD", "AUDUSD", sig3, ad[-n3:], oa[-n3:],
          TICK["AUDUSD"], SPR.get("AUDUSD", 0.6))
    probe("T1 commodity mom NZDUSD", "NZDUSD", sig3, nd[-n3:], onz[-n3:],
          TICK["NZDUSD"], SPR.get("NZDUSD", 0.6))

    # ---- 4. equity divergence: UK100 vs US500 -> GBPUSD ----
    _, _, ukc = daily("UK100.cash")
    _, _, usc = daily("US500.cash")
    gd, og, gbpc = daily("GBPUSD")
    n4 = min(len(ukc), len(usc), len(gbpc))
    ruk = np.log(ukc[-n4:] / np.roll(ukc, 5)[-n4:])
    rus = np.log(usc[-n4:] / np.roll(usc, 5)[-n4:])
    div = F.rolling_z(ruk - rus, 20)
    sig4 = np.where(div > 1.0, 1.0, np.where(div < -1.0, -1.0, 0.0))
    probe("T2 equity div GBPUSD", "GBPUSD", sig4, gd[-n4:], og[-n4:],
          TICK["GBPUSD"], SPR.get("GBPUSD", 0.6))

    # ---- 5. London-fix reversal: fade the pre-fix move on GBPUSD (M5) ----
    b = nfeed.bars_list_to_arrays(load_bars_cached("GBPUSD"))
    h = F.bar_hour(b["ts"])
    d5 = F.bar_day(b["ts"])
    pre = (h >= 16) & (h < 18)   # server 16:00-18:00 (London fix 16:00 BST = 15:00 UTC = server 18:00)
    post = (h >= 18) & (h < 20)
    prets = np.unique(d5[pre]); posts = np.unique(d5[post])
    days = np.intersect1d(prets, posts)
    rows = []
    for dd5 in days:
        pr = b["close"][np.where((d5 == dd5) & pre)[0][-1]] / b["close"][np.where((d5 == dd5) & pre)[0][0]] - 1.0
        po = b["close"][np.where((d5 == dd5) & post)[0][-1]] / b["close"][np.where((d5 == dd5) & post)[0][0]] - 1.0
        rows.append((pr, po))
    pr = np.array([r[0] for r in rows]); po = np.array([r[1] for r in rows])
    up, dn = po[pr > 0], po[pr <= 0]
    print(f"T2 London fix GBPUSD: n={len(rows)} days  pre>0 n={len(up)} post mean {up.mean()*1e4:+.2f}bp  "
          f"pre<=0 n={len(dn)} post mean {dn.mean()*1e4:+.2f}bp")
    # fade strategy: enter at pre-close (18:00), exit post-close (20:00); short if pre>0
    sigs5 = np.where(pr > 0, -1.0, np.where(pr < 0, 1.0, 0.0))
    if (sigs5 != 0).any():
        pnl5 = sigs5 * po  # enter at pre-close, exit at post-close
        win = (pnl5 > 0).mean() * 100
        print(f"    fade (enter pre-close): win={win:.1f}%  mean={pnl5.mean()*1e4:+.2f}bp/trade  "
              f"sum={pnl5.sum()*1e4:+.0f}bp (no costs; spread+comm ~1bp)")
        print(f"    unconditional short 18-20: mean={(-po).mean()*1e4:+.2f}bp/trade")


if __name__ == "__main__":
    main()
