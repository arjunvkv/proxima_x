"""Backtest: normal vs bug version to see if the bug would have saved us."""
import sys, time, random, argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, r"C:\Trading\Agentic_Trading\proxima_x")
from paper_trade.strategies.m1_z_reversal.strategy import PairState, TrailingStopManager, CONFIG

COST_BY_PAIR = {
    "EURUSD": 0.00002,  # 0.2 pips spread cost
    "EURJPY": 0.005,    # 0.5 pips (50 MP in *10000 units)
    "GBPJPY": 0.005,
}
DEFAULT_COST = 0.00002
TICK_DIR = Path(r"C:\Trading\Agentic_Trading\proxima_x\data\exness_ticks")

def load_ticks(pair="EURUSD", months=None):
    if months is None:
        months = [(2025, 10), (2025, 11), (2025, 12)]
    import pandas as pd
    dfs = []
    for y, m in months:
        p = TICK_DIR / f"{pair}_Raw_Spread_{y}_{m:02d}.zip"
        if not p.exists():
            continue
        d = pd.read_csv(p, names=["E","S","Ts","B","A"], skiprows=1, header=None,
                        dtype={"Ts": str, "B": np.float64, "A": np.float64})
        d["Ts"] = pd.to_datetime(d["Ts"].str.replace("Z","",regex=False),
            format="%Y-%m-%d %H:%M:%S.%f", errors="coerce")
        dfs.append(d.dropna(subset=["Ts"]))
    if not dfs:
        raise FileNotFoundError(f"No tick data for {pair}")
    t = pd.concat(dfs, ignore_index=True).sort_values("Ts").reset_index(drop=True)
    return t.set_index("Ts")


class GhostPositionManager:
    """Tracks positions that survive process restarts (lost in-memory stops)."""
    def __init__(self, cost=0.00002):
        self.cost = cost
        self.ghosts = {}  # pair -> {entry, dir, entry_time, ticket}

    def add_ghost(self, pair, direction, entry_price, entry_time, ticket):
        self.ghosts[pair] = {"dir": direction, "entry": entry_price,
                             "entry_time": entry_time, "ticket": ticket}

    def has_ghost(self, pair):
        return pair in self.ghosts

    def close_ghost(self, pair, exit_price, exit_time):
        g = self.ghosts.pop(pair, None)
        if g is None:
            return None
        pnl = (exit_price - g["entry"]) * g["dir"] - self.cost
        return {**g, "exit": exit_price, "exit_time": exit_time, "pnl": pnl,
                "dur_bars": (exit_time - g["entry_time"]) / 60, "exit_reason": "ghost_close"}

    def age_seconds(self, pair, now):
        g = self.ghosts.get(pair)
        return (now - g["entry_time"]) if g else 0

    def total_ghosts(self):
        return len(self.ghosts)


def backtest(pair="EURUSD", bug_prob=0.0, max_drift_hours=24, months=None):
    """Run backtest. bug_prob = fraction of tick-hours that trigger a restart."""
    import pandas as pd
    ticks = load_ticks(pair, months)
    cfg = {**CONFIG, "min_stop_pips": 1.5}
    cost = COST_BY_PAIR.get(pair, DEFAULT_COST)

    # Build M1 bars from mid for seeding
    mid = (ticks['B'] + ticks['A']) / 2.0
    ohlc = mid.resample('1min').ohlc().dropna()
    bars = pd.DataFrame({
        'open': ohlc['open'], 'high': ohlc['high'],
        'low': ohlc['low'], 'close': ohlc['close'],
    })
    bars.index = ohlc.index

    # Seed PairState
    ps = PairState(pair, cfg)
    seed_count = 60
    seed_end_idx = min(seed_count, len(bars) - 10)
    for i in range(seed_end_idx):
        bar = bars.iloc[i]
        ts_i = int(bar.name.timestamp()) if hasattr(bar, 'name') else 0
        ps.seed_bar({
            'open': bar['open'], 'high': bar['high'], 'low': bar['low'],
            'close': bar['close'], 'time': ts_i,
        })
    seed_end_ts = int(bars.index[seed_end_idx - 1].timestamp()) if seed_end_idx > 0 else 0

    # State
    tsm = TrailingStopManager(cfg)
    ghosts = GhostPositionManager(cost=cost)
    times = ticks.index.astype(np.int64) // 10**9
    bids = ticks['B'].values
    asks = ticks['A'].values

    trades = {}  # ticket -> trade dict
    prev_bar_min = -1
    restart_counter = 0
    retart_cooldown = 0  # prevent cascading restarts

    total_ticks = len(times)
    tick_hours = max(1, (times[-1] - times[0]) / 3600)
    bug_events_target = int(tick_hours * bug_prob)

    max_drift_s = max_drift_hours * 3600

    print(f"\n{'='*60}")
    print(f"Backtest: {pair} | months={months} | bug_prob={bug_prob} | drift_max={max_drift_hours}h")
    print(f"  Total ticks: {total_ticks:,}")
    print(f"  Span: {tick_hours:.0f}h")
    print(f"  Expected restarts: {bug_events_target}")
    print(f"{'='*60}")

    for i in range(total_ticks):
        ts = int(times[i])
        if ts <= seed_end_ts:
            continue
        bid = float(bids[i])
        ask = float(asks[i])
        mid = (bid + ask) / 2.0

        current_hour = (ts - seed_end_ts) / 3600.0

        # -- Check for ghost expiry (max drift time) --
        for pair_g in list(ghosts.ghosts.keys()):
            if ghosts.age_seconds(pair_g, ts) >= max_drift_s:
                g = ghosts.close_ghost(pair_g, ask if ghosts.ghosts[pair_g]["dir"] == 1 else bid, ts)
                if g:
                    trades[g["ticket"]] = g

        # -- Trailing stop checks (only for non-ghost positions) --
        closed = tsm.update(bid, ask, ts)
        for cp in closed:
            tr = trades.get(cp['ticket'])
            if tr:
                tr['exit'] = ask if tr['dir'] == -1 else bid  # close price matches direction
                # Actually direction: 1=BUY, exit with bid; -1=SELL, exit with ask
                if tr['dir'] == 1:
                    tr['exit'] = bid
                    tr['pnl'] = (tr['exit'] - tr['entry']) - cost
                else:
                    tr['exit'] = ask
                    tr['pnl'] = (tr['entry'] - tr['exit']) - cost
                tr['exit_time'] = ts
                tr['dur_bars'] = (ts - tr['entry_time']) / 60
                tr['exit_reason'] = 'stop'

        # -- Check for restart events --
        if bug_prob > 0 and retart_cooldown <= 0:
            # Poisson-ish: each tick has a tiny chance of a restart
            # Scale: if bug_prob=1.0 means ~1 restart per hour
            restart_chance = bug_prob / 3600.0  # per tick (1 tick ≈ 0.3s, so ~12000 ticks/hour)
            if random.random() < restart_chance:
                restart_counter += 1
                retart_cooldown = 60  # don't restart again for 60 ticks
                # Move ALL tsm positions to ghost
                for tkt in list(tsm.positions.keys()):
                    p = tsm.positions.pop(tkt)
                    ghosts.add_ghost(p["pair"], p["direction"], p["entry"],
                                     p["entry_time"], tkt)
                    tr = trades.get(tkt)
                    if tr:
                        tr['exit_reason'] = 'bug_lost_stop'
                # Any existing ghosts stay as ghosts
        retart_cooldown -= 1

        # -- Generate signals --
        sig = ps.update(mid, ts)
        if sig:
            bar_min = ts // 60
            if bar_min == prev_bar_min:
                continue
            prev_bar_min = bar_min

            direction = sig['direction']  # 1=BUY(fade low), -1=SELL(fade high)
            price_at_signal = ask if direction == 1 else bid

            # Check ghost: same direction = skip, opposite = close + new
            if ghosts.has_ghost(pair):
                g = ghosts.ghosts[pair]
                if g["dir"] == direction:
                    continue  # same direction, skip (don't add to position)
                else:
                    # Opposite direction: close ghost at market
                    exit_p = ask if g["dir"] == 1 else bid
                    closed_g = ghosts.close_ghost(pair, exit_p, ts)
                    if closed_g:
                        trades[closed_g["ticket"]] = closed_g
                    # Now open new position below

            # Open position in tsm
            entry = price_at_signal
            ticket = tsm.add(pair, direction, entry, sig['atr'], timestamp=ts,
                             spread=abs(ask - bid))
            trades[ticket] = {
                'bar_time': sig['bar_time'], 'dir': direction,
                'entry': entry, 'entry_time': ts,
                'z': sig['z_score'], 'atr': sig['atr'],
                'exit': None, 'exit_time': None, 'pnl': None,
                'dur_bars': None, 'exit_reason': None,
            }

    # Close all remaining at end
    final_ts = int(times[-1])
    final_bid = float(bids[-1])
    final_ask = float(asks[-1])

    # Ghost closes
    for pair_g in list(ghosts.ghosts.keys()):
        g = ghosts.close_ghost(pair_g, final_ask if ghosts.ghosts[pair_g]["dir"] == 1 else final_bid, final_ts)
        if g:
            trades[g["ticket"]] = g

    # TSM expiry
    expired = tsm.check_expiry(final_ts)
    for cp in expired:
        tr = trades.get(cp['ticket'])
        if tr and tr['exit'] is None:
            tr['exit'] = final_bid if tr['dir'] == 1 else final_ask
            tr['pnl'] = (tr['exit'] - tr['entry']) - cost if tr['dir'] == 1 else (tr['entry'] - tr['exit']) - cost
            tr['exit_time'] = final_ts
            tr['dur_bars'] = (tr['exit_time'] - tr['entry_time']) / 60
            tr['exit_reason'] = 'expiry'

    for tr in trades.values():
        if tr['exit'] is None:
            tr['exit'] = final_bid if tr['dir'] == 1 else final_ask
            tr['pnl'] = (tr['exit'] - tr['entry']) - cost if tr['dir'] == 1 else (tr['entry'] - tr['exit']) - cost
            tr['exit_time'] = final_ts
            tr['dur_bars'] = (tr['exit_time'] - tr['entry_time']) / 60
            tr['exit_reason'] = 'end_of_data'

    return list(trades.values()), restart_counter


def summary(trades, label="", restarts=0):
    if not trades:
        print(f"  No trades")
        return {}
    pnls = np.array([t['pnl'] for t in trades if t['pnl'] is not None])
    if len(pnls) == 0:
        print(f"  No completed trades")
        return {}

    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    total = len(pnls)
    wr = len(wins) / total * 100 if total > 0 else 0
    gross = float(pnls.sum())
    avg_w = float(wins.mean()) if len(wins) > 0 else 0
    avg_l = float(losses.mean()) if len(losses) > 0 else 0
    best = float(pnls.max())
    worst = float(pnls.min())

    # By exit reason
    reasons = {}
    for t in trades:
        r = t.get("exit_reason", "?")
        if r not in reasons:
            reasons[r] = {"pnls": [], "n": 0}
        if t['pnl'] is not None:
            reasons[r]["pnls"].append(t['pnl'])
            reasons[r]["n"] += 1

    # PnL in pips
    pip_size = 0.0001
    gross_pips = gross / pip_size
    avg_pips = (gross / total) / pip_size if total > 0 else 0

    print(f"\n--- {label} ---")
    print(f"  Trades: {total} | WR: {wr:.1f}% | Gross: {gross_pips:+.1f}p (${gross:+.2f})")
    print(f"  Avg: {avg_pips:+.1f}p | Avg Win: {avg_w/pip_size:+.1f}p | Avg Loss: {avg_l/pip_size:+.1f}p")
    print(f"  Best: {best/pip_size:+.1f}p | Worst: {worst/pip_size:+.1f}p")
    print(f"  Restarts: {restarts}")

    print(f"  Exit reasons:")
    for r, rd in sorted(reasons.items()):
        rpnls = np.array(rd["pnls"])
        rw = len(rpnls[rpnls > 0])
        rl = len(rpnls[rpnls < 0])
        print(f"    {r}: {rd['n']}t {rw}W/{rl}L {rpnls.sum()/pip_size:+.1f}p")

    return {
        "n": total, "wr": wr, "gross_pips": gross_pips, "gross": gross,
        "avg_win_pips": avg_w / pip_size, "avg_loss_pips": avg_l / pip_size,
        "worst_pips": worst / pip_size, "restarts": restarts,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pair", nargs="?", default="EURUSD")
    parser.add_argument("--months", default="10,11,12", help="months to test (comma-sep, e.g. 10,11)")
    args = parser.parse_args()

    months_list = []
    for m in args.months.split(","):
        months_list.append((2025, int(m.strip())))

    print(f"Loading {args.pair} for months {months_list}...")

    for bug_prob in [0, 0.25, 0.5, 1.0, 2.0, 5.0]:
        trades, restarts = backtest(pair=args.pair, bug_prob=bug_prob,
                                     max_drift_hours=24, months=months_list)
        summary(trades, label=f"bug_prob={bug_prob:.1f}/hr", restarts=restarts)
