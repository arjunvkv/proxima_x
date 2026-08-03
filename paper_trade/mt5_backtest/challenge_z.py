"""Challenge-Z: Fixed-hold z-score mean reversion for FundedNext 5-day challenge.

Design:
- Entry: |z| >= Z_THRESH at bar open (z uses previous bar's close → no look-ahead)
- Exit: After HOLD_BARS bars, at bar close (fixed hold, no trailing stop)
- Cost: Entry/exit spread + $3/lot commission
- Direction: Configurable LONG/SHORT/BOTH per pair

No trailing stop. No parameter sweeps. 3 parameters total.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone
import os


@dataclass
class Trade:
    entry_bar: int
    entry_price: float
    direction: int  # +1 LONG, -1 SHORT
    atr_entry: float
    z_entry: float
    sprd_entry: float
    exit_bar: int = 0
    exit_price: float = 0
    gross_pnl: float = 0
    net_pnl: float = 0
    spread_cost: float = 0
    commission: float = 0
    exit_reason: str = ""


@dataclass
class ChallengeZResult:
    pair: str
    config_name: str
    trades: list
    total_gross: float = 0
    total_net: float = 0
    total_commission: float = 0
    total_spread_cost: float = 0
    n_trades: int = 0
    n_wins: int = 0
    n_losses: int = 0
    win_rate: float = 0
    avg_win: float = 0
    avg_loss: float = 0
    max_consec_losses: int = 0
    max_drawdown: float = 0
    profit_factor: float = 0


class ChallengeZ:
    """Z-score mean reversion with fixed hold. No trailing stop."""

    def __init__(self,
                 z_threshold=3.5,
                 hold_bars=10,
                 max_spread=15,
                 lot_size=0.5,
                 trade_dir=0,  # 0=BOTH, 1=LONG, 2=SHORT
                 commission_per_lot=3.0,
                 contract_size=100000,
                 point_size=0.00001,
                 pip_value_usd=10.0,
                 z_window=50,
                 atr_period=20,
                 start_hour=0,
                 end_hour=24):

        self.z_threshold = z_threshold
        self.hold_bars = hold_bars
        self.max_spread = max_spread
        self.lot_size = lot_size
        self.trade_dir = trade_dir  # 0=BOTH, 1=LONG, 2=SHORT
        self.commission_per_lot = commission_per_lot
        self.contract_size = contract_size
        self.point_size = point_size
        self.pip_value_usd = pip_value_usd
        self.z_window = z_window
        self.atr_period = atr_period
        self.start_hour = start_hour
        self.end_hour = end_hour

        # Buffers
        self.close_buf = np.array([])
        self.atr_buf = np.array([])

        # Trade state
        self.trade: Optional[Trade] = None
        self.trades: list[Trade] = []

    def compute_zscore(self) -> float:
        n = self.z_window + 1
        if len(self.close_buf) < n + 1:
            return 0.0
        rets = np.diff(self.close_buf[-(n + 1):])
        cur_ret = rets[-1]
        mean = np.mean(rets[:-1])
        var = np.var(rets[:-1], ddof=1)
        if var < 1e-14:
            return 0.0
        return (cur_ret - mean) / np.sqrt(var)

    def get_atr(self) -> float:
        if len(self.atr_buf) < self.atr_period:
            return 0.0
        return float(np.mean(self.atr_buf))

    def check_hour(self, dt) -> bool:
        h = dt.hour
        if self.start_hour < self.end_hour:
            return self.start_hour <= h < self.end_hour
        return h >= self.start_hour or h < self.end_hour

    def get_spread_cost(self, spread_raw) -> float:
        """Spread cost for one side (half spread)."""
        return spread_raw * self.point_size * self.contract_size * self.lot_size

    def get_commission(self) -> float:
        """Commission per round-turn trade."""
        return self.lot_size * self.commission_per_lot

    def open_position(self, direction, entry_price, sprd, atr_v, z, bar_idx):
        self.trade = Trade(
            entry_bar=bar_idx,
            entry_price=entry_price,
            direction=direction,
            atr_entry=atr_v,
            z_entry=z,
            sprd_entry=sprd,
        )

    def close_position(self, reason, exit_price, bar_idx):
        t = self.trade
        if t is None:
            return
        t.exit_bar = bar_idx
        t.exit_price = exit_price
        t.exit_reason = reason

        # Gross PnL (price difference * contract_size * lot)
        if t.direction > 0:
            t.gross_pnl = (exit_price - t.entry_price) * self.contract_size * self.lot_size
        else:
            t.gross_pnl = (t.entry_price - exit_price) * self.contract_size * self.lot_size

        # Spread cost: entry and exit
        entry_sprd_cost = t.sprd_entry * self.point_size * self.contract_size * self.lot_size * 0.5
        exit_sprd_cost = entry_sprd_cost  # approximate (exit spread ∼ entry spread)
        t.spread_cost = entry_sprd_cost + exit_sprd_cost

        # Commission
        t.commission = self.get_commission()

        # Net PnL
        t.net_pnl = t.gross_pnl - t.spread_cost - t.commission

        self.trades.append(t)
        self.trade = None

    def check_entry(self, o, sprd, dt, bar_idx):
        if self.trade is not None:
            return
        if not self.check_hour(dt):
            return
        if sprd > self.max_spread:
            return

        z = self.compute_zscore()
        av = self.get_atr()
        if av <= 0:
            return

        if abs(z) >= self.z_threshold:
            direction = -1 if z > 0 else 1  # z>0 (overbought) → SHORT, z<0 (oversold) → LONG
            if self.trade_dir == 1 and direction == -1:
                return
            if self.trade_dir == 2 and direction == 1:
                return

            # Enter at open (with spread: LONG pays ask, SHORT gets bid)
            half_sprd_cost = sprd * self.point_size * 0.5
            entry_price = o + half_sprd_cost if direction > 0 else o - half_sprd_cost
            self.open_position(direction, entry_price, sprd, av, z, bar_idx)

    def run(self, times, opens, highs, lows, closes, spreads, volumes,
            start_dt=None):
        n = len(times)
        if start_dt is None:
            start_dt = datetime(2026, 6, 8, tzinfo=timezone.utc)

        warmup_bars = max(self.z_window + 3, 60)  # need at least z_window+2 closes

        for i in range(n):
            dt = datetime.fromtimestamp(times[i], tz=timezone.utc)

            # Warmup: fill buffers until enough data exists
            is_warmup = (dt < start_dt or i < warmup_bars)
            if is_warmup:
                self.close_buf = np.append(self.close_buf, closes[i])
                hl = highs[i] - lows[i]
                self.atr_buf = np.append(self.atr_buf, hl)[-self.atr_period:]
                continue

            # === MAIN PHASE ===
            # Before update_buffers: close_buf has closes[0..i-1]
            # z-score reflects return close[i-1] - close[i-2] (no look-ahead)

            # Check entry at open[i] using z from previous bar's close
            if self.trade is None:
                self.check_entry(opens[i], spreads[i], dt, i)

            # Manage existing trade
            if self.trade is not None:
                bars_held = i - self.trade.entry_bar
                if bars_held >= self.hold_bars:
                    half_sprd_cost = spreads[i] * self.point_size * 0.5
                    exit_price = closes[i] - half_sprd_cost if self.trade.direction > 0 else closes[i] + half_sprd_cost
                    self.close_position("expiry", exit_price, i)

            # Update buffers with current bar (after entry check)
            self.close_buf = np.append(self.close_buf, closes[i])
            hl = highs[i] - lows[i]
            self.atr_buf = np.append(self.atr_buf, hl)[-self.atr_period:]

        # Close any open trade at end
        if self.trade is not None:
            last_c = closes[-1]
            half_sprd = spreads[-1] * self.point_size * 0.5
            exit_price = last_c - half_sprd if self.trade.direction > 0 else last_c + half_sprd
            self.close_position("end", exit_price, n - 1)

        return self.trades


def compute_result(trades, pair, config_name) -> ChallengeZResult:
    r = ChallengeZResult(pair=pair, config_name=config_name, trades=trades)
    r.n_trades = len(trades)
    if r.n_trades == 0:
        return r

    r.total_gross = sum(t.gross_pnl for t in trades)
    r.total_net = sum(t.net_pnl for t in trades)
    r.total_commission = sum(t.commission for t in trades)
    r.total_spread_cost = sum(t.spread_cost for t in trades)

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]
    r.n_wins = len(wins)
    r.n_losses = len(losses)
    denom = r.n_trades - len([t for t in trades if abs(t.net_pnl) < 0.01])
    r.win_rate = r.n_wins / denom * 100 if denom else 0

    r.avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0
    r.avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0

    # Max consecutive losses
    streak = 0
    for t in trades:
        if t.net_pnl < 0:
            streak += 1
            r.max_consec_losses = max(r.max_consec_losses, streak)
        else:
            streak = 0

    # Max drawdown (peak-to-trough of cumulative net PnL)
    cum = np.cumsum([t.net_pnl for t in trades])
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    r.max_drawdown = float(np.max(dd)) if len(dd) > 0 else 0

    # Profit factor
    gross_wins = sum(t.net_pnl for t in wins) if wins else 0
    gross_losses = abs(sum(t.net_pnl for t in losses)) if losses else 0
    r.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    return r


def print_result(r: ChallengeZResult):
    print(f"  {r.pair:>8} | {r.config_name:<18} | "
          f"{r.n_trades:>3d} trades | WR {r.win_rate:>5.1f}% | "
          f"gross ${r.total_gross:>+7.2f} | "
          f"comm ${r.total_commission:>6.2f} | "
          f"sprd ${r.total_spread_cost:>6.2f} | "
          f"net ${r.total_net:>+7.2f} | "
          f"avgW ${r.avg_win:>+5.2f} avgL ${r.avg_loss:>+5.2f} | "
          f"maxDD ${r.max_drawdown:>5.2f} | "
          f"maxCL {r.max_consec_losses:>2d}")


def load_data(data_dir, pair):
    """Load .npy data file for a given pair."""
    fname = f"fundednext_{pair.lower()}_m1.npy"
    fpath = os.path.join(data_dir, fname)
    if not os.path.exists(fpath):
        return None
    data = np.load(fpath, allow_pickle=True)
    df = pd.DataFrame(data)
    return df


def run_config(df, config) -> list[Trade]:
    """Run ChallengeZ with given config and return trades."""
    sim = ChallengeZ(
        z_threshold=config['z'],
        hold_bars=config['hold_bars'],
        max_spread=config['max_spread'],
        lot_size=config['lot_size'],
        trade_dir=config.get('trade_dir', 0),
        commission_per_lot=3.0,
        contract_size=100000,
        point_size=0.00001,
        pip_value_usd=10.0,
        z_window=50,
        atr_period=20,
        start_hour=config.get('start_hour', 0),
        end_hour=config.get('end_hour', 24),
    )
    # time is in seconds (epoch) from .npy file
    if df['time'].max() > 1e12:
        # nanoseconds → seconds
        times = df['time'].values.astype(np.int64) // 10**9
    else:
        # already seconds
        times = df['time'].values.astype(np.int64)
    trades = sim.run(
        times,
        df['open'].values, df['high'].values,
        df['low'].values, df['close'].values,
        df['spread'].values, df['tick_volume'].values
    )
    return trades


def challenge_simulation(trades, target=500, daily_limit=250, account_size=5000):
    """Simulate 5-day challenge windows. Return pass rate."""
    if len(trades) == 0:
        return 0.0, 0, 0

    # Assign each trade to a day based on entry bar
    # (We don't have exact timestamps here, approximate by grouping)
    # For a real simulation, we'd use the actual timestamps
    pass


if __name__ == "__main__":
    DATA_DIR = os.path.dirname(__file__)
    PAIRS = ["AUDUSD", "EURAUD", "GBPAUD"]

    # Configs: z, hold_bars, max_spread, lot_size, trade_dir, start_hour, end_hour
    CONFIGS = [
        # Conservative: high z, 15 min hold
        {"name": "CZ-Conserv", "z": 4.0, "hold_bars": 15, "max_spread": 10,
         "lot_size": 0.5, "trade_dir": 0, "start_hour": 0, "end_hour": 24},
        # Moderate: mid z, 10 min hold
        {"name": "CZ-Moderate", "z": 3.5, "hold_bars": 10, "max_spread": 10,
         "lot_size": 0.5, "trade_dir": 0, "start_hour": 0, "end_hour": 24},
        # Aggressive: low z, 10 min hold
        {"name": "CZ-Aggress", "z": 3.0, "hold_bars": 10, "max_spread": 10,
         "lot_size": 0.5, "trade_dir": 0, "start_hour": 0, "end_hour": 24},
        # Moderate + LONG-only (CPPF showed LONG bias)
        {"name": "CZ-Mod-LONG", "z": 3.5, "hold_bars": 10, "max_spread": 10,
         "lot_size": 0.5, "trade_dir": 1, "start_hour": 0, "end_hour": 24},
    ]

    results = []
    print("=" * 140)
    print(f"{'Challenge-Z — FundedNext Server 3 (Jun 8 – Jul 24, 2026)'}")
    print(f"{'Costs: $3/lot commission + entry/exit spread. No trailing stop. Fixed hold.'}")
    print("=" * 140)

    for pair in PAIRS:
        df = load_data(DATA_DIR, pair)
        if df is None:
            print(f"\n{pair}: No data file found, skipping")
            continue
        print(f"\n--- {pair} ({len(df)} bars, med_sprd={int(df['spread'].median())} raw) ---")

        for cfg in CONFIGS:
            trades = run_config(df, cfg)
            r = compute_result(trades, pair, cfg['name'])
            results.append(r)
            print_result(r)

    # Summary: best net PnL per pair
    print("\n" + "=" * 140)
    print("BEST NET PnL PER PAIR")
    print("=" * 140)
    best_per_pair = {}
    for r in results:
        key = r.pair
        if key not in best_per_pair or r.total_net > best_per_pair[key].total_net:
            best_per_pair[key] = r

    for pair, r in sorted(best_per_pair.items()):
        surv = "✅ SURVIVES" if r.total_net > 0 else "❌ DIES"
        print(f"  {pair:<8} {r.config_name:<18} net=${r.total_net:>+7.2f}  "
              f"{r.n_trades:>3d}t  WR={r.win_rate:.1f}%  "
              f"avgW=${r.avg_win:>+.2f}  avgL=${r.avg_loss:>+.2f}  "
              f"maxDD=${r.max_drawdown:.2f}  {surv}")

    # Overall survivors summary
    print("\n" + "=" * 140)
    print("CONFIGS THAT SURVIVE COSTS (net PnL > 0)")
    print("=" * 140)
    survivors = [r for r in results if r.total_net > 0 and r.n_trades >= 10]
    if survivors:
        for r in sorted(survivors, key=lambda x: x.total_net, reverse=True):
            print_result(r)

        print(f"\n  → {len(survivors)} config/pair combos survive with {sum(r.n_trades for r in survivors)} total trades")
    else:
        print("  NONE. Challenge-Z fails on FundedNext data with current configs.")

    # For survivors, compute what it takes to hit $500 target
    print("\n" + "=" * 140)
    print("CHALLENGE MATH (for survivors)")
    print("=" * 140)
    for r in survivors:
        if r.total_net <= 0:
            continue
        n_days = (pd.to_datetime(df['time'], unit='s').max() -
                  pd.to_datetime(df['time'], unit='s').min()).days if locals().get('df') is not None else 47
        actual_days = n_days if n_days > 0 else 47
        trades_per_day = r.n_trades / actual_days if actual_days > 0 else 0
        net_per_trade = r.total_net / r.n_trades if r.n_trades > 0 else 0
        trades_needed = 500 / net_per_trade if net_per_trade > 0 else float('inf')
        days_needed = trades_needed / trades_per_day if trades_per_day > 0 else float('inf')
        max_loss_risk = abs(r.avg_loss) * r.max_consec_losses if r.max_consec_losses > 0 else 0

        print(f"  {r.pair:<8} {r.config_name:<18} "
              f"${r.total_net:>+7.2f} over {actual_days}d "
              f"({trades_per_day:.1f}t/d, ${net_per_trade:.2f}/t)")
        print(f"    → Need {trades_needed:.0f} trades ({days_needed:.0f}d) to hit $500 target")
        print(f"    → Max consecutive loss: {r.max_consec_losses} × ${abs(r.avg_loss):.2f} = ${max_loss_risk:.2f} "
              f"({'OK' if max_loss_risk < 250 else '⚠️ May breach $250 daily limit'})")
