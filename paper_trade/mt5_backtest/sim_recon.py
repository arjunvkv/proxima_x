"""Full Python sim of V2z_CPPF_RECON EA logic for parameter tweaking."""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Callable
from datetime import datetime, timezone


@dataclass
class Trade:
    entry_time: int
    entry_price: float
    direction: int
    atr_entry: float
    z_entry: float
    sprd_entry: float
    stop_loss: float
    best_price: float
    bars_held: int = 0
    pnl: float = 0
    exit_reason: str = ""
    exit_price: float = 0


class ReconSim:
    """Replica of V2z_CPPF_RECON.mq5 logic."""

    def __init__(self, z_threshold=4.0, stop_a=3.0, trig_a=1.0, gap_a=0.05,
                 max_hold=54, atr_period=20, z_window=50, base_lot=0.5,
                 max_spread=5, limit_entry_atr=0.0, enable_stability=False,
                 stab_thresh=0.5, z_cum_min=5.0, sprd_z_max=1.5, vol_z_max=2.0,
                 z_cum_bars=10, trade_dir=1, start_hour=0, end_hour=7,
                 contract_size=100000, on_entry: Callable = None):
        self.z_threshold = z_threshold
        self.stop_a = stop_a
        self.trig_a = trig_a
        self.gap_a = gap_a
        self.max_hold = max_hold
        self.atr_period = atr_period
        self.z_window = z_window
        self.base_lot = base_lot
        self.max_spread = max_spread
        self.limit_entry_atr = limit_entry_atr
        self.enable_stability = enable_stability
        self.stab_thresh = stab_thresh
        self.z_cum_min = z_cum_min
        self.sprd_z_max = sprd_z_max
        self.vol_z_max = vol_z_max
        self.z_cum_bars = z_cum_bars
        self.trade_dir = trade_dir
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.contract_size = contract_size
        self.on_entry = on_entry

        self.close_buf = np.array([])
        self.atr_buf = np.array([])
        self.z_hist = np.array([])
        self.spread_hist = np.array([])
        self.vol_hist = np.array([])
        self.close_count = 0
        self.atr_count = 0
        self.z_count = 0
        self.spread_count = 0
        self.vol_count = 0

        self.trade: Optional[Trade] = None
        self.daily_trades = 0
        self.daily_date = ""
        self.trades: list[Trade] = []
        self.entry_log: list[dict] = []

    def compute_zscore(self) -> float:
        n = self.z_window + 1
        if self.close_count < n + 1:
            return 0.0
        rets = np.diff(self.close_buf[-(n+1):])
        cur_ret = rets[-1]
        mean = np.mean(rets[:-1])
        var = np.var(rets[:-1], ddof=1)
        if var < 1e-14:
            return 0.0
        return (cur_ret - mean) / np.sqrt(var)

    def get_atr(self) -> float:
        if self.atr_count < self.atr_period:
            return 0.0
        return float(np.mean(self.atr_buf))

    def compute_cumulative_z(self) -> float:
        if self.z_count < self.z_cum_bars or self.z_cum_bars <= 0:
            return 0.0
        return float(np.sum(self.z_hist[-self.z_cum_bars:]))

    def compute_spread_z(self) -> float:
        if self.spread_count < 30:
            return 0.0
        mean = float(np.mean(self.spread_hist))
        std = float(np.std(self.spread_hist, ddof=1))
        if std < 0.01:
            return 0.0
        return (self.spread_hist[-1] - mean) / std

    def compute_volume_z(self) -> float:
        if self.vol_count < 20:
            return 0.0
        mean = float(np.mean(self.vol_hist))
        std = float(np.std(self.vol_hist, ddof=1))
        if std < 0.01:
            return 0.0
        return (self.vol_hist[-1] - mean) / std

    def compute_stability(self) -> float:
        z_cur = self.z_hist[-1]
        z_cum = self.compute_cumulative_z()
        spd_z = self.compute_spread_z()
        vol_z = self.compute_volume_z()
        z_norm = min(abs(z_cur) / 8.0, 1.0)
        z_cum_norm = min(abs(z_cum) / 15.0, 1.0)
        spd_norm = max(1.0 - spd_z / 3.0, 0.0)
        vol_norm = max(1.0 - vol_z / 3.0, 0.0)
        return 0.3 * z_norm + 0.3 * z_cum_norm + 0.2 * spd_norm + 0.2 * vol_norm

    def check_hour(self, dt) -> bool:
        h = dt.hour
        if self.start_hour < self.end_hour:
            return self.start_hour <= h < self.end_hour
        return h >= self.start_hour or h < self.end_hour

    def open_position(self, direction, atr_v, price, sprd, z, bar_idx):
        sl_dist = self.stop_a * atr_v
        if direction > 0:
            entry = price
            sl = entry - sl_dist
            best = entry
        else:
            entry = price
            sl = entry + sl_dist
            best = entry
        self.trade = Trade(
            entry_time=bar_idx, entry_price=entry, direction=direction,
            atr_entry=atr_v, z_entry=z, sprd_entry=sprd,
            stop_loss=sl, best_price=best
        )
        self.daily_trades += 1
        if self.on_entry:
            self.on_entry(self.trade, self)

    def close_position(self, reason, price, pnl):
        if self.trade is None:
            return
        self.trade.exit_reason = reason
        self.trade.exit_price = price
        self.trade.pnl = pnl
        self.trades.append(self.trade)
        self.trade = None

    def close_via_stop(self, stop_price):
        t = self.trade
        if t.direction > 0:
            pl = (stop_price - t.entry_price) * self.base_lot * self.contract_size
        else:
            pl = (t.entry_price - stop_price) * self.base_lot * self.contract_size
        self.close_position("stop", stop_price, pl)

    def manage_position(self, o, h, l, c, sprd):
        t = self.trade
        if t is None:
            return
        t.bars_held += 1
        atr_v = self.get_atr()
        if atr_v <= 0:
            return
        tg = self.trig_a * atr_v
        gp = self.gap_a * atr_v
        min_gap = 0.5 * 0.00001 * 10
        gp = max(gp, min_gap)

        if t.direction > 0:
            if h > t.best_price:
                t.best_price = h
                if t.best_price - t.entry_price > tg:
                    ns = t.best_price - gp
                    if ns > t.stop_loss:
                        t.stop_loss = ns
            if l <= t.stop_loss:
                self.close_via_stop(t.stop_loss)
                return
            if t.bars_held >= self.max_hold:
                pl = (c - t.entry_price) * self.base_lot * self.contract_size
                self.close_position("expiry", c, pl)
                return
        else:
            if l < t.best_price:
                t.best_price = l
                if t.entry_price - t.best_price > tg:
                    ns = t.best_price + gp
                    if ns < t.stop_loss:
                        t.stop_loss = ns
            if h >= t.stop_loss:
                self.close_via_stop(t.stop_loss)
                return
            if t.bars_held >= self.max_hold:
                pl = (t.entry_price - c) * self.base_lot * self.contract_size
                self.close_position("expiry", c, pl)
                return

    def check_entry(self, o, h, l, c, sprd, tick_vol, dt, bar_idx):
        if self.trade is not None:
            return
        if not self.check_hour(dt):
            return
        if sprd > self.max_spread:
            return

        d = dt.strftime("%Y%m%d")
        if d != self.daily_date:
            self.daily_date = d
            self.daily_trades = 0

        z = self.compute_zscore()
        av = self.get_atr()
        if av <= 0:
            return
        if self.limit_entry_atr > 0 and av < self.limit_entry_atr:
            return

        if abs(z) >= self.z_threshold:
            direction = -1 if z > 0 else 1
            if self.trade_dir == 1 and direction == -1:
                return
            if self.trade_dir == 2 and direction == 1:
                return

            if self.enable_stability:
                if self.z_count < self.z_cum_bars:
                    return
                z_cum = self.compute_cumulative_z()
                spd_z = self.compute_spread_z()
                vol_z = self.compute_volume_z()
                stability = self.compute_stability()
                if self.z_cum_min > 0 and abs(z_cum) < self.z_cum_min:
                    return
                if self.sprd_z_max > 0 and spd_z > self.sprd_z_max:
                    return
                if self.vol_z_max > 0 and vol_z > self.vol_z_max:
                    return
                if self.stab_thresh > 0 and stability < self.stab_thresh:
                    return

            entry_price = o if direction > 0 else o
            self.open_position(direction, av, entry_price, sprd, z, bar_idx)

    def update_buffers(self, o, h, l, c, sprd, tick_vol):
        self.close_buf = np.append(self.close_buf, c)
        self.close_count = len(self.close_buf)
        hl_range = h - l
        self.atr_buf = np.append(self.atr_buf, hl_range)[-self.atr_period:]
        self.atr_count = len(self.atr_buf)
        z = self.compute_zscore()
        self.z_hist = np.append(self.z_hist, z)[-30:]
        self.z_count = len(self.z_hist)
        self.spread_hist = np.append(self.spread_hist, float(sprd))[-30:]
        self.spread_count = len(self.spread_hist)
        self.vol_hist = np.append(self.vol_hist, float(tick_vol))[-20:]
        self.vol_count = len(self.vol_hist)

    def run(self, times, opens, highs, lows, closes, spreads, volumes, start_dt=None):
        n = len(times)
        if start_dt is None:
            start_dt = datetime(2026, 6, 8, tzinfo=timezone.utc)

        warmup_bars = max(self.z_window + 3, 60)

        for i in range(n):
            dt = datetime.fromtimestamp(times[i], tz=timezone.utc)

            # Warmup: fill buffers until enough data exists
            if dt < start_dt or i < warmup_bars:
                self.update_buffers(opens[i], highs[i], lows[i], closes[i],
                                    spreads[i], volumes[i])
                continue

            # Main phase: manage, entry check, then update
            if self.trade is not None:
                self.manage_position(opens[i], highs[i], lows[i], closes[i], spreads[i])
            if self.trade is None:
                self.check_entry(opens[i], highs[i], lows[i], closes[i],
                                 spreads[i], volumes[i], dt, i)
            self.update_buffers(opens[i], highs[i], lows[i], closes[i],
                                spreads[i], volumes[i])

        if self.trade is not None:
            last_c = closes[-1]
            if self.trade.direction > 0:
                pl = (last_c - self.trade.entry_price) * self.base_lot * self.contract_size
            else:
                pl = (self.trade.entry_price - last_c) * self.base_lot * self.contract_size
            self.close_position("end", last_c, pl)

        return self.trades


if __name__ == "__main__":
    import sys, os
    # Try FundedNext data first, fall back to FTMO
    data_path = r'C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\fundednext_audusd_m1.npy'
    if not os.path.exists(data_path):
        data_path = r'C:\Trading\Agentic_Trading\proxima_x\paper_trade\mt5_backtest\ftmo_audusd_m1.npy'
    data = np.load(data_path, allow_pickle=True)
    df = pd.DataFrame(data)
    # Time is in seconds (epoch) from .npy
    print(f"Data: {len(df)} bars, spread med={df['spread'].median()}")
    print(f"Time range: {datetime.fromtimestamp(df['time'].min(), tz=timezone.utc)} "
          f"to {datetime.fromtimestamp(df['time'].max(), tz=timezone.utc)}")

    configs = [
        {"name": "Baseline (no filters)",
         "limit_entry_atr": 0.0, "enable_stability": False},
        {"name": "ATR gate 0.00007",
         "limit_entry_atr": 0.00007, "enable_stability": False},
    ]

    for cfg in configs:
        sim = ReconSim(
            z_threshold=3.5, stop_a=3.0, trig_a=1.0, gap_a=0.05,
            max_hold=54, base_lot=0.5, max_spread=10,
            limit_entry_atr=cfg.get("limit_entry_atr", 0.0),
            enable_stability=cfg.get("enable_stability", False),
            stab_thresh=cfg.get("stab_thresh", 0.5),
            z_cum_min=cfg.get("z_cum_min", 5.0),
            trade_dir=0, start_hour=0, end_hour=24
        )
        trades = sim.run(
            df['time'].values.astype(np.int64),  # seconds, don't //10**9
            df['open'].values, df['high'].values,
            df['low'].values, df['close'].values,
            df['spread'].values, df['tick_volume'].values
        )
        total_pnl = sum(t.pnl for t in trades)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl < 0]
        neutral = [t for t in trades if abs(t.pnl) < 0.01]
        wr = len(wins)/(len(trades)-len(neutral))*100 if trades and (len(trades)-len(neutral)) > 0 else 0
        print(f"\n--- {cfg['name']} ---")
        print(f"  Trades: {len(trades)} ({len(wins)}W/{len(losses)}L/{len(neutral)}N) PnL: ${total_pnl:.2f} WR: {wr:.1f}%")
        for t in trades:
            print(f"  {t.entry_price:.5f} dir={t.direction:+d} ATR={t.atr_entry:.6f} "
                  f"z={t.z_entry:.2f} sprd={t.sprd_entry:.0f} -> {t.exit_reason} pnl=${t.pnl:.2f} held={t.bars_held}b")

    print("\n\n=== SUMMARY ===")
    for cfg in configs:
        print(f"{cfg['name']:40s}")
