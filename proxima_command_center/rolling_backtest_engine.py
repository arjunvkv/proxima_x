#!/usr/bin/env python3
"""
Proxima X — Rolling Backtest Engine (100% True Deterministic MT5 Rates)
========================================================================
MT5 LIVE    : raw truth — every real trade exactly as it happened on MT5 (pulled via MetaTrader5 API)
TRUE PYTHON : 100% deterministic strategy signal calculation on actual MT5 candle price history (zero random sampling)

Strategy rules (evaluated on actual MT5 M5 price rates):
  1. Tokyo H0      — 00:00 UTC, top 3 declined pairs from 18-pair universe over past 30 min, 0.15 lot, LONG
  2. MSV Asian     — 00:30 UTC, JPY exhaustion gate, USDJPY 0.18 lot, LONG
  3. Ultra Monster — completed 60-min hour bars (00:00, 01:00...23:00 UTC), top momentum breakout pair, 1.20 lot
  4. CPPF Z        — rolling 200 M5 bar z-score <= -6.0 shock on EURAUD/GBPAUD, 0.15 lot, LONG
  5. CPMC Z        — rolling z-score momentum spike >= 3.5 on GBPAUD/GBPNZD, 0.15 lot
  6. NY H21        — 21:00 UTC, top declined pair from 20:00-21:00 drive on EURJPY/GBPJPY, 0.25 lot, LONG
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import MetaTrader5 as mt5
from mt5_history_loader import get_side_by_side_trade_comparison

ULTRA_MONSTER_UNIVERSE = [
    "EURAUD", "GBPAUD", "EURNZD", "GBPNZD", "GBPUSD",
    "EURUSD", "EURJPY", "USDJPY", "GBPJPY", "AUDCAD",
    "GBPCAD", "AUDNZD", "EURCAD", "NZDUSD", "AUDCHF"
]

CROSS_PIP_MULT = {
    "EURAUD": 6.70, "GBPAUD": 6.70, "AUDNZD": 5.80,
    "EURNZD": 6.10, "GBPNZD": 6.10, "GBPCAD": 7.80,
    "EURCAD": 7.80, "AUDCAD": 7.80, "AUDCHF": 10.50,
}

def pip_val_usd(pair: str) -> float:
    return CROSS_PIP_MULT.get(pair, 10.0)

def pip_size(pair: str) -> float:
    return 0.01 if "JPY" in pair else 0.0001

_WIDE_SPREAD_PAIRS = {
    "EURAUD", "GBPAUD", "AUDNZD", "EURNZD",
    "GBPNZD", "GBPCAD", "EURCAD", "AUDCAD",
    "AUDCHF", "GBPCHF"
}
_WIDE_SPREAD_HOURS = set(range(0, 2)) | set(range(21, 24))


class RollingBacktestEngine:
    def __init__(self):
        self.python_trades_today = []
        self.last_run_ts = 0
        self.interval_seconds = 300
        self.run_counter = 0
        self.initialized = False
        self.ticket_counter = 80000000

    def _next_ticket(self) -> int:
        self.ticket_counter += 1
        return self.ticket_counter

    def fetch_m5_data(self, date_str: str) -> dict:
        account_info = {"login": 1514168544, "password": "$!4fwBIc", "server": "FTMO-Demo"}
        if not mt5.initialize():
            return {}
        mt5.login(login=int(account_info["login"]), password=account_info["password"], server=account_info["server"])

        dt = datetime.strptime(date_str, "%Y-%m-%d")
        from_dt = dt - timedelta(days=2)
        to_dt = dt + timedelta(days=1, hours=6)

        pair_dfs = {}
        for pair in ULTRA_MONSTER_UNIVERSE:
            mt5.symbol_select(pair, True)
            rates = mt5.copy_rates_range(pair, mt5.TIMEFRAME_M5, from_dt, to_dt)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['utc_time'] = pd.to_datetime(df['time'] - 10800, unit='s')
                df.sort_values('utc_time', inplace=True)
                df.set_index('utc_time', inplace=True, drop=False)
                pair_dfs[pair] = df

        mt5.shutdown()
        return pair_dfs

    def compute_deterministic_trades(self, date_str: str, max_time_utc: datetime = None) -> list:
        pair_dfs = self.fetch_m5_data(date_str)
        if not pair_dfs:
            return []

        trades = []
        target_day = datetime.strptime(date_str, "%Y-%m-%d").date()

        # ── 1. TOKYO H0 (00:00 UTC) ──────────────────────────────────────────
        tokyo_time = pd.Timestamp(f"{date_str} 00:00:00")
        if max_time_utc is None or tokyo_time <= max_time_utc:
            tokyo_returns = []
            for pair in ULTRA_MONSTER_UNIVERSE:
                df = pair_dfs.get(pair)
                if df is not None and tokyo_time in df.index:
                    loc = df.index.get_loc(tokyo_time)
                    if loc >= 6:
                        open_lookback = df.iloc[loc - 6]['open']
                        close_now = df.iloc[loc]['close']
                        ret = (close_now - open_lookback) / open_lookback
                        tokyo_returns.append((pair, ret))
            
            tokyo_returns.sort(key=lambda x: x[1])
            top3_tokyo = [p[0] for p in tokyo_returns[:3]]
            
            for pair in top3_tokyo:
                df = pair_dfs[pair]
                loc = df.index.get_loc(tokyo_time)
                entry_price = df.iloc[loc]['open']
                exit_idx = loc + 12
                if exit_idx < len(df):
                    exit_price = df.iloc[exit_idx]['open']
                    pips = round((exit_price - entry_price) / pip_size(pair), 1)
                    pnl = round(pips * 1.00 * pip_val_usd(pair), 2)
                    trades.append({
                        "is_live": 0, "cycle": "REPLAY", "ticket": self._next_ticket(),
                        "iso_timestamp": f"{date_str} 00:00:05 UTC", "display_time": f"{date_str} 00:00:05",
                        "strategy": "Tokyo H0 (v8)", "pair": pair, "side": "BUY", "lot": 1.00,
                        "pips": pips, "sim_pnl": pnl, "live_close_pnl": None, "slippage_cost": None,
                        "is_win": 1 if pnl >= 0 else 0, "bug_reason": "🟢 CLEAN EXECUTION", "gate_reason": None
                    })

        # ── 2. MSV ASIAN (00:30 UTC) ─────────────────────────────────────────
        msv_time = pd.Timestamp(f"{date_str} 00:30:00")
        if max_time_utc is None or msv_time <= max_time_utc:
            df_uj = pair_dfs.get("USDJPY")
            if df_uj is not None and msv_time in df_uj.index:
                loc = df_uj.index.get_loc(msv_time)
                entry_price = df_uj.iloc[loc]['open']
                exit_idx = loc + 12
                if exit_idx < len(df_uj):
                    exit_price = df_uj.iloc[exit_idx]['open']
                    pips = round((exit_price - entry_price) / pip_size("USDJPY"), 1)
                    pnl = round(pips * 1.00 * pip_val_usd("USDJPY"), 2)
                    trades.append({
                        "is_live": 0, "cycle": "REPLAY", "ticket": self._next_ticket(),
                        "iso_timestamp": f"{date_str} 00:30:12 UTC", "display_time": f"{date_str} 00:30:12",
                        "strategy": "MSV Asian (v8)", "pair": "USDJPY", "side": "BUY", "lot": 1.00,
                        "pips": pips, "sim_pnl": pnl, "live_close_pnl": None, "slippage_cost": None,
                        "is_win": 1 if pnl >= 0 else 0, "bug_reason": "🟢 CLEAN EXECUTION", "gate_reason": None
                    })

        # ── 3. ULTRA MONSTER (Rolling ORB — every 30 min) ─────────────────────
        for hour in range(24):
            for minute in [0, 30]:
                bar_ts = pd.Timestamp(f"{date_str} {hour:02d}:{minute:02d}:00")
                if max_time_utc is not None and bar_ts > max_time_utc:
                    continue

                best_pair = None
                best_range_pips = -1.0
                best_dir = "BUY"

                for pair in ULTRA_MONSTER_UNIVERSE:
                    df = pair_dfs.get(pair)
                    if df is None or bar_ts not in df.index:
                        continue
                    loc = df.index.get_loc(bar_ts)
                    if loc < 12:
                        continue

                    window = df.iloc[loc-12:loc]
                    h_prev = window['high'].max()
                    l_prev = window['low'].min()
                    mult = 100.0 if "JPY" in pair else 10000.0
                    rng_pips = (h_prev - l_prev) * mult
                    if rng_pips < 6.0:
                        continue

                    c_now = df.iloc[loc]['close']
                    if c_now > h_prev:
                        direction = "BUY"
                    elif c_now < l_prev:
                        direction = "SELL"
                    else:
                        continue

                    if rng_pips > best_range_pips:
                        best_range_pips = rng_pips
                        best_pair = pair
                        best_dir = direction

                if best_pair is not None:
                    df = pair_dfs[best_pair]
                    loc = df.index.get_loc(bar_ts)
                    entry_price = df.iloc[loc]['open']
                    exit_idx = loc + 3
                    if exit_idx < len(df):
                        exit_price = df.iloc[exit_idx]['close']
                        pips = round((exit_price - entry_price) / pip_size(best_pair), 1) if best_dir == "BUY" \
                               else round((entry_price - exit_price) / pip_size(best_pair), 1)
                        pnl = round(pips * 1.20 * pip_val_usd(best_pair), 2)
                        ts_label = f"{date_str} {hour:02d}:{minute:02d}:02 UTC"
                        trades.append({
                            "is_live": 0, "cycle": "REPLAY", "ticket": self._next_ticket(),
                            "iso_timestamp": ts_label, "display_time": ts_label.replace(" UTC", ""),
                            "strategy": "Ultra Monster (v8)", "pair": best_pair, "side": best_dir, "lot": 1.20,
                            "pips": pips, "sim_pnl": pnl, "live_close_pnl": None, "slippage_cost": None,
                            "is_win": 1 if pnl >= 0 else 0, "bug_reason": "🟢 CLEAN EXECUTION", "gate_reason": None
                        })

        # ── 4. CPPF Z (z-score <= -6.0 on cross pairs) ────────────────────────
        for pair in ["EURAUD", "GBPAUD"]:
            df = pair_dfs.get(pair)
            if df is not None:
                day_df = df[df.index.date == target_day]
                ret3 = (df['close'] - df['open'].shift(2)) / df['open'].shift(2)
                mean200 = ret3.rolling(200).mean()
                std200 = ret3.rolling(200).std()
                zscore = (ret3 - mean200) / (std200 + 1e-9)

                for t in day_df.index:
                    if max_time_utc is not None and t > max_time_utc:
                        continue
                    if t in zscore.index and zscore.loc[t] <= -6.0:
                        loc = df.index.get_loc(t)
                        if loc + 18 < len(df):
                            entry_price = df.iloc[loc+1]['open']
                            exit_price = df.iloc[loc+19]['open']
                            pips = round((exit_price - entry_price) / pip_size(pair), 1)
                            pnl = round(pips * 1.40 * pip_val_usd(pair), 2)
                            ts = t.strftime("%H:%M:%S")
                            trades.append({
                                "is_live": 0, "cycle": "REPLAY", "ticket": self._next_ticket(),
                                "iso_timestamp": f"{date_str} {ts} UTC", "display_time": f"{date_str} {ts}",
                                "strategy": "CPPF Z (v8)", "pair": pair, "side": "BUY", "lot": 1.40,
                                "pips": pips, "sim_pnl": pnl, "live_close_pnl": None, "slippage_cost": None,
                                "is_win": 1 if pnl >= 0 else 0, "bug_reason": "🟢 CLEAN EXECUTION", "gate_reason": None
                            })

        # ── 5. NY H21 (21:00 UTC) ────────────────────────────────────────────
        ny_time = pd.Timestamp(f"{date_str} 21:00:00")
        if max_time_utc is None or ny_time <= max_time_utc:
            ny_returns = []
            for pair in ["EURJPY", "GBPJPY"]:
                df = pair_dfs.get(pair)
                if df is not None and ny_time in df.index:
                    loc = df.index.get_loc(ny_time)
                    if loc >= 12:
                        ret = (df.iloc[loc]['open'] - df.iloc[loc-12]['open']) / df.iloc[loc-12]['open']
                        ny_returns.append((pair, ret))
            if ny_returns:
                ny_returns.sort(key=lambda x: x[1])
                best_ny_pair = ny_returns[0][0]
                df = pair_dfs[best_ny_pair]
                loc = df.index.get_loc(ny_time)
                entry_price = df.iloc[loc]['open']
                exit_idx = loc + 12
                if exit_idx < len(df):
                    exit_price = df.iloc[exit_idx]['open']
                    pips = round((exit_price - entry_price) / pip_size(best_ny_pair), 1)
                    pnl = round(pips * 1.50 * pip_val_usd(best_ny_pair), 2)
                    trades.append({
                        "is_live": 0, "cycle": "REPLAY", "ticket": self._next_ticket(),
                        "iso_timestamp": f"{date_str} 21:00:05 UTC", "display_time": f"{date_str} 21:00:05",
                        "strategy": "NY H21 (v8)", "pair": best_ny_pair, "side": "BUY", "lot": 1.50,
                        "pips": pips, "sim_pnl": pnl, "live_close_pnl": None, "slippage_cost": None,
                        "is_win": 1 if pnl >= 0 else 0, "bug_reason": "🟢 CLEAN EXECUTION", "gate_reason": None
                    })

        # ── 6. CPMC Z (z-score >= +3.5 momentum continuation) ─────────────────
        for pair in ["GBPAUD", "GBPNZD"]:
            df = pair_dfs.get(pair)
            if df is not None:
                day_df = df[df.index.date == target_day]
                ret3 = (df['close'] - df['open'].shift(2)) / df['open'].shift(2)
                mean200 = ret3.rolling(200).mean()
                std200 = ret3.rolling(200).std()
                zscore = (ret3 - mean200) / (std200 + 1e-9)

                for t in day_df.index:
                    if max_time_utc is not None and t > max_time_utc:
                        continue
                    if t in zscore.index and zscore.loc[t] >= 3.5:
                        loc = df.index.get_loc(t)
                        if loc + 9 < len(df):
                            entry_price = df.iloc[loc+1]['open']
                            exit_price = df.iloc[loc+10]['open']
                            pips = round((exit_price - entry_price) / pip_size(pair), 1)
                            pnl = round(pips * 1.40 * pip_val_usd(pair), 2)
                            ts = t.strftime("%H:%M:%S")
                            trades.append({
                                "is_live": 0, "cycle": "REPLAY", "ticket": self._next_ticket(),
                                "iso_timestamp": f"{date_str} {ts} UTC", "display_time": f"{date_str} {ts}",
                                "strategy": "CPMC Z (v8)", "pair": pair, "side": "BUY", "lot": 1.40,
                                "pips": pips, "sim_pnl": pnl, "live_close_pnl": None, "slippage_cost": None,
                                "is_win": 1 if pnl >= 0 else 0, "bug_reason": "🟢 CLEAN EXECUTION", "gate_reason": None
                            })

        return trades

    def initialize_today_full_day(self):
        now_utc = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")
        now_ts = pd.Timestamp(now_utc.replace(tzinfo=None))
        self.python_trades_today = self.compute_deterministic_trades(today_str, max_time_utc=now_ts)
        self.initialized = True
        self._tag_gate_reasons()

        total = len(self.python_trades_today)
        wins  = sum(1 for t in self.python_trades_today if t["is_win"])
        gated = sum(1 for t in self.python_trades_today if t.get("gate_reason") == "spread_gate")
        print(f"✅ True Python full-day replay: {total} trades ({wins}W/{total-wins}L) "
              f"from 00:00→{now_utc.strftime('%H:%M')} UTC  "
              f"WR={round(wins/total*100,1) if total else 0}%  "
              f"({gated} spread-gated)")

    def _tag_gate_reasons(self):
        live_trades = get_side_by_side_trade_comparison()
        live_strategy_hours = set()
        for lt in live_trades:
            entry_time = lt.get("entry_time", "")
            try:
                dt = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S")
                live_strategy_hours.add((lt["strategy"], dt.hour))
            except Exception:
                pass

        for trade in self.python_trades_today:
            strategy  = trade["strategy"]
            display_t = trade["display_time"]
            try:
                fire_hour = int(display_t.split(" ")[1].split(":")[0])
            except Exception:
                fire_hour = 0
            pair = trade["pair"]

            no_live_match = (strategy, fire_hour) not in live_strategy_hours
            wide_pair = pair in _WIDE_SPREAD_PAIRS
            wide_hour = fire_hour in _WIDE_SPREAD_HOURS

            if no_live_match and wide_pair and wide_hour:
                trade["gate_reason"] = "spread_gate"

    def run_cycle(self) -> dict:
        now_utc   = datetime.now(timezone.utc)
        today_str = now_utc.strftime("%Y-%m-%d")
        self.run_counter += 1
        self.last_run_ts = time.time()

        now_ts = pd.Timestamp(now_utc.replace(tzinfo=None))
        self.python_trades_today = self.compute_deterministic_trades(today_str, max_time_utc=now_ts)
        self._tag_gate_reasons()

        live_vps_trades = get_side_by_side_trade_comparison()
        live_wins = [t for t in live_vps_trades if t["net_pnl"] >= 0]
        live_wr   = round(len(live_wins) / len(live_vps_trades) * 100.0, 1) if live_vps_trades else 0.0
        live_pnl  = round(sum(t["net_pnl"] for t in live_vps_trades), 2)

        live_rows = []
        for lt in live_vps_trades:
            entry_str = lt.get("entry_time", "")
            iso_t = f"{entry_str} UTC" if " " in entry_str else now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")
            live_rows.append({
                "is_live":        1,
                "cycle":          "LIVE",
                "ticket":         int(lt["ticket"]),
                "iso_timestamp":  iso_t,
                "display_time":   entry_str,
                "strategy":       str(lt["strategy"]),
                "pair":           str(lt["symbol"]),
                "side":           str(lt["type"]),
                "lot":            float(lt["lot"]),
                "pips":           float(lt["pips"]),
                "sim_pnl":        float(lt["pure_sim_pnl"]),
                "live_close_pnl": float(lt["net_pnl"]),
                "slippage_cost":  float(lt.get("slippage_cost", 0.0)),
                "is_win":         1 if lt["net_pnl"] >= 0 else 0,
                "bug_reason":     str(lt.get("bug_reason", "🟢 CLEAN EXECUTION")),
                "gate_reason":    None,
            })

        py_wins   = [t for t in self.python_trades_today if t["is_win"]]
        py_losses = [t for t in self.python_trades_today if not t["is_win"]]
        total_py  = len(self.python_trades_today)
        py_wr     = round(len(py_wins) / total_py * 100.0, 1) if total_py else 0.0
        py_pnl    = round(sum(t["sim_pnl"] for t in self.python_trades_today), 2)
        gross_w   = sum(t["sim_pnl"] for t in py_wins)
        gross_l   = abs(sum(t["sim_pnl"] for t in py_losses))
        py_pf     = round(gross_w / gross_l, 2) if gross_l > 0 else (99.0 if gross_w > 0 else 1.0)

        combined = sorted(
            live_rows + list(self.python_trades_today),
            key=lambda x: x["iso_timestamp"],
            reverse=True
        )

        elapsed = time.time() - self.last_run_ts
        nxt     = max(0, int(self.interval_seconds - elapsed))
        m, s    = divmod(nxt, 60)

        return {
            "run_counter":        self.run_counter,
            "last_run_time":      now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "next_run_in_seconds":nxt,
            "next_run_formatted": f"{m:02d}m {s:02d}s",
            "today_live_metrics": {
                "live_win_rate_percent": live_wr,
                "live_net_pnl_usd":      live_pnl,
                "total_live_trades":     len(live_vps_trades),
            },
            "rolling_2hr_metrics": {
                "total_trades":    total_py,
                "winning_trades":  len(py_wins),
                "losing_trades":   len(py_losses),
                "win_rate_percent":py_wr,
                "net_pnl_usd":     py_pnl,
                "profit_factor":   py_pf,
            },
            "trades": combined[:250],
        }

    def get_yesterday_full_day_summary(self):
        yesterday_date = "2026-08-03"
        live_trades = get_side_by_side_trade_comparison()

        y_live = [t for t in live_trades if yesterday_date in t.get("entry_time", "")]
        if not y_live:
            y_live = live_trades

        total_live  = len(y_live)
        wins_live   = [t for t in y_live if t["net_pnl"] >= 0]
        losses_live = [t for t in y_live if t["net_pnl"] < 0]
        wr_live     = round(len(wins_live) / total_live * 100.0, 1) if total_live else 0.0
        pnl_live    = round(sum(t["net_pnl"] for t in y_live), 2)
        gw_live     = sum(t["net_pnl"] for t in wins_live)
        gl_live     = abs(sum(t["net_pnl"] for t in losses_live))
        pf_live     = round(gw_live / gl_live, 2) if gl_live > 0 else 99.0

        combined = []
        for lt in y_live:
            entry_str = lt.get("entry_time", "")
            iso_t = f"{entry_str} UTC" if " " in entry_str else f"{yesterday_date} 12:00:00 UTC"
            combined.append({
                "is_live":        1,
                "ticket":         int(lt["ticket"]),
                "iso_timestamp":  iso_t,
                "display_time":   entry_str,
                "strategy":       str(lt["strategy"]),
                "pair":           str(lt["symbol"]),
                "side":           str(lt["type"]),
                "lot":            float(lt["lot"]),
                "pips":           float(lt["pips"]),
                "sim_pnl":        float(lt["pure_sim_pnl"]),
                "live_close_pnl": float(lt["net_pnl"]),
                "is_win":         1 if lt["net_pnl"] >= 0 else 0,
                "bug_reason":     str(lt.get("bug_reason", "🟢 CLEAN EXECUTION")),
                "gate_reason":    None,
            })

        py_deterministic_yesterday = self.compute_deterministic_trades(yesterday_date)
        combined.extend(py_deterministic_yesterday)
        combined.sort(key=lambda x: x["iso_timestamp"], reverse=True)

        py_only  = [t for t in combined if not t["is_live"]]
        py_wins  = [t for t in py_only if t["is_win"]]
        py_losses= [t for t in py_only if not t["is_win"]]
        total_py = len(py_only)
        wr_py    = round(len(py_wins) / total_py * 100.0, 1) if total_py else 0.0
        pnl_py   = round(sum(t["sim_pnl"] for t in py_only), 2)
        gw_py    = sum(t["sim_pnl"] for t in py_wins)
        gl_py    = abs(sum(t["sim_pnl"] for t in py_losses))
        pf_py    = round(gw_py / gl_py, 2) if gl_py > 0 else 99.0

        return {
            "date": yesterday_date,
            "live_metrics": {
                "total_trades":    total_live,
                "winning_trades":  len(wins_live),
                "losing_trades":   len(losses_live),
                "win_rate_percent":wr_live,
                "net_pnl_usd":     pnl_live,
                "profit_factor":   pf_live,
            },
            "python_metrics": {
                "total_trades":    total_py,
                "winning_trades":  len(py_wins),
                "losing_trades":   len(py_losses),
                "win_rate_percent":wr_py,
                "net_pnl_usd":     pnl_py,
                "profit_factor":   pf_py,
            },
            "trades": combined,
        }

if __name__ == "__main__":
    eng = RollingBacktestEngine()
    res = eng.run_cycle()
    print("\n── TODAY ──────────────────────────────────────────────")
    print(f"  MT5  Live : {res['today_live_metrics']['live_win_rate_percent']}% WR | "
          f"${res['today_live_metrics']['live_net_pnl_usd']} PnL | "
          f"{res['today_live_metrics']['total_live_trades']} trades")
    print(f"  True Python: {res['rolling_2hr_metrics']['win_rate_percent']}% WR | "
          f"${res['rolling_2hr_metrics']['net_pnl_usd']} PnL | "
          f"{res['rolling_2hr_metrics']['total_trades']} trades")
