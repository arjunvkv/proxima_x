import sys
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta

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

class DeterministicPythonEngine:
    def __init__(self):
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
        from_dt = dt - timedelta(days=2) # 2 days back for rolling z-score
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

    def run_deterministic_replay(self, date_str: str) -> list:
        pair_dfs = self.fetch_m5_data(date_str)
        if not pair_dfs:
            return []

        trades = []
        target_day = datetime.strptime(date_str, "%Y-%m-%d").date()

        # ── 1. TOKYO H0 (00:00 UTC) ──────────────────────────────────────────
        tokyo_time = pd.Timestamp(f"{date_str} 00:00:00")
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
                pnl = round(pips * 0.15 * pip_val_usd(pair), 2)
                trades.append({
                    "is_live": False, "cycle": "REPLAY", "ticket": self._next_ticket(),
                    "iso_timestamp": f"{date_str} 00:00:05 UTC", "display_time": f"{date_str} 00:00:05",
                    "strategy": "Tokyo H0 (v107)", "pair": pair, "side": "BUY", "lot": 0.15,
                    "pips": pips, "sim_pnl": pnl, "live_close_pnl": None, "slippage_cost": None,
                    "is_win": pnl >= 0, "bug_reason": "🟢 CLEAN EXECUTION", "gate_reason": None
                })

        # ── 2. MSV ASIAN (00:30 UTC) ─────────────────────────────────────────
        msv_time = pd.Timestamp(f"{date_str} 00:30:00")
        df_uj = pair_dfs.get("USDJPY")
        if df_uj is not None and msv_time in df_uj.index:
            loc = df_uj.index.get_loc(msv_time)
            entry_price = df_uj.iloc[loc]['open']
            exit_idx = loc + 12
            if exit_idx < len(df_uj):
                exit_price = df_uj.iloc[exit_idx]['open']
                pips = round((exit_price - entry_price) / pip_size("USDJPY"), 1)
                pnl = round(pips * 0.18 * pip_val_usd("USDJPY"), 2)
                trades.append({
                    "is_live": False, "cycle": "REPLAY", "ticket": self._next_ticket(),
                    "iso_timestamp": f"{date_str} 00:30:12 UTC", "display_time": f"{date_str} 00:30:12",
                    "strategy": "MSV Asian (v107)", "pair": "USDJPY", "side": "BUY", "lot": 0.18,
                    "pips": pips, "sim_pnl": pnl, "live_close_pnl": None, "slippage_cost": None,
                    "is_win": pnl >= 0, "bug_reason": "🟢 CLEAN EXECUTION", "gate_reason": None
                })

        # ── 3. ULTRA MONSTER (Hourly 00:00 to 23:00 UTC) ──────────────────────
        for hour in range(24):
            bar_time = pd.Timestamp(f"{date_str} {hour:02d}:00:00")
            best_pair = None
            max_abs_ret = -1.0
            best_dir = "BUY"

            for pair in ULTRA_MONSTER_UNIVERSE:
                df = pair_dfs.get(pair)
                if df is not None and bar_time in df.index:
                    loc = df.index.get_loc(bar_time)
                    if loc >= 12:
                        window = df.iloc[loc-12:loc]
                        rng_pips = (window['high'].max() - window['low'].min()) / pip_size(pair)
                        if rng_pips >= 10.0:
                            ret = (df.iloc[loc]['open'] - df.iloc[loc-12]['open']) / pip_size(pair)
                            abs_ret = abs(ret)
                            if abs_ret > max_abs_ret:
                                max_abs_ret = abs_ret
                                best_pair = pair
                                best_dir = "BUY" if ret > 0 else "SELL"

            if best_pair is not None:
                df = pair_dfs[best_pair]
                loc = df.index.get_loc(bar_time)
                entry_price = df.iloc[loc]['open']
                exit_idx = loc + 6 # 30 min hold
                if exit_idx < len(df):
                    exit_price = df.iloc[exit_idx]['open']
                    pips = round((exit_price - entry_price)/pip_size(best_pair), 1) if best_dir == "BUY" else round((entry_price - exit_price)/pip_size(best_pair), 1)
                    pnl = round(pips * 1.20 * pip_val_usd(best_pair), 2)
                    trades.append({
                        "is_live": False, "cycle": "REPLAY", "ticket": self._next_ticket(),
                        "iso_timestamp": f"{date_str} {hour:02d}:00:02 UTC", "display_time": f"{date_str} {hour:02d}:00:02",
                        "strategy": "Ultra Monster (v107)", "pair": best_pair, "side": best_dir, "lot": 1.20,
                        "pips": pips, "sim_pnl": pnl, "live_close_pnl": None, "slippage_cost": None,
                        "is_win": pnl >= 0, "bug_reason": "🟢 CLEAN EXECUTION", "gate_reason": None
                    })

        # ── 4. CPPF Z & CPMC Z (z-score on cross pairs) ────────────────────────
        for pair in ["EURAUD", "GBPAUD"]:
            df = pair_dfs.get(pair)
            if df is not None:
                day_df = df[df.index.date == target_day]
                # compute 3-bar returns & 200-bar rolling z-score
                ret3 = (df['close'] - df['open'].shift(2)) / df['open'].shift(2)
                mean200 = ret3.rolling(200).mean()
                std200 = ret3.rolling(200).std()
                zscore = (ret3 - mean200) / (std200 + 1e-9)

                for t in day_df.index:
                    if t in zscore.index and zscore.loc[t] <= -6.0:
                        loc = df.index.get_loc(t)
                        if loc + 18 < len(df):
                            entry_price = df.iloc[loc+1]['open']
                            exit_price = df.iloc[loc+19]['open']
                            pips = round((exit_price - entry_price) / pip_size(pair), 1)
                            pnl = round(pips * 0.15 * pip_val_usd(pair), 2)
                            ts = t.strftime("%H:%M:%S")
                            trades.append({
                                "is_live": False, "cycle": "REPLAY", "ticket": self._next_ticket(),
                                "iso_timestamp": f"{date_str} {ts} UTC", "display_time": f"{date_str} {ts}",
                                "strategy": "CPPF Z (v107)", "pair": pair, "side": "BUY", "lot": 0.15,
                                "pips": pips, "sim_pnl": pnl, "live_close_pnl": None, "slippage_cost": None,
                                "is_win": pnl >= 0, "bug_reason": "🟢 CLEAN EXECUTION", "gate_reason": None
                            })

        # ── 5. NY H21 (21:00 UTC) ────────────────────────────────────────────
        ny_time = pd.Timestamp(f"{date_str} 21:00:00")
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
                pnl = round(pips * 0.25 * pip_val_usd(best_ny_pair), 2)
                trades.append({
                    "is_live": False, "cycle": "REPLAY", "ticket": self._next_ticket(),
                    "iso_timestamp": f"{date_str} 21:00:05 UTC", "display_time": f"{date_str} 21:00:05",
                    "strategy": "NY H21 (v107)", "pair": best_ny_pair, "side": "BUY", "lot": 0.25,
                    "pips": pips, "sim_pnl": pnl, "live_close_pnl": None, "slippage_cost": None,
                    "is_win": pnl >= 0, "bug_reason": "🟢 CLEAN EXECUTION", "gate_reason": None
                })

        return trades

engine = DeterministicPythonEngine()
trades = engine.run_deterministic_replay("2026-08-03")
print(f"\n✅ 100% Deterministic Engine generated {len(trades)} trades for 2026-08-03:")
for t in trades[:15]:
    print(f"  {t['iso_timestamp']} | #{t['ticket']} | {t['strategy']} | {t['pair']} {t['side']} | pips={t['pips']} | PnL=${t['sim_pnl']}")
