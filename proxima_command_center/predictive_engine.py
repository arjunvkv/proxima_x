#!/usr/bin/env python3
"""High-Performance Trading Intelligence Engine for Proxima X Command Center."""

import sys, os, json
from pathlib import Path
from datetime import datetime, timezone, timedelta

CONFIG_PATH = Path(__file__).parent / "dashboard_config.json"

class PredictiveEngine:
    """Config-driven predictive engine calculating upcoming strategy execution timetables."""
    def __init__(self):
        self.config = self.load_config()

    def load_config(self):
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def get_live_predictions(self):
        self.config = self.load_config() # Reload config dynamically if updated
        now_utc = datetime.now(timezone.utc)
        strategies = self.config.get("strategies", [])

        predictions = []
        for st in strategies:
            s_id = st["id"]

            if s_id == "ultra_monster":
                mins = now_utc.minute
                if mins < 30:
                    next_t = now_utc.replace(minute=30, second=0, microsecond=0)
                else:
                    next_t = (now_utc + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                seconds_left = int((next_t - now_utc).total_seconds())
            elif s_id == "tokyo_h0":
                next_t = (now_utc + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                seconds_left = int((next_t - now_utc).total_seconds())
            elif s_id == "sunday_h22":
                days_ahead = (6 - now_utc.weekday()) % 7
                if days_ahead == 0 and now_utc.hour >= 22:
                    days_ahead = 7
                next_t = (now_utc + timedelta(days=days_ahead)).replace(hour=22, minute=0, second=0, microsecond=0)
                seconds_left = int((next_t - now_utc).total_seconds())
            elif s_id == "cppf_z":
                seconds_left = 180
            elif s_id == "msv_asian":
                next_t = (now_utc + timedelta(days=1)).replace(hour=1, minute=0, second=0, microsecond=0) if now_utc.hour >= 7 else now_utc.replace(minute=0, second=0, microsecond=0)
                seconds_left = int((next_t - now_utc).total_seconds())
            elif s_id == "ny_h21":
                next_t = (now_utc + timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0) if now_utc.hour >= 21 else now_utc.replace(hour=21, minute=0, second=0, microsecond=0)
                seconds_left = int((next_t - now_utc).total_seconds())
            else:
                seconds_left = 420

            st_copy = dict(st)
            st_copy["seconds_until_fire"] = max(0, seconds_left)
            predictions.append(st_copy)

        return predictions, self.config

if __name__ == "__main__":
    eng = PredictiveEngine()
    preds, cfg = eng.get_live_predictions()
    print("CONFIG-DRIVEN ENGINE OUTPUT:")
    for p in preds:
        print(f"  • {p['name']:<28} | Symbol: {p['next_symbol']:<8} | Lot: {p['effective_lot']}L | Target Win: +${p['target_win_usd']}")
