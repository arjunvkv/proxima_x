"""
inject_swing_analysis.py
========================
Background helper daemon that watches logs/dashboard_latest.json and ensures
the swing_analysis data is populated correctly for the UI.
"""
import json
import time
from pathlib import Path

_HERE = Path(__file__).parent
LATEST_FILE = _HERE / "currency_decomposition" / "logs" / "dashboard_latest.json"

def classify_state(ssp, msp):
    if ssp > 1.0:
        return "BREAKOUT"
    if ssp > 0.85:
        return "EXHAUSTED"
    if ssp > 0.65:
        return "LATE"
    return "HEALTHY"

def main():
    print("Starting swing_analysis injector...")
    last_mtime = 0
    while True:
        try:
            if LATEST_FILE.exists():
                mtime = LATEST_FILE.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    with open(LATEST_FILE, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    updated = False
                    overlay = data.get("swing_overlay", {})
                    for sym, sw in overlay.items():
                        if "swing_analysis" not in sw or sw["swing_analysis"] is None:
                            avg_up = sw.get("avg_up", 1.0)
                            avg_dn = sw.get("avg_down", -1.0)
                            forming = sw.get("forming_pips", 0.0)
                            
                            # Calculate SSP
                            total_range = max(0.1, avg_up - avg_dn)
                            buy_ssp = round((forming - avg_dn) / total_range, 3)
                            sell_ssp = round((avg_up - forming) / total_range, 3)
                            
                            # Calculate MSP
                            buy_msp = round(forming / avg_up, 3) if forming > 0 and avg_up > 0 else 0.0
                            sell_msp = round(forming / avg_dn, 3) if forming < 0 and avg_dn < 0 else 0.0
                            
                            # Classify states
                            buy_state = classify_state(buy_ssp, buy_msp)
                            sell_state = classify_state(sell_ssp, sell_msp)
                            
                            position_state = "INSIDE_RANGE"
                            if buy_ssp > 1.0:
                                position_state = "BREAKOUT_UP"
                            elif sell_ssp > 1.0:
                                position_state = "BREAKOUT_DOWN"
                            elif total_range < 1.0:
                                position_state = "COMPRESSED_RANGE"
                            
                            sw["swing_analysis"] = {
                                "buy": {
                                    "state": buy_state,
                                    "ssp": buy_ssp,
                                    "msp": buy_msp,
                                },
                                "sell": {
                                    "state": sell_state,
                                    "ssp": sell_ssp,
                                    "msp": sell_msp,
                                },
                                "position_state": position_state,
                                "range_price": round(total_range, 1),
                                "range_expansion": 1.0,
                                "vol_expansion": 1.0,
                            }
                            updated = True
                    
                    if updated:
                        with open(LATEST_FILE, "w", encoding="utf-8") as f:
                            json.dump(data, f)
        except Exception as e:
            print("Error in injector:", e)
        time.sleep(0.5)

if __name__ == "__main__":
    main()
