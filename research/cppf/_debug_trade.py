"""Debug the single trade mismatch in live flow verification."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
import numpy as np
import paper_trade.strategies.v2z_bar.strategy as st
from research.cppf._verify_live_flow import run_live_flow, MockMT5

TARGET_ET = 1775247000
_orig_check = st.BarStopManager.check_stops
def debug_check(self, bar_data, current_time):
    # Check if our target position exists
    for pair, pos in list(self.positions.items()):
        if pos["entry_time"] == TARGET_ET:
            bar = bar_data.get(pair)
            if bar is not None:
                cb = current_time // 60
                eb = pos["entry_bar"]
                print(f"  [TARGET] cb={cb} eb={eb} diff={cb-eb} bar_close={bar['close']:.8f} bar_high={bar['high']:.8f} bar_low={bar['low']:.8f} best={pos['best']:.8f} stop={pos['stop']:.8f}")
    result = _orig_check(self, bar_data, current_time)
    for ct in result:
        if ct.get("entry_time") == TARGET_ET:
            print(f"  [CLOSED] ET={ct['entry_time']} exit={ct['exit']:.8f} reason={ct['exit_reason']}")
    return result

st.BarStopManager.check_stops = debug_check

df = pd.read_parquet("research/phase_dislocation/dukascopy_data/gbpnzd.parquet")
if "timestamp" in df.columns:
    df = df.set_index("timestamp")
df.index = pd.to_datetime(df.index)

mock_mt5 = MockMT5({"GBPNZD": df})
st._mt5 = mock_mt5

trades = run_live_flow(df, mock_mt5)

for t in trades:
    if t["entry_time"] == TARGET_ET:
        print(f"\nFINAL TRADE: entry={t['entry']:.8f} exit={t['exit']:.8f} reason={t['exit_reason']}")
