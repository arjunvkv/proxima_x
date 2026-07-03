import sys, os, json, time, logging
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from proxima_ops.execution.mt5_connector import MT5Connector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_trade_persistence")

STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "state", "trade_state_persistence.json")

def verify():
    print("=" * 60)
    print("TRADE STATE PERSISTENCE VERIFICATION")
    print("=" * 60)

    # 1. Check if persistence file exists
    print(f"\n[1] Persistence file: {STATE_PATH}")
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = json.load(f)
        print(f"    EXISTS with {len(state)} ticket(s)")
        for tid, s in state.items():
            print(f"    ticket={tid}: hold={s['hold_cycle_count']} "
                  f"mfe={s['mfe']} mae={s['mae']} "
                  f"entry_atr={s['entry_atr']} side={s['side']}")
    else:
        print("    NOT FOUND")
        state = {}

    # 2. Check MT5 positions
    print(f"\n[2] MT5 Positions:")
    mt5 = MT5Connector()
    if not mt5.connect():
        print("    Cannot connect to MT5")
        return

    positions = mt5.get_positions()
    if not positions:
        print("    No open positions")
        return

    for pos in positions:
        ticket = pos["ticket"]
        print(f"    ticket={ticket} symbol={pos['symbol']} side={pos['type']}")
        print(f"    entry={pos['price_open']} current={pos['price_current']}")
        print(f"    profit={pos['profit']:.2f}")

        # 3. Check stagnation eligibility
        print(f"\n[3] Stagnation Exit Eligibility for ticket={ticket}:")

        ticket_str = str(ticket)
        if ticket_str in state:
            ps = state[ticket_str]
            hold = ps["hold_cycle_count"]
            mfe = ps["mfe"]
            mae = ps["mae"]
            entry_atr = ps["entry_atr"]
        else:
            # Live calculation
            pos_time = pos["time"]
            elapsed_sec = time.time() - pos_time
            hold = max(1, int(elapsed_sec / 3))

            entry_price = float(pos["price_open"])
            side = pos["type"]
            current_price = float(pos["price_current"])
            adv = (entry_price - current_price) if side == "SELL" else (current_price - entry_price)
            mfe = round(max(0.0, adv), 6)
            mae = round(min(0.0, adv), 6)
            entry_atr = 0.0

        print(f"    hold_cycle_count={hold}")
        print(f"    Entry ATR={entry_atr}")
        print(f"    MFE={mfe}  MAE={mae}")
        mfe_mae_range = mfe - mae
        print(f"    MFE-MAE range={mfe_mae_range:.6f}")
        print(f"    0.5 x Entry ATR={0.5 * entry_atr:.6f}")

        if hold >= 60:
            print(f"    ✅ hold_time ({hold}) >= 60 — PASSES time condition")
            if entry_atr > 0 and mfe_mae_range < 0.5 * entry_atr:
                print(f"    ✅ STAGNATION_60 WOULD TRIGGER (range {mfe_mae_range:.6f} < 0.5*ATR {0.5*entry_atr:.6f})")
            elif entry_atr <= 0:
                print(f"    ⚠️ Entry ATR = 0 — cannot evaluate ATR condition")
            else:
                print(f"    ❌ Range {mfe_mae_range:.6f} >= 0.5*ATR {0.5*entry_atr:.6f} — will STAGNATE_WATCH")
        else:
            print(f"    ❌ hold_time ({hold}) < 60 — NOT eligible yet")

        # 4. Simulate restart
        print(f"\n[4] Restart simulation:")
        print(f"    Persisted hold_cycle_count={hold}")
        new_hold = hold + 1  # Simulate one cycle increment
        print(f"    After 1st cycle: {new_hold}")
        if new_hold >= 60:
            print(f"    ✅ Stagnation check active (hold >= 60)")
        else:
            restart_cycles_needed = 60 - hold
            print(f"    ❌ NEED {restart_cycles_needed} MORE cycles to reach 60")
        print(f"    (Without persistence, restart would reset hold to 0 → stagnation NEVER triggers)")

    mt5.disconnect()
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    verify()
