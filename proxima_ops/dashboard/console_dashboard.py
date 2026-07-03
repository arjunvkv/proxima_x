import json
import os
import time
import sys
from datetime import datetime

STATE_PATH = "state/unified_system_state.json"

BOX_CHARS = {
    "tl": "\u2554", "tr": "\u2557", "bl": "\u255a", "br": "\u255d",
    "h": "\u2550", "v": "\u2551",
    "tm": "\u2566", "bm": "\u2569", "lm": "\u2560", "rm": "\u2563",
    "c": "\u256c",
    "h_light": "\u2500", "v_light": "\u2502",
}

ANSI_CLS = "\033[2J\033[H"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"
ANSI_CYAN = "\033[96m"
ANSI_DIM = "\033[2m"


def _c(val, label="", width=10):
    s = str(val)
    if label:
        return f"{label}{s}"
    return s


def _colorize(val, good_vals=None, bad_vals=None):
    if good_vals and val in good_vals:
        return f"{ANSI_GREEN}{val}{ANSI_RESET}"
    if bad_vals and val in bad_vals:
        return f"{ANSI_RED}{val}{ANSI_RESET}"
    return str(val)


def _read_state() -> dict:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _bar(width: int = 46) -> str:
    return BOX_CHARS["h"] * width


def _v(s: str = "") -> str:
    return f"{BOX_CHARS['v']}  {s}"


def render(state: dict) -> str:
    ts = state.get("timestamp_human", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    bal = state.get("balance", 0.0)
    eq = state.get("equity", 0.0)
    pos = state.get("open_positions", 0)
    pnl = state.get("floating_pnl", 0.0)
    segl = state.get("segl_state", "N/A")
    mof_s = state.get("mof_state", "N/A")
    mof_sc = state.get("mof_score", 0.0)
    rf_ready = state.get("rf_ready", 0)
    rf_total = state.get("rf_total", 1)
    rf_drift = state.get("rf_drift", False)
    gov = state.get("gov_state", "WAITING")
    vel = state.get("vel_decision", "IDLE")
    cb = state.get("cb_ok", True)
    pip = state.get("pipeline", {})
    perf = state.get("performance", {})
    risk = state.get("risk", {})
    rsi_map = state.get("rsi_by_symbol", {})
    rates = state.get("rates", {})
    h = state.get("health", {})
    lat = state.get("latency_ms", 0.0)
    cyc = h.get("cycle", 0)
    pid = h.get("pid", 0)
    mt5_ok = h.get("mt5_ok", "?")
    dd = risk.get("drawdown_pct", 0.0)
    ls = risk.get("loss_streak", 0)
    wr = perf.get("win_rate", 0.0)
    amp = perf.get("amplitude", 1)
    sc = perf.get("current_phase", 1)
    total_trades = perf.get("total_trades", 0)
    wins = perf.get("wins", 0)
    losses = perf.get("losses", 0)
    gov_state_str = gov if gov else "WAITING"
    pip_sig = pip.get("signals_generated", 0)
    pip_confirm = pip.get("confirm_passes", 0)
    segl_str = _colorize(str(segl),
                         good_vals={"OBSERVE", "ARMED"},
                         bad_vals={"LOCKED", "COOLDOWN", "CRITICAL"})
    cb_str = _colorize("OK" if cb else "TRIPPED",
                       good_vals={"OK"}, bad_vals={"TRIPPED"})
    drift_str = _colorize("DRIFT" if rf_drift else "OK",
                          good_vals={"OK"}, bad_vals={"DRIFT"})
    vel_str = _colorize(str(vel),
                        good_vals={"IDLE", "ARMING"},
                        bad_vals={"BLOCKED", "DENIED"})

    rsi_parts = []
    for sym, rsi_val in sorted(rsi_map.items()):
        if sym in rates:
            bid = rates[sym].get("bid", "?")
            rsi_parts.append(f"{sym} RSI={rsi_val}@{bid}")
        else:
            rsi_parts.append(f"{sym} RSI={rsi_val}")
    market_line = " | ".join(rsi_parts[:4]) if rsi_parts else "No RSI data"

    bal_str = f"${bal:>,.2f}" if bal else "$0.00"
    eq_str = f"${eq:>,.2f}" if eq else "$0.00"
    pnl_str = f"${pnl:+,.2f}" if pnl else "$0.00"

    lines = []
    W = 54
    lines.append(f"{BOX_CHARS['tl']}{_bar(W)}{BOX_CHARS['tr']}")
    lines.append(f"{_v()}{ANSI_BOLD}PROXIMA LIVE SYSTEM DASHBOARD{ANSI_RESET}  {ts}")
    lines.append(f"{BOX_CHARS['lm']}{_bar(W)}{BOX_CHARS['rm']}")

    lines.append(f"{_v()}Market: {market_line}")
    lines.append(f"{_v()}SEGL: {segl_str}  MOF: {mof_s}({mof_sc:.2f})  RF: {rf_ready}/{rf_total} {drift_str}")
    lines.append(f"{_v()}Positions: {pos} | PnL: {pnl_str} | Balance: {bal_str} | Equity: {eq_str}")
    lines.append(f"{_v()}Pipeline: sig={pip_sig} confirm={pip_confirm} gov={gov_state_str} VEL={vel_str}")
    lines.append(f"{_v()}Risk: CB={cb_str} drawdown={dd:.1f}% loss_streak={ls}")
    lines.append(f"{_v()}Perf: staircase={sc} amp={amp} trades={total_trades} win={wr:.1f}% (W:{wins} L:{losses})")
    lines.append(f"{_v()}Health: PID={pid} alive MT5={mt5_ok} cycle={cyc} latency={lat:.0f}ms")

    lines.append(f"{BOX_CHARS['bl']}{_bar(W)}{BOX_CHARS['br']}")
    return "\n".join(lines)


def _main_loop(interval: float = 3.0):
    print(f"{ANSI_CLS}")
    print(f"Console Dashboard — reading {STATE_PATH} every {interval}s")
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            state = _read_state()
            if state:
                output = render(state)
            else:
                output = (f"{BOX_CHARS['tl']}{_bar(54)}{BOX_CHARS['tr']}\n"
                          f"{_v()}{ANSI_YELLOW}Waiting for state data...{ANSI_RESET}\n"
                          f"{BOX_CHARS['bl']}{_bar(54)}{BOX_CHARS['br']}")
            sys.stdout.write(f"{ANSI_CLS}{output}\n")
            sys.stdout.flush()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nConsole dashboard stopped by user.")


if __name__ == "__main__":
    _main_loop()
