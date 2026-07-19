"""Live terminal dashboard — refreshes every second. Displays a full stats snapshot."""
import time, threading, sys, os

_CLEAR = "cls" if os.name == "nt" else "clear"


def _clear_screen():
    try:
        os.system(_CLEAR)
    except Exception:
        sys.stderr.write("\033[2J\033[H")


class Dashboard:
    """Thread-safe terminal dashboard. Prints to stderr so stdout is clean."""

    def __init__(self):
        self._stats = {
            "status": "initializing", "uptime": 0, "total_trades": 0,
            "today_trades": 0, "open_positions": 0,
            "total_pnl": 0.0, "today_pnl": 0.0, "avg_pnl": 0.0,
            "win_rate": 0.0, "sharpe": 0.0, "profit_factor": 0.0,
            "max_dd": 0.0, "best_trade": 0.0, "worst_trade": 0.0,
            "rejects": 0, "last_signal": "", "per_pair": {},
        }
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._start = time.time()

    def set_status(self, s):
        with self._lock:
            self._stats["status"] = s

    def set_last_signal(self, s):
        with self._lock:
            self._stats["last_signal"] = s

    def push_snapshot(self, snapshot):
        """Replace all stats with a fresh snapshot from SessionStats."""
        with self._lock:
            self._stats.update(snapshot)

    def _render(self):
        with self._lock:
            s = dict(self._stats)
        s["elapsed"] = f"{int(time.time() - self._start)}s"

        lines = []
        lines.append("=" * 70)
        lines.append(f"  PAPER TRADE — {s['status'].upper()}")
        lines.append("=" * 70)
        lines.append(f"  Elapsed:   {s['elapsed']}")
        lines.append(f"  Last Sig:  {s['last_signal']}")
        lines.append("─" * 70)
        lines.append(f"  Trades:    {s['today_trades']} today  |  {s['total_trades']} total  |  {s['open_positions']} open")
        lines.append(f"  PnL:       ${s['today_pnl']:>8.2f} today  |  ${s['total_pnl']:>8.2f} total")
        lines.append(f"  Avg Trade: ${s['avg_pnl']:>8.2f}")
        lines.append(f"  Win Rate:  {s['win_rate']:>5.1f}%")
        lines.append(f"  Sharpe:    {s['sharpe']:>5.2f}")
        lines.append(f"  Profit F:  {s['profit_factor']:>5.2f}")
        lines.append(f"  Max DD:    ${s['max_dd']:>8.2f}")
        lines.append(f"  Best/Worst:${s['best_trade']:>8.2f} / ${s['worst_trade']:>8.2f}")
        lines.append(f"  Rejects:   {s['rejects']}")
        lines.append("─" * 70)
        if s["per_pair"]:
            lines.append("  Per Pair:")
            for pair, ps in s["per_pair"].items():
                lines.append(f"    {pair:>7s}  ${ps['pnl']:>8.2f}  ({ps['trades']} trades)")
        lines.append("=" * 70)

        _clear_screen()
        sys.stderr.write("\n".join(lines) + "\n")

    def _loop(self):
        while self._running:
            self._render()
            time.sleep(1)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
