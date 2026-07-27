"""Live session stats tracker — computes all validation metrics from trade events."""
import json, os, time
import numpy as np
from paper_trade.components import sharpe, win_rate, profit_factor, max_drawdown

_TRADE_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "trade_logs")


class SessionStats:
    """Tracks fills, closes, rejects per session. Produces a validation-ready snapshot."""

    def __init__(self, run_id=None):
        self._run_id = run_id or int(time.time())
        self._fills = []         # open positions
        self._trades = []        # completed trades: {pair, entry_t, exit_t, entry_p, exit_p, gross_pnl, direction, lot}
        self._rejects = []       # {time, pair, reason}
        self._total_pnl = 0.0
        self._today_pnl = 0.0
        self._today_trades = 0
        self._pnl_series = []    # cumulative PnL per close
        self._peak = 0.0
        self._pair_pnl = {}      # pair -> total PnL
        self._prev_day = None
        self._start_time = time.time()
        self._log_file = self._open_log()

    def _open_log(self):
        os.makedirs(_TRADE_LOG_DIR, exist_ok=True)
        path = os.path.join(_TRADE_LOG_DIR, f"run_{self._run_id}.jsonl")
        return open(path, "a")

    def _log_trade(self, event_type, data):
        record = {"run_id": self._run_id, "ts": int(time.time()), "event": event_type, **data}
        try:
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()
        except Exception:
            pass

    def record_fill(self, fill):
        self._fills.append(fill)
        self._log_trade("fill", {
            "ticket": fill.get("ticket"),
            "pair": fill["pair"], "direction": fill["direction"],
            "lot_size": fill["lot_size"], "entry_price": fill["entry_price"],
            "entry_time": fill["entry_time"], "spread": fill.get("spread"),
            "z_score": fill.get("z_score"),
        })

    def record_close(self, trade):
        """trade: dict from executor.close_position() merged with original fill."""
        gross = trade.get("gross_pnl", 0)
        self._trades.append(trade)
        self._total_pnl += gross

        day = time.strftime("%Y-%m-%d", time.gmtime(trade.get("exit_time", 0)))
        if day != self._prev_day:
            self._prev_day = day
            self._today_pnl = 0.0
            self._today_trades = 0
        self._today_pnl += gross
        self._today_trades += 1

        self._pnl_series.append(self._total_pnl)
        if self._total_pnl > self._peak:
            self._peak = self._total_pnl

        pair = trade.get("pair", "?")
        self._pair_pnl[pair] = self._pair_pnl.get(pair, 0) + gross

        self._log_trade("close", {
            "ticket": trade.get("ticket"),
            "pair": pair, "direction": trade.get("direction"),
            "lot_size": trade.get("lot_size"), "entry_price": trade.get("entry"),
            "exit_price": trade.get("exit_price"), "gross_pnl": gross,
            "entry_time": trade.get("entry_time"), "exit_time": trade.get("exit_time"),
        })

        # Remove from open fills (match on ticket)
        ticket = trade.get("ticket")
        if ticket is not None:
            self._fills = [f for f in self._fills if f.get("ticket") != ticket]
        else:
            self._fills = [f for f in self._fills if not (f.get("pair") == pair and f.get("entry_price") == trade.get("entry"))]

    def record_reject(self, pair, reason):
        self._rejects.append({"time": int(time.time()), "pair": pair, "reason": reason})

    def snapshot(self):
        """Return dict with all validation metrics."""
        pnls = np.array([t.get("gross_pnl", 0) for t in self._trades])
        cum = np.array(self._pnl_series) if self._pnl_series else np.array([])
        pair_summary = {
            p: {"pnl": round(v, 2), "trades": sum(1 for t in self._trades if t.get("pair") == p)}
            for p, v in sorted(self._pair_pnl.items(), key=lambda x: abs(x[1]), reverse=True)
        }

        return {
            "uptime": int(time.time() - self._start_time),
            "total_trades": len(self._trades),
            "today_trades": self._today_trades,
            "open_positions": len(self._fills),
            "total_pnl": round(self._total_pnl, 2),
            "today_pnl": round(self._today_pnl, 2),
            "avg_pnl": round(float(np.mean(pnls)), 2) if len(pnls) > 0 else 0.0,
            "win_rate": round(win_rate(pnls), 1),
            "sharpe": round(sharpe(pnls), 2),
            "profit_factor": round(profit_factor(pnls), 2),
            "max_dd": round(max_drawdown(cum), 2),
            "best_trade": round(float(np.max(pnls)), 2) if len(pnls) > 0 else 0.0,
            "worst_trade": round(float(np.min(pnls)), 2) if len(pnls) > 0 else 0.0,
            "rejects": len(self._rejects),
            "per_pair": pair_summary,
        }

    def close(self):
        try:
            self._log_file.flush()
            self._log_file.close()
        except Exception:
            pass

    @property
    def run_id(self):
        return self._run_id
