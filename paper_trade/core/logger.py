"""Trade logger — CSV + terminal output. No lookahead, append-only."""
import os, csv, json, time
from . import config

class Logger:
    """Logs every event to CSV. thread-safe for single process."""

    def __init__(self, strategy_name):
        self.name = strategy_name
        self._file = None
        self._writer = None
        self._events = []

    def start(self):
        os.makedirs(config.LOGS_DIR, exist_ok=True)
        path = os.path.join(config.LOGS_DIR, f"{self.name}_{int(time.time())}.csv")
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["timestamp", "type", "pair", "detail"])
        self._file.flush()
        return path

    def log(self, event_type, pair, detail):
        """detail can be a string or dict."""
        ts = int(time.time())
        detail_str = json.dumps(detail) if isinstance(detail, dict) else str(detail)
        row = [ts, event_type, pair, detail_str]
        self._events.append(row)
        if self._writer:
            self._writer.writerow(row)
            self._file.flush()

    def get_trades(self):
        """Return list of completed trades (FILL + CLOSE pairs)."""
        fills = {}
        trades = []
        for row in self._events:
            if row[1] == "FILL":
                key = (row[2], row[0])  # (pair, timestamp)
                fills[key] = json.loads(row[3]) if isinstance(row[3], str) else row[3]
            elif row[1] == "CLOSE":
                detail = json.loads(row[3]) if isinstance(row[3], str) else row[3]
                entry_price = detail.get("entry", 0)
                exit_price = detail.get("exit", 0)
                gross = detail.get("gross_pnl", 0)
                trades.append({
                    "pair": row[2], "entry_time": row[0], "exit_time": row[0],
                    "entry_price": entry_price, "exit_price": exit_price,
                    "gross_pnl": gross,
                })
        return trades

    def close(self):
        if self._file:
            self._file.close()
