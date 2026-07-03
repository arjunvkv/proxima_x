"""ReplayEngine — runs the full Wave12Executor pipeline against historical data.
Outputs: expectancy_report.json with trade-level results."""
import sys, os, json, time, logging, copy
from datetime import datetime

sys.path.insert(0, ".")
sys.path.insert(0, "proxima_x")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("proxima_ops.replay.engine")

import numpy as np
from proxima_x.proxima_ops.replay.replay_mt5_connector import ReplayMT5Connector
from proxima_x.proxima_ops.execution.wave12_executor import Wave12Executor
from proxima_x.proxima_ops.execution.execution_ledger import ExecutionLedger
from proxima_x.proxima_ops.monitoring.broker_reconciliation import BrokerReconciliation

SYMBOLS = ["EURUSD", "GBPUSD", "EURJPY", "USDJPY"]


class ReplayEngine:
    """Orchestrate historical replay of the Wave 12 pipeline."""

    def __init__(self, data_dir: str = "data",
                 replay_ledger_path: str = "state/replay_ledger.jsonl",
                 spec_path: str = "state/wave12_experiment_spec.json",
                 initial_balance: float = 24988.47):
        self.data_dir = data_dir
        self.spec_path = spec_path
        self.initial_balance = initial_balance
        self.replay_ledger_path = replay_ledger_path

        # Fresh ledger for replay
        self.ledger = ExecutionLedger(replay_ledger_path)
        self._clean_file(replay_ledger_path)

        self.connector = ReplayMT5Connector(data_dir=data_dir, initial_balance=initial_balance)
        self.total_bars = self.connector.total_bars

        # Create executor with the replay connector
        self.executor = Wave12Executor(
            spec_path=spec_path,
            ledger_path=replay_ledger_path,
            mt5_connector=self.connector,
        )
        # Reset staircase to start fresh
        self.executor._completed_trades = 0
        self.executor._current_phase = 1
        self.executor._save_staircase_state()

        self.trades: list[dict] = []
        self.cycle_results: list[dict] = []

    def _clean_file(self, path: str):
        if os.path.exists(path):
            os.remove(path)

    def _warm_up(self, n_bars: int = 100):
        """Advance data to build indicator windows before starting pipeline."""
        self.connector.advance(n_bars)

    def run(self, start_bar: int = 100, report_interval: int = 500) -> dict:
        """Run full replay from start_bar to end of data.
        Each cycle = advance 1 M1 bar + run executor.cycle()."""
        self._warm_up(start_bar)
        cycle = 0

        logger.info(
            f"[REPLAY] Starting {self.total_bars - start_bar} replay cycles "
            f"({self.total_bars} total M1 bars)"
        )

        while not self.connector.is_at_end:
            cycle += 1
            self.connector.advance(1)  # Advance one M1 bar
            result = self.executor.cycle()

            # Extract key info
            exec_result = result.get("execution_result", {})
            close_result = result.get("close_result", {})

            if exec_result and exec_result.get("success"):
                # Trade opened
                self.trades.append({
                    "type": "open",
                    "cycle": cycle,
                    "ticket": exec_result.get("ticket"),
                    "symbol": exec_result.get("symbol"),
                    "direction": exec_result.get("direction"),
                    "price": exec_result.get("price"),
                    "signal_id": exec_result.get("signal_id", "?"),
                    "fusion_sources": exec_result.get("fusion_sources", []),
                    "fusion_is_erl": exec_result.get("fusion_is_erl", False),
                    "time": datetime.now().isoformat(),
                })
                logger.info(f"[REPLAY_TRADE] OPEN ticket={exec_result.get('ticket')} "
                             f"{exec_result.get('symbol')} {exec_result.get('direction')} "
                             f"@{exec_result.get('price')}")

            if close_result and close_result.get("success"):
                # Trade(s) closed
                for r in close_result.get("results", []):
                    self.trades.append({
                        "type": "close",
                        "cycle": cycle,
                        "ticket": r.get("ticket"),
                        "symbol": r.get("symbol"),
                        "direction": r.get("direction"),
                        "pnl": r.get("pnl"),
                        "exit_price": r.get("exit_price"),
                        "time": datetime.now().isoformat(),
                    })
                    logger.info(f"[REPLAY_TRADE] CLOSE ticket={r.get('ticket')} "
                                 f"pnl={r.get('pnl')}")

            self.cycle_results.append({
                "cycle": cycle,
                "decision": result.get("decision"),
                "active_signals": result.get("active_signals", 0),
                "state": result.get("segl_state"),
                "balance": result.get("balance"),
                "execution": result.get("execution_result"),
                "close": close_result.get("success") if close_result else False,
            })

            if cycle % report_interval == 0:
                stats = self.ledger.get_stats()
                logger.info(
                    f"[REPLAY] cycle={cycle}/{self.total_bars - start_bar} "
                    f"trades={stats['total_trades']} pnl={stats['total_pnl']:.2f} "
                    f"balance={self.connector._balance:.2f}"
                )

        return self._generate_report()

    def _generate_report(self) -> dict:
        """Compile expectancy report from ledger and connector state."""
        stats = self.ledger.get_stats()

        # Build trade log from ledger events
        trade_log = []
        for evt in self.ledger._events:
            d = {
                "event_type": evt.event_type,
                "symbol": evt.symbol,
                "direction": evt.direction,
                "volume": evt.volume,
                "entry_price": evt.entry_price,
                "exit_price": evt.exit_price,
                "pnl": evt.pnl,
                "mt5_ticket": evt.mt5_ticket,
                "signal_id": evt.signal_id,
            }
            trade_log.append(d)

        # Count regime coverage
        rsi_extremes = {"lt_30": 0, "gt_70": 0}
        for sym in SYMBOLS:
            for i in range(100, min(self.connector._cursor, self.connector._m1_count)):
                bars = self.connector._m1_data.get(sym, [])
                if i < len(bars):
                    pass  # RSI computation would need full indicator state

        closed_trades = [t for t in trade_log if t["event_type"] == "trade_closed"]
        pnls = [t.get("pnl", 0) or 0 for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        opens = [t for t in self.trades if t["type"] == "open"]
        core_trades = [t for t in opens if not t.get("fusion_is_erl", False)]
        erl_trades = [t for t in opens if t.get("fusion_is_erl", False)]
        hybrid_trades = [t for t in opens if len(t.get("fusion_sources", [])) > 1]

        erl_pnl = 0
        core_pnl = 0
        for t in erl_trades:
            ticket = t.get("ticket")
            for ct in closed_trades:
                if ct.get("mt5_ticket") == ticket and ct.get("pnl") is not None:
                    erl_pnl += ct["pnl"]
        for t in core_trades:
            ticket = t.get("ticket")
            for ct in closed_trades:
                if ct.get("mt5_ticket") == ticket and ct.get("pnl") is not None:
                    core_pnl += ct["pnl"]

        report = {
            "replay_date": datetime.now().isoformat(),
            "data_summary": {
                "total_m1_bars": self.total_bars,
                "symbols": SYMBOLS,
                "data_dir": self.data_dir,
            },
            "execution_summary": {
                "total_cycles": len(self.cycle_results),
                "trades_opened": len(opens),
                "trades_closed": len(closed_trades),
                "final_balance": round(self.connector._balance, 2),
                "total_pnl": round(stats["total_pnl"], 2) if stats["total_pnl"] else 0,
                "win_rate": round(stats["win_rate"], 4) if stats["win_rate"] else 0,
                "profit_factor": round(stats["profit_factor"], 4) if stats["profit_factor"] else 0,
                "avg_win": round(sum(wins) / len(wins), 2) if wins else 0,
                "avg_loss": round(sum(losses) / len(losses), 2) if losses else 0,
                "max_drawdown": 0,  # TODO: compute
                "total_commission": 0,
            },
            "er_attribution": {
                "core_only": len(core_trades),
                "erl_driven": len(erl_trades),
                "hybrid": len(hybrid_trades),
                "core_pnl": round(core_pnl, 2),
                "erl_pnl": round(erl_pnl, 2),
            },
            "trade_log": trade_log,
            "trade_attribution": [t for t in opens],
            "raw_stats": stats,
        }

        os.makedirs("state", exist_ok=True)
        with open("state/expectancy_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"[REPLAY] Report saved to state/expectancy_report.json")
        return report


if __name__ == "__main__":
    engine = ReplayEngine(replay_ledger_path="state/replay_ledger_cross.jsonl")
    report = engine.run(start_bar=100, report_interval=500)
    print(json.dumps(report["execution_summary"], indent=2))
    print("\n--- ERL Attribution ---")
    print(json.dumps(report.get("er_attribution", {}), indent=2))
    if report.get("trade_attribution"):
        print("\n--- Trade Attribution ---")
        for t in report["trade_attribution"]:
            print(f"  cycle={t['cycle']} {t['symbol']} {t['direction']} signal={t['signal_id']} erl={t['fusion_is_erl']} sources={t['fusion_sources']}")
