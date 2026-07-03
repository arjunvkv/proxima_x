"""
UnifiedStateBuilder
====================
SINGLE canonical state builder — the only allowed aggregation layer between
live data sources and consumers (dashboard, monitors, logs).

Architecture:
    MT5 API ─────┐
    PipelineTrace ─┤ → UnifiedStateBuilder → UnifiedState → consumers
    RegimeTracker ─┘

NO raw file reads allowed outside this layer.
"""

import json
import logging
import os
import time
from typing import Any, Optional

from proxima_x.proxima_ops.execution.mt5_connector import MT5Connector

try:
    from proxima_x.proxima_ops.intelligence.symbol_universe_selector import (
        SymbolUniverseSelector,
    )
    _HAS_SIL = True
except ImportError:
    SymbolUniverseSelector = None
    _HAS_SIL = False

logger = logging.getLogger("proxima_ops.dashboard.unified_state_builder")

# Fallback symbols used when SIL (Symbol Intelligence Layer) is unavailable
FALLBACK_SYMBOLS = ["EURUSD", "GBPUSD", "EURJPY", "USDJPY"]

# Default MT5 magic number for this project
PROXIMA_MAGIC = 202406


class UnifiedStateBuilder:
    """Build the unified system state from MT5, pipeline trace, and regime data."""

    def __init__(
        self,
        mt5_connector: Optional[MT5Connector] = None,
        data_dir: str = "state",
    ):
        """
        Args:
            mt5_connector: Optional pre-connected MT5Connector instance.
                           If None, create a new one.
            data_dir: Path to the state directory. If a relative path is given,
                      it is resolved from the project root
                      (three levels up from this file's directory).
        """
        self._connector = mt5_connector or MT5Connector()
        self._data_dir = self._resolve_data_dir(data_dir)
        self._last_error: Optional[str] = None
        self._symbol_selector = SymbolUniverseSelector() if _HAS_SIL else None
        self._active_symbols = list(FALLBACK_SYMBOLS)
        self._sil_report = None

    # ------------------------------------------------------------------
    # Symbol universe (SIL integration)
    # ------------------------------------------------------------------

    def _refresh_active_symbols(self) -> list:
        """Get current symbol universe from SIL, with fallback to FALLBACK_SYMBOLS.

        Safety rules:
          - If SIL import failed → silent fallback to FALLBACK_SYMBOLS.
          - If select_universe() returns < 4 symbols → use FALLBACK_SYMBOLS.
          - Never raises.
        """
        try:
            if self._symbol_selector is not None:
                symbols = self._symbol_selector.get_active_universe()
                if symbols and len(symbols) >= 4:
                    self._active_symbols = symbols
                    return symbols
        except Exception as exc:
            logger.warning("SIL universe refresh failed: %s", exc)
        self._active_symbols = list(FALLBACK_SYMBOLS)
        return self._active_symbols

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> dict[str, Any]:
        """
        Build the UnifiedState object.

        Returns a dict with sections:
            market_state, execution_state, risk_state, performance_state,
            pipeline_state, system_health.

        This method MUST NOT raise exceptions.  All source reads are wrapped
        in try/except; failed sources are reflected in system_health.data_sources.
        """
        build_ts = time.time()

        # Refresh symbol universe from SIL
        self._refresh_active_symbols()

        # Track which data sources succeeded / failed
        data_sources = {
            "mt5_api": False,
            "pipeline_trace": False,
            "regime_tracker": False,
            "execution_ledger": False,
        }

        # ---- read raw data from all sources ----
        mt5_state = {"account": None, "positions": [], "ticks": {}}
        try:
            mt5_state = self._read_mt5_state()
            data_sources["mt5_api"] = True
        except Exception as exc:
            logger.warning("Failed to read MT5 state: %s", exc)
            self._last_error = str(exc)

        pipeline_trace = {}
        try:
            pipeline_trace = self._read_pipeline_trace()
            if pipeline_trace:
                data_sources["pipeline_trace"] = True
        except Exception as exc:
            logger.warning("Failed to read pipeline trace: %s", exc)
            self._last_error = str(exc)

        regime_state = {}
        try:
            regime_state = self._read_regime_state()
            if regime_state:
                data_sources["regime_tracker"] = True
        except Exception as exc:
            logger.warning("Failed to read regime state: %s", exc)
            self._last_error = str(exc)

        ledger = []
        try:
            ledger = self._read_ledger()
            if ledger:
                data_sources["execution_ledger"] = True
        except Exception as exc:
            logger.warning("Failed to read execution ledger: %s", exc)
            self._last_error = str(exc)

        # ---- build sections ----

        market_state = self._build_market_state(
            mt5_state, pipeline_trace, regime_state
        )

        execution_state = self._build_execution_state(mt5_state, pipeline_trace)

        risk_state = self._build_risk_state(mt5_state)

        performance_state = self._build_performance_state(
            mt5_state, ledger, execution_state.get("open_positions_detail", [])
        )

        pipeline_state = self._build_pipeline_state(pipeline_trace)

        system_health = self._build_system_health(
            mt5_connected=data_sources["mt5_api"],
            data_sources=data_sources,
            pipeline_trace=pipeline_trace,
            build_ts=build_ts,
        )

        return {
            "market_state": market_state,
            "execution_state": execution_state,
            "risk_state": risk_state,
            "performance_state": performance_state,
            "pipeline_state": pipeline_state,
            "system_health": system_health,
        }

    # ------------------------------------------------------------------
    # Data source readers
    # ------------------------------------------------------------------

    def _read_mt5_state(self) -> dict[str, Any]:
        """
        Connect to MT5 and read:
          - account info (balance, equity, margin, etc.)
          - open positions (filtered to PROXIMA_MAGIC if possible)
          - tick data for all active symbols (from SIL or fallback list)

        Returns a dict with keys: account, positions, ticks.
        """
        result: dict[str, Any] = {
            "account": None,
            "positions": [],
            "ticks": {},
        }

        # Ensure we are connected
        if not self._connector.is_connected:
            connected = self._connector.connect()
            if not connected:
                logger.warning(
                    "MT5 not connected: %s", self._connector.last_error
                )
                return result

        # Account info
        try:
            result["account"] = self._connector.get_account()
        except Exception as exc:
            logger.warning("Failed to read MT5 account info: %s", exc)

        # Positions — try filtering by magic first, fall back to all
        try:
            all_positions = self._connector.get_positions()
            # Filter to PROXIMA_MAGIC
            filtered = [
                p for p in all_positions if p.get("magic") == PROXIMA_MAGIC
            ]
            result["positions"] = filtered if filtered else all_positions
        except Exception as exc:
            logger.warning("Failed to read MT5 positions: %s", exc)

        # Tick data for tracked symbols
        for sym in self._active_symbols:
            try:
                tick = self._connector.get_tick(sym)
                if tick is not None:
                    result["ticks"][sym] = tick
            except Exception as exc:
                logger.debug("Failed to read tick for %s: %s", sym, exc)

        return result

    def _read_pipeline_trace(self) -> dict[str, Any]:
        """
        Read state/live_pipeline_trace.jsonl and return the LAST entry dict.

        Returns empty dict if file is missing or empty.
        """
        path = os.path.join(self._data_dir, "live_pipeline_trace.jsonl")
        if not os.path.isfile(path):
            logger.debug("Pipeline trace file not found: %s", path)
            return {}

        last_entry: Optional[dict] = None
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        last_entry = json.loads(stripped)
        except Exception as exc:
            logger.warning("Failed to parse pipeline trace: %s", exc)
            return {}

        if last_entry is None:
            logger.debug("Pipeline trace file is empty: %s", path)
            return {}

        return last_entry

    def _read_regime_state(self) -> dict[str, Any]:
        """
        Read state/regime_exposure_state.json.

        Returns parsed dict, or empty dict on failure.
        """
        path = os.path.join(self._data_dir, "regime_exposure_state.json")
        if not os.path.isfile(path):
            logger.debug("Regime state file not found: %s", path)
            return {}

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to read regime state: %s", exc)
            return {}

    def _read_ledger(self) -> list[dict[str, Any]]:
        """
        Read state/execution_ledger.jsonl.

        Returns list of parsed dicts, or empty list on failure.
        """
        path = os.path.join(self._data_dir, "execution_ledger.jsonl")
        if not os.path.isfile(path):
            logger.debug("Execution ledger not found: %s", path)
            return []

        entries: list[dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        entries.append(json.loads(stripped))
        except Exception as exc:
            logger.warning("Failed to parse execution ledger: %s", exc)
            return []

        return entries

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _compute_rsi(self, symbol: str, period: int = 14) -> tuple:
        """Compute RSI and ATR from live MT5 1-minute rates.

        Returns: (rsi, atr) where both are floats or None on failure.
        """
        try:
            import MetaTrader5 as mt5
            rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, period + 1)
            if rates is None or len(rates) < period + 1:
                return None, None
            closes = [r[4] for r in rates]
            gains, losses = [], []
            for i in range(1, len(closes)):
                diff = closes[i] - closes[i - 1]
                gains.append(max(diff, 0))
                losses.append(max(-diff, 0))
            avg_g = sum(gains) / len(gains)
            avg_l = sum(losses) / len(losses)
            rsi = 100 - 100 / (1 + avg_g / avg_l) if avg_l else 50.0
            # ATR from high-low
            hl = [abs(r[2] - r[3]) for r in rates[1:]]
            atr = sum(hl) / len(hl) if hl else None
            return round(rsi, 1), atr
        except Exception as exc:
            logger.debug("RSI compute failed for %s: %s", symbol, exc)
            return None, None

    def _build_market_state(
        self,
        mt5_state: dict[str, Any],
        pipeline_trace: dict[str, Any],
        regime_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the market_state section of the unified state."""
        symbols: dict[str, dict[str, Any]] = {}

        ticks = mt5_state.get("ticks", {})

        # Build symbol data from ticks (bid/ask/spread) and pipeline trace (rsi/atr)
        md_summary = pipeline_trace.get("md_summary", {}) if pipeline_trace else {}

        for sym in self._active_symbols:
            tick = ticks.get(sym, {})
            md = md_summary.get(sym, {}) if isinstance(md_summary, dict) else {}

            rsi = md.get("rsi")
            atr = md.get("atr")
            # Fall back to live computation if pipeline trace missing RSI
            if rsi is None:
                computed_rsi, computed_atr = self._compute_rsi(sym)
                if computed_rsi is not None:
                    rsi = computed_rsi
                    atr = atr or computed_atr

            # SIL score enrichment (best-effort)
            sil_score: Optional[float] = None
            try:
                report = getattr(self, "_sil_report", None)
                if report is None and self._symbol_selector is not None:
                    report = self._symbol_selector.get_full_report()
                    self._sil_report = report
                if report:
                    for scored in report.get("scored_symbols", []):
                        if scored.get("symbol") == sym:
                            sil_score = scored.get("total_score")
                            break
            except Exception:
                pass

            symbols[sym] = {
                "rsi": rsi,
                "atr": atr,
                "bid": tick.get("bid"),
                "ask": tick.get("ask"),
                "spread": tick.get("spread"),
                "timestamp": tick.get("time"),
                "sil_score": sil_score,
                "signal_state": None,
                "has_position": False,
            }

        # Derive market regime from regime_state
        market_regime: str = "UNKNOWN"
        regime_phase: str = "unknown"

        if regime_state and isinstance(regime_state, dict):
            # Attempt to infer regime from trend bins across symbols
            per_symbol = regime_state.get("per_symbol", {})
            if per_symbol and isinstance(per_symbol, dict):
                # Count how many symbols are trending vs ranging
                trending_count = 0
                ranging_count = 0
                for sym_data in per_symbol.values():
                    if not isinstance(sym_data, dict):
                        continue
                    trend_bins = sym_data.get("trend_bins", {})
                    if not isinstance(trend_bins, dict):
                        continue
                    bullish_total = trend_bins.get("bullish", 0)
                    bearish_total = trend_bins.get("bearish", 0)
                    neutral_total = trend_bins.get("neutral", 0)
                    total = bullish_total + bearish_total + neutral_total
                    if total > 0:
                        # If >60% of samples show direction, consider trending
                        directional_ratio = (
                            bullish_total + bearish_total
                        ) / total
                        if directional_ratio > 0.6:
                            trending_count += 1
                        else:
                            ranging_count += 1

                if trending_count > ranging_count:
                    market_regime = "TRENDING"
                    regime_phase = "trending"
                elif ranging_count >= trending_count and ranging_count > 0:
                    market_regime = "RANGING"
                    regime_phase = "ranging"

                # Check ATR percentile for volatility signal
                high_vol_count = 0
                for sym_data in per_symbol.values():
                    if not isinstance(sym_data, dict):
                        continue
                    atr_bins = sym_data.get("atr_percentile_bins", {})
                    if not isinstance(atr_bins, dict):
                        continue
                    high = atr_bins.get("80-100", 0)
                    if high > 0:
                        high_vol_count += 1

                if high_vol_count >= 2:
                    regime_phase = "volatile"

        return {
            "symbols": symbols,
            "market_regime": market_regime,
            "regime_phase": regime_phase,
        }

    def _build_execution_state(
        self,
        mt5_state: dict[str, Any],
        pipeline_trace: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the execution_state section of the unified state."""
        open_positions = mt5_state.get("positions", [])

        # ALWAYS trust MT5 API for position count — never fall back to pipeline trace
        trace_positions = len(open_positions)

        governor = pipeline_trace.get("governor", {}) if pipeline_trace else {}
        segl_state = governor.get(
            "segl_state", "OBSERVE"
        ) if governor else "OBSERVE"

        execution = (
            pipeline_trace.get("execution", {}) if pipeline_trace else {}
        )
        execution_decision = execution.get("decision", "HOLD")
        denial_reason = execution.get("denial_reason")

        governor_authorized = governor.get("authorized", False) if governor else False

        # Count active signals from signals_detail
        signals_detail = (
            pipeline_trace.get("signals_detail", []) if pipeline_trace else []
        )
        active_signals = len(signals_detail) if signals_detail else 0

        return {
            "cycle": pipeline_trace.get("cycle", 0) if pipeline_trace else 0,
            "segl_state": segl_state,
            "open_positions": len(open_positions),
            "open_positions_detail": open_positions,
            "active_signals": active_signals,
            "execution_decision": execution_decision,
            "denial_reason": denial_reason,
            "governor_authorized": governor_authorized,
        }

    def _build_risk_state(
        self,
        mt5_state: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the risk_state section of the unified state."""
        account = mt5_state.get("account")
        balance = 0.0
        equity = 0.0
        free_margin = 0.0
        margin_level = 0.0

        if account:
            balance = float(account.get("balance", 0))
            equity = float(account.get("equity", 0))
            free_margin = float(account.get("margin_free", 0))
            margin_level = float(account.get("margin_level", 0))

        # Drawdown from peak (simple: compare current equity to balance as reference)
        peak_equity = max(balance, equity)
        drawdown_pct = 0.0
        if peak_equity > 0:
            drawdown_pct = round(
                (peak_equity - equity) / peak_equity * 100, 2
            )

        # Read circuit_breaker_state.json
        circuit_breaker = self._read_circuit_breaker_state()

        # Daily loss estimate: sum of position profits (negative = loss)
        positions = mt5_state.get("positions", [])
        daily_loss = 0.0
        for p in positions:
            profit = p.get("profit", 0) or 0
            if profit < 0:
                daily_loss += profit
        daily_loss = round(daily_loss, 2)

        return {
            "balance": balance,
            "equity": equity,
            "free_margin": free_margin,
            "margin_level": margin_level,
            "drawdown_pct": drawdown_pct,
            "circuit_breaker": circuit_breaker,
            "daily_loss": daily_loss,
        }

    def _build_performance_state(
        self,
        mt5_state: dict[str, Any],
        ledger: list[dict[str, Any]],
        open_positions_detail: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build the performance_state section of the unified state."""
        perf = self._compute_performance(ledger, open_positions_detail)
        return {
            "total_trades": perf["total_trades"],
            "win_rate": perf["win_rate"],
            "total_pnl": perf["total_pnl"],
            "largest_winner": perf["largest_winner"],
            "largest_loser": perf["largest_loser"],
        }

    def _build_pipeline_state(
        self,
        pipeline_trace: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the pipeline_state section of the unified state."""
        funnel = (
            pipeline_trace.get("pipeline_funnel", {})
            if pipeline_trace
            else {}
        )

        governor = pipeline_trace.get("governor", {}) if pipeline_trace else {}

        return {
            "cycle": pipeline_trace.get("cycle", 0) if pipeline_trace else 0,
            "signal_generated": pipeline_trace.get("signals_generated", 0)
            if pipeline_trace
            else 0,
            "threshold_pass": pipeline_trace.get("threshold_pass_count", 0)
            if pipeline_trace
            else 0,
            "confirm_pass": pipeline_trace.get("confirm_gate", {}).get(
                "confirm_pass", False
            )
            if pipeline_trace
            else False,
            "governor_ready": funnel.get("governor_ready", 0),
            "vel_blocked": funnel.get("vel_blocked", 0),
            "executed": funnel.get("executed", 0),
        }

    def _build_system_health(
        self,
        mt5_connected: bool,
        data_sources: dict[str, bool],
        pipeline_trace: dict[str, Any],
        build_ts: float,
    ) -> dict[str, Any]:
        """Build the system_health section of the unified state."""
        uptime_cycles = pipeline_trace.get("cycle", 0) if pipeline_trace else 0

        return {
            "mt5_connected": mt5_connected,
            "last_error": self._last_error,
            "uptime_cycles": uptime_cycles,
            "build_timestamp": build_ts,
            "data_sources": data_sources,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_performance(
        self,
        ledger: list[dict[str, Any]],
        positions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Compute win_rate, total_pnl, largest_winner, largest_loser.

        Uses:
          - closed trades from the ledger (trade_closed events with pnl)
          - current open positions for unrealised PnL
        """
        total_trades = 0
        wins = 0
        total_pnl = 0.0
        largest_winner = 0.0
        largest_loser = 0.0

        # Closed trades from ledger
        closed_pnls: list[float] = []
        for entry in ledger:
            if not isinstance(entry, dict):
                continue
            event_type = entry.get("event_type", "")
            if event_type == "trade_closed":
                pnl = entry.get("pnl")
                if pnl is not None:
                    pnl_val = float(pnl)
                    closed_pnls.append(pnl_val)
                    total_pnl += pnl_val
                    if pnl_val > 0:
                        wins += 1
                        if pnl_val > largest_winner:
                            largest_winner = pnl_val
                    else:
                        if pnl_val < largest_loser:
                            largest_loser = pnl_val

        # Current open positions unrealised PnL
        for pos in positions:
            if not isinstance(pos, dict):
                continue
            profit = pos.get("profit", 0) or 0
            swap = pos.get("swap", 0) or 0
            pnl_val = float(profit) + float(swap)
            total_pnl += pnl_val
            if pnl_val > 0:
                wins += 1  # Counting in-the-money positions as "wins" contextually
                if pnl_val > largest_winner:
                    largest_winner = pnl_val
            else:
                if pnl_val < largest_loser:
                    largest_loser = pnl_val

        total_trades = len(closed_pnls) + len(positions)
        total_closed = len(closed_pnls)
        win_rate = round(wins / total_closed, 4) if total_closed > 0 else 0.0

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": round(total_pnl, 2),
            "largest_winner": round(largest_winner, 2),
            "largest_loser": round(largest_loser, 2),
        }

    def _read_circuit_breaker_state(self) -> str:
        """
        Read circuit_breaker_state.json and return its status string.

        Returns "CLOSED" if file missing or not triggered.
        """
        path = os.path.join(self._data_dir, "circuit_breaker_state.json")
        if not os.path.isfile(path):
            return "CLOSED"

        try:
            with open(path, "r", encoding="utf-8") as f:
                cb = json.load(f)
            if cb.get("triggered", False):
                return "TRIGGERED"
            return "CLOSED"
        except Exception as exc:
            logger.debug("Failed to read circuit breaker state: %s", exc)
            return "CLOSED"

    def _resolve_data_dir(self, data_dir: str) -> str:
        """
        Resolve the data directory path.

        If the given path is absolute, use it as-is.
        Otherwise, resolve relative to the project root
        (three levels up from this file's directory).
        """
        if os.path.isabs(data_dir):
            return data_dir

        # Resolve from this file: dashboard/unified_state_builder.py
        # → proxima_x/proxima_ops/dashboard/ → go up 3 levels
        this_dir = os.path.dirname(os.path.abspath(__file__))
        resolved = os.path.normpath(
            os.path.join(this_dir, "..", "..", "..", data_dir)
        )
        return resolved
