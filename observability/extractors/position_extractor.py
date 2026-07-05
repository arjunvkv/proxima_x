from typing import Optional
from ..schema.telemetry_schema import PositionSnapshot

class PositionExtractor:
    """
    Extracts all position and trade data from ProximaDemo.
    Pure extraction — NO string building, NO printing.
    """

    def __init__(self, demo):
        self._demo = demo

    def extract_open_positions(self) -> list[PositionSnapshot]:
        """
        Extract all currently open positions with full context.
        Mirrors the POSITIONS section of _print_dashboard().
        """
        positions = []
        open_positions = getattr(self._demo.positions, 'positions', []) or []

        for pos in open_positions:
            ticket = pos.get("ticket")
            meta = self._demo._active_positions_metadata.get(ticket, {})

            # Calculate bars elapsed
            entry_bar = meta.get("entry_bar_time")
            broker_sym = self._demo.mt5._get_broker_symbol(pos["symbol"])
            elapsed = self._demo._bars_elapsed(entry_bar, broker_sym) if hasattr(self._demo, '_bars_elapsed') else -1

            # Get trade context from eval_data
            sym_data = {}
            if hasattr(self._demo, '_current_eval_data'):
                sym_data = self._demo._current_eval_data.get(pos["symbol"], {})

            positions.append(PositionSnapshot(
                ticket=int(ticket) if ticket else 0,
                symbol=str(pos.get("symbol", "?")),
                side=str(pos.get("type", "?")),
                volume=float(pos.get("volume", 0.0)),
                entry_price=float(pos.get("price_open", 0.0)),
                current_price=float(pos.get("price_current", 0.0)),
                profit=float(pos.get("profit", 0.0)),
                bars_elapsed=max(0, elapsed) if elapsed >= 0 else -1,
                entry_es_rank=meta.get("entry_es_rank"),
                entry_at_rank=meta.get("entry_at_rank"),
                econ_ratio=sym_data.get("econ_ratio"),
                expected_move=sym_data.get("expected_move"),
                trigger_count_while_open=int(meta.get("trigger_count_while_open", 0)),
            ))

        return positions

    def extract_trade_context(self, ticket: int) -> dict:
        """Extract full trade context for a specific ticket."""
        meta = self._demo._active_positions_metadata.get(ticket, {})
        exit_state = getattr(self._demo, '_exit_state', {}).get(ticket, {})

        result = {
            "entry_bar_time": meta.get("entry_bar_time"),
            "entry_es_rank": meta.get("entry_es_rank"),
            "entry_at_rank": meta.get("entry_at_rank"),
            "entry_price": meta.get("entry_price"),
            "min_price": meta.get("min_price"),
            "max_price": meta.get("max_price"),
            "trigger_count": meta.get("trigger_count_while_open", 0),
            "direction": meta.get("direction"),
            "volume": meta.get("volume"),
            "expected_exit_reason": meta.get("expected_exit_reason"),
        }

        if exit_state:
            result["exit"] = {
                "direction": exit_state.get("direction"),
                "entry_tpi": exit_state.get("entry_tpi"),
                "inversion_count": exit_state.get("inversion_count"),
                "symbol": exit_state.get("symbol"),
            }

        return result

    def extract_account_positions_summary(self) -> dict:
        """Extract summary stats about open positions."""
        positions = self.extract_open_positions()
        return {
            "total_open": len(positions),
            "total_pnl": sum(p.profit for p in positions),
            "long_count": sum(1 for p in positions if p.side == "BUY"),
            "short_count": sum(1 for p in positions if p.side == "SELL"),
            "long_pnl": sum(p.profit for p in positions if p.side == "BUY"),
            "short_pnl": sum(p.profit for p in positions if p.side == "SELL"),
            "symbols": [p.symbol for p in positions],
        }

    def extract_account(self) -> dict:
        """Extract MT5 account info."""
        info = self._demo.mt5.get_account() or {}
        return {
            "login": str(info.get("login", "N/A")),
            "balance": float(info.get("balance", 0.0)),
            "equity": float(info.get("equity", 0.0)),
            "margin": float(info.get("margin", 0.0)),
            "profit": float(info.get("profit", 0.0)),
            "margin_level": float(info.get("margin_level", 0.0)),
        }
