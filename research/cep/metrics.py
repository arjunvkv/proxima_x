"""Economic metrics for CEP simulation."""
import math


class CEPMetrics:
    def compute(self, trades: list[dict]) -> dict:
        if not trades:
            return {
                "n_trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "expectancy": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                "sharpe": 0.0, "total_pnl": 0.0,
                "monetization_ratio": 0.0, "friction_collapse_point": 0.0,
            }

        wins = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
        losses = [t["net_pnl"] for t in trades if t["net_pnl"] < 0]
        n_wins = len(wins)
        n_losses = len(losses)
        total = len(trades)

        win_rate = n_wins / total if total else 0.0
        avg_win = sum(wins) / n_wins if n_wins else 0.0
        avg_loss = sum(losses) / n_losses if n_losses else 0.0
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss else 0.0
        expectancy = (win_rate * avg_win + (1 - win_rate) * avg_loss) if total else 0.0
        total_pnl = gross_profit - gross_loss

        pnls = [t["net_pnl"] for t in trades]
        sharpe = self._sharpe(pnls)

        round_trip_cost = trades[0].get("spread_cost", 0.001) if trades else 0.001
        monetization_ratio = avg_win / round_trip_cost if round_trip_cost > 0 and avg_win > 0 else 0.0

        return {
            "n_trades": total,
            "win_rate": round(win_rate, 4),
            "profit_factor": round(profit_factor, 4),
            "expectancy": round(expectancy, 6),
            "avg_win": round(avg_win, 6),
            "avg_loss": round(avg_loss, 6),
            "sharpe": round(sharpe, 4),
            "total_pnl": round(total_pnl, 6),
            "monetization_ratio": round(monetization_ratio, 4),
        }

    def _sharpe(self, pnls: list[float]) -> float:
        if len(pnls) < 2:
            return 0.0
        mean = sum(pnls) / len(pnls)
        var = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
        if var <= 0:
            return 0.0
        return mean / math.sqrt(var) * math.sqrt(len(pnls))
