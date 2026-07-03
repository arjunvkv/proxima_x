import csv
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.experiment_config import EXPERIMENTS
from research.layer_ablation_runner import run_ablation
from replay.environment import ReplayConfig


def build_matrix(replay_config: ReplayConfig):
    baseline = run_ablation(EXPERIMENTS["FULL_V4"], replay_config)
    print(f"[BASELINE FULL_V4] PnL={baseline['net_pnl']:.2f} PF={baseline['profit_factor']:.4f} "
          f"WR={baseline['win_rate']:.4f} Trades={baseline['trade_count']} "
          f"ExecRate={baseline['execution_rate']:.4f}")

    results = []
    for name, cfg in EXPERIMENTS.items():
        result = run_ablation(cfg, replay_config)
        result["experiment"] = name
        result["delta_pnl"] = result["net_pnl"] - baseline["net_pnl"]
        result["delta_pf"] = result["profit_factor"] - baseline["profit_factor"]
        result["delta_wr"] = result["win_rate"] - baseline["win_rate"]
        result["delta_trades"] = result["trade_count"] - baseline["trade_count"]
        result["delta_exec_rate"] = result["execution_rate"] - baseline["execution_rate"]
        results.append(result)
        print(f"[{name}] PnL={result['net_pnl']:.2f} ({result['delta_pnl']:+.2f}) "
              f"PF={result['profit_factor']:.4f} ({result['delta_pf']:+.4f}) "
              f"WR={result['win_rate']:.4f} Trades={result['trade_count']}")

    return results


def export_csv(results, path="research/results/layer_matrix.csv"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"[EXPORTED] {path}")


if __name__ == "__main__":
    cfg = ReplayConfig(
        symbols=["EURJPY", "USDJPY"],
        start="2026-03-12",
        end="2026-05-12",
        speed=1000.0,
        burst=True,
    )
    results = build_matrix(cfg)
    export_csv(results)
