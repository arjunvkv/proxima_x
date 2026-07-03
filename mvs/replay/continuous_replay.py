from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from mvs.reconstruction.mt5_history_loader import MT5HistoryLoader
from mvs.adaptation.observer_decay import ObserverDecayEngine
from mvs.adaptation.weak_day_detector import WeakDayDetector
from mvs.observer.observer_features import (
    normalize_tpi, persistence_ratio_from_streak,
    curvature_strength_from_state, compute_entropy_alignment,
    compute_confidence, state_from_confidence,
)
from data.tick_buffer import TickBuffer
from layer7.tpi_outcomes import TPIPersistenceTracker, TPICurvatureTracker
from layer7.entropy_compression import EntropyCompressionEngine

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")

SYMBOL_TRUST = {
    "CADJPY": 1.00, "EURJPY": 0.88, "EURGBP": 0.86,
    "USDJPY": 0.52, "XAUUSD": 0.34, "GBPJPY": 0.29,
}

CYCLE_TICKS = 128


class ContinuousReplay:
    def __init__(self, half_life_ticks: int = 600):
        self._loader = MT5HistoryLoader()
        self._tpi_persistence = TPIPersistenceTracker()
        self._tpi_curvature = TPICurvatureTracker()
        self._entropy = EntropyCompressionEngine()
        self._weak_day = WeakDayDetector()
        self._decay = ObserverDecayEngine(
            half_life_ticks=half_life_ticks,
            execute_threshold=0.75)
        self._tpi_buffer = TickBuffer()
        self._signal_ids: Dict[str, str] = {}
        self._observer_seq: Dict[str, int] = defaultdict(int)
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def run(self) -> Dict:
        print("Loading positions from MT5...")
        positions = self._loader.load_positions(days_back=90)
        print(f"Loaded {len(positions)} positions")

        sorted_pos = sorted(positions, key=lambda p: p.entry_time)
        results: List[Dict] = []

        for i, pos in enumerate(sorted_pos):
            if (i + 1) % 25 == 0:
                print(f"  Processing {i+1}/{len(sorted_pos)}...")

            result = self._replay_position(pos)
            if result is not None:
                results.append(result)

        return self._report(results)

    def _replay_position(self, pos) -> Optional[Dict]:
        tick_path = self._load_ticks(pos)
        if tick_path is None or len(tick_path) < 50:
            return None

        entry_ts = int(pos.entry_time.timestamp() * 1_000_000)
        sim_cycle_ticks = 0
        entry_eval: Optional[Dict] = None

        pre_warm_end = 0
        for i in range(1, len(tick_path)):
            ts = int(tick_path[i, 5] * 1000)
            if ts < entry_ts:
                bid = float(tick_path[i, 1])
                ask = float(tick_path[i, 2])
                self._tpi_buffer.append(pos.symbol, bid, ask, ts // 1_000_000)
                pre_warm_end = i
            else:
                break

        for i in range(pre_warm_end + 1, len(tick_path)):
            bid = float(tick_path[i, 1])
            ask = float(tick_path[i, 2])
            spread = ask - bid
            ts = int(tick_path[i, 5] * 1000)

            self._tpi_buffer.append(pos.symbol, bid, ask, ts // 1_000_000)
            self._weak_day.record_spread(spread)

            if i % CYCLE_TICKS == 0:
                sim_cycle_ticks += 1
                tpi_raw = self._tpi_buffer.get_tpi(pos.symbol)
                if tpi_raw is None:
                    continue
                tpi_val = tpi_raw["tpi"]
                d = 1 if tpi_val > 0 else (-1 if tpi_val < 0 else 0)
                self._tpi_persistence.update(pos.symbol, tpi_val, d)
                self._tpi_curvature.update(pos.symbol, tpi_val)
                self._entropy.update(pos.symbol, tpi_val)
                entropy_state = self._entropy.compute_state(pos.symbol) or {}
                ent = entropy_state.get("normalized_entropy", 0.5)
                pers_state = self._tpi_persistence.state(pos.symbol)
                curv_state = self._tpi_curvature.state(pos.symbol).get("state", "NEUTRAL")
                sig_conf = abs(tpi_val)
                regime_hash = 0

                ntpi = normalize_tpi(sig_conf)
                pers = persistence_ratio_from_streak(pers_state.get("streak", 0))
                curv = curvature_strength_from_state(curv_state)
                ent_align = compute_entropy_alignment(ent, max_entropy=1.0)

                conf = compute_confidence(ntpi, pers, curv, ent_align)
                state = state_from_confidence(conf)
                reality = min(1.0, max(0.0, conf + 0.1))

                signal_id = self._signal_ids.get(pos.symbol)
                if not signal_id:
                    self._observer_seq[pos.symbol] += 1
                    signal_id = f"{pos.symbol}:{self._observer_seq[pos.symbol]}"
                    self._signal_ids[pos.symbol] = signal_id
                    self._decay.birth(signal_id, reality, pers_state.get("streak", 0), ent, "NEUTRAL")
                    self._decay.tick(1)
                    decayed = self._decay.compute(signal_id, pers_state.get("streak", 0), ent, "NEUTRAL")
                else:
                    self._decay.tick(1)
                    decayed = self._decay.compute(signal_id, pers_state.get("streak", 0), ent, "NEUTRAL")

                self._weak_day.record_entropy_compression(ent < 0.3, curv_state == "ACCELERATION")
                self._weak_day.record_persistence_event(pers_state.get("streak", 0), pers_state.get("streak", 0) < 2)
                self._weak_day.record_regime(regime_hash)
                self._weak_day.tick()
                wds_result = self._weak_day.compute()

                final = decayed["confidence"]
                final *= wds_result["trade_multiplier"]
                final *= SYMBOL_TRUST.get(pos.symbol, 0.75)
                final = max(0.0, min(1.0, final))

                if ts >= entry_ts and entry_eval is None:
                    entry_eval = {
                        "state": state,
                        "decayed_state": decayed["state"],
                        "confidence": float(conf),
                        "decayed_confidence": float(decayed["confidence"]),
                        "reality_score": float(final),
                        "wds_multiplier": float(wds_result["trade_multiplier"]),
                        "wds_score": float(wds_result["weak_day_score"]),
                        "wds_fragility": wds_result["fragility_class"],
                        "tpi_confidence": float(sig_conf),
                        "persistence_streak": pers_state.get("streak", 0),
                        "curvature_state": curv_state,
                        "entropy": float(ent),
                        "ntpi": float(ntpi),
                        "persistence_norm": float(pers),
                        "curvature_norm": float(curv),
                        "entropy_alignment": float(ent_align),
                        "temporal_decay": float(decayed.get("temporal", 1.0)),
                        "survival": float(decayed.get("survival", 1.0)),
                        "survival_p": float(decayed.get("survival_components", {}).get("persistence", 1.0)),
                        "survival_e": float(decayed.get("survival_components", {}).get("entropy", 1.0)),
                        "survival_r": float(decayed.get("survival_components", {}).get("regime", 1.0)),
                        "total_cycles": sim_cycle_ticks,
                    }

        if entry_eval is None:
            return None

        is_win = pos.net_pnl > 0
        return {
            "position_id": pos.position_id,
            "symbol": pos.symbol,
            "actual_pnl": round(pos.net_pnl, 2),
            "actual_win": is_win,
            **entry_eval,
        }

    def _load_ticks(self, pos) -> Optional[np.ndarray]:
        exit_time = pos.final_exit_time
        try:
            if exit_time:
                return self._loader.load_tick_path(
                    pos.symbol, pos.entry_time, exit_time,
                    pre_buffer_minutes=30)
            else:
                return self._loader.load_tick_path(
                    pos.symbol, pos.entry_time, None,
                    pre_buffer_minutes=30)
        except Exception:
            return None

    def _report(self, results: List[Dict]) -> Dict:
        self._write_csv(results)

        total = len(results)
        winners = [r for r in results if r["actual_win"]]
        losers = [r for r in results if not r["actual_win"]]

        def avg(seq, key):
            vals = [r[key] for r in seq]
            return sum(vals) / max(len(vals), 1)

        def correlation(xs, ys):
            if len(xs) < 3:
                return 0.0
            xa = np.array(xs)
            ya = np.array(ys)
            if np.std(xa) < 1e-10 or np.std(ya) < 1e-10:
                return 0.0
            return float(np.corrcoef(xa, ya)[0, 1])

        reality_w = avg(winners, "reality_score") if winners else 0
        reality_l = avg(losers, "reality_score") if losers else 0
        conf_w = avg(winners, "confidence") if winners else 0
        conf_l = avg(losers, "confidence") if losers else 0
        all_reality = [r["reality_score"] for r in results]
        all_pnls = [r["actual_pnl"] for r in results]
        all_tpi = [r["tpi_confidence"] for r in results]
        reality_pnl_corr = correlation(all_reality, all_pnls)
        tpi_pnl_corr = correlation(all_tpi, all_pnls)

        state_dist = defaultdict(int)
        decayed_dist = defaultdict(int)
        for r in results:
            state_dist[r["state"]] += 1
            decayed_dist[r["decayed_state"]] += 1

        wds_states = defaultdict(int)
        for r in results:
            wds_states[r["wds_fragility"]] += 1

        symbol_summary = defaultdict(lambda: {
            "total": 0, "wins": 0, "pnls": [], "reality_scores": [], "confidences": []})
        for r in results:
            sym = r["symbol"]
            symbol_summary[sym]["total"] += 1
            symbol_summary[sym]["wins"] += 1 if r["actual_win"] else 0
            symbol_summary[sym]["pnls"].append(r["actual_pnl"])
            symbol_summary[sym]["reality_scores"].append(r["reality_score"])
            symbol_summary[sym]["confidences"].append(r["confidence"])

        sym_out = {}
        for sym, st in sorted(symbol_summary.items()):
            rc = correlation(st["reality_scores"], st["pnls"])
            sym_out[sym] = {
                "total": st["total"],
                "wr": round(st["wins"] / max(st["total"], 1), 4),
                "avg_reality": round(sum(st["reality_scores"]) / max(len(st["reality_scores"]), 1), 4),
                "avg_confidence": round(sum(st["confidences"]) / max(len(st["confidences"]), 1), 4),
                "reality_pnl_corr": round(rc, 4),
            }

        summary = {
            "total_positions": total,
            "win_rate": round(len(winners) / max(total, 1), 4),
            "avg_confidence": round(avg(results, "confidence"), 4),
            "avg_reality": round(avg(results, "reality_score"), 4),
            "avg_reality_winner": round(reality_w, 4),
            "avg_reality_loser": round(reality_l, 4),
            "avg_confidence_winner": round(conf_w, 4),
            "avg_confidence_loser": round(conf_l, 4),
            "reality_pnl_correlation": round(reality_pnl_corr, 4),
            "tpi_pnl_correlation": round(tpi_pnl_corr, 4),
            "state_distribution": dict(state_dist),
            "decayed_state_distribution": dict(decayed_dist),
            "wds_fragility_distribution": dict(wds_states),
            "symbols": sym_out,
        }

        report_path = os.path.join(OUTPUT_DIR, "continuous_replay_report.json")
        with open(report_path, "w") as f:
            json.dump(summary, f, indent=2)

        self._print(summary)
        return summary

    def _write_csv(self, results: List[Dict]) -> None:
        path = os.path.join(OUTPUT_DIR, "continuous_replay_results.csv")
        with open(path, "w") as f:
            cols = ["position_id","symbol","actual_pnl","actual_win",
                    "state","decayed_state","confidence","decayed_confidence",
                    "reality_score","wds_multiplier","wds_score","wds_fragility",
                    "tpi_confidence","persistence_streak","curvature_state",
                    "entropy","ntpi","persistence_norm","curvature_norm",
                    "entropy_alignment","temporal_decay","survival",
                    "survival_p","survival_e","survival_r","total_cycles"]
            f.write(",".join(cols) + "\n")
            for r in results:
                f.write(f"{r['position_id']},{r['symbol']},{r['actual_pnl']},"
                        f"{int(r['actual_win'])},{r['state']},{r['decayed_state']},"
                        f"{r['confidence']},{r['decayed_confidence']},"
                        f"{r['reality_score']},{r['wds_multiplier']},"
                        f"{r['wds_score']},{r['wds_fragility']},"
                        f"{r['tpi_confidence']},{r['persistence_streak']},"
                        f"{r['curvature_state']},{r['entropy']},{r['ntpi']},"
                        f"{r['persistence_norm']},{r['curvature_norm']},"
                        f"{r['entropy_alignment']},{r['temporal_decay']},"
                        f"{r['survival']},{r['survival_p']},{r['survival_e']},"
                        f"{r['survival_r']},{r['total_cycles']}\n")
        print(f"Results written to {path}")

    def _print(self, s: Dict) -> None:
        print("=" * 60)
        print("CONTINUOUS REPLAY — REALITY vs OUTCOME")
        print("=" * 60)
        print(f"Positions:             {s['total_positions']}")
        print(f"Win rate:              {s['win_rate']:.1%}")
        print(f"Avg confidence:        {s['avg_confidence']:.4f}")
        print(f"Avg reality score:     {s['avg_reality']:.4f}")
        print(f"Avg reality (winners): {s['avg_reality_winner']:.4f}")
        print(f"Avg reality (losers):  {s['avg_reality_loser']:.4f}")
        print(f"Reality–PnL corr:      {s['reality_pnl_correlation']:.4f}")
        print(f"TPI–PnL corr:          {s['tpi_pnl_correlation']:.4f}")
        print("-" * 60)
        print("State distribution (pre-decay):")
        for st, cnt in sorted(s["state_distribution"].items(), key=lambda x: -x[1]):
            print(f"  {st:20s}: {cnt:3d}")
        print("Decayed state distribution:")
        for st, cnt in sorted(s["decayed_state_distribution"].items(), key=lambda x: -x[1]):
            print(f"  {st:20s}: {cnt:3d}")
        print("WDS fragility:")
        for fg, cnt in sorted(s["wds_fragility_distribution"].items(), key=lambda x: -x[1]):
            print(f"  {fg:20s}: {cnt:3d}")
        print("-" * 60)
        print("Symbol analysis:")
        for sym, sa in sorted(s["symbols"].items(), key=lambda x: -abs(x[1]["reality_pnl_corr"])):
            print(f"  {sym:8s}: corr={sa['reality_pnl_corr']:+.4f}  "
                  f"reality={sa['avg_reality']:.3f}  conf={sa['avg_confidence']:.3f}  "
                  f"wr={sa['wr']:.0%}  ({sa['total']} trades)")
        print("-" * 60)
        passes = [
            ("reality_pnl_correlation > 0.15", s["reality_pnl_correlation"] > 0.15),
            ("avg_reality_winner > avg_reality_loser", s["avg_reality_winner"] > s["avg_reality_loser"]),
            ("avg_confidence > 0.30", s["avg_confidence"] > 0.30),
            ("avg_reality > 0.15", s["avg_reality"] > 0.15),
            ("less than 50% WDS=WEAK", s["wds_fragility_distribution"].get("WEAK", 0) / max(s["total_positions"], 1) < 0.50),
        ]
        for label, passed in passes:
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
        all_pass = all(p for _, p in passes)
        print("=" * 60)
        print(f"VERDICT: {'PASS — Monday ready' if all_pass else 'FAIL — tune before Monday'}")
        print("=" * 60)


if __name__ == "__main__":
    cr = ContinuousReplay()
    cr.run()
