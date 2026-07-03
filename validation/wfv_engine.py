from typing import Dict, List, Any


class WalkForwardValidator:
    def __init__(self, train_size: int = 200, test_size: int = 50):
        self.train_size = train_size
        self.test_size = test_size

    def run(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(records) < self.train_size + self.test_size:
            return {"error": f"Need >= {self.train_size + self.test_size} records, got {len(records)}", "accuracy": 0.5, "pnl_proxy": 0.0, "edge_detected": False}

        results = []
        i = 0
        while i + self.train_size + self.test_size <= len(records):
            test = records[i + self.train_size:i + self.train_size + self.test_size]
            preds = [r.get("signal", 0) for r in test]
            outcomes = [r.get("outcome", 0) for r in test]
            results.append({
                "accuracy": self._accuracy(preds, outcomes),
                "pnl_proxy": self._pnl(preds, outcomes),
                "window_start": i + self.train_size,
                "window_end": i + self.train_size + self.test_size,
            })
            i += self.test_size

        return self._summary(results)

    def _accuracy(self, preds: List[int], outcomes: List[float]) -> float:
        correct = 0
        total = 0
        for p, o in zip(preds, outcomes):
            if p == 0:
                continue
            total += 1
            if (p > 0 and o > 0) or (p < 0 and o < 0):
                correct += 1
        return correct / total if total else 0.0

    def _pnl(self, preds: List[int], outcomes: List[float]) -> float:
        return sum(p * o for p, o in zip(preds, outcomes))

    def _summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not results:
            return {"accuracy": 0.5, "pnl_proxy": 0.0, "edge_detected": False}
        acc = sum(r["accuracy"] for r in results) / len(results)
        pnl = sum(r["pnl_proxy"] for r in results)
        return {
            "accuracy": round(acc, 4),
            "pnl_proxy": round(pnl, 4),
            "windows": len(results),
            "total_records": self.train_size + self.test_size * len(results),
            "edge_detected": acc > 0.55 and pnl > 0,
            "windows_detail": results,
        }


class StatisticalEdgeTest:
    @staticmethod
    def run(results: Dict[str, Any]) -> Dict[str, Any]:
        acc = results.get("accuracy", 0.5)
        pnl = results.get("pnl_proxy", 0.0)
        return {
            "accuracy": acc,
            "pnl_proxy": pnl,
            "edge_detected": acc > 0.55 and pnl > 0,
            "marginal": 0.55 < acc <= 0.60,
            "strong": acc > 0.60,
            "verdict": "EDGE DETECTED" if (acc > 0.55 and pnl > 0) else "NO EDGE OR NOISE",
        }
