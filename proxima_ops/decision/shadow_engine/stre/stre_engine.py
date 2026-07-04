import json
import numpy as np
import os
from collections import deque


STATE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")


class STREngine:
    def __init__(self, window=100):
        self.window = window
        self.gt_buffer = deque(maxlen=window)
        self.sy_buffer = deque(maxlen=window)
        self.pnl_buffer = deque(maxlen=window)

    def ingest(self, gt_similarity, sy_similarity, pnl_delta):
        self.gt_buffer.append(gt_similarity)
        self.sy_buffer.append(sy_similarity)
        self.pnl_buffer.append(pnl_delta)

    def _corr(self, a, b):
        if len(a) < 10:
            return 0.0
        try:
            return float(np.corrcoef(a, b)[0][1])
        except Exception:
            return 0.0

    def compute(self):
        return {
            "gt_corr": round(self._corr(list(self.gt_buffer), list(self.pnl_buffer)), 4),
            "sy_corr": round(self._corr(list(self.sy_buffer), list(self.pnl_buffer)), 4),
            "stas": round(
                self._corr(list(self.gt_buffer), list(self.pnl_buffer))
                - self._corr(list(self.sy_buffer), list(self.pnl_buffer)),
                4,
            ),
            "winner": "GT" if self._corr(list(self.gt_buffer), list(self.pnl_buffer))
                       > self._corr(list(self.sy_buffer), list(self.pnl_buffer))
                       else "SY_LEGACY",
            "samples": len(self.pnl_buffer),
        }

    def save(self, path=None):
        path = path or os.path.join(STATE_DIR, "str_e_state.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "window": self.window,
            "gt_buffer": list(self.gt_buffer),
            "sy_buffer": list(self.sy_buffer),
            "pnl_buffer": list(self.pnl_buffer),
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path=None, window=100):
        path = path or os.path.join(STATE_DIR, "str_e_state.json")
        engine = cls(window=window)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                engine.gt_buffer.extend(data.get("gt_buffer", []))
                engine.sy_buffer.extend(data.get("sy_buffer", []))
                engine.pnl_buffer.extend(data.get("pnl_buffer", []))
            except Exception as e:
                pass
        return engine


class STRECoordinator:
    def __init__(self, stre_engine):
        self.stre = stre_engine
        self.phase2_enabled = False

    def step(self, gt_sim, sy_sim, pnl):
        self.stre.ingest(gt_sim, sy_sim, pnl)
        result = self.stre.compute()
        self.phase2_enabled = result["gt_corr"] > result["sy_corr"]
        result["phase2_blocked"] = not self.phase2_enabled
        return result
