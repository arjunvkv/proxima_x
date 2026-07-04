import json
import os
from collections import deque


class GTSuppressionTracker:
    def __init__(self, window=100):
        self.window = window
        self.layer_history = deque(maxlen=window)

    def ingest(self, layer, symbol, event):
        self.layer_history.append({
            "layer": layer,
            "symbol": symbol,
            "oss_ev": float(event.get("oss_ev", 0)),
            "oss_conf": float(event.get("oss_conf", 0)),
            "ecdf_rank": float(event.get("ecdf_rank", 0)),
            "at_rank": float(event.get("at_rank", 0)),
            "research_p_cont": float(event.get("research_p_cont", 0)),
        })

    def per_layer_mean(self):
        layers = {}
        for entry in self.layer_history:
            l = entry["layer"]
            if l not in layers:
                layers[l] = []
            layers[l].append(entry["research_p_cont"])
        result = {}
        for l, vals in layers.items():
            result[l] = sum(vals) / len(vals) if vals else 0
        return result

    def suppression_flow(self):
        means = self.per_layer_mean()
        layers_ordered = sorted(means.keys())
        flow = []
        for i in range(len(layers_ordered) - 1):
            u = layers_ordered[i]
            v = layers_ordered[i + 1]
            flow.append({
                "source": u,
                "target": v,
                "drop": round(means[u] - means[v], 4),
            })
        return flow


class CounterfactualConvictionGT:
    def compute(self, events_by_layer):
        if len(events_by_layer) < 2:
            return {"suppression": 0, "amplification": 0}
        l0 = events_by_layer[0]
        l5 = events_by_layer[-1]
        raw = l0.get("oss_ev", 0) if isinstance(l0, dict) else 0
        final = l5.get("oss_ev", 0) if isinstance(l5, dict) else 0
        return {
            "raw": raw,
            "final": final,
            "suppression": max(0, raw - final),
            "amplification": max(0, final - raw),
        }
