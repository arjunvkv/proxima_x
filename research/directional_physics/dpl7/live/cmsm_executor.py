from ..features.feature_bridge import FeatureBridge
from ..core.state_encoder import StateEncoder
from ..core.market_manifold import MarketManifold
from ..policy.signal_head import SignalHead
from ..policy.sizing_policy import SizingPolicy
from ..config import dpl7_config


class CMSMExecutor:
    def __init__(self, ed=None, tt=None):
        self.bridge = FeatureBridge(ed, tt)
        self.encoder = StateEncoder(dim=dpl7_config.LATENT_DIM)
        self.manifold = MarketManifold(alpha=dpl7_config.MANIFOLD_ALPHA)
        self.head = SignalHead(dim=dpl7_config.LATENT_DIM)
        self.sizing = SizingPolicy(
            thresholds=dpl7_config.CONFIDENCE_THRESHOLDS,
            multipliers=dpl7_config.SIZE_MULTIPLIERS
        )

    def process(self, data: dict, idx: int) -> dict:
        features = self.bridge.extract(data, idx)
        z = self.encoder.encode(features)
        z_smooth = self.manifold.update(z)
        signal = self.head.predict(z_smooth)
        size = self.sizing.compute(signal)
        return {
            "direction": signal["direction"],
            "confidence": signal["confidence"],
            "score": signal["score"],
            "size_multiplier": size,
            "features": features,
            "z": z_smooth.tolist()
        }

    def reset(self):
        self.bridge.reset()
        self.manifold.reset()
        self.head.reset()
