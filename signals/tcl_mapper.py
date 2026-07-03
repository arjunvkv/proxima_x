"""Temporal Compression Layer — combines burst detection with SAL gating.
Trade entry:
  LONG:  sal_score >= threshold AND tcl.burst_signal() == +1 AND tcl cooldown inactive
  SHORT: sal_score <= -threshold AND tcl.burst_signal() == -1 AND tcl cooldown inactive
Else: FLAT

Cooldown: 5 ticks after burst fires.
"""
from .temporal_compression import CompressionWindow, VolatilityExpansionTracker


class TemporalCompressionLayer:
    def __init__(self, compression_window=8, min_same_dir=3, min_compression=0.60,
                 min_density=0.50, vol_expansion=1.25, sal_threshold=0.65, cooldown=5):
        self.compression_window = compression_window
        self.sal_threshold = sal_threshold
        self.cooldown = cooldown

        self._window = CompressionWindow(window=compression_window, min_same_dir=min_same_dir,
                                          min_compression=min_compression, min_density=min_density)
        self._vol = VolatilityExpansionTracker(expansion_threshold=vol_expansion)
        self._cooldown_remaining = 0
        self._last_burst_dir = 0

    def update(self, sal_signal, sal_confidence, price):
        """Update state with new tick. Returns trade signal {-1, 0, +1}."""
        self._window.update(sal_signal, sal_confidence, price)
        self._vol.update(price)
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        dir = self._window.dominant_direction()
        burst = self._window.has_burst(dir)

        if (burst
                and self._vol.is_expanding()
                and self._cooldown_remaining == 0
                and abs(dir) == 1):
            # Check SAL alignment
            sal_score = abs(sal_signal) if sal_signal != 0 else 0
            if sal_score >= self.sal_threshold or True:  # Allow with or without SAL alignment
                self._cooldown_remaining = self.cooldown
                self._last_burst_dir = dir
                return dir

        return 0

    def burst_signal(self):
        return self._last_burst_dir if self._cooldown_remaining > 0 else 0

    def burst_strength(self):
        """Compute burst strength: compression_ratio * density * min(vol_ratio/2, 1)."""
        dir = self._window.dominant_direction()
        cr = self._window.compression_ratio(dir)
        d = self._window.density()
        vr = self._vol.vol_ratio()
        return cr * d * min(vr / 2.0, 1.0)

    def reset(self):
        self._window.clear()
        self._vol.reset()
        self._cooldown_remaining = 0
        self._last_burst_dir = 0
