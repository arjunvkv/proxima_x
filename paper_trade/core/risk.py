"""Risk checks — spread limit, max concurrent, session filter, max daily loss, market hours."""
import time

class Risk:
    """Stateless risk checks. Each check returns (ok: bool, reason: str)."""

    def __init__(self, config):
        self.max_concurrent = config.get("max_concurrent", 3)
        self.max_spread_mult = config.get("max_spread_mult", 1.5)
        self.max_spread_pips = config.get("max_spread_pips", 0.0)
        self.session_start = config.get("session_start", 7)
        self.session_end = config.get("session_end", 21)
        self.max_daily_loss = config.get("max_daily_loss", 500)
        self.today_pnl = 0.0
        self._today = None
        self._spread_baselines = {}

    def update_spread_baseline(self, pair, spread):
        track = self._spread_baselines.setdefault(pair, {"count": 0, "total": 0.0})
        track["count"] += 1
        track["total"] += spread

    def normal_spread(self, pair):
        t = self._spread_baselines.get(pair)
        if t and t["count"] > 10:
            return t["total"] / t["count"]
        return 0.0

    def check_market_hours(self, gmtime_struct):
        wd = gmtime_struct.tm_wday
        hr = gmtime_struct.tm_hour
        # FX market: closed Fri 22:00 UTC → Sun 22:00 UTC
        if wd == 4 and hr >= 22:
            return False, "weekend:Fri_after_22UTC"
        if wd == 5:
            return False, "weekend:Sat"
        if wd == 6 and hr < 22:
            return False, "weekend:Sun_before_22UTC"
        if hr < self.session_start or hr > self.session_end:
            return False, f"outside_session:H{hr:02d}"
        return True, ""

    def check_session(self, hour_utc):
        if hour_utc < self.session_start or hour_utc > self.session_end:
            return False, f"outside_session:H{hour_utc:02d}"
        return True, ""

    def check_spread(self, current_spread, normal_spread):
        if normal_spread > 0 and current_spread > normal_spread * self.max_spread_mult:
            return False, f"spread_widen:{current_spread:.1f}x{normal_spread:.1f}"
        return True, ""

    def check_absolute_spread(self, pair, current_spread):
        if self.max_spread_pips > 0 and current_spread > 0:
            pip_size = 0.01 if "JPY" in pair else 0.0001
            spread_pips = current_spread / pip_size
            if spread_pips > self.max_spread_pips:
                return False, f"spread_exceeds_max:{spread_pips:.1f}p>{self.max_spread_pips:.1f}p"
        return True, ""

    def check_concurrent(self, open_count):
        if open_count >= self.max_concurrent:
            return False, f"max_concurrent:{open_count}"
        return True, ""

    def check_daily_loss(self, pnl):
        today = time.strftime("%Y-%m-%d")
        if today != self._today:
            self._today = today
            self.today_pnl = 0.0
        self.today_pnl += pnl
        if self.today_pnl < -self.max_daily_loss:
            return False, f"daily_loss_limit:${self.today_pnl:.0f}"
        return True, ""

    def check_all(self, gmtime_struct, current_spread, normal_spread, open_count, pnl=0.0, pair=None):
        ok, reason = self.check_market_hours(gmtime_struct)
        if not ok: return False, reason
        ok, reason = self.check_spread(current_spread, normal_spread)
        if not ok: return False, reason
        if pair:
            ok, reason = self.check_absolute_spread(pair, current_spread)
            if not ok: return False, reason
        ok, reason = self.check_concurrent(open_count)
        if not ok: return False, reason
        if pnl != 0:
            ok, reason = self.check_daily_loss(pnl)
            if not ok: return False, reason
        return True, ""
