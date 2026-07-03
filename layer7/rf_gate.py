"""LiveRFGate — rolling TPI+BFD buffer with RF prob gate for live execution."""
import os, logging, json
import numpy as np
import joblib

logger = logging.getLogger(__name__)

# Feature order for model input (must match training)
FEAT_ORDER = ["fft_low","fft_mid","fft_high","spectral_entropy","corr","lag_corr","max_corr","zero_sync","tpi_flip","bfd_burst","bfd_density","tpi_mean","tpi_std","tpi_skew"]

_DRIFT_BASELINE_WINDOW = 500  # first N preds form the reference distribution

# TPI computation window
_TPI_WIN = 200

def _feat(tpi, bfd):
    Nl=len(tpi); f={}
    fft=np.fft.rfft(tpi-np.mean(tpi)); fp=np.abs(fft)**2; n=len(fp)
    l1,l2=n//3,2*n//3
    f["fft_low"]=float(np.sum(fp[1:l1])/max(np.sum(fp[1:]),1))
    f["fft_mid"]=float(np.sum(fp[l1:l2])/max(np.sum(fp[1:]),1))
    f["fft_high"]=float(np.sum(fp[l2:])/max(np.sum(fp[1:]),1))
    p=fp[1:]/max(np.sum(fp[1:]),1e-12); p=p[p>0]
    f["spectral_entropy"]=float(-np.sum(p*np.log2(p))) if len(p)>0 else 0
    if np.std(tpi)>1e-12 and np.std(bfd)>1e-12:
        f["corr"]=float(np.corrcoef(tpi,bfd)[0,1])
        from scipy import signal as _sg
        cc=_sg.correlate(tpi-np.mean(tpi),bfd-np.mean(bfd),mode="same")
        cc=cc/(Nl*np.std(tpi)*np.std(bfd)+1e-12)
        mi=np.argmax(np.abs(cc))
        f["lag_corr"]=float(mi-Nl//2); f["max_corr"]=float(cc[mi])
    else: f["corr"]=f["lag_corr"]=f["max_corr"]=0.0
    ts=np.sign(tpi); bs=np.sign(bfd)
    f["zero_sync"]=float(np.mean((ts!=0)&(bs!=0)&(ts==bs)))
    tn=tpi[np.abs(tpi)>1e-12]
    f["tpi_flip"]=float(np.sum(np.abs(np.diff(np.sign(tn))))/2/max(len(tn),1)) if len(tn)>1 else 0
    inter=np.diff(np.where(bfd>0.05)[0]) if np.sum(bfd>0.05)>1 else [0]
    f["bfd_burst"]=float(np.std(inter)/max(np.mean(inter),1e-12)) if len(inter)>0 else 0
    f["bfd_density"]=float(np.mean(bfd>0.05))
    f["tpi_mean"]=float(np.mean(tpi)); f["tpi_std"]=float(np.std(tpi))
    f["tpi_skew"]=float(np.mean((tpi-np.mean(tpi))**3)/max(np.std(tpi)**3,1e-12)) if np.std(tpi)>1e-12 else 0
    return f

class LiveRfGate:
    """Rolling TPI+BFD gate using the EURJPY-trained RF model.

    Maintains per-symbol:
      - mid price buffer (latest 200 ticks for per-tick TPI)
      - timestamp buffer (latest 6 ticks for per-tick BFD)
      - TPI value buffer (latest 2000 ticks for RF features)
      - BFD value buffer (latest 2000 ticks for RF features)

    RF inference runs every `predict_every` ticks when 2000-tick windows are full.
    """
    def __init__(self, model_path=None, window=2000, predict_every=500, prob_thresh=0.60):
        self.window = window
        self.predict_every = predict_every
        self.prob_thresh = prob_thresh
        self._initd = False
        self._tpi_vals = {}   # symbol -> list of per-tick TPI values (2000)
        self._bfd_vals = {}   # symbol -> list of per-tick BFD values (2000)
        self._mid_vals = {}   # symbol -> list of mid prices (200)
        self._ts_vals = {}    # symbol -> list of timestamps (6)
        self._prob = {}       # symbol -> latest RF prob
        self._last_feat_time = {}
        self._tick_count = {} # symbol -> tick counter
        self.model = None
        self._model_missing = False
        self._drift_reference = []   # first N probs for drift baseline
        self._drift_mean = None
        self._drift_std = None
        self._drift_alert = False
        # Feature-space drift tracking
        self._feat_ref: list[dict] = []
        self._feat_means: dict = {}
        self._feat_stds: dict = {}
        self._feat_drift_alert = False
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        else:
            self._model_missing = True
            logger.error(f"[RF GATE] model not found at {model_path} — gate will REJECT all signals (fail-closed)")

    def load(self, path):
        mdl = joblib.load(path)
        self.model = mdl["model"]
        self._initd = True
        logger.info(f"[RF GATE] loaded model from {path}")

    def _ensure(self, sym):
        for d in (self._tpi_vals, self._bfd_vals, self._mid_vals, self._ts_vals, self._prob, self._last_feat_time, self._tick_count):
            if sym not in d:
                if isinstance(d, dict):
                    if d is self._tpi_vals or d is self._bfd_vals:
                        d[sym] = []
                    elif d is self._mid_vals:
                        d[sym] = []
                    elif d is self._ts_vals:
                        d[sym] = []
                    elif d is self._prob:
                        d[sym] = 0.0
                    elif d is self._last_feat_time:
                        d[sym] = -1
                    elif d is self._tick_count:
                        d[sym] = 0

    def feed_tick(self, symbol, bid, ask, timestamp, tick_idx):
        """Feed raw tick data. Computes TPI+BFD internally, rolls buffers, runs inference."""
        self._ensure(symbol)
        mid = (bid + ask) / 2.0

        # 1. Maintain mid price buffer for TPI
        mbuf = self._mid_vals[symbol]
        mbuf.append(mid)
        if len(mbuf) > _TPI_WIN:
            mbuf.pop(0)

        # 2. Compute winsorized magnitude-weighted TPI from mid buffer
        tpi_val = 0.0
        if len(mbuf) >= _TPI_WIN:
            arr = np.array(mbuf)
            delta = np.diff(arr)
            # Winsorize: clip outliers at P5/P95 to prevent single spike dominance
            if len(delta) >= 10:
                p5, p95 = np.percentile(delta, [5, 95])
                delta = np.clip(delta, p5, p95)
            su = float(np.sum(delta[delta > 1e-8]))
            sd = float(np.abs(np.sum(delta[delta < -1e-8])))
            tm = su + sd
            if tm > 1e-10:
                tpi_val = (su - sd) / tm

        # 3. Maintain timestamp buffer for BFD
        tbuf = self._ts_vals[symbol]
        tbuf.append(timestamp)
        if len(tbuf) > 6:
            tbuf.pop(0)

        # 4. Compute per-tick BFD from timestamp buffer
        bfd_val = 0.0
        if len(tbuf) >= 6:
            gaps = [tbuf[i+1] - tbuf[i] for i in range(len(tbuf)-1)]
            mg = sum(gaps) / len(gaps)
            sg = (sum((g - mg)**2 for g in gaps) / len(gaps))**0.5
            if sg > 0:
                bfd_val = float(np.clip((gaps[-1] - mg) / sg, -1, 1))

        # 5. Feed to rolling 2000-tick buffers
        vbuf = self._tpi_vals[symbol]
        vbuf.append(tpi_val)
        if len(vbuf) > self.window:
            vbuf.pop(0)

        bbuf = self._bfd_vals[symbol]
        bbuf.append(bfd_val)
        if len(bbuf) > self.window:
            bbuf.pop(0)

        # 6. Increment tick count
        self._tick_count[symbol] += 1

        # 7. Run RF inference periodically
        if self._initd and \
           len(vbuf) >= self.window and \
           len(bbuf) >= self.window and \
           self._tick_count[symbol] % self.predict_every == 0 and \
           self._tick_count[symbol] != self._last_feat_time.get(symbol, -1):
            try:
                arr_tpi = np.array(vbuf)
                arr_bfd = np.array(bbuf)
                fv = _feat(arr_tpi, arr_bfd)
                import pandas as pd
                d = pd.DataFrame([fv])[FEAT_ORDER]
                prob = float(self.model.predict_proba(d)[0, 1])
                self._prob[symbol] = prob

                # Drift monitoring: probability-space
                if len(self._drift_reference) < _DRIFT_BASELINE_WINDOW:
                    self._drift_reference.append(prob)
                elif self._drift_mean is None:
                    arr = np.array(self._drift_reference)
                    self._drift_mean = float(np.mean(arr))
                    self._drift_std = float(np.std(arr)) + 1e-10
                    logger.info(f"[RF GATE] drift baseline: mean={self._drift_mean:.3f} std={self._drift_std:.3f}")
                else:
                    z = abs(prob - self._drift_mean) / self._drift_std
                    if z > 3.0 and not self._drift_alert:
                        self._drift_alert = True
                        logger.warning(f"[RF GATE] DRIFT ALERT: prob={prob:.3f} z={z:.1f} — gate will reject")

                # Drift monitoring: feature-space
                if len(self._feat_ref) < _DRIFT_BASELINE_WINDOW:
                    self._feat_ref.append(fv)
                elif not self._feat_means:
                    keys = list(fv.keys())
                    arrs = np.array([[f[k] for k in keys] for f in self._feat_ref])
                    self._feat_means = {k: float(np.mean(arrs[:, i])) for i, k in enumerate(keys)}
                    self._feat_stds = {k: float(np.std(arrs[:, i])) + 1e-10 for i, k in enumerate(keys)}
                    logger.info(f"[RF GATE] feature drift baseline: {len(keys)} features")
                else:
                    n_drifted = 0
                    for k, v in fv.items():
                        if k in self._feat_means:
                            fz = abs(v - self._feat_means[k]) / self._feat_stds[k]
                            if fz > 3.5:
                                n_drifted += 1
                    if n_drifted >= 3 and not self._feat_drift_alert:
                        self._feat_drift_alert = True
                        logger.warning(f"[RF GATE] FEATURE DRIFT ALERT: {n_drifted} features drifted — gate will reject")

                self._last_feat_time[symbol] = self._tick_count[symbol]
            except Exception as e:
                logger.warning(f"[RF GATE] predict failed for {symbol}: {e}")

    def prob(self, symbol):
        return self._prob.get(symbol, 0.0)

    def ready(self, symbol):
        return self._initd and len(self._tpi_vals.get(symbol, [])) >= self.window

    def pre_seed(self, symbol: str, ticks: list) -> int:
        """Pre-seed the rolling buffer with historical tick data for fast warmup.
        Feeds ticks in chronological order (oldest-first) to build up TPI/BFD buffers.
        Returns the number of ticks actually loaded into the buffer."""
        count = 0
        for tick_data in ticks:
            bid = tick_data.get("bid", 0.0)
            ask = tick_data.get("ask", 0.0)
            ts = tick_data.get("time", 0)
            self.feed_tick(symbol, bid, ask, ts, count)
            count += 1
        return count

    def gate(self, symbol, signal, threshold=None):
        """Return signal if RF prob >= threshold, else 0 (HOLD).
        Fail-closed: if model is missing or drift detected (prob or feature space), reject all."""
        if self._model_missing or self.model is None or self._drift_alert or self._feat_drift_alert:
            return 0
        th = threshold or self.prob_thresh
        if not self.ready(symbol):
            return 0
        if self._prob.get(symbol, 0.0) < th:
            return 0
        return signal
