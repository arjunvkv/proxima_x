"""OMS-2: Volatility Expansion Test.
Does residual marker predict future volatility/range/entropy expansion?
Hypothesis: Direction is SECONDARY — a consequence of volatility expansion in a drift-biased market.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SRC))

from research.observable_market_state.oms_core import OMSCore, SYMBOLS, save_oms_report
from research.directional_state.dsr_core import WalkForwardValidator, DSRCore, HORIZONS

LOOKBACK = 20
VOL_HORIZONS = [5, 20, 50]
HL_MAP = {5: "H5", 20: "H20", 50: "H50"}
REPORTS_DIR = Path(__file__).resolve().parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def _recent_vol(es, i, w=LOOKBACK):
    if i < w:
        return np.nan
    r = np.diff(es[i - w:i])
    return np.std(r) if len(r) >= 3 else np.nan


def _future_vol(es, i, h):
    if i + h > len(es):
        return np.nan
    r = np.diff(es[i:i + h])
    return np.std(r) if len(r) >= 3 else np.nan


def _recent_range(es, i, w=LOOKBACK):
    if i < w:
        return np.nan
    return np.max(es[i - w:i]) - np.min(es[i - w:i])


def _future_range(es, i, h):
    if i + h > len(es):
        return np.nan
    return np.max(es[i:i + h]) - np.min(es[i:i + h])


def _entropy(arr):
    x = arr[~np.isnan(arr)]
    if len(x) < 3:
        return np.nan
    s = np.sign(x)
    n_pos = float(np.sum(s > 0))
    n_neg = float(np.sum(s < 0))
    t = n_pos + n_neg
    if t == 0:
        return np.nan
    pp = n_pos / t
    pn = n_neg / t
    if pp > 0 and pn > 0:
        return -(pp * np.log2(pp) + pn * np.log2(pn))
    return 0.0


def _recent_entropy(es, i, w=LOOKBACK):
    if i < w:
        return np.nan
    return _entropy(np.diff(es[i - w:i]))


def _future_entropy(es, i, h):
    if i + h > len(es):
        return np.nan
    return _entropy(np.diff(es[i:i + h]))


def cond_prob(marker, condition):
    valid = ~np.isnan(marker) & ~np.isnan(condition)
    m = marker[valid].astype(bool)
    c = condition[valid].astype(bool)
    n1 = int(np.sum(m))
    n0 = int(np.sum(~m))
    p1 = float(np.mean(c[m])) if n1 > 0 else np.nan
    p0 = float(np.mean(c[~m])) if n0 > 0 else np.nan
    d = p1 - p0 if not (np.isnan(p1) or np.isnan(p0)) else np.nan
    return {
        "p_given_marker_1": None if np.isnan(p1) else round(p1, 4),
        "p_given_marker_0": None if np.isnan(p0) else round(p0, 4),
        "delta": None if np.isnan(d) else round(d, 4),
        "n_marker_1": n1,
        "n_marker_0": n0,
    }


def analyze_symbol(oms, sym, marker=None, mask=None):
    """Full OMS-2 analysis for a symbol. If mask is provided, only evaluate on those bars."""
    m = marker if marker is not None else oms.marker_present(sym)
    es = oms.get_es(sym)
    fut_ret = oms.get_future_returns(sym)
    n = len(es)

    if mask is not None:
        m = m.copy().astype(float)
        min_len = min(len(m), len(mask))
        m[:min_len][~mask[:min_len]] = np.nan
        if len(m) > min_len:
            m[min_len:] = np.nan

    horizon_idx = {h: HORIZONS.index(h) for h in VOL_HORIZONS}
    out = {"n_bars": int(n), "marker_rate": round(float(np.nanmean(m)), 4)}

    for h in VOL_HORIZONS:
        hk = HL_MAP[h]
        hidx = horizon_idx[h]

        vol_inc = np.full(n, np.nan)
        range_exp = np.full(n, np.nan)
        ent_inc = np.full(n, np.nan)

        for i in range(LOOKBACK + h, n):
            rv = _recent_vol(es, i, LOOKBACK)
            fv = _future_vol(es, i, h)
            if not (np.isnan(rv) or np.isnan(fv)):
                vol_inc[i] = float(fv > rv)
            rr = _recent_range(es, i, LOOKBACK)
            fr = _future_range(es, i, h)
            if not (np.isnan(rr) or np.isnan(fr)) and rr > 0:
                range_exp[i] = float(fr > 2.0 * rr)
            re = _recent_entropy(es, i, LOOKBACK)
            fe = _future_entropy(es, i, h)
            if not (np.isnan(re) or np.isnan(fe)):
                ent_inc[i] = float(fe > re)

        out.setdefault("vol_prediction", {})[hk] = cond_prob(m, vol_inc)
        out.setdefault("range_prediction", {})[hk] = cond_prob(m, range_exp)
        out.setdefault("entropy_prediction", {})[hk] = cond_prob(m, ent_inc)

        # Direction prediction
        fwd = fut_ret[:, hidx].copy()
        dir_up = np.full(n, np.nan)
        valid_dir = ~np.isnan(fwd)
        dir_up[valid_dir] = (fwd[valid_dir] > 0).astype(float)
        dir_pred = cond_prob(m, dir_up)
        vol_pred = out["vol_prediction"][hk]

        out.setdefault("direction_vs_volatility", {})[hk] = {
            "direction": dir_pred,
            "volatility": vol_pred,
        }

        # Direction controlling for volatility state
        vol_state = np.full(n, np.nan)
        for i in range(LOOKBACK, n):
            rv = _recent_vol(es, i, LOOKBACK)
            if not np.isnan(rv):
                vol_state[i] = rv

        vs = vol_state.copy()
        vs_valid = ~np.isnan(vs)
        if np.sum(vs_valid) > 10:
            t1 = np.nanpercentile(vs[vs_valid], 33.33)
            t2 = np.nanpercentile(vs[vs_valid], 66.67)
        else:
            t1, t2 = np.nan, np.nan

        def cp_ctrl(vstate_mask):
            valid = ~np.isnan(m) & vstate_mask & ~np.isnan(dir_up)
            mm = m[valid].astype(bool)
            dd = dir_up[valid]
            n1 = int(np.sum(mm))
            n0 = int(np.sum(~mm))
            p1 = float(np.mean(dd[mm])) if n1 > 0 else np.nan
            p0 = float(np.mean(dd[~mm])) if n0 > 0 else np.nan
            d = p1 - p0 if not (np.isnan(p1) or np.isnan(p0)) else np.nan
            return {
                "p_up_given_marker_1": None if np.isnan(p1) else round(p1, 4),
                "p_up_given_marker_0": None if np.isnan(p0) else round(p0, 4),
                "delta": None if np.isnan(d) else round(d, 4),
                "n_marker_1": n1,
                "n_marker_0": n0,
            }

        out.setdefault("direction_controlling_vol", {})[hk] = {
            "low_vol": cp_ctrl(vs <= t1) if not np.isnan(t1) else {},
            "med_vol": cp_ctrl((vs > t1) & (vs <= t2)) if not np.isnan(t1) else {},
            "high_vol": cp_ctrl(vs > t2) if not np.isnan(t1) else {},
            "uncontrolled": dir_pred,
        }

        # Volatility regime change prediction
        vol_reg = np.full(n, np.nan, dtype=float)
        if not np.isnan(t1):
            for i in range(LOOKBACK, n):
                rv = _recent_vol(es, i, LOOKBACK)
                if not np.isnan(rv):
                    if rv <= t1:
                        vol_reg[i] = 0.0
                    elif rv <= t2:
                        vol_reg[i] = 1.0
                    else:
                        vol_reg[i] = 2.0

        vol_chg = np.full(n, np.nan, dtype=float)
        vol_chg_up = np.full(n, np.nan, dtype=float)
        vol_chg_dn = np.full(n, np.nan, dtype=float)
        for i in range(1, n):
            if vol_reg[i] >= 0 and vol_reg[i - 1] >= 0 and vol_reg[i] != vol_reg[i - 1]:
                vol_chg[i] = 1.0
                if vol_reg[i] > vol_reg[i - 1]:
                    vol_chg_up[i] = 1.0
                else:
                    vol_chg_dn[i] = 1.0

        fwd_chg = np.full(n, np.nan, dtype=float)
        fwd_up = np.full(n, np.nan, dtype=float)
        fwd_dn = np.full(n, np.nan, dtype=float)
        for i in range(n - h):
            sl = slice(i + 1, i + h + 1)
            if np.any(vol_chg[sl] == 1.0):
                fwd_chg[i] = 1.0
                fwd_up[i] = 1.0 if np.any(vol_chg_up[sl] == 1.0) else 0.0
                fwd_dn[i] = 1.0 if np.any(vol_chg_dn[sl] == 1.0) else 0.0
            else:
                fwd_chg[i] = 0.0
                fwd_up[i] = 0.0
                fwd_dn[i] = 0.0

        out.setdefault("vol_regime_change", {})[hk] = {
            "any_change": cond_prob(m, fwd_chg),
            "change_up": cond_prob(m, fwd_up),
            "change_down": cond_prob(m, fwd_dn),
        }

    return out


def walk_forward(oms):
    """Run 3-split walk-forward validation."""
    wfv = WalkForwardValidator(oms.dsr)
    wf = {}
    for sym in SYMBOLS:
        wfv.prepare(sym)
        wf[sym] = {}
        for name, (train_name, test_name) in enumerate(WalkForwardValidator.SPLITS):
            _, test_mask = wfv.split(sym, train_name, test_name)
            key = f"{train_name}->{test_name}"
            m = oms.marker_present(sym)
            wf[sym][key] = analyze_symbol(oms, sym, marker=m, mask=test_mask)
    return wf


def aggregate_across_symbols(per_symbol):
    """Average conditional probabilities across symbols (weighted by N)."""
    agg = {}
    for hk in ["H5", "H20", "H50"]:
        wp = agg.setdefault("vol_prediction", {}).setdefault(hk, {})
        rp = agg.setdefault("range_prediction", {}).setdefault(hk, {})
        ep = agg.setdefault("entropy_prediction", {}).setdefault(hk, {})
        dv = agg.setdefault("direction_vs_volatility", {}).setdefault(hk, {})
        dc = agg.setdefault("direction_controlling_vol", {}).setdefault(hk, {})

        total_p1_v, total_p0_v, tw_v = 0.0, 0.0, 0
        total_p1_r, total_p0_r, tw_r = 0.0, 0.0, 0
        total_p1_e, total_p0_e, tw_e = 0.0, 0.0, 0
        total_d1, total_d0, tw_d = 0.0, 0.0, 0
        dc_acc = {"low_vol": {}, "med_vol": {}, "high_vol": {}}
        rc_acc = {"any_change": {}, "change_up": {}, "change_down": {}}
        rc_w = {"any_change": 0, "change_up": 0, "change_down": 0}
        dc_w = {"low_vol": 0, "med_vol": 0, "high_vol": 0}

        for sym_data in per_symbol.values():
            def _sum(d, k1, k2):
                v = d.get(k1, {}).get(k2, {})
                p1 = v.get("p_given_marker_1")
                p0 = v.get("p_given_marker_0")
                n1 = v.get("n_marker_1", 0)
                n0 = v.get("n_marker_0", 0)
                return p1, p0, n1, n0

            p1_v, p0_v, n1_v, n0_v = _sum(sym_data, "vol_prediction", hk)
            if p1_v is not None:
                total_p1_v += p1_v * (n1_v + n0_v)
                total_p0_v += p0_v * (n1_v + n0_v)
                tw_v += n1_v + n0_v

            p1_r, p0_r, n1_r, n0_r = _sum(sym_data, "range_prediction", hk)
            if p1_r is not None:
                total_p1_r += p1_r * (n1_r + n0_r)
                total_p0_r += p0_r * (n1_r + n0_r)
                tw_r += n1_r + n0_r

            p1_e, p0_e, n1_e, n0_e = _sum(sym_data, "entropy_prediction", hk)
            if p1_e is not None:
                total_p1_e += p1_e * (n1_e + n0_e)
                total_p0_e += p0_e * (n1_e + n0_e)
                tw_e += n1_e + n0_e

            dird = sym_data.get("direction_vs_volatility", {}).get(hk, {}).get("direction", {})
            dd1 = dird.get("p_given_marker_1")
            dd0 = dird.get("p_given_marker_0")
            nd1 = dird.get("n_marker_1", 0)
            nd0 = dird.get("n_marker_0", 0)
            if dd1 is not None:
                total_d1 += dd1 * (nd1 + nd0)
                total_d0 += dd0 * (nd1 + nd0)
                tw_d += nd1 + nd0

            for vl in ["low_vol", "med_vol", "high_vol"]:
                ctrl = sym_data.get("direction_controlling_vol", {}).get(hk, {}).get(vl, {})
                cp1 = ctrl.get("p_up_given_marker_1")
                cp0 = ctrl.get("p_up_given_marker_0")
                cn1 = ctrl.get("n_marker_1", 0)
                cn0 = ctrl.get("n_marker_0", 0)
                if cp1 is not None:
                    dc_acc[vl].setdefault("p1", 0)
                    dc_acc[vl].setdefault("p0", 0)
                    dc_acc[vl].setdefault("w", 0)
                    dc_acc[vl]["p1"] += cp1 * (cn1 + cn0)
                    dc_acc[vl]["p0"] += cp0 * (cn1 + cn0)
                    dc_acc[vl]["w"] += cn1 + cn0
                    dc_w[vl] += cn1 + cn0

            for ct in ["any_change", "change_up", "change_down"]:
                rcd = sym_data.get("vol_regime_change", {}).get(hk, {}).get(ct, {})
                rp1 = rcd.get("p_given_marker_1")
                rp0 = rcd.get("p_given_marker_0")
                rn1 = rcd.get("n_marker_1", 0)
                rn0 = rcd.get("n_marker_0", 0)
                if rp1 is not None:
                    rc_acc[ct].setdefault("p1", 0)
                    rc_acc[ct].setdefault("p0", 0)
                    rc_acc[ct].setdefault("w", 0)
                    rc_acc[ct]["p1"] += rp1 * (rn1 + rn0)
                    rc_acc[ct]["p0"] += rp0 * (rn1 + rn0)
                    rc_acc[ct]["w"] += rn1 + rn0
                    rc_w[ct] += rn1 + rn0

        def _fin(ttl_p1, ttl_p0, tw):
            p1 = round(ttl_p1 / tw, 4) if tw > 0 else None
            p0 = round(ttl_p0 / tw, 4) if tw > 0 else None
            delta = round(p1 - p0, 4) if (p1 is not None and p0 is not None) else None
            return {
                "p_given_marker_1": p1,
                "p_given_marker_0": p0,
                "delta": delta,
                "n": tw,
            }

        wp.update(_fin(total_p1_v, total_p0_v, tw_v))
        rp.update(_fin(total_p1_r, total_p0_r, tw_r))
        ep.update(_fin(total_p1_e, total_p0_e, tw_e))
        dv["direction"] = _fin(total_d1, total_d0, tw_d)

        for vl in ["low_vol", "med_vol", "high_vol"]:
            a = dc_acc[vl]
            wgt = dc_w[vl]
            dc[vl] = {
                "p_up_given_marker_1": round(a["p1"] / wgt, 4) if wgt > 0 else None,
                "p_up_given_marker_0": round(a["p0"] / wgt, 4) if wgt > 0 else None,
                "n": wgt,
            } if a else {}

        for ct in ["any_change", "change_up", "change_down"]:
            a = rc_acc[ct]
            wgt = rc_w[ct]
            rc_val = agg.setdefault("vol_regime_change", {}).setdefault(hk, {})
            rc_val[ct] = _fin(a.get("p1", 0), a.get("p0", 0), wgt) if wgt > 0 else {}

    return agg


def generate_md(report, per_symbol, wf, agg):
    lines = []
    lines.append("# OMS-2: Volatility Expansion Test")
    lines.append("")
    lines.append("**Hypothesis:** If the residual marker detects 'information absorption state' rather than direction, then it should predict volatility/range expansion/entropy increase BEFORE direction. Direction may be SECONDARY — a consequence of volatility expansion in a drift-biased market.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Aggregate key findings
    lines.append("## Aggregate Results (Weighted Across Symbols)")
    lines.append("")
    lines.append("### RQ1: Volatility Prediction — P(vol_increase | marker)")
    lines.append("")
    lines.append("| Horizon | P(vol_inc \\| marker=1) | P(vol_inc \\| marker=0) | Delta | N |")
    lines.append("|---------|----------------------|----------------------|-------|---|")
    for hk in ["H5", "H20", "H50"]:
        d = agg.get("vol_prediction", {}).get(hk, {})
        p1 = d.get("p_given_marker_1", "N/A")
        p0 = d.get("p_given_marker_0", "N/A")
        n = d.get("n", 0)
        delta = round(p1 - p0, 4) if isinstance(p1, (int, float)) and isinstance(p0, (int, float)) else "N/A"
        lines.append(f"| {hk} | {p1} | {p0} | {delta} | {n} |")
    lines.append("")

    lines.append("### RQ2: Range Expansion — P(range_expansion | marker)")
    lines.append("")
    lines.append("| Horizon | P(range \\| marker=1) | P(range \\| marker=0) | Delta | N |")
    lines.append("|---------|---------------------|---------------------|-------|---|")
    for hk in ["H5", "H20", "H50"]:
        d = agg.get("range_prediction", {}).get(hk, {})
        p1 = d.get("p_given_marker_1", "N/A")
        p0 = d.get("p_given_marker_0", "N/A")
        n = d.get("n", 0)
        delta = round(p1 - p0, 4) if isinstance(p1, (int, float)) and isinstance(p0, (int, float)) else "N/A"
        lines.append(f"| {hk} | {p1} | {p0} | {delta} | {n} |")
    lines.append("")

    lines.append("### RQ3: Entropy Increase — P(entropy_increase | marker)")
    lines.append("")
    lines.append("| Horizon | P(ent \\| marker=1) | P(ent \\| marker=0) | Delta | N |")
    lines.append("|---------|-------------------|-------------------|-------|---|")
    for hk in ["H5", "H20", "H50"]:
        d = agg.get("entropy_prediction", {}).get(hk, {})
        p1 = d.get("p_given_marker_1", "N/A")
        p0 = d.get("p_given_marker_0", "N/A")
        n = d.get("n", 0)
        delta = round(p1 - p0, 4) if isinstance(p1, (int, float)) and isinstance(p0, (int, float)) else "N/A"
        lines.append(f"| {hk} | {p1} | {p0} | {delta} | {n} |")
    lines.append("")

    lines.append("### RQ4: Volatility vs Direction Prediction — Absolute Delta Comparison")
    lines.append("")
    lines.append("| Horizon | Volatility Delta (abs) | Direction Delta (abs) | Stronger Predictor |")
    lines.append("|---------|----------------------|----------------------|-------------------|")
    for hk in ["H5", "H20", "H50"]:
        vd = agg.get("vol_prediction", {}).get(hk, {})
        dd = agg.get("direction_vs_volatility", {}).get(hk, {}).get("direction", {})
        vp1 = vd.get("p_given_marker_1", 0) or 0
        vp0 = vd.get("p_given_marker_0", 0) or 0
        dp1 = dd.get("p_given_marker_1", 0) or 0
        dp0 = dd.get("p_given_marker_0", 0) or 0
        v_delta = abs(vp1 - vp0)
        d_delta = abs(dp1 - dp0)
        stronger = "Volatility" if v_delta > d_delta else ("Direction" if d_delta > v_delta else "Equal")
        lines.append(f"| {hk} | {v_delta:.4f} | {d_delta:.4f} | {stronger} |")
    lines.append("")

    lines.append("### RQ5: Direction Controlling for Volatility — P(up | marker) by Vol State")
    lines.append("")
    lines.append("| Horizon | Vol State | P(up \\| marker=1) | P(up \\| marker=0) | Delta | N |")
    lines.append("|---------|-----------|------------------|------------------|-------|---|")
    for hk in ["H5", "H20", "H50"]:
        dc = agg.get("direction_controlling_vol", {}).get(hk, {})
        for vl in ["low_vol", "med_vol", "high_vol"]:
            d = dc.get(vl, {})
            p1 = d.get("p_up_given_marker_1", "N/A")
            p0 = d.get("p_up_given_marker_0", "N/A")
            n = d.get("n", 0)
            delta = round(p1 - p0, 4) if isinstance(p1, (int, float)) and isinstance(p0, (int, float)) else "N/A"
            lines.append(f"| {hk} | {vl} | {p1} | {p0} | {delta} | {n} |")
        # uncontrolled
        d = dc.get("uncontrolled", {})
        p1 = d.get("p_given_marker_1", "N/A")
        p0 = d.get("p_given_marker_0", "N/A")
        n = d.get("n_marker_1", 0) + d.get("n_marker_0", 0)
        delta = round(p1 - p0, 4) if isinstance(p1, (int, float)) and isinstance(p0, (int, float)) else "N/A"
        lines.append(f"| {hk} | uncontrolled | {p1} | {p0} | {delta} | {n} |")
    lines.append("")

    lines.append("### RQ6: Volatility Regime Change — P(regime_change | marker)")
    lines.append("")
    lines.append("| Horizon | Change Type | P(change \\| marker=1) | P(change \\| marker=0) | Delta | N |")
    lines.append("|---------|-------------|----------------------|----------------------|-------|---|")
    for hk in ["H5", "H20", "H50"]:
        rc = agg.get("vol_regime_change", {}).get(hk, {})
        for ct in ["any_change", "change_up", "change_down"]:
            d = rc.get(ct, {})
            p1 = d.get("p_given_marker_1", "N/A")
            p0 = d.get("p_given_marker_0", "N/A")
            n = d.get("n", 0)
            delta = round(p1 - p0, 4) if isinstance(p1, (int, float)) and isinstance(p0, (int, float)) else "N/A"
            lines.append(f"| {hk} | {ct} | {p1} | {p0} | {delta} | {n} |")
    lines.append("")

    # Per-symbol summary
    lines.append("---")
    lines.append("## Per-Symbol Summary")
    lines.append("")
    for sym, sd in per_symbol.items():
        lines.append(f"### {sym}")
        lines.append(f"- **N bars:** {sd.get('n_bars', 'N/A')} | **Marker rate:** {sd.get('marker_rate', 'N/A')}")
        lines.append("")
        lines.append("| Horizon | Metric | P(\\|M=1) | P(\\|M=0) | Delta |")
        lines.append("|---------|--------|---------|---------|-------|")
        for hk in ["H5", "H20", "H50"]:
            for metric, label in [("vol_prediction", "Vol Inc"), ("range_prediction", "Range Exp"), ("entropy_prediction", "Ent Inc")]:
                d = sd.get(metric, {}).get(hk, {})
                p1 = d.get("p_given_marker_1", "N/A")
                p0 = d.get("p_given_marker_0", "N/A")
                delta = d.get("delta", "N/A")
                lines.append(f"| {hk} | {label} | {p1} | {p0} | {delta} |")
        lines.append("")

    # Walk-forward
    lines.append("---")
    lines.append("## Walk-Forward Validation")
    lines.append("")
    for sym, wf_sym in wf.items():
        lines.append(f"### {sym}")
        lines.append("")
        for split_key, sd in wf_sym.items():
            lines.append(f"**{split_key}** — N={sd.get('n_bars', 'N/A')}, Marker rate={sd.get('marker_rate', 'N/A')}")
            lines.append("")
            lines.append("| Horizon | Metric | P(\\|M=1) | P(\\|M=0) | Delta |")
            lines.append("|---------|--------|---------|---------|-------|")
            for hk in ["H5", "H20", "H50"]:
                for metric, label in [("vol_prediction", "Vol Inc"), ("range_prediction", "Range Exp"), ("entropy_prediction", "Ent Inc")]:
                    d = sd.get(metric, {}).get(hk, {})
                    p1 = d.get("p_given_marker_1", "N/A")
                    p0 = d.get("p_given_marker_0", "N/A")
                    delta = d.get("delta", "N/A")
                    lines.append(f"| {hk} | {label} | {p1} | {p0} | {delta} |")
            lines.append("")

    lines.append("---")
    lines.append("## Answers to Research Questions")
    lines.append("")
    lines.append("**RQ1: Does residual marker presence predict FUTURE realized volatility?**")
    for hk in ["H5", "H20", "H50"]:
        d = agg.get("vol_prediction", {}).get(hk, {})
        p1 = d.get("p_given_marker_1", "N/A")
        p0 = d.get("p_given_marker_0", "N/A")
        delta = d.get("delta", "N/A")
        lines.append(f"- {hk}: P(vol_inc|M=1)={p1} vs P(vol_inc|M=0)={p0}, diff={delta}")
    lines.append("")
    lines.append("**RQ2: Does marker predict range expansion (future range > 2x recent range)?**")
    for hk in ["H5", "H20", "H50"]:
        d = agg.get("range_prediction", {}).get(hk, {})
        p1 = d.get("p_given_marker_1", "N/A")
        p0 = d.get("p_given_marker_0", "N/A")
        lines.append(f"- {hk}: P(range|M=1)={p1} vs P(range|M=0)={p0}")
    lines.append("")
    lines.append("**RQ3: Does marker predict entropy increase?**")
    for hk in ["H5", "H20", "H50"]:
        d = agg.get("entropy_prediction", {}).get(hk, {})
        p1 = d.get("p_given_marker_1", "N/A")
        p0 = d.get("p_given_marker_0", "N/A")
        lines.append(f"- {hk}: P(entropyup|M=1)={p1} vs P(entropyup|M=0)={p0}")
    lines.append("")
    lines.append("**RQ4: Is the volatility prediction stronger than the direction prediction?**")
    for hk in ["H5", "H20", "H50"]:
        vd = agg.get("vol_prediction", {}).get(hk, {})
        dd = agg.get("direction_vs_volatility", {}).get(hk, {}).get("direction", {})
        vp1 = vd.get("p_given_marker_1", 0) or 0
        vp0 = vd.get("p_given_marker_0", 0) or 0
        dp1 = dd.get("p_given_marker_1", 0) or 0
        dp0 = dd.get("p_given_marker_0", 0) or 0
        v_delta = abs(vp1 - vp0)
        d_delta = abs(dp1 - dp0)
        lines.append(f"- {hk}: |diffvol|={v_delta:.4f} vs |diffdir|={d_delta:.4f} — {'Volatility wins' if v_delta > d_delta else 'Direction wins' if d_delta > v_delta else 'Equal'}")
    lines.append("")
    lines.append("**RQ5: Does direction only appear as a secondary effect in high-volatility states?**")
    for hk in ["H5", "H20", "H50"]:
        dc = agg.get("direction_controlling_vol", {}).get(hk, {})
        lines.append(f"- {hk}:")
        for vl in ["low_vol", "med_vol", "high_vol"]:
            d = dc.get(vl, {})
            p1 = d.get("p_up_given_marker_1", "N/A")
            p0 = d.get("p_up_given_marker_0", "N/A")
            lines.append(f"  - {vl}: P(up|M=1)={p1}, P(up|M=0)={p0}")
    lines.append("")
    lines.append("**RQ6: Does marker presence predict volatility REGIME CHANGES?**")
    for hk in ["H5", "H20", "H50"]:
        rc = agg.get("vol_regime_change", {}).get(hk, {})
        for ct in ["any_change", "change_up", "change_down"]:
            d = rc.get(ct, {})
            p1 = d.get("p_given_marker_1", "N/A")
            p0 = d.get("p_given_marker_0", "N/A")
            lines.append(f"- {hk} {ct}: P(change|M=1)={p1} vs P(change|M=0)={p0}")
        lines.append("")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by OMS-2: Volatility Expansion Test*")

    return "\n".join(lines)


def print_key_findings(agg):
    print("\n" + "=" * 70)
    print("OMS-2: KEY FINDINGS")
    print("=" * 70)

    print("\nRQ1: Volatility Prediction:")
    for hk in ["H5", "H20", "H50"]:
        d = agg.get("vol_prediction", {}).get(hk, {})
        p1 = d.get("p_given_marker_1", "N/A")
        p0 = d.get("p_given_marker_0", "N/A")
        delta = d.get("delta")
        n = d.get("n", 0)
        print(f"  {hk}: P(vol_inc|M=1)={p1}, P(vol_inc|M=0)={p0}, diff={delta}, n={n}")

    print("\nRQ2: Range Expansion:")
    for hk in ["H5", "H20", "H50"]:
        d = agg.get("range_prediction", {}).get(hk, {})
        p1 = d.get("p_given_marker_1", "N/A")
        p0 = d.get("p_given_marker_0", "N/A")
        delta = d.get("delta")
        print(f"  {hk}: P(range|M=1)={p1}, P(range|M=0)={p0}, diff={delta}")

    print("\nRQ3: Entropy Increase:")
    for hk in ["H5", "H20", "H50"]:
        d = agg.get("entropy_prediction", {}).get(hk, {})
        p1 = d.get("p_given_marker_1", "N/A")
        p0 = d.get("p_given_marker_0", "N/A")
        delta = d.get("delta")
        print(f"  {hk}: P(entropyup|M=1)={p1}, P(entropyup|M=0)={p0}, diff={delta}")

    print("\nRQ4: Volatility vs Direction:")
    for hk in ["H5", "H20", "H50"]:
        vd = agg.get("vol_prediction", {}).get(hk, {})
        dd = agg.get("direction_vs_volatility", {}).get(hk, {}).get("direction", {})
        vp1 = vd.get("p_given_marker_1", 0) or 0
        vp0 = vd.get("p_given_marker_0", 0) or 0
        dp1 = dd.get("p_given_marker_1", 0) or 0
        dp0 = dd.get("p_given_marker_0", 0) or 0
        v_delta = abs(vp1 - vp0)
        d_delta = abs(dp1 - dp0)
        winner = "Vol" if v_delta > d_delta else "Dir" if d_delta > v_delta else "Tie"
        print(f"  {hk}: |diffvol|={v_delta:.4f}, |diffdir|={d_delta:.4f} -> {winner}")

    print("\nRQ5: Direction x Volatility Control:")
    for hk in ["H5", "H20", "H50"]:
        dc = agg.get("direction_controlling_vol", {}).get(hk, {})
        for vl in ["low_vol", "med_vol", "high_vol"]:
            d = dc.get(vl, {})
            p1 = d.get("p_up_given_marker_1", "N/A")
            p0 = d.get("p_up_given_marker_0", "N/A")
            print(f"  {hk} {vl}: P(up|M=1)={p1}, P(up|M=0)={p0}")

    print("\nRQ6: Volatility Regime Change:")
    for hk in ["H5", "H20", "H50"]:
        rc = agg.get("vol_regime_change", {}).get(hk, {})
        for ct in ["any_change", "change_up", "change_down"]:
            d = rc.get(ct, {})
            p1 = d.get("p_given_marker_1", "N/A")
            p0 = d.get("p_given_marker_0", "N/A")
            print(f"  {hk} {ct}: P(change|M=1)={p1}, P(change|M=0)={p0}")

    print("=" * 70)


def run_oms2():
    print("Loading OMS core...")
    oms = OMSCore()
    oms.load_all()
    print("OMS core ready.")

    per_symbol = {}
    for sym in SYMBOLS:
        sym_result = analyze_symbol(oms, sym)
        per_symbol[sym] = sym_result

    wf = walk_forward(oms)

    agg = aggregate_across_symbols(per_symbol)

    report = {
        "metadata": {
            "title": "OMS-2: Volatility Expansion Test",
            "description": "Tests whether residual marker predicts volatility/range/entropy expansion before direction",
            "lookback_bars": LOOKBACK,
            "horizons": VOL_HORIZONS,
            "symbols": SYMBOLS,
        },
        "aggregate": agg,
        "per_symbol": per_symbol,
        "walk_forward": wf,
    }

    json_path = REPORTS_DIR / "oms2_volatility_expansion.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved {json_path}")

    md_content = generate_md(report, per_symbol, wf, agg)
    md_path = REPORTS_DIR / "OMS2_VOLATILITY_EXPANSION.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved {md_path}")

    print_key_findings(agg)
    return report


if __name__ == "__main__":
    report = run_oms2()
