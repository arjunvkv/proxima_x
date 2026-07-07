"""
Directional Energy Validation Agent -- Offline Analysis
========================================================
Compares exec_drift with alternative directional sensors:
  - Candidate B: Directional Energy (ecdf-change based flow)
  - Candidate C: Multi-scale drift (D5, D15 moving average)
  - TPI as direct directional energy proxy

Reads:  proxima_demo.log
Output: Full comparison statistics + confusion matrix
"""

import re
import os
import math
from collections import defaultdict, deque

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "proxima_demo.log")
LOG_PATH = os.path.normpath(LOG_PATH)

OSS_RE = re.compile(
    r"\[OSS SURFACE\]\s+(?P<symbol>\w+)\s+"
    r"ecdf=(?P<ecdf>[\d.]+)\s+"
    r"exec_drift=(?P<exec_drift>-?[\d.]+)\s+"
    r"live_drift=(?P<live_drift>-?[\d.]+)\s+"
    r"horizon=blended\(w3=(?P<w3>[\d.]+),w10=(?P<w10>[\d.]+),w20=(?P<w20>[\d.]+)\)\s+"
    r"regime=(?P<regime>\S+)\s+"
    r"p_cont=(?P<p_cont>[\d.]+)\s+"
    r"ph=(?P<ph>\d+)\s+pt=(?P<pt>\d+)\s+"
    r"r_pc=(?P<r_pc>[\d.]+)\s+"
    r"r_ph=(?P<r_ph>\d+)\s+r_pt=(?P<r_pt>\d+)\s+"
    r"r_bucket=(?P<r_bucket>\S+)\s+"
    r"r_fb=(?P<r_fb>\S+)\s+"
    r"signal=(?P<signal>-?[\d.]+)\s+"
    r"up=(?P<up>[\d.]+)%\s+dn=(?P<dn>[\d.]+)%"
)

TPI_RE = re.compile(
    r"\[TPI_SOURCE\]\s+(?P<symbol>\w+)\s+"
    r"source=(?P<source>\S+)\s+"
    r"direction=(?P<direction>FLAT|LONG|SHORT)\s+"
    r"conf=(?P<conf>[\d.]+)\s+"
    r"n_ticks=(?P<n_ticks>\d+)"
)


def parse_log(filepath):
    oss_entries = []
    tpi_entries = defaultdict(list)

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = OSS_RE.search(line)
            if m:
                ts = line[:23].strip()
                d = m.groupdict()
                d["_ts"] = ts
                d["ecdf"] = float(d["ecdf"])
                d["exec_drift"] = int(float(d["exec_drift"]))
                d["live_drift"] = int(float(d["live_drift"]))
                d["p_cont"] = float(d["p_cont"])
                d["ph"] = int(d["ph"])
                d["pt"] = int(d["pt"])
                d["r_pc"] = float(d["r_pc"])
                d["signal"] = int(float(d["signal"]))
                d["up"] = float(d["up"])
                d["dn"] = float(d["dn"])
                d["w3"] = float(d["w3"])
                d["w10"] = float(d["w10"])
                d["w20"] = float(d["w20"])
                oss_entries.append(d)
                continue

            m = TPI_RE.search(line)
            if m:
                d = m.groupdict()
                d["conf"] = float(d["conf"])
                d["n_ticks"] = int(d["n_ticks"])
                val = 1.0 if d["direction"] == "LONG" else (-1.0 if d["direction"] == "SHORT" else 0.0)
                d["tpi_value"] = val
                tpi_entries[d["symbol"]].append(d)

    return oss_entries, tpi_entries


def compute_directional_energy(entries):
    """Candidate B: Directional Energy from ecdf changes (proxy for tick flow)."""
    results = []
    prev_ecdf = {}
    for e in entries:
        sym = e["symbol"]
        ecdf = e["ecdf"]
        prev = prev_ecdf.get(sym, ecdf)
        ecdf_change = ecdf - prev
        prev_ecdf[sym] = ecdf
        energy = abs(ecdf_change)
        de = ecdf_change / max(energy, 1e-9) if energy > 1e-6 else 0.0
        results.append({
            "symbol": sym, "ts": e["_ts"],
            "ecdf": ecdf, "ecdf_change": ecdf_change,
            "energy": energy, "directional_energy": de,
            "exec_drift": e["exec_drift"], "p_cont": e["p_cont"],
            "regime": e["regime"], "signal": e["signal"],
        })
    return results


def compute_multiscale_drift(entries, windows=(5, 15)):
    """Candidate C: D5, D15 moving averages of exec_drift per symbol."""
    buffers = {}
    results = []
    max_w = max(windows)
    for e in entries:
        sym = e["symbol"]
        if sym not in buffers:
            buffers[sym] = deque(maxlen=max_w)
        buffers[sym].append(e["exec_drift"])
        buf = list(buffers[sym])
        d5 = sum(buf[-5:]) / min(len(buf[-5:]), 5) if buf else 0.0
        d15 = sum(buf) / len(buf) if buf else 0.0
        results.append({
            "symbol": sym, "ts": e["_ts"],
            "D5": d5, "D15": d15,
            "exec_drift": e["exec_drift"],
            "p_cont": e["p_cont"], "regime": e["regime"],
        })
    return results


def compute_tpi_proxy(oss_entries, tpi_by_symbol):
    """Match TPI readings to OSS entries by symbol (sequential cursor)."""
    tpi_cursor = defaultdict(int)
    tpi_sorted = {s: sorted(v, key=lambda x: x.get("_ts", "")) for s, v in tpi_by_symbol.items()}
    results = []
    for e in oss_entries:
        sym = e["symbol"]
        idx = tpi_cursor[sym]
        readings = tpi_sorted.get(sym, [])
        best = readings[idx] if idx < len(readings) else None
        if best is None:
            tpi_val, tpi_conf = 0.0, 0.0
        else:
            tpi_val = best["tpi_value"]
            tpi_conf = best["conf"]
        results.append({
            "symbol": sym, "ts": e["_ts"],
            "tpi_value": tpi_val, "tpi_conf": tpi_conf,
            "exec_drift": e["exec_drift"],
            "p_cont": e["p_cont"], "regime": e["regime"],
        })
    return results


def histogram(values, bins=21, lo=-1.05, hi=1.05):
    bcount = [0] * bins
    bw = (hi - lo) / bins
    for v in values:
        v = max(lo, min(hi, v))
        idx = min(int((v - lo) / bw), bins - 1)
        bcount[idx] += 1
    edges = [lo + i * bw for i in range(bins + 1)]
    return bcount, edges


def fmt_hist(values, label, bins=21, width=40):
    bcount, edges = histogram(values, bins=bins)
    total = sum(bcount)
    if total == 0:
        return "  {}: (no data)".format(label)
    mx = max(bcount)
    lines = []
    lines.append("  {}: (n={})".format(label, total))
    lines.append("    {:>12s} : {:>6s}  {}".format("Range", "Count", "Bar"))
    lines.append("    {:>12s} : {:>6s}  {}".format("-" * 12, "-" * 6, "-" * 40))
    for cnt, lo, hi in zip(bcount, edges[:-1], edges[1:]):
        bar = chr(9608) * int(cnt / mx * width) if mx > 0 else ""
        lines.append("    {:>7.3f}-{:<7.3f} : {:>6d}  {}".format(lo, hi, cnt, bar))
    return "\n".join(lines)


def classify_signal(value, threshold=0.1):
    if abs(value) < threshold:
        return 0
    return 1 if value > 0 else -1


def correlate(xs, ys):
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    xm = sum(xs[:n]) / n
    ym = sum(ys[:n]) / n
    num = sum((xs[i] - xm) * (ys[i] - ym) for i in range(n))
    dx = math.sqrt(sum((xs[i] - xm) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - ym) ** 2 for i in range(n)))
    if dx * dy == 0:
        return 0.0
    return num / (dx * dy)


def print_confusion(actual, predicted, label):
    n = min(len(actual), len(predicted))
    m = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(n):
        a = actual[i] + 1
        p = predicted[i] + 1
        if 0 <= a <= 2 and 0 <= p <= 2:
            m[a][p] += 1

    print("\n  {}:".format(label))
    print("    {:>14s}  {:>8s}  {:>8s}  {:>8s}".format("", "PRED -1", "PRED 0", "PRED +1"))
    print("    " + "-" * 42)
    for ri, rl in enumerate(["ACTUAL -1", "ACTUAL  0", "ACTUAL +1"]):
        print("    {:>12s} : {:>8d}  {:>8d}  {:>8d}".format(rl, m[ri][0], m[ri][1], m[ri][2]))

    # Sign agreement on non-zero actual
    pairs = [(a, p) for a, p in zip(actual, predicted) if abs(a) > 0.1]
    if pairs:
        agree = sum(1 for a, p in pairs if classify_signal(a) == classify_signal(p))
        same = sum(1 for a, p in pairs if (a > 0) == (p > 0))
        print("    Sign agreement (non-zero actual): {}/{} = {:.1f}%".format(agree, len(pairs), agree/len(pairs)*100))
        print("    Same-sign: {}/{} = {:.1f}%".format(same, len(pairs), same/len(pairs)*100))
    total = sum(sum(r) for r in m)
    correct = m[0][0] + m[1][1] + m[2][2]
    if total:
        print("    Overall accuracy: {}/{} = {:.1f}%".format(correct, total, correct/total*100))


def main():
    print("=" * 72)
    print("  DIRECTIONAL ENERGY VALIDATION AGENT")
    print("  Offline comparison of exec_drift vs alternative sensors")
    print("=" * 72)
    print("\n  Log file: {}".format(LOG_PATH))
    print("  Exists: {}".format(os.path.exists(LOG_PATH)))

    if not os.path.exists(LOG_PATH):
        print("  ERROR: Log file not found!")
        return

    # Phase 1: Parse
    print("\n-- Phase 1: Parsing Log " + "-" * 40)
    oss_entries, tpi_entries = parse_log(LOG_PATH)
    print("  OSS SURFACE entries: {}".format(len(oss_entries)))
    print("  TPI_SOURCE entries:  {}".format(sum(len(v) for v in tpi_entries.values())))
    print("  Symbols tracked:     {}".format(len(tpi_entries)))

    drifts = [e["exec_drift"] for e in oss_entries]
    zero_count = sum(1 for d in drifts if d == 0)
    non_zero_count = len(drifts) - zero_count
    pct_zero = zero_count / len(drifts) * 100 if drifts else 0
    print("\n  exec_drift = 0:  {} / {} ({:.1f}%)".format(zero_count, len(drifts), pct_zero))
    print("  exec_drift != 0: {} / {} ({:.1f}%)".format(non_zero_count, len(drifts), 100 - pct_zero))

    # Phase 2: Compute sensors
    print("\n-- Phase 2: Computing Candidates " + "-" * 37)

    print("  Candidate B: Directional Energy (ecdf-change flow)...")
    de_results = compute_directional_energy(oss_entries)
    de_values = [r["directional_energy"] for r in de_results]
    de_nz = [v for v in de_values if abs(v) > 0.1]
    de_pct_nz = len(de_nz) / len(de_values) * 100 if de_values else 0
    de_mean_abs = sum(abs(v) for v in de_nz) / len(de_nz) if de_nz else 0

    print("  Candidate C: Multi-scale Drift (D5, D15)...")
    ms_results = compute_multiscale_drift(oss_entries)
    d5_values = [r["D5"] for r in ms_results]
    d15_values = [r["D15"] for r in ms_results]
    d5_nz = [v for v in d5_values if abs(v) > 0.1]
    d15_nz = [v for v in d15_values if abs(v) > 0.1]
    d5_pct_nz = len(d5_nz) / len(d5_values) * 100 if d5_values else 0
    d15_pct_nz = len(d15_nz) / len(d15_values) * 100 if d15_values else 0
    d5_mean_abs = sum(abs(v) for v in d5_nz) / len(d5_nz) if d5_nz else 0
    d15_mean_abs = sum(abs(v) for v in d15_nz) / len(d15_nz) if d15_nz else 0

    print("  TPI Proxy (direct directional energy sensor)...")
    tpi_results = compute_tpi_proxy(oss_entries, tpi_entries)
    tpi_values = [r["tpi_value"] for r in tpi_results]
    tpi_nz = [v for v in tpi_values if abs(v) > 0.1]
    tpi_pct_nz = len(tpi_nz) / len(tpi_values) * 100 if tpi_values else 0
    tpi_mean_abs = sum(abs(v) for v in tpi_nz) / len(tpi_nz) if tpi_nz else 0

    # Phase 3: Summary
    print("\n" + "=" * 72)
    print("  PHASE 3: SUMMARY COMPARISON")
    print("=" * 72)
    print()
    print("    {:30s} {:>12s} {:>12s} {:>8s}".format("Sensor", "% Non-zero", "Mean |val|", "Count"))
    print("    " + "-" * 62)
    print("    {:30s} {:>12.1f}% {:>12s} {:>8d}".format("exec_drift (current)", 100-pct_zero, "--", len(drifts)))
    print("    {:30s} {:>12.1f}% {:>12.4f} {:>8d}".format("Cand-B: Directional Energy", de_pct_nz, de_mean_abs, len(de_values)))
    print("    {:30s} {:>12.1f}% {:>12.4f} {:>8d}".format("Cand-C: D5 (MA-5)", d5_pct_nz, d5_mean_abs, len(d5_values)))
    print("    {:30s} {:>12.1f}% {:>12.4f} {:>8d}".format("Cand-C: D15 (MA-15)", d15_pct_nz, d15_mean_abs, len(d15_values)))
    print("    {:30s} {:>12.1f}% {:>12.4f} {:>8d}".format("TPI (direct DE proxy)", tpi_pct_nz, tpi_mean_abs, len(tpi_values)))

    # Phase 4: Histograms
    print("\n" + "=" * 72)
    print("  PHASE 4: DISTRIBUTION HISTOGRAMS")
    print("=" * 72)
    print()
    print(fmt_hist(drifts, "exec_drift"))
    print()
    print(fmt_hist(de_values, "Directional Energy"))
    print()
    print(fmt_hist(d5_values, "D5 (MA-5)", bins=11))
    print()
    print(fmt_hist(d15_values, "D15 (MA-15)", bins=11))
    print()
    print(fmt_hist(tpi_values, "TPI Value"))

    # Phase 5: Correlation
    print("\n" + "=" * 72)
    print("  PHASE 5: CORRELATION ANALYSIS")
    print("=" * 72)

    print("\n    {:35s} {:>12s}".format("Pair", "Correlation"))
    print("    " + "-" * 47)
    print("    {:35s} {:>12.4f}".format("exec_drift vs Directional Energy", correlate(drifts, de_values)))
    print("    {:35s} {:>12.4f}".format("exec_drift vs D5", correlate(drifts, d5_values)))
    print("    {:35s} {:>12.4f}".format("exec_drift vs D15", correlate(drifts, d15_values)))
    print("    {:35s} {:>12.4f}".format("exec_drift vs TPI", correlate(drifts, tpi_values)))
    print("    {:35s} {:>12.4f}".format("DE vs TPI", correlate(de_values, tpi_values)))
    print("    {:35s} {:>12.4f}".format("D5 vs D15", correlate(d5_values, d15_values)))

    p_cont_values = [e["p_cont"] for e in oss_entries]
    print("\n    {:35s} {:>12s}".format("Pair", "Correlation"))
    print("    " + "-" * 47)
    print("    {:35s} {:>12.4f}".format("exec_drift vs p_cont", correlate(drifts, p_cont_values)))
    print("    {:35s} {:>12.4f}".format("DE vs p_cont", correlate(de_values, p_cont_values)))
    print("    {:35s} {:>12.4f}".format("D5 vs p_cont", correlate(d5_values, p_cont_values)))
    print("    {:35s} {:>12.4f}".format("D15 vs p_cont", correlate(d15_values, p_cont_values)))
    print("    {:35s} {:>12.4f}".format("TPI vs p_cont", correlate(tpi_values, p_cont_values)))

    # Phase 6: The critical question
    print("\n" + "=" * 72)
    print("  PHASE 6: CRITICAL QUESTION -- Can sensors detect signal")
    print("  when exec_drift = 0 ({:.1f}% of cycles)?".format(pct_zero))
    print("=" * 72)

    z_idx = [i for i, d in enumerate(drifts) if d == 0]
    print("\n  Cycles with exec_drift=0: {}".format(len(z_idx)))

    de_z = [de_values[i] for i in z_idx if i < len(de_values)]
    de_znz = [v for v in de_z if abs(v) > 0.1]
    d5_z = [d5_values[i] for i in z_idx if i < len(d5_values)]
    d5_znz = [v for v in d5_z if abs(v) > 0.1]
    d15_z = [d15_values[i] for i in z_idx if i < len(d15_values)]
    d15_znz = [v for v in d15_z if abs(v) > 0.1]
    tpi_z = [tpi_values[i] for i in z_idx if i < len(tpi_values)]
    tpi_znz = [v for v in tpi_z if abs(v) > 0.1]

    print("\n    Cand-B: Directional Energy:")
    print("      Non-zero: {}/{} ({:.1f}%)".format(len(de_znz), len(de_z), len(de_znz)/len(de_z)*100 if de_z else 0))
    print("    Cand-C: D5:")
    print("      Non-zero: {}/{} ({:.1f}%)".format(len(d5_znz), len(d5_z), len(d5_znz)/len(d5_z)*100 if d5_z else 0))
    print("    Cand-C: D15:")
    print("      Non-zero: {}/{} ({:.1f}%)".format(len(d15_znz), len(d15_z), len(d15_znz)/len(d15_z)*100 if d15_z else 0))
    print("    TPI Proxy:")
    print("      Non-zero: {}/{} ({:.1f}%)".format(len(tpi_znz), len(tpi_z), len(tpi_znz)/len(tpi_z)*100 if tpi_z else 0))

    # Phase 7: Confusion matrix
    print("\n" + "=" * 72)
    print("  PHASE 7: CONFUSION MATRIX -- When exec_drift != 0,")
    print("  do other sensors agree on sign?")
    print("=" * 72)

    nz_mask = [i for i, d in enumerate(drifts) if d != 0]
    nz_act = [classify_signal(drifts[i]) for i in nz_mask]
    de_nz_sig = [classify_signal(de_values[i]) for i in nz_mask if i < len(de_values)]
    d5_nz_sig = [classify_signal(d5_values[i]) for i in nz_mask if i < len(d5_values)]
    d15_nz_sig = [classify_signal(d15_values[i]) for i in nz_mask if i < len(d15_values)]
    tpi_nz_sig = [classify_signal(tpi_values[i]) for i in nz_mask if i < len(tpi_values)]

    trim = min(len(nz_act), len(de_nz_sig), len(d5_nz_sig), len(d15_nz_sig), len(tpi_nz_sig))
    nz_act = nz_act[:trim]
    de_nz_sig = de_nz_sig[:trim]
    d5_nz_sig = d5_nz_sig[:trim]
    d15_nz_sig = d15_nz_sig[:trim]
    tpi_nz_sig = tpi_nz_sig[:trim]

    print("\n  Non-zero exec_drift cycles analyzed: {}".format(trim))
    print("  exec_drift=+1: {}".format(sum(1 for a in nz_act if a > 0)))
    print("  exec_drift=-1: {}".format(sum(1 for a in nz_act if a < 0)))

    print_confusion(nz_act, de_nz_sig, "DE vs exec_drift")
    print_confusion(nz_act, d5_nz_sig, "D5 vs exec_drift")
    print_confusion(nz_act, d15_nz_sig, "D15 vs exec_drift")
    print_confusion(nz_act, tpi_nz_sig, "TPI vs exec_drift")

    # Phase 8: Ensemble
    print("\n" + "=" * 72)
    print("  PHASE 8: MULTI-SENSOR ENSEMBLE VOTE")
    print("=" * 72)

    votes = []
    for i in range(trim):
        s = de_nz_sig[i] + d5_nz_sig[i] + d15_nz_sig[i] + tpi_nz_sig[i]
        votes.append(classify_signal(s))
    print_confusion(nz_act, votes, "Ensemble (DE+D5+D15+TPI) vs exec_drift")

    # Ensemble on zero-drift
    z_trim = min(len(z_idx), len(de_values), len(d5_values), len(d15_values), len(tpi_values))
    zv = []
    for i in range(z_trim):
        idx = z_idx[i]
        s = (classify_signal(de_values[idx]) +
             classify_signal(d5_values[idx]) +
             classify_signal(d15_values[idx]) +
             classify_signal(tpi_values[idx]))
        zv.append(classify_signal(s))
    zvnz = sum(1 for v in zv if v != 0)
    zvp = sum(1 for v in zv if v > 0)
    zvn = sum(1 for v in zv if v < 0)
    print("\n  Ensemble signal when exec_drift=0:")
    print("    Non-zero: {}/{} ({:.1f}%)".format(zvnz, z_trim, zvnz/z_trim*100 if z_trim else 0))
    print("    Positive: {}  Negative: {}  Neutral: {}".format(zvp, zvn, z_trim - zvnz))

    # Phase 9: Regime sensitivity
    print("\n" + "=" * 72)
    print("  PHASE 9: REGIME SENSITIVITY")
    print("=" * 72)

    rdata = defaultdict(lambda: {"de": [], "d5": [], "d15": [], "tpi": [], "exec_drift": []})
    for i, e in enumerate(oss_entries):
        reg = e["regime"]
        if i < len(de_values): rdata[reg]["de"].append(de_values[i])
        if i < len(d5_values): rdata[reg]["d5"].append(d5_values[i])
        if i < len(d15_values): rdata[reg]["d15"].append(d15_values[i])
        if i < len(tpi_values): rdata[reg]["tpi"].append(tpi_values[i])
        rdata[reg]["exec_drift"].append(e["exec_drift"])

    print("\n    {:25s} {:>8s} {:>12s} {:>10s} {:>10s} {:>10s} {:>10s}".format(
        "Regime", "Count", "ExecDr%!=0", "DE_%!=0", "D5_%!=0", "D15_%!=0", "TPI_%!=0"))
    print("    " + "-" * 85)
    for reg, d in sorted(rdata.items(), key=lambda x: -len(x[1]["exec_drift"])):
        n = len(d["exec_drift"])
        dr_pct = sum(1 for x in d["exec_drift"] if x != 0) / n * 100
        de_pct = sum(1 for v in d["de"] if abs(v) > 0.1) / n * 100 if d["de"] else 0
        d5_pct = sum(1 for v in d["d5"] if abs(v) > 0.1) / n * 100 if d["d5"] else 0
        d15_pct = sum(1 for v in d["d15"] if abs(v) > 0.1) / n * 100 if d["d15"] else 0
        tpi_pct = sum(1 for v in d["tpi"] if abs(v) > 0.1) / n * 100 if d["tpi"] else 0
        print("    {:25s} {:>8d} {:>11.1f}% {:>9.1f}% {:>9.1f}% {:>9.1f}% {:>9.1f}%".format(
            reg, n, dr_pct, de_pct, d5_pct, d15_pct, tpi_pct))

    # Mean values per regime
    print("\n    {:25s} {:>12s} {:>12s} {:>12s} {:>12s}".format(
        "Regime", "Mean DE", "Mean D5", "Mean D15", "Mean TPI"))
    print("    " + "-" * 73)
    for reg, d in sorted(rdata.items(), key=lambda x: -len(x[1]["exec_drift"])):
        n = len(d["exec_drift"])
        if n < 10:
            continue
        print("    {:25s} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}".format(
            reg,
            sum(d["de"]) / n,
            sum(d["d5"]) / n,
            sum(d["d15"]) / n,
            sum(d["tpi"]) / n))

    # Final verdict
    print("\n" + "=" * 72)
    print("  FINAL VERDICT")
    print("=" * 72)

    print("""
  Q1: Can any candidate produce non-zero signals when exec_drift=0 ({:.1f}% cycles)?
  """.format(pct_zero))

    de_znz_rt = len(de_znz)/len(de_z)*100 if de_z else 0
    d5_znz_rt = len(d5_znz)/len(d5_z)*100 if d5_z else 0
    d15_znz_rt = len(d15_znz)/len(d15_z)*100 if d15_z else 0
    tpi_znz_rt = len(tpi_znz)/len(tpi_z)*100 if tpi_z else 0

    print("     Directional Energy: {} ({}/{} = {:.1f}%)".format(
        "YES" if de_znz else "NO", len(de_znz), len(de_z), de_znz_rt))
    print("     D5:                 {} ({}/{} = {:.1f}%)".format(
        "YES" if d5_znz else "NO", len(d5_znz), len(d5_z), d5_znz_rt))
    print("     D15:                {} ({}/{} = {:.1f}%)".format(
        "YES" if d15_znz else "NO", len(d15_znz), len(d15_z), d15_znz_rt))
    print("     TPI:                {} ({}/{} = {:.1f}%)".format(
        "YES" if tpi_znz else "NO", len(tpi_znz), len(tpi_z), tpi_znz_rt))

    print("""
  Q2: When exec_drift IS non-zero, how often do sensors agree on sign?
  """)

    fa = 0
    ma = 0
    for i in range(trim):
        act = nz_act[i]
        sensors = [de_nz_sig[i], d5_nz_sig[i], d15_nz_sig[i], tpi_nz_sig[i]]
        nz_sens = [s for s in sensors if s != 0]
        if nz_sens:
            if all(s == act for s in nz_sens):
                fa += 1
            # Majority: count sensors matching actual > sensors not matching
            match = sum(1 for s in nz_sens if s == act)
            if match > len(nz_sens) / 2:
                ma += 1

    if trim:
        print("     Full consensus: {} ({} = {:.1f}%)".format(
            fa, trim, fa/trim*100))
        print("     Majority agreement: {} ({} = {:.1f}%)".format(
            ma, trim, ma/trim*100))

    de_a = sum(1 for a, p in zip(nz_act, de_nz_sig) if a == p)
    d5_a = sum(1 for a, p in zip(nz_act, d5_nz_sig) if a == p)
    d15_a = sum(1 for a, p in zip(nz_act, d15_nz_sig) if a == p)
    tpi_a = sum(1 for a, p in zip(nz_act, tpi_nz_sig) if a == p)
    ev_a = sum(1 for a, p in zip(nz_act, votes) if a == p)

    print("\n     Sign-match against exec_drift (non-zero cycles):")
    print("       Directional Energy: {}/{} ({:.1f}%)".format(de_a, trim, de_a/trim*100 if trim else 0))
    print("       D5:                 {}/{} ({:.1f}%)".format(d5_a, trim, d5_a/trim*100 if trim else 0))
    print("       D15:                {}/{} ({:.1f}%)".format(d15_a, trim, d15_a/trim*100 if trim else 0))
    print("       TPI:                {}/{} ({:.1f}%)".format(tpi_a, trim, tpi_a/trim*100 if trim else 0))
    print("       Ensemble:           {}/{} ({:.1f}%)".format(ev_a, trim, ev_a/trim*100 if trim else 0))

    # Best candidates
    zd_cands = [("Directional Energy", de_znz_rt), ("D5", d5_znz_rt), ("D15", d15_znz_rt), ("TPI", tpi_znz_rt)]
    zd_cands.sort(key=lambda x: -x[1])
    sig_cands = [("DE", de_a/trim) if trim else ("DE", 0),
                 ("D5", d5_a/trim) if trim else ("D5", 0),
                 ("D15", d15_a/trim) if trim else ("D15", 0),
                 ("TPI", tpi_a/trim) if trim else ("TPI", 0),
                 ("Ensemble", ev_a/trim) if trim else ("Ensemble", 0)]
    sig_cands.sort(key=lambda x: -x[1])

    print("""
  Best zero-drift signal detector: {} ({:.1f}% non-zero)
  Best sign predictor:             {} ({:.1f}% agreement)

  Architecture implication:
  - Current exec_drift is 0 in {:.1f}% of cycles -> near-dead sensor
  - Directional Energy fires in {:.1f}% of cycles -> live sensor
  - Moving averages (D5/D15) provide drift memory but lag
  """.format(
        zd_cands[0][0], zd_cands[0][1],
        sig_cands[0][0], sig_cands[0][1] * 100,
        pct_zero, de_pct_nz))

    print("=" * 72)
    print("  ANALYSIS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
