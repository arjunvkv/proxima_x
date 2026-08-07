"""run_three_5min_trials.py — 3 live-vs-engine trials, 5 minutes apart.

Each trial: full 18-symbol micro-batch (54 x 0.01-lot round trips on live
FTMO-Demo) + broker-ledger reconcile vs offline engine. Outputs saved per trial
(trial_N_batch.json / trial_N_reconcile.json) so results are preserved.
Run with the terminal up and NO daemon attached (batch uses the IPC).
"""
import subprocess, sys, time, json, shutil, os

REPO = r"C:\Trading\Proxima_X"
BATCH = os.path.join(REPO, "scripts", "verify_live_micro_batch.py")
RECON = os.path.join(REPO, "scripts", "reconcile_live_history_deals.py")
PY = os.path.join(REPO, ".venv", "Scripts", "python.exe")
GAP_S = 300   # 5 minutes between trials

def run(label, script):
    env = dict(os.environ)
    env["MT5_PATH"] = r"C:\Program Files\FTMO Global Markets MT5 Terminal\terminal64.exe"
    env.pop("PYTHONPATH", None)
    r = subprocess.run([PY, script], capture_output=True, text=True, timeout=540, env=env)
    ok = r.returncode == 0
    tail = "\n".join((r.stdout or "").splitlines()[-6:])
    print(f"[{label}] exit={r.returncode} :: {tail}", flush=True)
    if not ok:
        print(f"[{label}] STDERR: {(r.stderr or '')[-800:]}", flush=True)
    return ok

def main():
    results = []
    for trial in (1, 2, 3):
        print(f"\n========== TRIAL {trial} @ {time.strftime('%H:%M:%S')} ==========", flush=True)
        ok1 = run(f"trial{trial} batch", BATCH)
        if ok1:
            shutil.copy(os.path.join(REPO, "micro_batch_alignment.json"),
                        os.path.join(REPO, f"trial_{trial}_batch.json"))
        ok2 = run(f"trial{trial} reconcile", RECON)
        if ok2:
            shutil.copy(os.path.join(REPO, "reconcile_live_history_deals.json"),
                        os.path.join(REPO, f"trial_{trial}_reconcile.json"))
        res = {"trial": trial, "batch_ok": ok1, "reconcile_ok": ok2}
        if ok2:
            d = json.load(open(os.path.join(REPO, "reconcile_live_history_deals.json")))
            res.update({"positions": d["positions"], "gross_aligned": d["gross_aligned"],
                        "net_aligned": d["net_aligned"], "sum_ledger": d["sum_ledger_net"],
                        "sum_engine": d["sum_engine_net"], "sum_diff": d["sum_diff"],
                        "verdict": d["verdict"]})
        results.append(res)
        print(f"[trial{trial}] {json.dumps(res, default=str)}", flush=True)
        if trial < 3:
            print(f"--- waiting {GAP_S}s to next trial ---", flush=True)
            time.sleep(GAP_S)

    print("\n========== ALL TRIALS ==========")
    for r in results:
        print(json.dumps(r, default=str))
    all_pass = all(r.get("verdict") == "PASS" for r in results) and len(results) == 3
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())