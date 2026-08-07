"""Live-port — emit a deployable MT5 runtime from a validated StrategySpec.

Lessons encoded (from run_tokyo_h0_live.py + the MT5 IPC gauntlet):
  * ATTACH-ONLY: never re-init to a different terminal path, never login to
    wrong-account creds. Pass MT5_PATH env; neutralize settings creds before
    connector.connect().
  * SERVER-CLOCK gating: gate on (server_tick_time//3600)%24, never host wall clock.
  * FILL-BAR fidelity: only fill while the fill M5 bar is live (POST_FILL_TOL_S);
    once closed, skip the day.
  * single terminal64.exe; JSON state file for day-dedup; --execute/--manage live.
  * NO cron: the live daemon is a manual --daemon process.

A StrategySpec is plain data, so the same JSON that backtested ships to live
unchanged. emit_live_manifest writes the runtime contract; the generic MT5
engine entrypoint consumes it.
"""
from __future__ import annotations
import json, os

from .spec import StrategySpec

LIVE_DEFAULTS = {
    "post_only": True,
    "post_fill_tol_s": 300,
    "server_clock": True,          # server-tick gating, never host wall clock
    "attach_only": True,           # never re-init path / never re-login
    "no_cron": True,               # manual --daemon, kept alive by hand
    "max_positions": 5,
    "state_file": "proxima_ops/state/{name}_state.json",
}


def emit_live_manifest(spec: StrategySpec, out_path: str,
                       terminal_path: str,
                       account: str = "",
                       extra: dict | None = None) -> str:
    """Write the live runtime manifest (spec + runtime knobs) as JSON."""
    manifest = {
        "spec": spec.to_dict(),
        "runtime": {**LIVE_DEFAULTS,
                    "terminal64": terminal_path,
                    "account": account,
                    **(extra or {})},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return out_path