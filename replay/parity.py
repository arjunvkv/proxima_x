import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime
from typing import Optional

logger = logging.getLogger("proxima.replay.parity")

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
ENGINE_VERSION = "tick-time-machine-1.0"


def _sha(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonicalize(obj, decimals: int = 8):
    if isinstance(obj, float):
        return round(obj, decimals)
    if isinstance(obj, dict):
        return {k: _canonicalize(v, decimals) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(v, decimals) for v in obj]
    return obj


def _find_repo_root(path: str) -> str:
    """Walk up from path until a .git directory is found."""
    current = os.path.abspath(path)
    while True:
        parent = os.path.dirname(current)
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        if current == parent:
            return None
        current = parent


def _get_git_sha() -> str:
    try:
        root = _find_repo_root(os.path.dirname(os.path.abspath(__file__)))
        if root:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=root,
            )
            if result.returncode == 0:
                return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _get_archive_hash(archive_dir: str = None) -> str:
    if archive_dir is None:
        archive_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ticks")
    if not os.path.isdir(archive_dir):
        return "unknown"
    try:
        hasher = hashlib.sha256()
        for fname in sorted(os.listdir(archive_dir)):
            if fname.endswith(".parquet"):
                fpath = os.path.join(archive_dir, fname)
                with open(fpath, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return "unknown"


class ParityLedger:
    def __init__(self, symbol: str = "", seed: int = 42):
        self._symbol = symbol
        self._seed = seed
        self._ticks: list[dict] = []
        self._signals: list[dict] = []
        self._trades: list[dict] = []
        self._state: dict = {}

    def add_tick(self, tick: dict):
        self._ticks.append({
            "symbol": tick.get("symbol"),
            "bid": tick.get("bid"),
            "ask": tick.get("ask"),
            "time_sec": tick.get("time_sec", tick.get("timestamp")),
        })

    def add_signal(self, signal: dict):
        self._signals.append(signal)

    def add_trade(self, trade: dict):
        self._trades.append(trade)

    def finalize(self, broker_state: dict):
        self._state = dict(broker_state) if broker_state else {}

    @property
    def h_ticks(self) -> str:
        return _sha(_canonicalize(self._ticks))

    @property
    def h_signals(self) -> str:
        return _sha(_canonicalize(self._signals))

    @property
    def h_trades(self) -> str:
        return _sha(_canonicalize(self._trades))

    @property
    def h_state(self) -> str:
        return _sha(_canonicalize(self._state))

    def build(self) -> dict:
        return {
            "H_ticks": self.h_ticks,
            "H_signals": self.h_signals,
            "H_trades": self.h_trades,
            "H_state": self.h_state,
            "tick_count": len(self._ticks),
            "signal_count": len(self._signals),
            "trade_count": len(self._trades),
        }

    def _build_meta(self) -> dict:
        return {
            "_meta": {
                "engine_version": ENGINE_VERSION,
                "git_sha": _get_git_sha(),
                "archive_hash": _get_archive_hash(),
                "generated_at": datetime.utcnow().isoformat(),
            },
        }

    def save(self, name: str = None) -> str:
        if name is None:
            name = f"{self._symbol}_seed{self._seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        path = os.path.join(GOLDEN_DIR, f"{name}.json")
        payload = {**self.build(), **self._build_meta()}
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Parity snapshot saved to {path}")
        return path

    def compare(self, other: dict) -> list[str]:
        mine = self.build()
        diffs = []
        for k in mine:
            if k not in other:
                diffs.append(f"{k}: missing in other")
            elif mine[k] != other[k]:
                diffs.append(f"{k}: {mine[k][:16]}... != {other[k][:16]}...")
        return diffs

    def assert_match(self, other: dict):
        diffs = self.compare(other)
        if diffs:
            raise AssertionError(f"Parity mismatch:\n" + "\n".join(diffs))
        logger.info("Parity check PASSED — all hashes match")


def load_golden(name: str) -> Optional[dict]:
    path = os.path.join(GOLDEN_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
