import json
import time
import hashlib
import os
from pathlib import Path
from typing import Optional
from data.models import StateEnvelope

SCHEMA_VERSION = 1

class SnapshotManager:
    def __init__(self, state_dir: Optional[str] = None):
        if state_dir is None:
            appdata = os.environ.get("APPDATA", str(Path.home() / ".local"))
            state_dir = str(Path(appdata) / "CurrencyDecomposition" / "state")
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.current_path = self.state_dir / "current.json"
        self.previous_path = self.state_dir / "previous.json"

    def save(self, payload: dict) -> bool:
        try:
            envelope = {
                "market_timestamp": payload.get("market_timestamp", time.time()),
                "wall_timestamp": time.time(),
                "schema_version": SCHEMA_VERSION,
                "payload": payload
            }
            serialized = json.dumps(envelope, indent=2, default=str)
            envelope["checksum"] = hashlib.sha256(serialized.encode()).hexdigest()
            serialized = json.dumps(envelope, indent=2, default=str)

            tmp_path = self.state_dir / "current.tmp"
            with open(tmp_path, "w") as f:
                f.write(serialized)
                f.flush()
                os.fsync(f.fileno())

            if self.current_path.exists():
                if self.previous_path.exists():
                    self.previous_path.unlink()
                self.current_path.rename(self.previous_path)

            tmp_path.rename(self.current_path)
            return True
        except Exception:
            return False

    def load(self) -> Optional[dict]:
        for path in [self.current_path, self.previous_path]:
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    data = json.load(f)
                checksum = data.get("checksum", "")
                data_no_cs = {k: v for k, v in data.items() if k != "checksum"}
                serialized = json.dumps(data_no_cs, indent=2, default=str)
                expected = hashlib.sha256(serialized.encode()).hexdigest()
                if checksum and checksum != expected:
                    continue
                payload = data.get("payload", data)
                if not isinstance(payload, dict):
                    continue
                if "market_timestamp" not in payload:
                    payload["market_timestamp"] = data.get("market_timestamp", 0.0)
                return payload
            except (json.JSONDecodeError, KeyError, IOError):
                continue
        return None

    def clear(self) -> None:
        for path in [self.current_path, self.previous_path]:
            if path.exists():
                path.unlink()

