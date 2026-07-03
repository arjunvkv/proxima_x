import json
import os
from datetime import datetime, date
from typing import Optional


class ResearchMemory:
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        self._dir = data_dir
        self._dailies: list[dict] = []
        self._weeklies: list[dict] = []
        self._load()

    def _path_for(self, kind: str) -> str:
        return os.path.join(self._dir, f"{kind}.json")

    def _load(self):
        for kind in ("dailies", "weeklies"):
            p = self._path_for(kind)
            if os.path.exists(p):
                try:
                    with open(p, encoding="utf-8") as f:
                        setattr(self, f"_{kind}", json.load(f))
                except json.JSONDecodeError:
                    backup = p + ".corrupt"
                    os.replace(p, backup)
                    setattr(self, f"_{kind}", [])
                    import logging
                    logging.getLogger("proxima.director").warning(
                        f"Corrupt {kind}.json quarantined to {backup}")

    def _save(self, kind: str):
        p = self._path_for(kind)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(getattr(self, f"_{kind}"), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)

    def store_daily(self, report: dict):
        report["_stored"] = date.today().isoformat()
        self._dailies.append(report)
        if len(self._dailies) > 365:
            self._dailies = self._dailies[-365:]
        self._save("dailies")

    def store_weekly(self, report: dict):
        report["_stored"] = date.today().isoformat()
        self._weeklies.append(report)
        if len(self._weeklies) > 104:
            self._weeklies = self._weeklies[-104:]
        self._save("weeklies")

    def latest_daily(self) -> Optional[dict]:
        return self._dailies[-1] if self._dailies else None

    def recent_dailies(self, n: int = 7) -> list[dict]:
        return self._dailies[-n:]

    def latest_weekly(self) -> Optional[dict]:
        return self._weeklies[-1] if self._weeklies else None

    def count_dailies(self) -> int:
        return len(self._dailies)

    def count_weeklies(self) -> int:
        return len(self._weeklies)
