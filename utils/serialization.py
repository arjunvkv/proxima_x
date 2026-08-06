"""utils/serialization.py — JSON read/write helpers (restored)."""
import json
import logging
import os
from typing import Any

logger = logging.getLogger("proxima.utils.serialization")


def save_json(path, obj: Any) -> None:
    """Serialize obj to JSON at path (mkdir -p parents)."""
    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, default=str, indent=1)


def load_json(path) -> Any:
    """Deserialize JSON from path (returns {} if missing/corrupt)."""
    path = os.fspath(path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError) as e:
        logger.warning(f"[SERIALIZATION] Failed to load {path}: {e}")
        return {}