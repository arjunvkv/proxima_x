"""Account registry — prevents two strategies from using the same MT5 account.

File-based lock at live/account_registry.json. Each running strategy claims its
account on start and releases on stop. Heartbeat detects crashed processes.
"""
import json, os, time

from . import config

REGISTRY_PATH = os.path.join(config.LIVE_DIR, "account_registry.json")
HEARTBEAT_INTERVAL = 30
STALE_AFTER = 120


def _read():
    if not os.path.exists(REGISTRY_PATH):
        return {}
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _write(data):
    os.makedirs(os.path.dirname(REGISTRY_PATH), exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(data, f, indent=2)


def stale_entries(data):
    now = time.time()
    stale = []
    for login, entry in list(data.items()):
        if now - entry.get("heartbeat", 0) > STALE_AFTER:
            stale.append(login)
    return stale


def claim(login, strategy_name):
    """Claim an MT5 account for a strategy. Returns (ok: bool, reason: str)."""
    login_s = str(login)
    data = _read()

    # Remove stale first
    for s_login in stale_entries(data):
        del data[s_login]

    if login_s in data:
        existing = data[login_s]
        # Same strategy re-claiming is allowed (e.g. restart)
        if existing["strategy"] == strategy_name:
            data[login_s]["claimed_at"] = time.time()
            data[login_s]["heartbeat"] = time.time()
            data[login_s]["pid"] = os.getpid()
            _write(data)
            return True, "re-claimed"
        return False, f"account {login} already claimed by '{existing['strategy']}' (pid {existing.get('pid','?')}, last heartbeat {int(time.time()-existing.get('heartbeat',0))}s ago)"

    data[login_s] = {
        "strategy": strategy_name,
        "claimed_at": time.time(),
        "heartbeat": time.time(),
        "pid": os.getpid(),
    }
    _write(data)
    return True, ""


def release(login, strategy_name):
    """Release a claimed account. Only the claiming strategy can release it."""
    login_s = str(login)
    data = _read()
    if login_s in data and data[login_s]["strategy"] == strategy_name:
        del data[login_s]
        _write(data)
        return True
    return False


def heartbeat(login, strategy_name):
    """Update heartbeat timestamp for a claimed account."""
    login_s = str(login)
    data = _read()
    if login_s in data and data[login_s]["strategy"] == strategy_name:
        data[login_s]["heartbeat"] = time.time()
        _write(data)
        return True
    return False


def list_claims():
    """Return dict of all active claims (login -> info)."""
    data = _read()
    now = time.time()
    return {
        k: {**v, "age_sec": int(now - v.get("claimed_at", 0))}
        for k, v in data.items()
        if now - v.get("heartbeat", 0) <= STALE_AFTER
    }


def cleanup():
    """Remove all stale entries. Returns number removed."""
    data = _read()
    stale = stale_entries(data)
    for s in stale:
        del data[s]
    _write(data)
    return len(stale)
