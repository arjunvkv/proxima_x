"""Central magic number generator for Proxima X.

Encodes strategy + direction into every order's magic number so that
any trade can be attributed back to its originating signal source.

Magic number layout:
    BASE + direction_offset(0|100) + instance(1..99)

    BUY  → BASE + 0 + instance  (range:  [BASE+1, BASE+99])
    SELL → BASE + 100 + instance (range:  [BASE+101, BASE+199])
"""

MAGIC_BASES = {
    "OSS": 406000,
    "SHADOW": 407000,
    "FUSION": 408000,
    "ERL_PRESSURE": 409000,
    "ERL_MOMENTUM": 410000,
    "EDGE_SIGNAL": 411000,
    "EXPLORATION": 412000,
    "PROXIMA_V2": 202000,
    "PROXIMA_V4": 234000,
    "PROXIMA_V5": 235000,
}

# Legacy magic numbers kept for backward compatibility in position lookups
LEGACY_MAGICS = {
    202406: "PROXIMA_V2",
    20240630: "EXPLORATION",
    234000: "PROXIMA_V4",
}


def generate_magic(strategy_id: str, direction: str, instance: int = 1) -> int:
    """Generate a structured magic number for a given strategy + direction.

    Args:
        strategy_id: One of the keys in MAGIC_BASES.
        direction: "BUY" or "SELL".
        instance: Instance counter (1..99) for disambiguation.

    Returns:
        Unique magic integer for the order stream.
    """
    base = MAGIC_BASES.get(strategy_id, 406000)
    dir_offset = 100 if direction == "SELL" else 0
    return base + dir_offset + instance


def parse_magic(magic: int) -> dict:
    """Reverse-lookup what strategy/direction/instance a magic number represents.

    Also recognizes legacy magic numbers via LEGACY_MAGIC.

    Returns:
        dict with keys: strategy, direction, instance (or 'UNKNOWN').
    """
    # Check legacy first
    if magic in LEGACY_MAGICS:
        return {"strategy": LEGACY_MAGICS[magic], "direction": "UNKNOWN", "instance": 1}

    for strategy, base in MAGIC_BASES.items():
        if base <= magic < base + 200:
            offset = magic - base
            if offset >= 100:
                return {"strategy": strategy, "direction": "SELL", "instance": offset - 100}
            else:
                return {"strategy": strategy, "direction": "BUY", "instance": offset}
    return {"strategy": "UNKNOWN", "magic": magic}


def infer_strategy_from_comment(comment: str) -> str:
    """Best-effort strategy inference from an order comment."""
    if not comment:
        return "UNKNOWN"
    upper = comment.upper()
    for sid in ("OSS", "SHADOW", "FUSION", "ERL_PRESSURE", "ERL_MOMENTUM",
                "EDGE_SIGNAL", "EXPLORATION", "PROXIMA_V2", "PROXIMA_V4", "PROXIMA_V5"):
        if sid in upper:
            return sid
    return "UNKNOWN"