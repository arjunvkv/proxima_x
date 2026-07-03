from mvs.models.market_plane import MarketRealityPlane
from mvs.models.perception_plane import PerceptionStatePlane
from mvs.models.action_plane import ActionStatePlane
from mvs.models.outcome_plane import OutcomeStatePlane
from mvs.storage.duckdb_store import TruthStore


def truther(db_path: str = "mvs.duckdb", flush_batch_size: int = 256) -> TruthStore:
    return TruthStore(db_path=db_path, flush_batch_size=flush_batch_size)


__all__ = [
    "MarketRealityPlane",
    "PerceptionStatePlane",
    "ActionStatePlane",
    "OutcomeStatePlane",
    "TruthStore",
    "truther",
]
