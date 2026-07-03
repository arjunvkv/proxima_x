import time
from typing import Callable, Optional
from ..core.fsv_engine import FSVEngine
from ..ingestion.event_normalizer import MacroEventNormalizer, NormalizedEvent


class FSVEventLoop:
    def __init__(self, engine: FSVEngine = None, normalizer: MacroEventNormalizer = None) -> None:
        self.engine: FSVEngine = engine or FSVEngine()
        self.normalizer: MacroEventNormalizer = normalizer or MacroEventNormalizer()
        self.event_queue: list[NormalizedEvent] = []
        self.running: bool = False
        self.cycle_count: int = 0
        self.max_queue_size: int = 10000
        self.last_decay_time: float = time.time()

    def start_loop(self) -> bool:
        self.running = True
        return True

    def stop_loop(self) -> bool:
        self.running = False
        return False

    def ingest_event(self, event: NormalizedEvent) -> bool:
        if len(self.event_queue) >= self.max_queue_size:
            return False
        self.event_queue.append(event)
        return True

    def ingest_raw(self, raw_event: dict, source: str) -> bool:
        try:
            event = self.normalizer.normalize(raw_event, source)
        except ValueError:
            return False
        return self.ingest_event(event)

    def run_cycle(self, current_time: float = None) -> int:
        if not self.running:
            return 0
        processed = 0
        for event in self.event_queue:
            self.engine.update_with_event(event)
            processed += 1
        self.engine.decay_all(current_time or time.time())
        self.event_queue.clear()
        self.cycle_count += 1
        return processed

    def process_events(self, max_events: int = 100) -> int:
        count = 0
        to_process = self.event_queue[:max_events]
        for event in to_process:
            self.engine.update_with_event(event)
            count += 1
        self.event_queue = self.event_queue[max_events:]
        return count

    def get_queue_size(self) -> int:
        return len(self.event_queue)

    def get_engine_stats(self) -> dict:
        return self.engine.get_stats()

    def health_check(self) -> dict:
        return {
            "running": self.running,
            "queue_size": self.get_queue_size(),
            "cycle_count": self.cycle_count,
            "engine_update_count": self.engine.update_count,
            "event_log_size": len(self.engine.event_log),
        }


class BackgroundIngestor:
    def __init__(self, event_loop: FSVEventLoop, poll_interval: float = 1.0) -> None:
        self.loop: FSVEventLoop = event_loop
        self.poll_interval: float = poll_interval

    def poll_source(self, source_func: Callable[[], list[dict]], source_name: str) -> int:
        raw_events = source_func()
        count = 0
        for event in raw_events:
            if self.loop.ingest_raw(event, source_name):
                count += 1
        return count

    def run_once(self, sources: list[tuple[Callable[[], list[dict]], str]]) -> dict:
        counts: dict = {}
        for source_func, source_name in sources:
            counts[source_name] = self.poll_source(source_func, source_name)
        return counts
