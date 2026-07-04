import threading
import json
import time


class ShadowWorker:

    def __init__(self, shadow_core, output_path="state/shadow_trace.jsonl"):
        self.shadow_core = shadow_core
        self.output_path = output_path
        self.running = False
        self.latest_state = {}
        self.lock = threading.Lock()

    def start(self):
        self.running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def _run(self):
        with open(self.output_path, "a") as f:
            while self.running:
                try:
                    event = self.shadow_core.queue.get(timeout=1)
                except Exception:
                    continue

                f.write(json.dumps(event) + "\n")

                with self.lock:
                    self.latest_state = event

    def get_latest(self):
        with self.lock:
            return self.latest_state

    def stop(self):
        self.running = False
