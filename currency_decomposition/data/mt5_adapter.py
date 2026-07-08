"""MT5 adapter using 1-minute rates. Single-owner worker thread for all MT5 API calls."""
import MetaTrader5 as mt5
import time
import threading
import uuid
from queue import Queue, Empty
from typing import Optional, Any
from config.settings import SYMBOLS
from data.models import Tick, TickBatch


class MT5Adapter:
    def __init__(self):
        self.connected = False
        self._sequence = 0
        self._last_bar_close: dict[str, float] = {}
        self._cmd_queue: Queue = Queue()
        self._results: dict[str, Any] = {}
        self._result_events: dict[str, threading.Event] = {}
        self._results_lock = threading.Lock()
        self._worker_thread: Optional[threading.Thread] = None
        self.running = False

    def _mt5_worker(self) -> None:
        try:
            ok = mt5.initialize()
            if ok:
                for sym in SYMBOLS:
                    try:
                        mt5.symbol_select(sym, True)
                    except Exception:
                        pass
        except Exception:
            ok = False
        self._deliver("__init__", ok)
        if not ok:
            self.running = False
            return

        while self.running:
            try:
                rid, command, args = self._cmd_queue.get(timeout=0.1)
                try:
                    result = command(*args)
                except Exception as exc:
                    result = exc
                self._deliver(rid, result)
            except Empty:
                continue

        try:
            mt5.shutdown()
        except Exception:
            pass

    def _deliver(self, rid: str, result: Any) -> None:
        with self._results_lock:
            self._results[rid] = result
            ev = self._result_events.pop(rid, None)
        if ev is not None:
            ev.set()

    def call_mt5(self, fn, *args, timeout: float = 10.0) -> Any:
        rid = str(uuid.uuid4())
        ev = threading.Event()
        with self._results_lock:
            self._result_events[rid] = ev
        self._cmd_queue.put((rid, fn, args))
        if not ev.wait(timeout=timeout):
            with self._results_lock:
                self._result_events.pop(rid, None)
                self._results.pop(rid, None)
            raise TimeoutError(f"MT5 call {fn.__name__} timed out after {timeout}s")
        with self._results_lock:
            result = self._results.pop(rid)
        if isinstance(result, BaseException):
            raise result
        return result

    def connect(self, retries: int = 3) -> bool:
        self.running = True
        self._worker_thread = threading.Thread(target=self._mt5_worker, daemon=True)
        self._worker_thread.start()
        deadline = time.time() + retries * 2.0
        while time.time() < deadline:
            with self._results_lock:
                if "__init__" in self._results:
                    ok = self._results.pop("__init__")
                    self.connected = bool(ok)
                    return self.connected
            time.sleep(0.2)
        self.connected = False
        return False

    def disconnect(self) -> None:
        self.running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=3.0)
        self.connected = False

    def _poll_ticks_impl(self) -> TickBatch:
        ticks = []
        latest_ts = 0.0
        for symbol in SYMBOLS:
            try:
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 2)
                if rates is None or len(rates) == 0:
                    continue
                latest = rates[-1]
                ts = float(latest[0])
                high = float(latest[2])
                low = float(latest[3])
                close = float(latest[4])
                vol = int(latest[5]) if len(latest) > 5 else 0

                prev_close = self._last_bar_close.get(symbol)
                if prev_close is not None and close == prev_close:
                    continue

                self._last_bar_close[symbol] = close

                tick = Tick(
                    symbol=symbol,
                    timestamp=ts,
                    bid=low,
                    ask=high,
                    volume=vol
                )
                ticks.append(tick)
                latest_ts = max(latest_ts, ts)
            except Exception:
                pass
        self._sequence += 1
        return TickBatch(
            ticks=ticks,
            market_timestamp=latest_ts or time.time(),
            sequence=self._sequence,
            received_timestamp=time.time()
        )

    def poll_ticks(self) -> TickBatch:
        return self.call_mt5(self._poll_ticks_impl)

    def get_symbol_info(self, symbol: str):
        return self.call_mt5(mt5.symbol_info, symbol)

    def get_rates(self, symbol: str, timeframe: int, count: int):
        return self.call_mt5(mt5.copy_rates_from_pos, symbol, timeframe, 0, count)

    def audit_symbols(self) -> dict:
        from config.settings import SYMBOLS
        symbols = self.call_mt5(mt5.symbols_get)
        if symbols is None:
            return {"available": {}, "missing": list(SYMBOLS)}
        names = {s.name for s in symbols}
        available = {}
        missing = []
        for expected in SYMBOLS:
            matches = [x for x in names if x.startswith(expected)]
            if matches:
                available[expected] = matches
            else:
                missing.append(expected)
        return {"available": available, "missing": missing}

    def latest_tick(self, symbol: str) -> Optional[Tick]:
        rates = self.call_mt5(mt5.copy_rates_from_pos, symbol, mt5.TIMEFRAME_M1, 0, 1)
        if rates is not None and len(rates) > 0:
            r = rates[-1]
            return Tick(symbol=symbol, timestamp=float(r[0]), bid=float(r[3]), ask=float(r[4]),
                        volume=int(r[5]) if len(r) > 5 else 0)
        return None
