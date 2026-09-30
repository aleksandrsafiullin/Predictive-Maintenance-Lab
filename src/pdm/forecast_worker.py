"""One cancellable forecast at a time per replay session, without Streamlit calls."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from threading import Event, Lock


@dataclass
class _Job:
    key: tuple
    stop: Event = field(default_factory=Event)
    result: dict | None = None
    error: str | None = None
    done: bool = False


class ForecastWorker:
    """Repeated polls reuse a job; a new key cancels and replaces old work.

    A single executor bounds CPU/memory and queued requests are cancelled before
    replacement. Results belong to their immutable key, including the cursor;
    stale publications can never become the active chart's prediction.
    """

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="signal-replay")
        self._lock = Lock()
        self._job = None
        self._future = None

    def request(self, key: tuple, compute) -> dict:
        with self._lock:
            if self._job is None or self._job.key != key:
                self._cancel_locked()
                self._job = _Job(key)
                self._future = self._pool.submit(self._run, self._job, compute)
        return self.snapshot()

    def snapshot(self) -> dict:
        with self._lock:
            job = self._job
            if job is None:
                return {"key": None, "result": None, "error": None, "done": False}
            return {"key": job.key, "result": deepcopy(job.result), "error": job.error, "done": job.done}

    def _run(self, job: _Job, compute) -> None:
        def publish(result):
            with self._lock:
                if not job.stop.is_set() and self._job is job:
                    job.result = deepcopy(result)

        try:
            if job.stop.is_set():
                return
            result = compute(should_stop=job.stop.is_set, progress_cb=publish)
            publish(result)
        except InterruptedError:
            pass
        except Exception as exc:
            with self._lock:
                if not job.stop.is_set() and self._job is job:
                    job.error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                job.done = True

    def _cancel_locked(self) -> None:
        if self._job is not None:
            self._job.stop.set()
        if self._future is not None:
            self._future.cancel()

    def cancel(self) -> None:
        with self._lock:
            self._cancel_locked()
            self._job = None

    def close(self) -> None:
        self.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
