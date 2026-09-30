from __future__ import annotations

import time
from threading import Event

from pdm.forecast_worker import ForecastWorker


def _completed(worker):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        snapshot = worker.snapshot()
        if snapshot["done"]:
            return snapshot
        time.sleep(.005)
    raise AssertionError("Background forecast did not complete")


def test_polling_is_nonblocking_and_does_not_restart_the_calculation():
    worker = ForecastWorker()
    published, release = Event(), Event()
    calls = []

    def compute(*, should_stop, progress_cb):
        calls.append(1)
        payload = {"points": [{"value": 1}]}
        progress_cb(payload)
        payload["points"][0]["value"] = 999
        published.set()
        assert release.wait(3)
        return {"points": [{"value": 2}]}

    try:
        worker.request((1,), compute)
        assert published.wait(3)
        for _ in range(10):
            poll = worker.request((1,), compute)
            assert not poll["done"] and poll["result"]["points"][0]["value"] == 1
            poll["result"]["points"][0]["value"] = -1
        assert calls == [1] and not release.is_set()
        release.set()
        assert _completed(worker)["result"] == {"points": [{"value": 2}]}
    finally:
        release.set()
        worker.close()


def test_new_cursor_cancels_old_work_and_only_latest_queued_request_runs():
    worker = ForecastWorker()
    started, release, cancelled = Event(), Event(), Event()
    calls = []

    def old(*, should_stop, progress_cb):
        started.set()
        assert release.wait(3)
        if should_stop():
            cancelled.set()
        # Even a misbehaving worker ignoring cancellation cannot publish stale data.
        progress_cb({"cursor": 1})
        return {"cursor": 1}

    def new(*, should_stop, progress_cb):
        calls.append(3)
        return {"cursor": 3}

    try:
        worker.request((1,), old)
        assert started.wait(3)
        worker.request((2,), lambda **kwargs: calls.append(2))
        worker.request((3,), new)
        assert worker.snapshot()["result"] is None
        release.set()
        latest = _completed(worker)
        assert cancelled.is_set() and calls == [3]
        assert latest["key"] == (3,) and latest["result"] == {"cursor": 3}
        worker.cancel()
        assert worker.snapshot()["key"] is None
    finally:
        release.set()
        worker.close()


def test_background_errors_surface_once_instead_of_restarting_forever():
    worker = ForecastWorker()
    calls = []

    def broken(**kwargs):
        calls.append(1)
        raise ValueError("Invalid artifact")

    try:
        worker.request((1,), broken)
        result = _completed(worker)
        assert result["error"] == "ValueError: Invalid artifact"
        assert worker.request((1,), broken)["done"]
        assert calls == [1]
    finally:
        worker.close()
