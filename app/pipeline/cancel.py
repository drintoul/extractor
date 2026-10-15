"""Cooperative cancellation registry. Nodes poll is_cancelled() so a kill
request aborts the graph at the next node boundary — completed fan-out work
stays checkpointed and the run can be resumed with the same run_id."""
import threading


class Cancelled(Exception):
    pass


_lock = threading.Lock()
_cancelled: set[str] = set()


def request(run_id: str) -> None:
    with _lock:
        _cancelled.add(run_id)


def is_cancelled(run_id: str) -> bool:
    with _lock:
        return run_id in _cancelled


def clear(run_id: str) -> None:
    with _lock:
        _cancelled.discard(run_id)
