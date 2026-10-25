from __future__ import annotations

import contextvars
import dataclasses
import json
import logging
import math
import queue
import threading
import time
import traceback
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

TRACE_SCHEMA_VERSION = 1
_STOP = object()


@dataclasses.dataclass(frozen=True)
class _TraceContext:
    events: list[dict[str, Any]]
    started_monotonic_s: float
    scope: dict[str, Any]


_active_context: contextvars.ContextVar[_TraceContext | None] = contextvars.ContextVar(
    "dex_teleop_control_trace", default=None
)


def _protobuf_to_dict(value: Any) -> dict[str, Any] | None:
    descriptor = getattr(value, "DESCRIPTOR", None)
    fields = getattr(descriptor, "fields", None)
    if fields is None:
        return None
    result: dict[str, Any] = {}
    try:
        for field in fields:
            result[field.name] = _to_jsonable(getattr(value, field.name))
    except Exception:
        return None
    return result


def _to_jsonable(value: Any) -> Any:
    """Snapshot common control values without letting tracing affect control."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isfinite(number):
            return number
        if math.isnan(number):
            return "NaN"
        return "Infinity" if number > 0 else "-Infinity"
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _to_jsonable(value.tolist())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _to_jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_to_jsonable(item) for item in sorted(value, key=repr)]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, BaseException):
        return _exception_data(value)
    protobuf = _protobuf_to_dict(value)
    if protobuf is not None:
        return protobuf
    if hasattr(value, "items"):
        try:
            return {str(key): _to_jsonable(item) for key, item in value.items()}
        except Exception:
            pass
    return repr(value)


def _exception_data(exc: BaseException) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "message": str(exc),
    }
    if exc.__traceback__ is not None:
        result["traceback"] = traceback.format_exception(
            type(exc), exc, exc.__traceback__
        )
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        result["cause"] = {
            "type": f"{type(cause).__module__}.{type(cause).__qualname__}",
            "message": str(cause),
        }
    return result


def trace_event(name: str, /, **data: Any) -> None:
    """Append a structured event to the active trace span, if one exists."""
    context = _active_context.get()
    if context is None:
        return
    try:
        now = time.monotonic()
        context.events.append(
            {
                "name": str(name),
                "monotonic_s": now,
                "offset_s": now - context.started_monotonic_s,
                "scope": _to_jsonable(context.scope),
                "data": _to_jsonable(data),
            }
        )
    except Exception as exc:  # diagnostics must never change a control outcome
        logger.warning("Failed to capture control trace event %s: %s", name, exc)


@contextmanager
def trace_scope(**labels: Any) -> Iterator[None]:
    """Add labels such as arm_id to events emitted in a nested control call."""
    context = _active_context.get()
    if context is None:
        yield
        return
    scoped = _TraceContext(
        events=context.events,
        started_monotonic_s=context.started_monotonic_s,
        scope={**context.scope, **_to_jsonable(labels)},
    )
    token = _active_context.set(scoped)
    try:
        yield
    finally:
        _active_context.reset(token)


class ControlTracer:
    """Non-blocking writer for ordered, structured control trace records."""

    def __init__(self, path: str | Path, *, queue_size: int = 4096) -> None:
        if int(queue_size) <= 0:
            raise ValueError("queue_size must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex
        self._queue: queue.Queue[dict[str, Any] | object] = queue.Queue(
            maxsize=int(queue_size)
        )
        self._file = self.path.open("a", encoding="utf-8", buffering=1)
        self._closed = False
        self._writer_error: BaseException | None = None
        self._enqueued_records = 0
        self._written_records = 0
        self._dropped_records = 0
        self._warned_full = False
        self._thread = threading.Thread(
            target=self._writer_loop,
            name="dex-teleop-control-trace",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped_records(self) -> int:
        return self._dropped_records

    @property
    def writer_error(self) -> BaseException | None:
        return self._writer_error

    def _base_record(self, record_type: str) -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "session_id": self.session_id,
            "record_type": str(record_type),
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "recorded_monotonic_s": time.monotonic(),
        }

    def record(self, record_type: str, /, **data: Any) -> bool:
        if self._closed:
            return False
        record = self._base_record(record_type)
        record.update(_to_jsonable(data))
        if self._writer_error is not None:
            self._dropped_records += 1
            return False
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_records += 1
            if not self._warned_full:
                self._warned_full = True
                logger.warning(
                    "Control trace queue is full; records are being dropped (%s)",
                    self.path,
                )
            return False
        self._enqueued_records += 1
        return True

    def event(self, name: str, /, **data: Any) -> bool:
        return self.record("event", name=name, data=data)

    @contextmanager
    def span(self, record_type: str, /, **metadata: Any) -> Iterator[None]:
        started = time.monotonic()
        events: list[dict[str, Any]] = []
        context = _TraceContext(events=events, started_monotonic_s=started, scope={})
        token = _active_context.set(context)
        status = "ok"
        exception: dict[str, Any] | None = None
        try:
            yield
        except KeyboardInterrupt as exc:
            status = "interrupted"
            exception = _exception_data(exc)
            raise
        except BaseException as exc:
            status = "error"
            exception = _exception_data(exc)
            raise
        finally:
            ended = time.monotonic()
            _active_context.reset(token)
            payload = {
                **metadata,
                "started_monotonic_s": started,
                "ended_monotonic_s": ended,
                "duration_s": ended - started,
                "status": status,
                "events": events,
            }
            if exception is not None:
                payload["exception"] = exception
            self.record(record_type, **payload)

    def _writer_loop(self) -> None:
        try:
            while True:
                record = self._queue.get()
                try:
                    if record is _STOP:
                        summary = self._base_record("trace_summary")
                        summary.update(
                            enqueued_records=self._enqueued_records,
                            written_records_before_summary=self._written_records,
                            dropped_records=self._dropped_records,
                            writer_error=None,
                        )
                        self._file.write(
                            json.dumps(
                                summary,
                                ensure_ascii=False,
                                allow_nan=False,
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        return
                    line = json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    self._file.write(line + "\n")
                    self._written_records += 1
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            self._writer_error = exc
            logger.error("Control trace writer failed for %s: %s", self.path, exc)
        finally:
            try:
                self._file.flush()
            finally:
                self._file.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer_error is None:
            try:
                self._queue.put(_STOP, timeout=2.0)
            except queue.Full:
                self._dropped_records += 1
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            logger.error("Timed out closing control trace writer for %s", self.path)
        if self._writer_error is not None:
            logger.error("Control trace is incomplete: %s", self._writer_error)

    def __enter__(self) -> ControlTracer:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
