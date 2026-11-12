"""Bounded, subprocess-isolated batch resolver API (``resolve_many``).

This module runs a *finite* batch of typed
:class:`pipenv.resolver.schema.ResolverRequest` envelopes — each carrying
a caller-assigned stable ID — through a hard-bounded pool of fresh
``spawn``-style resolver subprocesses and returns one complete
:class:`pipenv.resolver.schema.ResolverResponse` per input ID, including
deterministic cancel / timeout / crash terminal states.

Why a process per task (and never a thread)?
    The existing resolver entries (``resolve_for_pipenv`` /
    ``resolve_packages`` and the pip marker helpers they reach) mutate
    process-global state: ``os.environ`` (``_apply_request_env`` /
    ``HackedPythonVersion``) and ``packaging.markers.default_environment``
    (``_patched_marker_environment``).  Calling them concurrently from
    threads would cross-contaminate marker environments, index
    credentials and working directories between requests.  Every batch
    task therefore gets its OWN freshly spawned interpreter, built from
    an explicit per-task environment dict and ``cwd``.  The parent
    process never mutates ``os.environ`` itself — it only snapshots it
    once and hands copies to children — so the caller's environment and
    every sibling request stay untouched.

Transport safety:
    * Tasks cross the process boundary as restricted JSON files written
      by the parent and read by the canonical
      ``pipenv-resolver --request-file / --response-file`` entry point.
      Both directions are rejected past an explicit byte cap
      (:data:`MAX_REQUEST_MESSAGE_BYTES` /
      :data:`MAX_RESPONSE_MESSAGE_BYTES`).
    * Nothing is ever pickled; request fields are never interpolated
      into a shell string (argv is always a pre-tokenized list passed
      to :class:`subprocess.Popen` with ``shell=False``).

Process ownership / cancellation:
    Each child is started in its own process group / new-process-group
    (``start_new_session=True`` on POSIX,
    ``CREATE_NEW_PROCESS_GROUP`` on Windows).  Cancellation reaps ONLY
    the batch-owned child and the managed processes in its tree
    (``killpg`` / ``taskkill /T``), never unrelated pip tasks the
    caller may have running.  Tasks cancelled before their turn never
    create a process at all.  Every ``Popen`` is waited on and every
    pipe is closed, so no zombie or leaked handle survives a run.

Caching:
    Children may share a :class:`pipenv.resolver.manifest_cache.
    ParsedManifestCache` root (e.g. via a common ``PIPENV_CACHE_DIR``).
    That cache's atomic temp-file + ``os.replace`` protocol already
    gives cross-process safety; a corrupt entry is an independent miss
    that is refetched rather than a batch-wide failure.  This module
    deliberately performs NO in-memory cache sharing across processes,
    so reuse can never leak credentials or parsed manifests between
    requests.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from pipenv.resolver.schema import (
    SCHEMA_VERSION,
    InternalError,
    ResolverRequest,
    ResolverResponse,
)

# ---------------------------------------------------------------------------
# Hard limits / tunables
# ---------------------------------------------------------------------------

#: Absolute ceiling for ``max_workers`` regardless of caller input.
MAX_WORKERS_HARD_CAP = 8
#: Absolute ceiling for ``max_queued`` (tasks waiting for a worker slot).
MAX_QUEUED_HARD_CAP = 64
#: Default worker concurrency.
DEFAULT_MAX_WORKERS = 4
#: Default queue depth.
DEFAULT_MAX_QUEUED = 16

#: Largest request envelope the parent will write / a child may read.
MAX_REQUEST_MESSAGE_BYTES = 8 * 1024 * 1024
#: Largest response envelope the parent will parse.  Oversized responses
#: are treated as a crashed child rather than read into memory.
MAX_RESPONSE_MESSAGE_BYTES = 16 * 1024 * 1024

#: Stable IDs are kept short and filesystem-safe; this caps the raw ID.
MAX_ID_LENGTH = 128
#: Per-task captured-log capacity (lines).  A flooding child can never
#: grow the parent's memory past this.
MAX_TASK_LOG_LINES = 500
#: Grace period between a soft tree-kill and the forceful one, and the
#: upper bound for the final ``wait()`` reaping pass.
PROCESS_EXIT_GRACE_S = 5.0
#: Scheduler / monitor tick.  Kept small so cancellation and deadline
#: enforcement are responsive without busy-spinning.
_SCHEDULER_TICK_S = 0.05

# ---------------------------------------------------------------------------
# Terminal statuses (distinct from the response's own result.kind)
# ---------------------------------------------------------------------------

#: Child exited and produced a parseable ``ResolverResponse`` (the
#: response itself may still be ``success`` / ``resolution_error`` /
#: ``internal_error`` — those variants stay distinguishable inside it).
STATUS_COMPLETED = "completed"
#: Task was cancelled — either before it ever spawned or by reaping a
#: running child process group.
STATUS_CANCELLED = "cancelled"
#: The per-task deadline elapsed and the child process group was reaped.
STATUS_TIMEOUT = "timeout"
#: Child started but died without a parseable structured response.
STATUS_CRASHED = "crashed"
#: Task never ran: schema/message validation failure or spawn error.
STATUS_REJECTED = "rejected"

_TERMINAL_STATUSES = frozenset(
    {
        STATUS_COMPLETED,
        STATUS_CANCELLED,
        STATUS_TIMEOUT,
        STATUS_CRASHED,
        STATUS_REJECTED,
    }
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchResolveItem:
    """One element of a :func:`resolve_many` batch.

    Attributes
    ----------
    id:
        Caller-assigned stable identifier, unique within the batch.
        Results are returned keyed by this value.
    request:
        The fully-formed typed resolver envelope.  Its
        ``metadata.deadline_seconds`` (when set) is the per-task
        wall-clock deadline enforced by the parent.
    python_executable:
        Interpreter used for this task's subprocess.  Defaults to the
        parent's ``sys.executable``.
    env:
        Per-task environment *overlays* (marker variables, index
        selection, etc.).  Values must be strings.  Applied on top of
        the parent environment snapshot inside the CHILD only; the
        parent's ``os.environ`` is never modified.
    netrc_file:
        Optional per-task credentials file, exposed to the child as
        ``NETRC``.  Credentials therefore never appear on argv.
    workdir:
        Per-task working directory.  Defaults to a private temp
        directory created for this task.
    """

    id: str
    request: ResolverRequest
    python_executable: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    netrc_file: str | None = None
    workdir: str | None = None


@dataclass(frozen=True)
class BatchItemResult:
    """The deterministic terminal outcome of one batch item.

    Every input ID always produces exactly one of these — there is no
    "lost" outcome, including after parent cancellation.
    """

    id: str
    #: One of the ``STATUS_*`` constants.
    status: str
    #: Always present.  For :data:`STATUS_COMPLETED` this is the child's
    #: own response; for the other terminals it is a synthesized
    #: ``InternalError`` response describing the terminal reason.
    response: ResolverResponse
    #: Bounded, redacted, per-task log lines (stderr/stdout capture).
    log: tuple[str, ...] = ()
    #: Child exit code when a process was started, else ``None``.
    returncode: int | None = None
    #: Whether a subprocess was ever created for this item.
    started: bool = False


# ---------------------------------------------------------------------------
# Log redaction
# ---------------------------------------------------------------------------

# ``https://user:pass@host/...`` -> ``https://user:***@host/...``
_RE_URL_CREDENTIALS = re.compile(r"(https?://)([^/\s:@]+):([^/\s@]+)@")
# ``Authorization: Bearer <secret>`` / ``Basic <secret>``
_RE_AUTH_HEADER = re.compile(
    r"(?i)(authorization\s*:\s*(?:bearer|basic)\s+)(\S+)"
)
# netrc-style ``password <secret>`` lines
_RE_NETRC_PASSWORD = re.compile(r"(?i)(password\s+)(\S+)")
# Generic ``SOME_TOKEN=secret`` style assignments
_RE_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b([A-Z0-9_]*(?:TOKEN|PASSWORD|SECRET|API_?KEY)[A-Z0-9_]*)\s*[=:]\s*(\S+)"
)

_REDACTION = "***"


def _redact(line: str) -> str:
    """Redact credentials and secrets from one captured log line."""
    line = _RE_URL_CREDENTIALS.sub(rf"\1\2:{_REDACTION}@", line)
    line = _RE_AUTH_HEADER.sub(rf"\1{_REDACTION}", line)
    line = _RE_NETRC_PASSWORD.sub(rf"\1{_REDACTION}", line)
    line = _RE_SECRET_ASSIGNMENT.sub(rf"\1={_REDACTION}", line)
    return line


def _synthetic_response(message: str) -> ResolverResponse:
    """Build a typed ``InternalError`` response for a batch-level terminal."""
    return ResolverResponse(
        schema_version=SCHEMA_VERSION,
        result=InternalError(kind="internal_error", message=message),
    )


def _default_worker_entry() -> Path:
    """Resolve the canonical ``pipenv-resolver`` worker script path.

    Mirrors the parent-side subprocess launcher in
    ``pipenv/utils/resolver.py::_run_resolver_subprocess``: the package
    ``__init__`` re-exports ``main`` as a function, so the module is
    located explicitly via :func:`importlib.import_module`.
    """
    from importlib import import_module

    module = import_module("pipenv.resolver.main")
    assert module.__file__ is not None
    return Path(module.__file__.rstrip("co"))


def _safe_id(task_id: str) -> str:
    """Turn an arbitrary stable ID into a collision-resistant directory
    name component (sanitized prefix + content hash)."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)[:48].strip(".-") or "task"
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:12]
    return f"{sanitized}-{digest}"


# ---------------------------------------------------------------------------
# Internal per-task runtime state
# ---------------------------------------------------------------------------


class _RunningTask:
    """Mutable bookkeeping for one spawned subprocess (parent-side only)."""

    def __init__(
        self,
        item: BatchResolveItem,
        workdir: Path,
        request_path: Path,
        response_path: Path,
    ) -> None:
        self.item = item
        self.workdir = workdir
        self.request_path = request_path
        self.response_path = response_path
        self.started_at = time.monotonic()
        self.proc: subprocess.Popen | None = None
        self.cancel_event = threading.Event()
        self.done_event = threading.Event()
        # Combined, bounded capture of both pipes (tagged per stream).
        self.log: deque[str] = deque(maxlen=MAX_TASK_LOG_LINES)
        self.dropped_lines = 0
        self._reader_threads: list[threading.Thread] = []
        # Terminal bookkeeping, filled by the monitor.
        self.status: str | None = None
        self.response: ResolverResponse | None = None
        self.returncode: int | None = None

    def append_log(self, line: str) -> None:
        if len(self.log) == MAX_TASK_LOG_LINES:
            self.dropped_lines += 1
        self.log.append(_redact(line.rstrip()))

    def snapshot_log(self) -> tuple[str, ...]:
        lines = tuple(self.log)
        if self.dropped_lines:
            lines = lines + (
                f"... ({self.dropped_lines} log lines elided by capacity cap)",
            )
        return lines


# ---------------------------------------------------------------------------
# The batch resolver
# ---------------------------------------------------------------------------


class BatchResolver:
    """Run a bounded batch of independent resolver requests in subprocesses.

    One instance owns one batch run: it snapshots the parent environment
    at construction, tracks only the subprocesses it spawned itself, and
    guarantees every submitted ID reaches a terminal state.

    Use :func:`resolve_many` for the simple synchronous path; construct
    this class directly when another thread needs to call
    :meth:`cancel` / :meth:`cancel_all` while the batch is in flight.
    """

    def __init__(
        self,
        *,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_queued: int = DEFAULT_MAX_QUEUED,
        worker_entry: str | Path | None = None,
        work_root: str | Path | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        if not isinstance(max_workers, int) or isinstance(max_workers, bool):
            raise TypeError("max_workers must be an int")
        if not isinstance(max_queued, int) or isinstance(max_queued, bool):
            raise TypeError("max_queued must be an int")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if max_queued < 0:
            raise ValueError("max_queued must be >= 0")
        if max_workers > MAX_WORKERS_HARD_CAP:
            raise ValueError(
                f"max_workers={max_workers} exceeds hard cap {MAX_WORKERS_HARD_CAP}"
            )
        if max_queued > MAX_QUEUED_HARD_CAP:
            raise ValueError(
                f"max_queued={max_queued} exceeds hard cap {MAX_QUEUED_HARD_CAP}"
            )
        self.max_workers = max_workers
        self.max_queued = max_queued
        self.worker_entry = Path(worker_entry) if worker_entry else _default_worker_entry()
        self._external_work_root = Path(work_root) if work_root else None
        self._work_root: Path | None = None
        self._cleanup_work_root = False

        # Parent environment snapshot — the ONLY environment data handed
        # to children.  The parent process itself never mutates os.environ.
        self._parent_env_snapshot: dict[str, str] = dict(os.environ)

        self._state_lock = threading.Lock()
        self._running: dict[str, _RunningTask] = {}
        self._cancel_requested: set[str] = set()
        self._external_cancel = cancel_event
        self._internal_cancel = threading.Event()
        # Every Popen this instance creates, for post-run zombie audits.
        self.processes: list[subprocess.Popen] = []

    # ------------------------------------------------------------------
    # Introspection / cancellation
    # ------------------------------------------------------------------

    @property
    def parent_env_snapshot(self) -> dict[str, str]:
        """Return a copy of the parent environment captured at construction."""
        return dict(self._parent_env_snapshot)

    def cancel(self, task_id: str) -> None:
        """Request cancellation of one item.

        A still-queued item is skipped without ever spawning; a running
        item has its own process group reaped.  Unknown / already
        terminal IDs are a no-op.
        """
        with self._state_lock:
            self._cancel_requested.add(task_id)
            task = self._running.get(task_id)
        if task is not None:
            task.cancel_event.set()

    def cancel_all(self) -> None:
        """Cancel every queued and running item owned by this batch."""
        self._internal_cancel.set()
        with self._state_lock:
            self._cancel_requested.update(self._running)
            tasks = list(self._running.values())
        for task in tasks:
            task.cancel_event.set()

    def _is_cancelled(self, task_id: str) -> bool:
        if self._internal_cancel.is_set():
            return True
        if self._external_cancel is not None and self._external_cancel.is_set():
            return True
        with self._state_lock:
            return task_id in self._cancel_requested

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_batch_shape(self, items: Sequence[BatchResolveItem]) -> None:
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, Mapping)):
            raise TypeError("items must be a finite sequence of BatchResolveItem")
        if len(items) == 0:
            raise ValueError("resolve_many requires at least one item")
        if len(items) > self.max_workers + self.max_queued:
            raise ValueError(
                f"batch size {len(items)} exceeds hard limit "
                f"max_workers({self.max_workers}) + max_queued({self.max_queued}) "
                f"= {self.max_workers + self.max_queued}"
            )
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, BatchResolveItem):
                raise TypeError(
                    f"every batch item must be a BatchResolveItem, got {type(item)!r}"
                )
            task_id = item.id
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError("batch item id must be a non-empty string")
            if len(task_id) > MAX_ID_LENGTH:
                raise ValueError(
                    f"batch item id exceeds {MAX_ID_LENGTH} characters: {task_id!r}"
                )
            if task_id in seen:
                raise ValueError(f"duplicate batch item id: {task_id!r}")
            seen.add(task_id)

    def _validate_item(self, item: BatchResolveItem) -> str | None:
        """Return an error string when the item cannot be dispatched,
        else ``None``.  Never raises — per-item rejection is a terminal
        result, not a batch-wide failure."""
        if not isinstance(item.request, ResolverRequest):
            return (
                f"item {item.id!r}: request must be a ResolverRequest, "
                f"got {type(item.request).__name__}"
            )
        if not isinstance(item.env, Mapping):
            return f"item {item.id!r}: env must be a mapping of str -> str"
        for key, value in item.env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                return f"item {item.id!r}: env keys and values must all be strings"
        if item.python_executable is not None and (
            not isinstance(item.python_executable, str)
            or not item.python_executable.strip()
        ):
            return f"item {item.id!r}: python_executable must be a non-empty string or None"
        if item.netrc_file is not None and (
            not isinstance(item.netrc_file, str) or not item.netrc_file.strip()
        ):
            return f"item {item.id!r}: netrc_file must be a non-empty string or None"
        # Schema round-trip: proves the envelope is restricted-JSON
        # representable (no arbitrary objects) and re-validates it.
        try:
            payload = item.request.to_json_dict()
            ResolverRequest.from_json_dict(payload)
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
        except (TypeError, ValueError, KeyError) as exc:
            return f"item {item.id!r}: schema validation failed: {exc}"
        if len(body) > MAX_REQUEST_MESSAGE_BYTES:
            return (
                f"item {item.id!r}: request message is {len(body)} bytes, "
                f"over the {MAX_REQUEST_MESSAGE_BYTES}-byte limit"
            )
        return None

    # ------------------------------------------------------------------
    # Process construction / teardown
    # ------------------------------------------------------------------

    def _child_environment(self, item: BatchResolveItem) -> dict[str, str]:
        """Build the isolated environment dict for one child process."""
        env = dict(self._parent_env_snapshot)
        env.update({str(k): str(v) for k, v in item.env.items()})
        if item.netrc_file:
            env["NETRC"] = str(item.netrc_file)
        # Make the child easy to attribute in process listings / logs.
        env["PIPENV_RESOLVE_MANY_TASK_ID"] = item.id
        return env

    def _popen_kwargs(self) -> dict:
        """Platform-specific isolation flags: each child gets its own
        process group so a cancel can reap exactly that tree."""
        kwargs: dict = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "shell": False,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        return kwargs

    def _start_task(self, item: BatchResolveItem) -> _RunningTask:
        """Spawn the subprocess for one item.

        Always returns the task object.  A task whose ``status`` is
        already set on return was rejected at the spawn boundary (no
        process exists, ``done_event`` is pre-set); otherwise it is
        registered as running and owns a live :class:`subprocess.Popen`.
        """
        assert self._work_root is not None
        task_dir = self._work_root / _safe_id(item.id)
        task_dir.mkdir(parents=True, exist_ok=True)
        workdir = Path(item.workdir) if item.workdir else task_dir
        workdir.mkdir(parents=True, exist_ok=True)
        request_path = task_dir / "request.json"
        response_path = task_dir / "response.json"

        task = _RunningTask(item, workdir, request_path, response_path)

        # Re-check the message cap at the actual write boundary.
        try:
            body = json.dumps(item.request.to_json_dict(), sort_keys=True).encode(
                "utf-8"
            )
            if len(body) > MAX_REQUEST_MESSAGE_BYTES:
                task.status = STATUS_REJECTED
                task.response = _synthetic_response(
                    f"request message over {MAX_REQUEST_MESSAGE_BYTES}-byte limit"
                )
                task.done_event.set()
                return task
            request_path.write_bytes(body)
            argv = [
                item.python_executable or sys.executable,
                str(self.worker_entry),
                "--request-file",
                str(request_path),
                "--response-file",
                str(response_path),
            ]
            proc = subprocess.Popen(  # noqa: S603 — fixed list argv, shell=False
                argv,
                env=self._child_environment(item),
                cwd=str(workdir),
                **self._popen_kwargs(),
            )
        except OSError as exc:
            task.status = STATUS_REJECTED
            task.response = _synthetic_response(
                f"could not spawn resolver subprocess: {exc}"
            )
            task.returncode = None
            task.done_event.set()
            return task

        task.proc = proc
        self.processes.append(proc)
        with self._state_lock:
            self._running[item.id] = task
        return task

    def _start_pipe_readers(self, task: _RunningTask) -> None:
        proc = task.proc
        assert proc is not None

        def _drain(stream, tag: str) -> None:
            try:
                for line in iter(stream.readline, ""):
                    task.append_log(f"{tag}: {line}")
            except (OSError, ValueError):
                # Pipe closed underneath us during teardown — normal.
                pass

        for stream, tag in ((proc.stderr, "err"), (proc.stdout, "out")):
            if stream is None:
                continue
            reader = threading.Thread(
                target=_drain,
                args=(stream, tag),
                name=f"resolve-many-io-{task.item.id}",
                daemon=True,
            )
            reader.start()
            task._reader_threads.append(reader)

    def _close_pipes(self, task: _RunningTask) -> None:
        proc = task.proc
        if proc is None:
            return
        for reader in task._reader_threads:
            reader.join(timeout=PROCESS_EXIT_GRACE_S)
        for stream in (proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _reap_process_tree(self, task: _RunningTask) -> None:
        """Terminate ONLY this batch-owned child and its managed subtree.

        Unrelated pip tasks the caller owns are never touched because we
        signal the child's own process group / use ``taskkill /T`` on
        the single tracked PID.
        """
        proc = task.proc
        assert proc is not None
        if proc.poll() is None:
            if os.name == "nt":
                # /T = whole tree rooted at this PID, /F = force.
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        timeout=PROCESS_EXIT_GRACE_S,
                        check=False,
                    )
                except (OSError, subprocess.SubprocessError):
                    proc.kill()
            else:
                try:
                    os.killpg(  # type: ignore[attr-defined]
                        os.getpgid(proc.pid), signal.SIGTERM  # type: ignore[attr-defined]
                    )
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=PROCESS_EXIT_GRACE_S)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(  # type: ignore[attr-defined]
                            os.getpgid(proc.pid),  # type: ignore[attr-defined]
                            signal.SIGKILL,  # type: ignore[attr-defined]
                        )
                    except ProcessLookupError:
                        pass
        try:
            proc.wait(timeout=PROCESS_EXIT_GRACE_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=PROCESS_EXIT_GRACE_S)
        task.returncode = proc.returncode

    # ------------------------------------------------------------------
    # Monitor (one daemon thread per running task)
    # ------------------------------------------------------------------

    def _monitor(self, task: _RunningTask) -> None:
        """Wait on one child, enforcing cancellation and the deadline,
        then translate the outcome into a terminal status.

        The whole body is guarded: any unexpected monitor failure still
        reaps the child and settles a deterministic terminal instead of
        stranding the scheduler on an event that will never fire.
        """
        proc = task.proc
        assert proc is not None
        try:
            self._start_pipe_readers(task)
            while True:
                if task.cancel_event.is_set() or self._is_cancelled(task.item.id):
                    self._reap_process_tree(task)
                    task.status = STATUS_CANCELLED
                    task.response = _synthetic_response(
                        f"task {task.item.id!r} was cancelled by the parent"
                    )
                    break
                returncode = proc.poll()
                if returncode is not None:
                    self._finish_completed(task, returncode)
                    break
                deadline = task.item.request.metadata.deadline_seconds
                if deadline is not None and deadline > 0:
                    elapsed = time.monotonic() - task.started_at
                    if elapsed >= float(deadline):
                        self._reap_process_tree(task)
                        task.status = STATUS_TIMEOUT
                        task.response = _synthetic_response(
                            f"task {task.item.id!r} timed out after "
                            f"{float(deadline):g}s and its process group was reaped"
                        )
                        break
                task.done_event.wait(timeout=_SCHEDULER_TICK_S)
        except Exception as exc:  # noqa: BLE001 — monitor must never die silently
            try:
                self._reap_process_tree(task)
            except Exception:  # noqa: BLE001 — preserve the original failure
                pass
            if task.status is None:
                task.status = STATUS_CRASHED
                task.response = _synthetic_response(
                    f"resolver monitor for {task.item.id!r} failed: {exc}"
                )
                task.returncode = proc.returncode
        finally:
            self._close_pipes(task)
            with self._state_lock:
                self._running.pop(task.item.id, None)
            task.done_event.set()

    def _finish_completed(self, task: _RunningTask, returncode: int) -> None:
        """Read + validate the structured response of an exited child."""
        task.returncode = returncode
        response = self._read_response(task)
        if response is None:
            task.status = STATUS_CRASHED
            task.response = _synthetic_response(
                f"resolver subprocess for {task.item.id!r} exited with code "
                f"{returncode} but produced no parseable response"
            )
            return
        task.status = STATUS_COMPLETED
        task.response = response

    def _read_response(self, task: _RunningTask) -> ResolverResponse | None:
        try:
            if not task.response_path.exists():
                return None
            size = task.response_path.stat().st_size
            if size > MAX_RESPONSE_MESSAGE_BYTES:
                task.append_log(
                    f"err: response file ({size} bytes) exceeds "
                    f"{MAX_RESPONSE_MESSAGE_BYTES}-byte cap; discarded"
                )
                return None
            raw = task.response_path.read_text(encoding="utf-8")
            if not raw.strip():
                return None
            return ResolverResponse.from_json_dict(json.loads(raw))
        except (OSError, ValueError, KeyError):
            # JSONDecodeError is a ValueError subclass; any parse failure
            # or schema mismatch means the child's response is unusable.
            return None

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def run(self, items: Sequence[BatchResolveItem]) -> dict[str, BatchItemResult]:
        """Execute the batch and return ``{id: BatchItemResult}``.

        The returned mapping is rebuilt in input order.  Every submitted
        ID is present exactly once and holds a terminal state.
        """
        self._validate_batch_shape(items)

        if self._external_work_root is not None:
            self._work_root = self._external_work_root
            self._work_root.mkdir(parents=True, exist_ok=True)
        else:
            self._work_root = Path(
                tempfile.mkdtemp(prefix="pipenv-resolve-many-")
            )
            self._cleanup_work_root = True

        results: dict[str, BatchItemResult] = {}
        pending: deque[BatchResolveItem] = deque()
        for item in items:
            error = self._validate_item(item)
            if error is not None:
                results[item.id] = BatchItemResult(
                    id=item.id,
                    status=STATUS_REJECTED,
                    response=_synthetic_response(error),
                    started=False,
                )
            else:
                pending.append(item)

        monitor_threads: list[threading.Thread] = []
        # Every task object this run creates, including ones rejected at
        # the spawn boundary.  This is the harvest registry: a task can
        # finish (and remove itself from ``_running``) before the
        # scheduler's next tick, so ``_running`` alone is not a reliable
        # completion channel.
        all_tasks: dict[str, _RunningTask] = {}
        # The loop condition covers unharvested tasks explicitly: a fast
        # child can finish (and unregister itself from ``_running``)
        # during the park wait below, so keying the loop only on
        # ``_running`` would exit before its terminal state is collected.
        try:
            while pending or any(
                task.item.id not in results for task in all_tasks.values()
            ):
                # Fill free worker slots in strict input order so process
                # start order is deterministic for any given batch.
                while pending and len(self._running) < self.max_workers:
                    item = pending.popleft()
                    if self._is_cancelled(item.id):
                        # Cancelled before launch: no process is created.
                        results[item.id] = BatchItemResult(
                            id=item.id,
                            status=STATUS_CANCELLED,
                            response=_synthetic_response(
                                f"task {item.id!r} was cancelled before it started"
                            ),
                            started=False,
                        )
                        continue
                    task = self._start_task(item)
                    all_tasks[item.id] = task
                    if task.status is not None:
                        # Rejected at the spawn boundary — no monitor.
                        continue
                    monitor = threading.Thread(
                        target=self._monitor,
                        args=(task,),
                        name=f"resolve-many-monitor-{item.id}",
                        daemon=True,
                    )
                    monitor.start()
                    monitor_threads.append(monitor)

                # Harvest every task whose terminal state is settled,
                # irrespective of whether its monitor already unregistered
                # it from the active ``_running`` set.
                for task in list(all_tasks.values()):
                    if task.item.id in results or not task.done_event.is_set():
                        continue
                    results[task.item.id] = BatchItemResult(
                        id=task.item.id,
                        status=task.status or STATUS_CRASHED,
                        response=task.response
                        or _synthetic_response(
                            "resolver task ended without a response"
                        ),
                        log=task.snapshot_log(),
                        returncode=task.returncode,
                        started=task.proc is not None,
                    )

                if not self._running:
                    # No active children: either everything finished or
                    # only rejected tasks remain (harvested above). The
                    # outer while condition decides whether to continue.
                    continue

                # Park until a child ticks or finishes; the loop then
                # re-evaluates cancellation, deadlines and free slots.
                for task in list(self._running.values()):
                    task.done_event.wait(timeout=_SCHEDULER_TICK_S)
        finally:
            # Guarantee reaping: any still-running child (e.g. an error
            # in the scheduler itself) is cancelled and joined so no
            # zombie process or daemon thread escapes the run.
            with self._state_lock:
                stragglers = list(self._running.values())
            for task in stragglers:
                task.cancel_event.set()
                self._reap_process_tree(task)
                task.done_event.set()
            for monitor in monitor_threads:
                monitor.join(timeout=PROCESS_EXIT_GRACE_S * 2)
            for task in stragglers:
                results.setdefault(
                    task.item.id,
                    BatchItemResult(
                        id=task.item.id,
                        status=task.status or STATUS_CANCELLED,
                        response=task.response
                        or _synthetic_response(
                            "resolver task was reaped during shutdown"
                        ),
                        log=task.snapshot_log(),
                        returncode=task.returncode,
                        started=True,
                    ),
                )
            if self._cleanup_work_root and self._work_root is not None:
                shutil.rmtree(self._work_root, ignore_errors=True)

        # Rebuild in the caller's input order; assert the terminal contract.
        ordered: dict[str, BatchItemResult] = {}
        for item in items:
            result = results.get(item.id)
            if result is None or result.status not in _TERMINAL_STATUSES:
                # Defensive: should be unreachable given the loop above.
                result = BatchItemResult(
                    id=item.id,
                    status=STATUS_CRASHED,
                    response=_synthetic_response(
                        f"task {item.id!r} reached no deterministic terminal state"
                    ),
                )
            ordered[item.id] = result
        return ordered


# ---------------------------------------------------------------------------
# Public convenience API
# ---------------------------------------------------------------------------


def resolve_many(
    items: Sequence[BatchResolveItem],
    *,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_queued: int = DEFAULT_MAX_QUEUED,
    cancel_event: threading.Event | None = None,
    worker_entry: str | Path | None = None,
    work_root: str | Path | None = None,
) -> dict[str, BatchItemResult]:
    """Resolve a finite batch of independent :class:`ResolverRequest` objects.

    Each item runs in its own bounded spawn subprocess with a private
    marker environment, index credentials, working directory and
    deadline.  Returns one complete :class:`ResolverResponse` per input
    ID alongside a deterministic batch terminal status
    (completed / cancelled / timeout / crashed / rejected).

    Parameters
    ----------
    items:
        Non-empty sequence of :class:`BatchResolveItem` with unique
        stable IDs.  Validated against the wire schema before any
        subprocess is created.
    max_workers:
        Hard upper bound on concurrently running resolver subprocesses
        (capped at :data:`MAX_WORKERS_HARD_CAP`).
    max_queued:
        Hard upper bound on items waiting for a worker slot (capped at
        :data:`MAX_QUEUED_HARD_CAP`); the batch itself may not exceed
        ``max_workers + max_queued``.
    cancel_event:
        Optional shared event: setting it cancels every queued and
        running item in this batch without touching processes the
        caller owns outside the batch.
    worker_entry:
        Override for the resolver worker script (testing / embedding).
    work_root:
        Override for the directory that holds per-task working
        directories and JSON message files.

    Notes
    -----
    No merged/cross-platform lock is produced — callers own any
    aggregation of the per-item :class:`ResolverResponse` objects.
    """
    resolver = BatchResolver(
        max_workers=max_workers,
        max_queued=max_queued,
        worker_entry=worker_entry,
        work_root=work_root,
        cancel_event=cancel_event,
    )
    return resolver.run(items)


__all__ = [
    "DEFAULT_MAX_QUEUED",
    "DEFAULT_MAX_WORKERS",
    "MAX_ID_LENGTH",
    "MAX_QUEUED_HARD_CAP",
    "MAX_REQUEST_MESSAGE_BYTES",
    "MAX_RESPONSE_MESSAGE_BYTES",
    "MAX_TASK_LOG_LINES",
    "MAX_WORKERS_HARD_CAP",
    "STATUS_CANCELLED",
    "STATUS_COMPLETED",
    "STATUS_CRASHED",
    "STATUS_REJECTED",
    "STATUS_TIMEOUT",
    "BatchItemResult",
    "BatchResolveItem",
    "BatchResolver",
    "resolve_many",
]
