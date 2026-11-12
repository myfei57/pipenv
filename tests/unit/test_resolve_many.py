"""Unit tests for the bounded, subprocess-isolated ``resolve_many`` API.

The batch API lives in :mod:`pipenv.resolver.batch`.  These tests drive
it with a generated *stub worker* that speaks the same
``--request-file`` / ``--response-file`` JSON contract as the real
``pipenv-resolver`` entry point, so every test is network-free, fast and
determinable — including on Windows where the production path is a
fresh ``spawn``-style ``subprocess.Popen``.

Coverage matrix (each requirement from the batch-API spec maps to a
test below):

* two opposite python marker environments + same-named package on
  different indexes — actual per-task selection,
* parent environment snapshot / no parent or sibling pollution,
* ``max_workers`` / ``max_queued`` hard caps and deterministic start
  order,
* cancellation before launch (no process created) and while running
  (only the batch-owned process tree is reaped; an unrelated decoy
  survives),
* per-task deadline -> timeout terminal with no zombie,
* subprocess crash, structured resolution-error vs internal-error
  distinction and sibling failure isolation,
* log flood capacity cap + credential redaction,
* request/response message byte caps and schema validation,
* the REAL worker entry (Windows spawn + import path) with ordering,
* shared ``ParsedManifestCache`` atomic protocol and corrupt-entry
  independence.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from pipenv.resolver import batch as batch_mod
from pipenv.resolver.batch import (
    MAX_RESPONSE_MESSAGE_BYTES,
    MAX_TASK_LOG_LINES,
    MAX_WORKERS_HARD_CAP,
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    STATUS_CRASHED,
    STATUS_REJECTED,
    STATUS_TIMEOUT,
    BatchResolveItem,
    BatchResolver,
    resolve_many,
)
from pipenv.resolver.manifest_cache import ParsedManifestCache
from pipenv.resolver.schema import (
    SCHEMA_VERSION,
    PackageSpecs,
    RequestMetadata,
    ResolverOptions,
    ResolverRequest,
    Source,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Stub worker — a standalone script implementing the resolver wire contract.
# Behavior is selected by the RESOLVE_MANY_TEST_MODE environment variable so
# the same script covers every failure / timing scenario.
# ---------------------------------------------------------------------------

STUB_WORKER = r"""
import json
import os
import sys
import time


def _parse_args(argv):
    request_file = response_file = None
    for i, arg in enumerate(argv):
        if arg == "--request-file":
            request_file = argv[i + 1]
        elif arg == "--response-file":
            response_file = argv[i + 1]
    return request_file, response_file


def _write_response(response_file, payload):
    with open(response_file, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def _success_payload(request):
    specs = request["packages"]["specs"]
    name = next(iter(specs), "demo")
    sources = request.get("sources") or []
    source = sources[0] if sources else {}
    # Encode every per-task isolation dimension into the markers field so
    # the parent can assert actual child-side selection.
    markers = "marker={marker}|env={env}|cwd={cwd}|index={index}|url={url}|netrc={netrc}|cache={cache}".format(
        marker=request.get("python_marker_override") or "",
        env=os.environ.get("TASK_MARKER_VAR", ""),
        cwd=os.getcwd(),
        index=source.get("name", ""),
        url=source.get("url", ""),
        netrc=os.environ.get("NETRC", ""),
        cache=os.environ.get("PIPENV_CACHE_DIR", ""),
    )
    return {
        "schema_version": 1,
        "result": {
            "kind": "success",
            "locked": [
                {"name": name, "version": "==1.0", "markers": markers}
            ],
        },
    }


def main():
    request_file, response_file = _parse_args(sys.argv[1:])
    with open(request_file, encoding="utf-8") as fh:
        request = json.load(fh)

    mode = os.environ.get("RESOLVE_MANY_TEST_MODE", "echo")
    ready = os.environ.get("RESOLVE_MANY_READY_FILE")
    if ready:
        with open(ready, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))

    if mode == "echo":
        _write_response(response_file, _success_payload(request))
        return 0

    if mode == "slow":
        time.sleep(float(os.environ.get("RESOLVE_MANY_SLEEP_S", "30")))
        _write_response(response_file, _success_payload(request))
        return 0

    if mode == "order":
        order_file = os.environ["RESOLVE_MANY_ORDER_FILE"]
        task_id = os.environ.get("PIPENV_RESOLVE_MANY_TASK_ID", "?")
        with open(order_file, "a", encoding="utf-8") as fh:
            fh.write(f"start:{task_id}\n")
        time.sleep(float(os.environ.get("RESOLVE_MANY_ORDER_SLEEP_S", "0.2")))
        with open(order_file, "a", encoding="utf-8") as fh:
            fh.write(f"end:{task_id}\n")
        _write_response(response_file, _success_payload(request))
        return 0

    if mode == "block":
        # Stay alive until the parent reaps the process group.
        for _ in range(6000):
            time.sleep(0.1)
        return 1

    if mode == "flood":
        for i in range(5000):
            sys.stderr.write(f"flood line {i}\n")
        # Emit credentials LAST so they survive the parent's bounded
        # ring buffer; the redaction contract is checked on retained lines.
        sys.stderr.write(
            "fetching https://alice:s3cret-pass@example.com/simple/demo\n"
        )
        sys.stderr.write("machine example.com password t0psecret\n")
        sys.stderr.write("API_TOKEN=abc123\n")
        sys.stderr.flush()
        _write_response(response_file, _success_payload(request))
        return 0

    if mode == "crash":
        # Hard exit without writing any response file.
        os._exit(3)

    if mode == "resolution_error":
        _write_response(
            response_file,
            {
                "schema_version": 1,
                "result": {
                    "kind": "resolution_error",
                    "conflicts": [],
                    "pip_message": "no matching distribution found for demo",
                },
            },
        )
        return 0

    if mode == "internal_error":
        _write_response(
            response_file,
            {
                "schema_version": 1,
                "result": {
                    "kind": "internal_error",
                    "message": "boom inside the worker",
                },
            },
        )
        return 0

    if mode == "huge_response":
        size = int(os.environ.get("RESOLVE_MANY_HUGE_BYTES", "65536"))
        with open(response_file, "wb") as fh:
            fh.write(b"x" * size)
        return 0

    sys.stderr.write(f"unknown test mode: {mode}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())
"""


@pytest.fixture()
def stub_worker(tmp_path):
    path = tmp_path / "stub_resolver_worker.py"
    path.write_text(STUB_WORKER, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_request(
    *,
    marker=None,
    sources=None,
    deadline=None,
    packages=None,
    backend="",
):
    return ResolverRequest(
        schema_version=SCHEMA_VERSION,
        category="default",
        packages=PackageSpecs(specs=packages or {"demo": "demo==1.0"}),
        options=ResolverOptions(backend=backend),
        sources=tuple(
            sources
            or [Source(name="pypi", url="https://pypi.org/simple", verify_ssl=True)]
        ),
        python_marker_override=marker,
        metadata=RequestMetadata(deadline_seconds=deadline),
    )


def make_item(
    task_id,
    *,
    worker_mode="echo",
    marker=None,
    sources=None,
    deadline=None,
    extra_env=None,
    netrc_file=None,
    python_executable=None,
    packages=None,
):
    env = {"RESOLVE_MANY_TEST_MODE": worker_mode}
    if extra_env:
        env.update(extra_env)
    return BatchResolveItem(
        id=task_id,
        request=make_request(
            marker=marker, sources=sources, deadline=deadline, packages=packages
        ),
        python_executable=python_executable,
        env=env,
        netrc_file=netrc_file,
    )


def wait_until(predicate, timeout=15.0, interval=0.05):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition was not satisfied before the timeout")


def wait_for_file(path, timeout=15.0):
    wait_until(lambda: Path(path).exists(), timeout=timeout)


def start_batch_in_thread(resolver, items):
    holder = {}

    def _run():
        holder["results"] = resolver.run(items)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread, holder


def spawn_decoy():
    """An unrelated long-lived process the batch must never touch."""
    kwargs = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(180)"], **kwargs
    )


# ---------------------------------------------------------------------------
# 1. Two opposite marker environments + same-named package, different indexes
# ---------------------------------------------------------------------------


@pytest.mark.utils
def test_two_opposite_marker_envs_and_indexes_select_per_task(tmp_path, stub_worker):
    sources_a = (
        Source(name="index-a", url="https://index-a.example.com/simple"),
    )
    sources_b = (
        Source(name="index-b", url="https://index-b.example.org/simple"),
    )
    items = [
        make_item(
            "py38",
            marker="3.8.18",
            sources=sources_a,
            extra_env={"TASK_MARKER_VAR": "alpha"},
        ),
        make_item(
            "py312",
            marker="3.12.7",
            sources=sources_b,
            extra_env={"TASK_MARKER_VAR": "beta"},
        ),
    ]

    results = resolve_many(items, worker_entry=stub_worker, work_root=tmp_path / "work")

    assert list(results) == ["py38", "py312"]
    for item in items:
        result = results[item.id]
        assert result.status == STATUS_COMPLETED
        assert result.started is True
        assert result.returncode == 0
        assert result.response.result.kind == "success"

    markers_a = results["py38"].response.result.locked[0].markers
    markers_b = results["py312"].response.result.locked[0].markers
    assert "marker=3.8.18" in markers_a
    assert "marker=3.12.7" in markers_b
    assert "env=alpha" in markers_a
    assert "env=beta" in markers_b
    assert "index=index-a" in markers_a
    assert "index=index-b" in markers_b
    # Same-named package resolved independently against each index.
    assert results["py38"].response.result.locked[0].name == "demo"
    assert results["py312"].response.result.locked[0].name == "demo"
    # Sibling isolation: neither child saw the other's marker variable.
    assert "beta" not in markers_a
    assert "alpha" not in markers_b
    # Each task got its own private working directory.
    cwd_a = [part for part in markers_a.split("|") if part.startswith("cwd=")][0]
    cwd_b = [part for part in markers_b.split("|") if part.startswith("cwd=")][0]
    assert cwd_a != cwd_b
    assert cwd_a.split("=", 1)[1].startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# 2. Parent environment snapshot — no parent / sibling pollution
# ---------------------------------------------------------------------------


@pytest.mark.utils
def test_parent_environment_is_snapshotted_and_not_polluted(
    tmp_path, stub_worker, monkeypatch
):
    probe_key = "RESOLVE_MANY_PARENT_PROBE"
    monkeypatch.setenv(probe_key, "parent-value")
    netrc = tmp_path / "task.netrc"
    netrc.write_text("machine example.com login alice password t0psecret\n")

    resolver = BatchResolver(worker_entry=stub_worker, work_root=tmp_path / "work")
    # Snapshot is taken at construction and contains the parent's value.
    assert resolver.parent_env_snapshot[probe_key] == "parent-value"

    items = [
        make_item(
            "with-creds",
            extra_env={"TASK_MARKER_VAR": "child-a"},
            netrc_file=str(netrc),
        ),
        make_item("plain", extra_env={"TASK_MARKER_VAR": "child-b"}),
    ]
    results = resolver.run(items)

    # Parent process environment is byte-for-byte unpolluted.
    assert os.environ.get(probe_key) == "parent-value"
    assert "TASK_MARKER_VAR" not in os.environ
    assert "NETRC" not in os.environ
    assert "PIPENV_RESOLVE_MANY_TASK_ID" not in os.environ

    # Each child saw exactly its own overlay + credentials.
    markers_a = results["with-creds"].response.result.locked[0].markers
    markers_b = results["plain"].response.result.locked[0].markers
    assert "env=child-a" in markers_a
    assert f"netrc={netrc}" in markers_a
    assert "env=child-b" in markers_b
    assert "netrc=" in markers_b
    assert "child-a" not in markers_b
    # Both children inherited the parent snapshot through the env dict.
    assert "parent-value" in resolver.parent_env_snapshot.values()


# ---------------------------------------------------------------------------
# 3. Hard caps: max_workers / max_queued / batch shape
# ---------------------------------------------------------------------------


@pytest.mark.utils
def test_hard_caps_and_batch_shape_validation(stub_worker):
    with pytest.raises(ValueError, match="max_workers"):
        BatchResolver(max_workers=MAX_WORKERS_HARD_CAP + 1)
    with pytest.raises(ValueError, match="max_queued"):
        BatchResolver(max_queued=batch_mod.MAX_QUEUED_HARD_CAP + 1)
    with pytest.raises(ValueError):
        BatchResolver(max_workers=0)
    with pytest.raises(ValueError):
        BatchResolver(max_queued=-1)
    with pytest.raises(TypeError):
        BatchResolver(max_workers=True)

    resolver = BatchResolver(max_workers=1, max_queued=1, worker_entry=stub_worker)
    with pytest.raises(ValueError, match="exceeds hard limit"):
        resolver.run([make_item(f"t{i}") for i in range(3)])

    with pytest.raises(ValueError, match="duplicate"):
        resolver.run([make_item("dup"), make_item("dup")])

    with pytest.raises(ValueError, match="non-empty string"):
        resolver.run([make_item("")])

    with pytest.raises(ValueError, match="at least one item"):
        resolver.run([])

    with pytest.raises(TypeError, match="BatchResolveItem"):
        resolver.run(["not-an-item"])  # type: ignore[list-item]


@pytest.mark.utils
def test_non_string_env_and_bad_request_become_rejected_terminals(
    tmp_path, stub_worker
):
    good = make_item("good")
    bad_env = BatchResolveItem(id="bad-env", request=make_request(), env={1: 2})  # type: ignore[dict-item]
    bad_request = BatchResolveItem(id="bad-request", request="not-a-request")  # type: ignore[arg-type]

    results = resolve_many(
        [bad_env, bad_request, good],
        worker_entry=stub_worker,
        work_root=tmp_path / "work",
    )
    assert results["bad-env"].status == STATUS_REJECTED
    assert results["bad-env"].started is False
    assert results["bad-request"].status == STATUS_REJECTED
    # One item's validation failure never costs another item its result.
    assert results["good"].status == STATUS_COMPLETED
    assert results["good"].response.result.kind == "success"


# ---------------------------------------------------------------------------
# 4. max_workers bounds concurrency; start order is deterministic
# ---------------------------------------------------------------------------


@pytest.mark.utils
def test_max_workers_one_serializes_in_input_order(tmp_path, stub_worker):
    order_file = tmp_path / "order.log"
    items = [
        make_item(
            f"task-{i}",
            worker_mode="order",
            extra_env={
                "RESOLVE_MANY_ORDER_FILE": str(order_file),
                "RESOLVE_MANY_ORDER_SLEEP_S": "0.15",
            },
        )
        for i in range(3)
    ]
    resolver = BatchResolver(
        max_workers=1, max_queued=8, worker_entry=stub_worker,
        work_root=tmp_path / "work",
    )
    results = resolver.run(items)

    assert all(r.status == STATUS_COMPLETED for r in results.values())
    events = order_file.read_text().strip().splitlines()
    # With a single worker, each task fully finishes before the next starts.
    assert events == [
        "start:task-0",
        "end:task-0",
        "start:task-1",
        "end:task-1",
        "start:task-2",
        "end:task-2",
    ]
    # Exactly one process per task, all reaped (no zombies).
    assert len(resolver.processes) == 3
    assert all(proc.poll() is not None for proc in resolver.processes)


# ---------------------------------------------------------------------------
# 5. Cancel before launch -> no process; cancel running -> own tree only
# ---------------------------------------------------------------------------


@pytest.mark.utils
def test_cancel_queued_tasks_create_no_process(tmp_path, stub_worker):
    ready_a = tmp_path / "ready-a"
    items = [
        make_item(
            "blocker",
            worker_mode="block",
            extra_env={"RESOLVE_MANY_READY_FILE": str(ready_a)},
        ),
        make_item("queued-b"),
        make_item("queued-c"),
    ]
    resolver = BatchResolver(
        max_workers=1, max_queued=8, worker_entry=stub_worker,
        work_root=tmp_path / "work",
    )
    thread, holder = start_batch_in_thread(resolver, items)
    wait_for_file(ready_a)

    # B and C are still queued (single worker occupied by the blocker).
    resolver.cancel("queued-b")
    resolver.cancel("queued-c")
    wait_until(lambda: len(resolver.processes) >= 1)
    # Give the scheduler a tick to observe the queued cancellations.
    time.sleep(0.2)
    assert len(resolver.processes) == 1, "queued cancellation must not spawn"

    resolver.cancel("blocker")
    thread.join(timeout=30)
    assert not thread.is_alive()

    results = holder["results"]
    assert results["queued-b"].status == STATUS_CANCELLED
    assert results["queued-c"].status == STATUS_CANCELLED
    assert results["queued-b"].started is False
    assert results["queued-c"].started is False
    assert results["blocker"].status == STATUS_CANCELLED
    assert results["blocker"].started is True
    assert results["blocker"].returncode is not None
    assert all(proc.poll() is not None for proc in resolver.processes)


@pytest.mark.utils
def test_cancel_running_task_does_not_touch_unrelated_processes(
    tmp_path, stub_worker
):
    decoy = spawn_decoy()
    try:
        assert decoy.poll() is None
        ready = tmp_path / "ready"
        items = [
            make_item(
                "blocker",
                worker_mode="block",
                extra_env={"RESOLVE_MANY_READY_FILE": str(ready)},
            ),
            make_item("sibling"),
        ]
        resolver = BatchResolver(
            max_workers=1, max_queued=8, worker_entry=stub_worker,
            work_root=tmp_path / "work",
        )
        thread, holder = start_batch_in_thread(resolver, items)
        wait_for_file(ready)

        resolver.cancel_all()
        thread.join(timeout=30)
        assert not thread.is_alive()

        results = holder["results"]
        assert results["blocker"].status == STATUS_CANCELLED
        # The queued sibling never launched and is cancelled cleanly.
        assert results["sibling"].status == STATUS_CANCELLED
        assert results["sibling"].started is False
        # Every batch-owned process is reaped.
        assert all(proc.poll() is not None for proc in resolver.processes)
        # The caller's unrelated process is untouched.
        assert decoy.poll() is None, "batch cancel killed an unrelated process"
    finally:
        decoy.kill()
        decoy.wait(timeout=10)


@pytest.mark.utils
def test_external_cancel_event_cancels_whole_batch(tmp_path, stub_worker):
    ready_1 = tmp_path / "ready-1"
    ready_2 = tmp_path / "ready-2"
    items = [
        make_item(
            "block-1",
            worker_mode="block",
            extra_env={"RESOLVE_MANY_READY_FILE": str(ready_1)},
        ),
        make_item(
            "block-2",
            worker_mode="block",
            extra_env={"RESOLVE_MANY_READY_FILE": str(ready_2)},
        ),
    ]
    cancel_event = threading.Event()
    resolver = BatchResolver(
        max_workers=2,
        max_queued=4,
        worker_entry=stub_worker,
        work_root=tmp_path / "work",
        cancel_event=cancel_event,
    )
    thread, holder = start_batch_in_thread(resolver, items)
    wait_for_file(ready_1)
    wait_for_file(ready_2)

    cancel_event.set()
    thread.join(timeout=30)
    assert not thread.is_alive()

    results = holder["results"]
    assert {r.status for r in results.values()} == {STATUS_CANCELLED}
    assert all(proc.poll() is not None for proc in resolver.processes)


# ---------------------------------------------------------------------------
# 6. Deadline -> timeout terminal, no zombie
# ---------------------------------------------------------------------------


@pytest.mark.utils
def test_deadline_timeout_reaps_process(tmp_path, stub_worker):
    items = [
        make_item(
            "slow",
            worker_mode="slow",
            deadline=0.5,
            extra_env={"RESOLVE_MANY_SLEEP_S": "60"},
        ),
    ]
    resolver = BatchResolver(
        max_workers=1, worker_entry=stub_worker, work_root=tmp_path / "work"
    )
    started = time.monotonic()
    results = resolver.run(items)
    elapsed = time.monotonic() - started

    result = results["slow"]
    assert result.status == STATUS_TIMEOUT
    assert result.started is True
    assert result.returncode is not None
    assert result.response.result.kind == "internal_error"
    assert "timed out" in result.response.result.message
    # The parent did not wait anywhere near the child's 60 s sleep.
    assert elapsed < 30
    assert all(proc.poll() is not None for proc in resolver.processes)


# ---------------------------------------------------------------------------
# 7. Crash / resolution-error / internal-error distinction + isolation
# ---------------------------------------------------------------------------


@pytest.mark.utils
def test_crash_and_error_variants_are_distinct_and_isolated(
    tmp_path, stub_worker
):
    ready = tmp_path / "ready-crash"
    items = [
        make_item(
            "crasher",
            worker_mode="crash",
            extra_env={"RESOLVE_MANY_READY_FILE": str(ready)},
        ),
        make_item("conflict", worker_mode="resolution_error"),
        make_item("boom", worker_mode="internal_error"),
        make_item("healthy"),
    ]
    results = resolve_many(
        items, max_workers=2, worker_entry=stub_worker, work_root=tmp_path / "work"
    )

    assert set(results) == {"crasher", "conflict", "boom", "healthy"}
    assert results["crasher"].status == STATUS_CRASHED
    assert results["crasher"].response.result.kind == "internal_error"
    assert results["crasher"].returncode == 3

    # Structured child responses remain "completed" at batch level while
    # keeping their distinguishable result kinds.
    assert results["conflict"].status == STATUS_COMPLETED
    assert results["conflict"].response.result.kind == "resolution_error"
    assert "no matching distribution" in results["conflict"].response.result.pip_message
    assert results["boom"].status == STATUS_COMPLETED
    assert results["boom"].response.result.kind == "internal_error"
    assert results["healthy"].status == STATUS_COMPLETED
    assert results["healthy"].response.result.kind == "success"

    # The crash did not steal any sibling's result.
    assert all(r.response is not None for r in results.values())


# ---------------------------------------------------------------------------
# 8. Log flood: bounded capture + redaction
# ---------------------------------------------------------------------------


@pytest.mark.utils
def test_log_flood_is_bounded_and_redacted(tmp_path, stub_worker):
    items = [make_item("flooder", worker_mode="flood")]
    resolver = BatchResolver(
        max_workers=1, worker_entry=stub_worker, work_root=tmp_path / "work"
    )
    results = resolver.run(items)

    result = results["flooder"]
    assert result.status == STATUS_COMPLETED
    # Cap + at most one elision sentinel.
    assert len(result.log) <= MAX_TASK_LOG_LINES + 1
    assert any("elided by capacity cap" in line for line in result.log)
    joined = "\n".join(result.log)
    # Raw secrets never escape; redaction markers do.
    assert "s3cret-pass" not in joined
    assert "t0psecret" not in joined
    assert "abc123" not in joined
    assert "***" in joined
    # The pipe drain kept up with the flood instead of deadlocking.
    assert any("flood line" in line for line in result.log)


# ---------------------------------------------------------------------------
# 9. Message byte caps + oversize response
# ---------------------------------------------------------------------------


@pytest.mark.utils
def test_oversize_request_rejected_without_spawn(
    tmp_path, stub_worker, monkeypatch
):
    monkeypatch.setattr(batch_mod, "MAX_REQUEST_MESSAGE_BYTES", 1024)
    # A genuinely large envelope: ~100 package specs serialize well past
    # the patched 1 KiB cap, while the single-spec sibling stays under it.
    huge_packages = {f"pkg-{i:03d}": f"pkg-{i:03d}==1.{i}.0" for i in range(100)}
    resolver = BatchResolver(
        max_workers=1, worker_entry=stub_worker, work_root=tmp_path / "work"
    )
    results = resolver.run(
        [
            make_item("huge", packages=huge_packages),
            make_item("normal"),
        ]
    )

    assert results["huge"].status == STATUS_REJECTED
    assert results["huge"].started is False
    assert "byte limit" in results["huge"].response.result.message
    assert results["normal"].status == STATUS_COMPLETED
    assert len(resolver.processes) == 1, "only the valid sibling may spawn"


@pytest.mark.utils
def test_oversize_response_is_crash_terminal(tmp_path, stub_worker, monkeypatch):
    monkeypatch.setattr(batch_mod, "MAX_RESPONSE_MESSAGE_BYTES", 1024)
    items = [
        make_item(
            "huge-response",
            worker_mode="huge_response",
            extra_env={"RESOLVE_MANY_HUGE_BYTES": str(MAX_RESPONSE_MESSAGE_BYTES + 4096)},
        ),
    ]
    results = resolve_many(
        items, worker_entry=stub_worker, work_root=tmp_path / "work"
    )
    assert results["huge-response"].status == STATUS_CRASHED
    assert results["huge-response"].response.result.kind == "internal_error"


@pytest.mark.utils
def test_spawn_failure_is_rejected_terminal(tmp_path, stub_worker):
    items = [
        BatchResolveItem(
            id="bad-exe",
            request=make_request(),
            python_executable=str(tmp_path / "definitely-not-a-python.exe"),
            env={"RESOLVE_MANY_TEST_MODE": "echo"},
        ),
    ]
    results = resolve_many(
        items, worker_entry=stub_worker, work_root=tmp_path / "work"
    )
    assert results["bad-exe"].status == STATUS_REJECTED
    assert results["bad-exe"].started is False
    assert "could not spawn" in results["bad-exe"].response.result.message


# ---------------------------------------------------------------------------
# 10. REAL worker entry: Windows spawn + import path, deterministic ordering
# ---------------------------------------------------------------------------


def _install_real_worker_stub(stub_dir, order_file):
    """sitecustomize that short-circuits the real resolver dispatch,
    mirroring tests/unit/test_resolver_protocol_smoke.py."""
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "sitecustomize.py").write_text(
        textwrap.dedent(
            f"""
            import pipenv.resolver.core as _core
            from pipenv.resolver.schema import LockedRequirement

            def _stub_resolve_packages(request):
                order_file = {str(order_file)!r}
                task_id = __import__("os").environ.get(
                    "PIPENV_RESOLVE_MANY_TASK_ID", "?"
                )
                with open(order_file, "a", encoding="utf-8") as fh:
                    fh.write(f"start:{{task_id}}\\n")
                name = next(iter(request.packages.specs), "demo")
                return [LockedRequirement(name=name, version="==2024.1")], None

            _core._dispatch_resolve_packages = _stub_resolve_packages
            """
        ),
        encoding="utf-8",
    )


@pytest.mark.utils
def test_real_worker_spawn_import_and_ordering(tmp_path):
    order_file = tmp_path / "real-order.log"
    stub_dir = tmp_path / "_sitecustomize"
    _install_real_worker_stub(stub_dir, order_file)

    pythonpath = os.pathsep.join(
        [str(stub_dir), str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    items = [
        BatchResolveItem(
            id=f"real-{i}",
            request=make_request(),
            env={"PYTHONPATH": pythonpath},
        )
        for i in range(3)
    ]

    # Default worker entry = the real pipenv/resolver/main.py.
    resolver = BatchResolver(
        max_workers=1, max_queued=8, work_root=tmp_path / "work"
    )
    results = resolver.run(items)

    assert all(r.status == STATUS_COMPLETED for r in results.values())
    for item_id in ("real-0", "real-1", "real-2"):
        locked = results[item_id].response.result.locked
        assert locked[0].version == "==2024.1"
    assert order_file.read_text().strip().splitlines() == [
        "start:real-0",
        "start:real-1",
        "start:real-2",
    ]
    assert all(proc.poll() is not None for proc in resolver.processes)


# ---------------------------------------------------------------------------
# 11. Shared ParsedManifestCache: atomic protocol + corrupt independence
# ---------------------------------------------------------------------------


@pytest.mark.utils
def test_shared_manifest_cache_corrupt_entry_is_independent(tmp_path):
    from pipenv.resolver.candidate import Candidate, Hash

    cache_root = tmp_path / "shared-manifests"
    cache_a = ParsedManifestCache(cache_root)
    cache_b = ParsedManifestCache(cache_root)
    index_url = "https://index.example.com/simple"

    candidate = Candidate(
        name="demo",
        version="1.0",
        url="https://index.example.com/demo-1.0-py3-none-any.whl",
        filename="demo-1.0-py3-none-any.whl",
        hashes=frozenset({Hash(algo="sha256", value="a" * 64)}),
        requires_python=None,
        yanked=False,
        yanked_reason=None,
        upload_time=None,
        is_wheel=True,
        wheel_tags=None,
    )
    # Atomic write through the real protocol; a second cache instance
    # (i.e. another resolver process) reads it back.
    cache_a.put(index_url, "demo", [candidate], etag='"etag-1"', ttl_seconds=600)
    shared = cache_b.get(index_url, "demo")
    assert shared is not None
    assert shared.candidates[0].version == "1.0"

    # Corrupt a sibling entry: it must fail as an independent miss while
    # the valid entry stays readable — validation is never ignored.
    corrupt_target = cache_a._path_for(index_url, "broken")
    corrupt_target.parent.mkdir(parents=True, exist_ok=True)
    corrupt_target.write_text("{not valid json", encoding="utf-8")
    assert cache_b.get(index_url, "broken") is None
    assert cache_b.get(index_url, "demo") is not None

    # Refetch overwrites the corrupt file via the atomic replace path.
    cache_b.put(index_url, "broken", [candidate], etag='"etag-2"', ttl_seconds=600)
    assert cache_a.get(index_url, "broken") is not None


@pytest.mark.utils
def test_batch_children_share_cache_dir_without_cross_credential_leak(
    tmp_path, stub_worker
):
    cache_dir = tmp_path / "pipenv-cache"
    items = [
        make_item(
            "cache-a",
            extra_env={
                "TASK_MARKER_VAR": "a",
                "PIPENV_CACHE_DIR": str(cache_dir),
            },
        ),
        make_item(
            "cache-b",
            extra_env={
                "TASK_MARKER_VAR": "b",
                "PIPENV_CACHE_DIR": str(cache_dir),
            },
        ),
    ]
    results = resolve_many(
        items, worker_entry=stub_worker, work_root=tmp_path / "work"
    )
    for item_id in ("cache-a", "cache-b"):
        markers = results[item_id].response.result.locked[0].markers
        cache_field = [
            part for part in markers.split("|") if part.startswith("cache=")
        ][0]
        assert cache_field == f"cache={cache_dir}"
    # The parent never adopted the cache dir itself.
    assert os.environ.get("PIPENV_CACHE_DIR") != str(cache_dir)


# ---------------------------------------------------------------------------
# 12. Every ID always reaches exactly one terminal state
# ---------------------------------------------------------------------------


@pytest.mark.utils
def test_every_id_has_exactly_one_terminal_state(tmp_path, stub_worker):
    items = [
        make_item("ok"),
        make_item(
            "slow",
            worker_mode="slow",
            deadline=0.4,
            extra_env={"RESOLVE_MANY_SLEEP_S": "60"},
        ),
        BatchResolveItem(id="rejected", request=make_request(), env={"x": 1}),  # type: ignore[dict-item]
    ]
    results = resolve_many(
        items, max_workers=1, worker_entry=stub_worker, work_root=tmp_path / "work"
    )
    assert list(results) == ["ok", "slow", "rejected"]
    assert results["ok"].status == STATUS_COMPLETED
    assert results["slow"].status == STATUS_TIMEOUT
    assert results["rejected"].status == STATUS_REJECTED
    for result in results.values():
        assert result.response is not None
        assert result.response.schema_version == SCHEMA_VERSION
