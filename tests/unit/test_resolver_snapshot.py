"""Record/replay resolution-snapshot tests for the pure-Python backend.

These tests drive the REAL resolver stack end-to-end
(``PEP691Client`` + ``ParallelFetcher`` + ``ParsedManifestCache`` +
``resolvelib``) against an in-process fake index and assert the
record/replay contract:

* recording captures the exact simple-index manifests, PEP 658 wheel
  metadata, canonical request and marker environment the resolve used;
* replay is genuinely offline (sockets are blocked, the live index is
  never contacted and the host's mutable manifest cache is never read)
  and returns comparable lock entries — version, markers, hashes and
  source — not merely a success flag;
* deleting/mutating live candidates after recording cannot change a
  replay; same-named packages on different indexes stay distinct;
* corrupted blobs, missing metadata/manifests, format skew and
  incomplete recordings are all refused loudly with the right typed
  error (SnapshotCorrupt / ReplayMiss / IncompleteSnapshot /
  RequestMismatch), never a silent network fallback;
* credentials never reach the snapshot directory and principal
  namespaces partition authorization views instead of merging
  credential-stripped URLs.
"""
from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from pipenv.resolver.backends.pure_python import PurePythonBackend
from pipenv.resolver.fetcher import ParallelFetcher
from pipenv.resolver.manifest_cache import ParsedManifestCache
from pipenv.resolver.pep691 import PEP691Client
from pipenv.resolver.pure_python_metadata import MetadataCache
from pipenv.resolver.schema import (
    SCHEMA_VERSION,
    InternalError,
    PackageSpecs,
    ResolverOptions,
    ResolverRequest,
    ResolverSuccess,
    SnapshotMode,
    Source,
)
from pipenv.resolver.snapshot import (
    SNAPSHOT_FORMAT_VERSION,
    IncompleteSnapshot,
    ReplayMiss,
    SnapshotBundle,
    SnapshotCorrupt,
    classify_exchange,
    compare_lock_entries,
    detect_unrecordable_top_level_specs,
    load_snapshot,
    write_snapshot,
)

_INDEX_A = "https://primary.example/simple"
_INDEX_B = "https://secondary.example/simple"
_FILES_A = "https://files-primary.example/packages"
_FILES_B = "https://files-secondary.example/packages"

_TARGET_ENV = {
    "python_version": "3.11",
    "python_full_version": "3.11.7",
    "sys_platform": "linux",
    "platform_machine": "x86_64",
    "platform_system": "Linux",
    "os_name": "posix",
    "implementation_name": "cpython",
}

# Credential fixture values — kept as constants so the leak assertions
# below never trip the magic-value lint and stay greppable in one place.
_SECRET_USER = "tokenuser"
_SECRET_PASSWORD = "secrettoken123"
_BASIC_AUTH_PREFIX = b"Basic "


# ---------------------------------------------------------------------------
# Fake index infrastructure
# ---------------------------------------------------------------------------


class _FakeResponse:
    """urllib3-response-shaped object (`.status` / `.data` / `.headers`)."""

    def __init__(self, status: int, body: bytes, headers: dict[str, str] | None = None):
        self.status = status
        self.status_code = status
        self.data = body
        self.content = body
        self.headers = headers or {}

    def release_conn(self) -> None:
        return None


class FakeIndexSession:
    """In-process simple index + PEP 658 sidecar server.

    ``pages`` maps full simple-page URLs to JSON bodies; ``sidecars``
    maps ``<wheel_url>.metadata`` URLs to METADATA bytes.  Every request
    is logged so tests can prove replay never touches the live world.
    """

    def __init__(
        self,
        pages: dict[str, bytes],
        sidecars: dict[str, bytes],
    ) -> None:
        self._pages = dict(pages)
        self._sidecars = dict(sidecars)
        self.request_log: list[tuple[str, str]] = []

    @property
    def request_count(self) -> int:
        return len(self.request_log)

    def request(self, method, url, headers=None, timeout=None, verify=None, cert=None):
        self.request_log.append((method.upper(), url))
        if url in self._sidecars:
            return _FakeResponse(
                200,
                self._sidecars[url],
                {"Content-Type": "application/octet-stream"},
            )
        if url in self._pages:
            return _FakeResponse(
                200,
                self._pages[url],
                {"Content-Type": "application/vnd.pypi.simple.v1+json", "ETag": '"etag"'},
            )
        return _FakeResponse(404, b"", {"Content-Type": "text/plain"})

    # Test helpers to mutate the *live* world after a recording exists.
    def delete_page(self, url: str) -> None:
        self._pages.pop(url, None)

    def replace_page(self, url: str, body: bytes) -> None:
        self._pages[url] = body


def _metadata_body(
    name: str,
    version: str,
    *,
    requires_dist: tuple[str, ...] = (),
    requires_python: str | None = None,
) -> bytes:
    lines = ["Metadata-Version: 2.1", f"Name: {name}", f"Version: {version}"]
    if requires_python:
        lines.append(f"Requires-Python: {requires_python}")
    lines.extend(f"Requires-Dist: {line}" for line in requires_dist)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _wheel_entry(
    name: str,
    version: str,
    metadata: bytes,
    *,
    files_base: str,
    requires_python: str | None = None,
) -> tuple[dict, str, str]:
    """Return (PEP691 file entry, wheel URL, sidecar URL) for a wheel."""
    filename = f"{name}-{version}-py3-none-any.whl"
    wheel_url = f"{files_base}/{filename}"
    sidecar_url = wheel_url + ".metadata"
    entry = {
        "filename": filename,
        "url": wheel_url,
        "hashes": {
            "sha256": hashlib.sha256(wheel_url.encode("utf-8")).hexdigest()
        },
        "core-metadata": {"sha256": hashlib.sha256(metadata).hexdigest()},
        "requires-python": requires_python,
        "yanked": False,
    }
    return entry, wheel_url, sidecar_url


def _page(name: str, entries: list[dict]) -> bytes:
    return json.dumps(
        {"meta": {"api-version": "1.0"}, "name": name, "files": entries}
    ).encode("utf-8")


def _page_url(index_url: str, name: str) -> str:
    return f"{index_url.rstrip('/')}/{name}/"


def build_world() -> tuple[dict[str, bytes], dict[str, bytes]]:
    """Build the standard fixture world.

    * ``app`` 1.0.0 (no deps) and 2.0.0 (requires ``lib>=1.0`` and the
      marker-gated ``py311dep`` on Python >= 3.11);
    * ``lib`` 1.0.0 / 1.1.0, leaf;
    * ``py311dep`` 1.0.0, leaf;
    * ``dup`` 1.0.0 on index A and 2.0.0 on index B — same canonical
      name, two authorization-distinct sources.
    """
    pages: dict[str, bytes] = {}
    sidecars: dict[str, bytes] = {}

    app_10_meta = _metadata_body("app", "1.0.0")
    app_20_meta = _metadata_body(
        "app",
        "2.0.0",
        requires_dist=(
            "lib>=1.0",
            "py311dep; python_version >= '3.11'",
        ),
    )
    lib_10_meta = _metadata_body("lib", "1.0.0")
    lib_11_meta = _metadata_body("lib", "1.1.0")
    py311_meta = _metadata_body("py311dep", "1.0.0")
    dup_a_meta = _metadata_body("dup", "1.0.0")
    dup_b_meta = _metadata_body("dup", "2.0.0")

    app_10, _u, s = _wheel_entry("app", "1.0.0", app_10_meta, files_base=_FILES_A)
    sidecars[s] = app_10_meta
    app_20, _u, s = _wheel_entry("app", "2.0.0", app_20_meta, files_base=_FILES_A)
    sidecars[s] = app_20_meta
    pages[_page_url(_INDEX_A, "app")] = _page("app", [app_10, app_20])

    lib_10, _u, s = _wheel_entry("lib", "1.0.0", lib_10_meta, files_base=_FILES_A)
    sidecars[s] = lib_10_meta
    lib_11, _u, s = _wheel_entry("lib", "1.1.0", lib_11_meta, files_base=_FILES_A)
    sidecars[s] = lib_11_meta
    pages[_page_url(_INDEX_A, "lib")] = _page("lib", [lib_10, lib_11])

    py311, _u, s = _wheel_entry("py311dep", "1.0.0", py311_meta, files_base=_FILES_A)
    sidecars[s] = py311_meta
    pages[_page_url(_INDEX_A, "py311dep")] = _page("py311dep", [py311])

    dup_a, _u, s = _wheel_entry("dup", "1.0.0", dup_a_meta, files_base=_FILES_A)
    sidecars[s] = dup_a_meta
    pages[_page_url(_INDEX_A, "dup")] = _page("dup", [dup_a])

    dup_b, _u, s = _wheel_entry("dup", "2.0.0", dup_b_meta, files_base=_FILES_B)
    sidecars[s] = dup_b_meta
    pages[_page_url(_INDEX_B, "dup")] = _page("dup", [dup_b])

    return pages, sidecars


# ---------------------------------------------------------------------------
# Backend / request helpers
# ---------------------------------------------------------------------------


def _sources(*urls: str) -> tuple[Source, ...]:
    names = {_INDEX_A: "primary", _INDEX_B: "secondary"}
    return tuple(
        Source(name=names.get(url, f"src-{i}"), url=url, verify_ssl=True)
        for i, url in enumerate(urls)
    )


def _request(
    specs: dict[str, str],
    *,
    sources: tuple[Source, ...],
    snapshot: SnapshotMode | None = None,
    target_python: dict | None = None,
) -> ResolverRequest:
    return ResolverRequest(
        schema_version=SCHEMA_VERSION,
        category="default",
        packages=PackageSpecs(specs=specs),
        options=ResolverOptions(),
        sources=sources,
        snapshot=snapshot,
    )


def _online_backend(session: FakeIndexSession, tmp_path: Path, env=None):
    cache = ParsedManifestCache(tmp_path / "host-style-manifests")
    client = PEP691Client(session)
    fetcher = ParallelFetcher(client, cache)
    metadata_cache = MetadataCache(tmp_path / "host-style-metadata")
    return PurePythonBackend(
        cache=cache,
        fetcher=fetcher,
        session=session,
        metadata_cache=metadata_cache,
        target_env=dict(env or _TARGET_ENV),
    )


def _locked_map(response) -> dict[str, dict]:
    assert isinstance(response.result, ResolverSuccess)
    return {entry.name: entry.to_json_dict() for entry in response.result.locked}


@pytest.fixture
def blocked_socket(monkeypatch):
    """Hard network block: any real socket construction fails."""

    def _refuse(*args, **kwargs):
        raise OSError("network access is forbidden in replay tests")

    monkeypatch.setattr(socket, "socket", _refuse)
    monkeypatch.setattr(socket, "create_connection", _refuse)


# ---------------------------------------------------------------------------
# Schema-level tests
# ---------------------------------------------------------------------------


class TestSnapshotModeSchema:
    def test_wire_suppressed_when_unset(self):
        req = _request({"app": "app"}, sources=_sources(_INDEX_A))
        encoded = req.to_json_dict()
        assert "snapshot" not in encoded
        # Round-trip stays None.
        assert ResolverRequest.from_json_dict(encoded).snapshot is None

    def test_wire_roundtrip(self):
        mode = SnapshotMode(
            mode="record",
            path="/tmp/snap",
            principal_namespace="team-a",
            strict=False,
        )
        req = _request({"app": "app"}, sources=_sources(_INDEX_A), snapshot=mode)
        encoded = req.to_json_dict()
        restored = ResolverRequest.from_json_dict(encoded).snapshot
        assert restored == mode
        assert encoded["snapshot"]["principal_namespace"] == "team-a"

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            SnapshotMode(mode="time-travel", path="/tmp/snap")
        with pytest.raises(ValueError):
            SnapshotMode(mode="record", path="")

    def test_strict_defaults_true(self):
        assert SnapshotMode(mode="replay", path="/tmp/snap").strict is True


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


class TestRecording:
    def test_publishes_content_addressed_complete_snapshot(self, tmp_path):
        pages, sidecars = build_world()
        session = FakeIndexSession(pages, sidecars)
        backend = _online_backend(session, tmp_path)
        snap_dir = tmp_path / "snapshot"

        response = backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )

        assert isinstance(response.result, ResolverSuccess)
        manifest_path = snap_dir / "snapshot.json"
        assert manifest_path.is_file()
        bundle = load_snapshot(snap_dir)
        assert bundle.complete is True
        assert bundle.incomplete_reasons == ()
        assert bundle.principal_namespace == "default"

        # Content addressing: every blob file is named by its sha256
        # and the commit manifest enumerates all of them with sizes.
        manifest = json.loads(manifest_path.read_text("utf-8"))
        blobs_dir = snap_dir / "blobs"
        on_disk = {p.name for p in blobs_dir.iterdir()}
        committed = {b["digest"] for b in manifest["blobs"]}
        assert committed == on_disk
        for digest in committed:
            assert hashlib.sha256((blobs_dir / digest).read_bytes()).hexdigest() == digest

        # The exact candidate world used: manifests + wheel metadata,
        # canonical names, source identity and candidate file hashes.
        roles = {e.role for e in bundle.exchanges}
        assert "simple-manifest" in roles
        assert "wheel-metadata" in roles
        manifest_records = {e.package: e for e in bundle.exchanges if e.role == "simple-manifest"}
        assert set(manifest_records) == {"app", "lib", "py311dep"}
        for record in manifest_records.values():
            assert record.source_name == "primary"
            assert record.candidate_identities
            assert all(
                identity["hashes"] and identity["filename"]
                for identity in record.candidate_identities
            )

        # Canonical request + target marker environment are frozen.
        assert bundle.canonical_request["packages"] == {"app": "app"}
        assert bundle.target_env["python_version"] == "3.11"
        assert bundle.frozen_config["allow_prereleases"] is False
        assert bundle.frozen_config["index_urls"] == [_INDEX_A]

        # Recorded lock entries carry full comparable detail.
        recorded = {e["name"]: e for e in bundle.recorded_locked}
        assert set(recorded) == {"app", "lib", "py311dep"}
        assert recorded["app"]["version"] == "==2.0.0"
        assert recorded["app"]["index"] == "primary"
        assert recorded["app"]["hashes"]
        assert recorded["py311dep"]["markers"] is not None

    def test_marker_branch_is_part_of_recorded_world(self, tmp_path):
        pages, sidecars = build_world()
        session = FakeIndexSession(pages, sidecars)
        backend = _online_backend(session, tmp_path)
        snap_dir = tmp_path / "snapshot"
        response = backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )
        names = set(_locked_map(response))
        # python_version 3.11 activates the marker-gated transitive.
        assert "py311dep" in names
        bundle = load_snapshot(snap_dir)
        assert any(e.package == "py311dep" for e in bundle.manifest_exchanges())

    def test_dual_index_same_name_kept_distinct(self, tmp_path):
        pages, sidecars = build_world()
        session = FakeIndexSession(pages, sidecars)
        backend = _online_backend(session, tmp_path)
        snap_dir = tmp_path / "snapshot"

        response = backend.resolve(
            _request(
                {"dup": "dup"},
                sources=_sources(_INDEX_A, _INDEX_B),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )
        assert isinstance(response.result, ResolverSuccess)
        bundle = load_snapshot(snap_dir)
        dup_manifests = [
            e for e in bundle.manifest_exchanges() if e.package == "dup"
        ]
        # Two distinct source identities for the same canonical name.
        assert len(dup_manifests) == 2
        source_names = {e.source_name for e in dup_manifests}
        assert source_names == {"primary", "secondary"}
        urls = {e.url for e in dup_manifests}
        assert len(urls) == 2
        # The higher version, served by the secondary index, wins.
        locked = _locked_map(response)
        assert locked["dup"]["version"] == "==2.0.0"

    def test_failed_resolve_publishes_nothing(self, tmp_path):
        pages, sidecars = build_world()
        session = FakeIndexSession(pages, sidecars)
        backend = _online_backend(session, tmp_path)
        snap_dir = tmp_path / "snapshot"

        # An unsatisfiable pin makes resolvelib fail; no recording may
        # be published for a failed resolve.
        response = backend.resolve(
            _request(
                {"app": "app==99.0.0"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )
        assert response.result.kind == "resolution_error"
        assert not snap_dir.exists() or not (snap_dir / "snapshot.json").exists()

    def test_write_failure_never_publishes_manifest(self, tmp_path, monkeypatch):
        pages, sidecars = build_world()
        session = FakeIndexSession(pages, sidecars)
        backend = _online_backend(session, tmp_path)
        snap_dir = tmp_path / "snapshot"

        import pipenv.resolver.snapshot.bundle as bundle_mod

        def _explode(target, body):
            raise OSError("simulated disk failure")

        monkeypatch.setattr(bundle_mod, "_atomic_write", _explode)
        response = backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )
        assert isinstance(response.result, InternalError)
        assert "simulated disk failure" in response.result.message
        assert not (snap_dir / "snapshot.json").exists()


# ---------------------------------------------------------------------------
# Replay — offline fidelity
# ---------------------------------------------------------------------------


class TestReplay:
    def test_replay_offline_matches_recording(
        self, tmp_path, blocked_socket
    ):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        recorded_response = record_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )
        assert isinstance(recorded_response.result, ResolverSuccess)
        requests_after_record = live.request_count

        # Fresh backend with NO collaborators: replay must not bootstrap
        # a host cache or network session, and sockets are blocked.
        replay_backend = PurePythonBackend(target_env=dict(_TARGET_ENV))
        replay_response = replay_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("replay", str(snap_dir)),
            )
        )

        assert isinstance(replay_response.result, ResolverSuccess)
        # The live index was never contacted during replay.
        assert live.request_count == requests_after_record
        # Replay collaborators live in private temp dirs, never the
        # host's mutable manifest cache.
        assert str(replay_backend._cache._root).startswith(
            str(Path(tempfile_gettempdir()))
        )
        assert "host-style" not in str(replay_backend._cache._root)

        recorded_map = _locked_map(recorded_response)
        replay_map = _locked_map(replay_response)
        assert set(replay_map) == set(recorded_map)
        for name, recorded_entry in recorded_map.items():
            # Full entry comparison: version + markers + hashes + source.
            assert replay_map[name] == recorded_entry

        # The persisted comparable report agrees and is not a bare
        # success marker.
        report = json.loads((snap_dir / "replay-report.json").read_text("utf-8"))
        assert report["comparison"]["equal"] is True
        assert report["comparison"]["mismatches"] == []
        assert {e["name"] for e in report["replay_locked"]} == set(recorded_map)

    def test_live_index_deletion_and_mutation_cannot_change_replay(
        self, tmp_path, blocked_socket
    ):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        recorded = record_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )
        recorded_map = _locked_map(recorded)
        original_app_hash = recorded_map["app"]["hashes"]

        # Mutate the live world: delete the app page entirely and
        # change lib's advertised hash on another index session.
        live.delete_page(_page_url(_INDEX_A, "app"))
        mutated_lib_page = json.loads(pages[_page_url(_INDEX_A, "lib")])
        for entry in mutated_lib_page["files"]:
            entry["hashes"] = {"sha256": "f" * 64}
        live.replace_page(
            _page_url(_INDEX_A, "lib"),
            json.dumps(mutated_lib_page).encode("utf-8"),
        )

        replay_backend = PurePythonBackend(target_env=dict(_TARGET_ENV))
        replay = replay_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("replay", str(snap_dir)),
            )
        )
        replay_map = _locked_map(replay)
        assert replay_map["app"]["version"] == "==2.0.0"
        assert replay_map["app"]["hashes"] == original_app_hash
        # The mutated live hash never entered the replay.
        assert "f" * 64 not in "".join(
            "".join(e["hashes"]) for e in replay_map.values()
        )

    def test_dual_index_replay_survives_live_removal(
        self, tmp_path, blocked_socket
    ):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        record_backend.resolve(
            _request(
                {"dup": "dup"},
                sources=_sources(_INDEX_A, _INDEX_B),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )
        # Remove the secondary index's live page.
        live.delete_page(_page_url(_INDEX_B, "dup"))

        replay_backend = PurePythonBackend(target_env=dict(_TARGET_ENV))
        replay = replay_backend.resolve(
            _request(
                {"dup": "dup"},
                sources=_sources(_INDEX_A, _INDEX_B),
                snapshot=SnapshotMode("replay", str(snap_dir)),
            )
        )
        # Still the recorded secondary-index winner.
        assert _locked_map(replay)["dup"]["version"] == "==2.0.0"

    def test_marker_environment_mismatch_refused_strict(
        self, tmp_path, blocked_socket
    ):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        record_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )

        other_env = dict(_TARGET_ENV)
        other_env["python_version"] = "3.10"
        replay_backend = PurePythonBackend(target_env=other_env)
        response = replay_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("replay", str(snap_dir)),
            )
        )
        assert isinstance(response.result, InternalError)
        assert "target marker environment differs" in response.result.message

    def test_unrecorded_manifest_is_explicit_replay_miss(
        self, tmp_path, blocked_socket
    ):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        record_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )

        # Drop lib's simple-manifest exchange + blob from the bundle.
        manifest_path = snap_dir / "snapshot.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        removed_digests = set()
        kept_exchanges = []
        for exchange in manifest["exchanges"]:
            if exchange["role"] == "simple-manifest" and exchange["package"] == "lib":
                removed_digests.add(exchange["body_digest"])
                continue
            kept_exchanges.append(exchange)
        manifest["exchanges"] = kept_exchanges
        manifest["blobs"] = [
            b for b in manifest["blobs"] if b["digest"] not in removed_digests
        ]
        for digest in removed_digests:
            (snap_dir / "blobs" / digest).unlink()
        manifest_path.write_text(json.dumps(manifest), "utf-8")

        replay_backend = PurePythonBackend(target_env=dict(_TARGET_ENV))
        response = replay_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("replay", str(snap_dir)),
            )
        )
        assert isinstance(response.result, InternalError)
        assert "ReplayMiss" in response.result.message
        assert "/lib/" in response.result.message

    def test_missing_top_level_manifest_caught_at_prefetch(
        self, tmp_path, blocked_socket
    ):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        record_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )

        # Drop the TOP-LEVEL app manifest: the miss fires in pre-fetch,
        # and ``resolve`` must still return a structured response
        # rather than letting ReplayMiss escape the backend.
        manifest_path = snap_dir / "snapshot.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        removed_digests = set()
        kept_exchanges = []
        for exchange in manifest["exchanges"]:
            if exchange["role"] == "simple-manifest" and exchange["package"] == "app":
                removed_digests.add(exchange["body_digest"])
                continue
            kept_exchanges.append(exchange)
        manifest["exchanges"] = kept_exchanges
        manifest["blobs"] = [
            b for b in manifest["blobs"] if b["digest"] not in removed_digests
        ]
        for digest in removed_digests:
            (snap_dir / "blobs" / digest).unlink()
        manifest_path.write_text(json.dumps(manifest), "utf-8")

        replay_backend = PurePythonBackend(target_env=dict(_TARGET_ENV))
        response = replay_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("replay", str(snap_dir)),
            )
        )
        assert isinstance(response.result, InternalError)
        assert "ReplayMiss" in response.result.message
        assert "/app/" in response.result.message

    def test_missing_wheel_metadata_is_replay_miss(
        self, tmp_path, blocked_socket
    ):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        record_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )

        # Remove the recorded sidecar for lib 1.1.0 (the resolved lib).
        manifest_path = snap_dir / "snapshot.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        removed_digests = set()
        kept_exchanges = []
        for exchange in manifest["exchanges"]:
            if (
                exchange["role"] == "wheel-metadata"
                and "lib-1.1.0" in exchange["url"]
            ):
                removed_digests.add(exchange["body_digest"])
                continue
            kept_exchanges.append(exchange)
        manifest["exchanges"] = kept_exchanges
        manifest["blobs"] = [
            b for b in manifest["blobs"] if b["digest"] not in removed_digests
        ]
        for digest in removed_digests:
            (snap_dir / "blobs" / digest).unlink()
        manifest_path.write_text(json.dumps(manifest), "utf-8")

        replay_backend = PurePythonBackend(target_env=dict(_TARGET_ENV))
        response = replay_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("replay", str(snap_dir)),
            )
        )
        assert isinstance(response.result, InternalError)
        assert "ReplayMiss" in response.result.message
        assert ".metadata" in response.result.message

    def test_corrupt_blob_detected(self, tmp_path, blocked_socket):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        record_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )
        # Tamper a blob on disk (appended bytes → size check fires).
        any_blob = next((snap_dir / "blobs").iterdir())
        any_blob.write_bytes(any_blob.read_bytes() + b"tampered")

        replay_backend = PurePythonBackend(target_env=dict(_TARGET_ENV))
        response = replay_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("replay", str(snap_dir)),
            )
        )
        assert isinstance(response.result, InternalError)
        assert "mismatch" in response.result.message

    def test_same_length_tamper_hits_digest_check(
        self, tmp_path, blocked_socket
    ):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        record_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )
        # Same-length rewrite keeps the committed size valid but
        # invalidates the sha256 content digest.
        blob = next((snap_dir / "blobs").iterdir())
        original = blob.read_bytes()
        flipped = bytearray(original)
        flipped[0] = flipped[0] ^ 0xFF
        blob.write_bytes(bytes(flipped))

        replay_backend = PurePythonBackend(target_env=dict(_TARGET_ENV))
        response = replay_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("replay", str(snap_dir)),
            )
        )
        assert isinstance(response.result, InternalError)
        assert "digest mismatch" in response.result.message

    def test_format_version_skew_refused(self, tmp_path, blocked_socket):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        record_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )
        manifest_path = snap_dir / "snapshot.json"
        manifest = json.loads(manifest_path.read_text("utf-8"))
        manifest["format_version"] = SNAPSHOT_FORMAT_VERSION + 999
        manifest_path.write_text(json.dumps(manifest), "utf-8")

        replay_backend = PurePythonBackend(target_env=dict(_TARGET_ENV))
        response = replay_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("replay", str(snap_dir)),
            )
        )
        assert isinstance(response.result, InternalError)
        assert "format_version" in response.result.message

    def test_principal_namespace_mismatch_refused(
        self, tmp_path, blocked_socket
    ):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        record_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode(
                    "record", str(snap_dir), principal_namespace="team-a"
                ),
            )
        )
        bundle = load_snapshot(snap_dir)
        assert bundle.principal_namespace == "team-a"
        assert all(
            e.principal_namespace == "team-a" for e in bundle.exchanges
        )

        replay_backend = PurePythonBackend(target_env=dict(_TARGET_ENV))
        response = replay_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode(
                    "replay", str(snap_dir), principal_namespace="team-b"
                ),
            )
        )
        assert isinstance(response.result, InternalError)
        assert "principal_namespace mismatch" in response.result.message

    def test_non_strict_replay_warns_instead_of_refusing(
        self, tmp_path, blocked_socket
    ):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        record_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode(
                    "record", str(snap_dir), principal_namespace="team-a"
                ),
            )
        )

        replay_backend = PurePythonBackend(target_env=dict(_TARGET_ENV))
        response = replay_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode(
                    "replay",
                    str(snap_dir),
                    principal_namespace="team-b",
                    strict=False,
                ),
            )
        )
        # Non-strict mode proceeds but records the compatibility note.
        assert isinstance(response.result, ResolverSuccess)
        warnings = " ".join(response.diagnostics.warnings)
        assert "principal_namespace mismatch" in warnings


# ---------------------------------------------------------------------------
# Credential safety
# ---------------------------------------------------------------------------


class TestCredentialSafety:
    def test_no_credentials_leak_into_snapshot(self, tmp_path):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        secret_url = (
            f"https://{_SECRET_USER}:{_SECRET_PASSWORD}@primary.example/simple"
        )

        response = record_backend.resolve(
            _request(
                {"app": "app"},
                sources=(
                    Source(name="primary", url=secret_url, verify_ssl=True),
                ),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )
        assert isinstance(response.result, ResolverSuccess)

        all_bytes = b"".join(p.read_bytes() for p in snap_dir.rglob("*") if p.is_file())
        assert _SECRET_PASSWORD.encode("utf-8") not in all_bytes
        assert _SECRET_USER.encode("utf-8") not in all_bytes
        assert _BASIC_AUTH_PREFIX not in all_bytes

        bundle = load_snapshot(snap_dir)
        # The stripped URL — not the credential-bearing one — is stored.
        assert bundle.canonical_request["sources"] == [
            {"name": "primary", "url": _INDEX_A, "verify_ssl": True}
        ]
        assert all("authorization" not in e.request_headers for e in bundle.exchanges)

    def test_credential_views_partitioned_by_principal(self, tmp_path):
        # Same stripped URL, two principals: snapshots must not merge.
        from pipenv.resolver.snapshot import RecordingSession

        class _Inner:
            def request(self, *a, **k):  # pragma: no cover - exercised below
                return _FakeResponse(200, b"{}", {"Content-Type": "application/vnd.pypi.simple.v1+json"})

        rec_a = RecordingSession(
            _Inner(), principal_namespace="team-a", source_by_url={_INDEX_A: "primary"}
        )
        rec_b = RecordingSession(
            _Inner(), principal_namespace="team-b", source_by_url={_INDEX_A: "primary"}
        )
        rec_a.request(
            "GET",
            _page_url(_INDEX_A, "app"),
            headers={"Accept": "application/vnd.pypi.simple.v1+json",
                     "Authorization": "Basic xyz",
                     "Cookie": "session=abc"},
        )
        rec_b.request(
            "GET",
            _page_url(_INDEX_A, "app"),
            headers={"Accept": "application/vnd.pypi.simple.v1+json"},
        )
        assert rec_a.records[0].principal_namespace == "team-a"
        assert rec_b.records[0].principal_namespace == "team-b"
        assert "authorization" not in rec_a.records[0].request_headers
        assert "cookie" not in rec_a.records[0].request_headers


# ---------------------------------------------------------------------------
# Incomplete-scope handling
# ---------------------------------------------------------------------------


class TestIncompleteScope:
    def test_unrecordable_spec_detection(self):
        req = _request(
            {
                "vcs-pkg": "git+https://example.com/repo.git",
                "local-pkg": "file:///wheels/local.whl",
                "normal": "normal>=1",
            },
            sources=_sources(_INDEX_A),
        )
        reasons = detect_unrecordable_top_level_specs(req)
        joined = " ".join(reasons)
        assert "vcs-pkg" in joined
        assert "local-pkg" in joined
        assert "normal" not in joined

    def test_role_classification(self):
        assert (
            classify_exchange(
                "GET", "https://x/packages/app-1.0.tar.gz", {}
            )
            == "sdist-build"
        )
        assert (
            classify_exchange("GET", "https://x/app-1.0.whl.metadata", {})
            == "wheel-metadata"
        )
        assert (
            classify_exchange("HEAD", "https://x/app-1.0.whl", {})
            == "wheel-range"
        )
        assert (
            classify_exchange(
                "GET",
                "https://x/app-1.0.whl",
                {"Range": "bytes=0-1"},
            )
            == "wheel-range"
        )
        assert (
            classify_exchange(
                "GET",
                "git+https://example.com/repo.git",
                {},
            )
            == "vcs"
        )
        assert (
            classify_exchange(
                "GET",
                _page_url(_INDEX_A, "app"),
                {"Accept": "application/vnd.pypi.simple.v1+json"},
            )
            == "simple-manifest"
        )

    def test_recorder_marks_sdist_exchange_incomplete(self):
        from pipenv.resolver.snapshot import RecordingSession

        class _Inner:
            def request(self, method, url, headers=None, **kwargs):
                return _FakeResponse(200, b"sdist-bytes", {})

        recorder = RecordingSession(
            _Inner(), principal_namespace="default"
        )
        recorder.request("GET", "https://x/packages/legacy-1.0.tar.gz")
        assert recorder.incomplete_reasons
        assert any(
            "sdist-build" in reason for reason in recorder.incomplete_reasons
        )

    def test_strict_replay_refuses_incomplete_bundle(
        self, tmp_path, blocked_socket
    ):
        # Hand-build an incomplete bundle (e.g. an sdist-only resolve).
        bundle = SnapshotBundle(
            format_version=SNAPSHOT_FORMAT_VERSION,
            principal_namespace="default",
            created_at="2026-01-01T00:00:00+00:00",
            complete=False,
            incomplete_reasons=(
                "sdist-build exchange is outside the recordable scope",
            ),
            canonical_request={},
            target_env={},
            frozen_config={},
            algorithm={},
            exchanges=(),
            recorded_locked=(),
            bodies={},
        )
        snap_dir = tmp_path / "incomplete-snap"
        write_snapshot(bundle, snap_dir)
        # The manifest exists but is explicitly marked incomplete.
        assert json.loads((snap_dir / "snapshot.json").read_text())["complete"] is False

        replay_backend = PurePythonBackend(target_env=dict(_TARGET_ENV))
        response = replay_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("replay", str(snap_dir)),
            )
        )
        assert isinstance(response.result, InternalError)
        assert "incomplete" in response.result.message.lower()

    def test_loading_corrupt_manifest_raises(self, tmp_path):
        snap_dir = tmp_path / "broken"
        snap_dir.mkdir()
        (snap_dir / "snapshot.json").write_text("{not json", "utf-8")
        with pytest.raises(SnapshotCorrupt):
            load_snapshot(snap_dir)


# ---------------------------------------------------------------------------
# Lock comparison API
# ---------------------------------------------------------------------------


class TestLockComparison:
    def test_detects_hash_and_source_differences(self):
        recorded = [
            {
                "name": "app",
                "version": "==2.0.0",
                "markers": None,
                "hashes": ["sha256:aaaa"],
                "index": "primary",
            }
        ]
        replay_same = [dict(recorded[0])]
        assert compare_lock_entries(recorded, replay_same).equal is True

        replay_diff = [
            {
                **recorded[0],
                "hashes": ["sha256:bbbb"],
                "index": "secondary",
            }
        ]
        comparison = compare_lock_entries(recorded, replay_diff)
        assert comparison.equal is False
        mismatch = comparison.mismatches[0]
        assert mismatch["name"] == "app"
        assert mismatch["recorded"]["hashes"] == ("sha256:aaaa",)
        assert mismatch["replay"]["index"] == "secondary"

    def test_detects_presence_differences(self):
        recorded = [{"name": "a", "version": "==1", "hashes": (), "index": None, "markers": None}]
        replay = [
            {"name": "a", "version": "==1", "hashes": (), "index": None, "markers": None},
            {"name": "b", "version": "==2", "hashes": (), "index": None, "markers": None},
        ]
        comparison = compare_lock_entries(recorded, replay)
        assert comparison.equal is False
        assert comparison.replay_only == ("b",)


# ---------------------------------------------------------------------------
# Replay transport unit behaviour
# ---------------------------------------------------------------------------


class TestReplayTransport:
    def test_replay_session_raises_replay_miss_for_unknown_url(self, tmp_path):
        pages, sidecars = build_world()
        live = FakeIndexSession(pages, sidecars)
        record_backend = _online_backend(live, tmp_path / "rec")
        snap_dir = tmp_path / "snapshot"
        record_backend.resolve(
            _request(
                {"app": "app"},
                sources=_sources(_INDEX_A),
                snapshot=SnapshotMode("record", str(snap_dir)),
            )
        )
        bundle = load_snapshot(snap_dir)
        from pipenv.resolver.snapshot import ReplaySession

        session = ReplaySession(bundle)
        with pytest.raises(ReplayMiss) as exc:
            session.request("GET", "https://evil.example/never-recorded/")
        assert "never-recorded" in str(exc.value)

    def test_network_blocked_session_always_refuses(self):
        from pipenv.resolver.snapshot import NetworkBlockedSession

        with pytest.raises(ReplayMiss) as exc:
            NetworkBlockedSession().request(
                "GET", "https://should-never-be-contacted.example/"
            )
        assert "network is disabled" in str(exc.value)

    def test_incomplete_exception_is_public(self):
        assert issubclass(IncompleteSnapshot, Exception)


def tempfile_gettempdir() -> str:
    import tempfile

    return tempfile.gettempdir()
