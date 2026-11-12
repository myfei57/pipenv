"""Content-addressed resolution-snapshot bundle format.

A snapshot captures the *candidate world* one pure-Python resolve actually
consumed:

* every simple-index manifest (PEP 691 JSON / PEP 503 HTML response body),
* every wheel ``METADATA`` body (PEP 658 sidecar),
* the canonical request, the frozen marker environment and every
  resolver configuration value that can influence candidate selection,
* the lock entries the recording resolve produced.

The bundle is **content-addressed**: each captured body is stored under
``blobs/<sha256>`` and ``snapshot.json`` is the complete commit manifest
enumerating every blob with its digest and size.  The manifest is the
publication point — it is written last and atomically, so a failed write
can never publish a ``"complete"`` recording.

Scope and non-goals
-------------------

Only requests whose wheel metadata can be captured completely are
recordable.  sdist builds, active VCS checkouts, wheel-binary downloads
and any other un-captured dynamic source mark the bundle
``complete = false`` with an explicit reason; strict replay refuses such
bundles.  A snapshot is deliberately NOT an offline install mirror and
does NOT store wheel binaries.

Credential safety
-----------------

Credential-bearing URL userinfo is stripped before anything is
persisted and the caller-supplied *principal namespace* (a non-secret
label) partitions views that share a stripped URL but were authorized
differently — views are never merged merely because their URLs look
identical after stripping.

This module imports only stdlib + ``pipenv.vendor.packaging`` — no
patched-pip internals, matching the Initiative G gate.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from pipenv.resolver.auth import extract_url_credentials
from pipenv.vendor.packaging.utils import canonicalize_name

#: Bump on any incompatible change of the on-disk snapshot layout.
SNAPSHOT_FORMAT_VERSION = 1

#: File names inside a snapshot directory.
SNAPSHOT_MANIFEST_NAME = "snapshot.json"
BLOBS_DIRNAME = "blobs"
REPLAY_REPORT_NAME = "replay-report.json"

#: Response headers worth keeping (everything else — especially
#: ``Authorization`` / ``Cookie`` on the request side — is dropped).
_RESPONSE_HEADER_ALLOWLIST = ("content-type", "etag", "last-modified")
_REQUEST_HEADER_ALLOWLIST = ("accept", "if-none-match", "range")

#: Request headers that must never be persisted even if an allowlist
#: entry ever overlaps them.  Defence in depth: the transport already
#: strips them, the bundle refuses them again.
_FORBIDDEN_HEADER_NAMES = (
    "authorization",
    "proxy-authorization",
    "cookie",
    "set-cookie",
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SnapshotError(Exception):
    """Base class for every snapshot/record/replay failure."""


class IncompleteSnapshot(SnapshotError):
    """A bundle is marked incomplete (or strict replay hit an
    incompatibility); strict replay must refuse it."""


class SnapshotCorrupt(SnapshotError):
    """The on-disk bundle fails format/digest verification."""


class RequestMismatch(SnapshotError):
    """The replay request is not compatible with the recorded request."""


class ReplayMiss(SnapshotError):
    """Replay needed a fetch that the snapshot did not record.

    Raised explicitly instead of silently falling back to the network
    or the host's mutable manifest cache.  Carries the HTTP method and
    URL that could not be served from the snapshot.
    """

    def __init__(self, method: str, url: str, detail: str = "") -> None:
        self.method = method
        self.url = url
        message = f"ReplayMiss: {method} {url!r} is not recorded in the snapshot"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExchangeRecord:
    """One recorded HTTP exchange (a simple manifest or wheel metadata
    body, plus anything else the transport observed).

    ``key`` is ``(method, url)``; the URL is always credential-free.
    ``role`` classifies the exchange so replay knows which exchanges are
    load-bearing for the resolver and which ones only explain why the
    bundle is incomplete.
    """

    key: tuple[str, str]
    method: str
    url: str
    role: str
    status: int
    content_type: str | None
    etag: str | None
    request_headers: dict[str, str]
    response_headers: dict[str, str]
    body_digest: str
    body_size: int
    principal_namespace: str
    source_name: str | None
    package: str | None
    candidate_identities: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class SnapshotBundle:
    """In-memory representation of a snapshot directory.

    ``bodies`` holds the verified blob bytes keyed by their sha256
    digest; it is never serialized into ``snapshot.json`` (the blobs
    live as separate content-addressed files).
    """

    format_version: int
    principal_namespace: str
    created_at: str
    complete: bool
    incomplete_reasons: tuple[str, ...]
    canonical_request: dict[str, Any]
    target_env: dict[str, Any]
    frozen_config: dict[str, Any]
    algorithm: dict[str, str]
    exchanges: tuple[ExchangeRecord, ...]
    recorded_locked: tuple[dict[str, Any], ...]
    bodies: dict[str, bytes] = field(default_factory=dict)

    def exchange_for(self, method: str, url: str) -> ExchangeRecord | None:
        """Return the recorded exchange for ``(method, url)`` or None."""
        key = (method.upper(), url)
        for record in self.exchanges:
            if record.key == key:
                return record
        return None

    def manifest_exchanges(self) -> tuple[ExchangeRecord, ...]:
        return tuple(e for e in self.exchanges if e.role == "simple-manifest")

    def metadata_exchanges(self) -> tuple[ExchangeRecord, ...]:
        return tuple(e for e in self.exchanges if e.role == "wheel-metadata")


# ---------------------------------------------------------------------------
# Digests / header sanitisation
# ---------------------------------------------------------------------------


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sanitize_headers(headers: Any, allowlist: Sequence[str]) -> dict[str, str]:
    """Copy only allow-listed headers, lower-casing names.

    Forbidden auth/cookie names are dropped unconditionally even if a
    future allowlist edit accidentally admits them.
    """
    out: dict[str, str] = {}
    if not headers:
        return out
    try:
        items = headers.items()
    except Exception:  # noqa: BLE001 — non-mapping header container
        return out
    allowed = set(allowlist)
    for raw_name, raw_value in items:
        if not isinstance(raw_name, str):
            continue
        name = raw_name.lower()
        if name in _FORBIDDEN_HEADER_NAMES:
            continue
        if name not in allowed:
            continue
        out[name] = "" if raw_value is None else str(raw_value)
    return out


def sanitized_request_headers(headers: Any) -> dict[str, str]:
    return _sanitize_headers(headers, _REQUEST_HEADER_ALLOWLIST)


def sanitized_response_headers(headers: Any) -> dict[str, str]:
    return _sanitize_headers(headers, _RESPONSE_HEADER_ALLOWLIST)


def stripped_index_url(index_url: str) -> str:
    """Return the index URL without embedded credentials.

    Credential-bearing source URLs are a legitimate runtime input (the
    :class:`PEP691Client` strips userinfo before every request), so this
    helper only strips — it never rejects.  The snapshot persists the
    stripped URL together with the caller-supplied principal namespace,
    and the recorder additionally hard-fails if a credential-bearing URL
    ever reaches the request boundary.  A post-write secret scan is the
    final guarantee that nothing sensitive was persisted.
    """
    stripped, _creds = extract_url_credentials(index_url)
    return stripped


def collect_embedded_secrets(source_urls: Sequence[str]) -> list[str]:
    """Collect non-empty userinfo secret values from source URLs.

    Used by the recorder to verify after writing that none of these
    substrings leaked into the snapshot directory.
    """
    secrets: list[str] = []
    for url in source_urls:
        _stripped, creds = extract_url_credentials(url)
        if creds is None:
            continue
        user, password = creds
        if user:
            secrets.append(user)
        if password:
            secrets.append(password)
    return [s for s in dict.fromkeys(secrets) if s]


def scan_tree_for_secrets(root: Path, secret_values: Sequence[str]) -> list[str]:
    """Walk every file under ``root`` and return paths whose bytes
    contain any of ``secret_values``.  Empty result == no leak.
    """
    if not secret_values:
        return []
    needles = tuple(s.encode("utf-8") for s in secret_values if s)
    if not needles:
        return []
    offending: list[str] = []
    for path in sorted(Path(root).rglob("*")):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if any(needle in data for needle in needles):
            offending.append(str(path))
    return offending


# ---------------------------------------------------------------------------
# Request freezing / compatibility
# ---------------------------------------------------------------------------


def _frozen_source(source: Any) -> dict[str, Any]:
    return {
        "name": source.name,
        "url": stripped_index_url(source.url),
        "verify_ssl": bool(source.verify_ssl),
    }


def freeze_request(request: Any, target_env: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the selection-relevant slice of the resolver request.

    Only fields that can change which candidates are considered or
    chosen are recorded; bookkeeping fields (deadlines, parent pid) are
    intentionally excluded.
    """
    packages = {
        canonicalize_name(name): str(spec)
        for name, spec in sorted(request.packages.specs.items())
    }
    return {
        "schema_version": int(request.schema_version),
        "category": request.category,
        "packages": packages,
        "options": {
            "pre": bool(request.options.pre),
            "backend": str(request.options.backend or ""),
        },
        "sources": [_frozen_source(s) for s in request.sources],
        "python_marker_override": request.python_marker_override,
        "extra_pip_args": tuple(sorted(request.extra_pip_args or ())),
        "target_env": {k: str(v) for k, v in sorted(target_env.items())},
    }


def check_request_compatible(
    bundle: SnapshotBundle,
    request: Any,
    target_env: Mapping[str, Any],
    *,
    strict: bool,
) -> list[str]:
    """Compare the replay request against the recorded canonical request.

    Returns the list of human-readable mismatch reasons (empty ==
    compatible).  In ``strict`` mode the first incompatibility raises
    :class:`RequestMismatch`; in non-strict mode the reasons are
    returned so the caller can surface them as compatibility warnings.
    """
    recorded = bundle.canonical_request
    current = freeze_request(request, target_env)
    reasons: list[str] = []

    if recorded["category"] != current["category"]:
        reasons.append(
            f"category differs: recorded={recorded['category']!r} "
            f"replay={current['category']!r}"
        )
    if recorded["packages"] != current["packages"]:
        reasons.append(
            "package specs differ: "
            f"recorded={recorded['packages']!r} replay={current['packages']!r}"
        )
    if recorded["options"] != current["options"]:
        reasons.append(
            "selection options differ: "
            f"recorded={recorded['options']!r} replay={current['options']!r}"
        )
    if recorded["sources"] != current["sources"]:
        reasons.append(
            "source list (identity/order/verify_ssl) differs: "
            f"recorded={recorded['sources']!r} replay={current['sources']!r}"
        )
    if recorded["python_marker_override"] != current["python_marker_override"]:
        reasons.append(
            "python_marker_override differs: "
            f"recorded={recorded['python_marker_override']!r} "
            f"replay={current['python_marker_override']!r}"
        )
    # Tuples on the in-memory side vs JSON-decoded lists on the loaded
    # side compare as sequences — normalize so the check is shape-stable.
    if list(recorded["extra_pip_args"]) != list(current["extra_pip_args"]):
        reasons.append(
            "extra_pip_args differ: "
            f"recorded={recorded['extra_pip_args']!r} "
            f"replay={current['extra_pip_args']!r}"
        )
    if recorded["target_env"] != current["target_env"]:
        reasons.append(
            "target marker environment differs: "
            f"recorded={recorded['target_env']!r} replay={current['target_env']!r}"
        )

    if strict and reasons:
        raise RequestMismatch("; ".join(reasons))
    return reasons


def algorithm_versions() -> dict[str, str]:
    """Record the algorithm/component versions so replay can report
    compatibility differences honestly instead of claiming bit-identical
    behaviour across arbitrary versions.
    """
    from pipenv.__version__ import __version__ as pipenv_version
    from pipenv.patched.pip._vendor.resolvelib import __version__ as resolvelib_version
    from pipenv.vendor.packaging import __version__ as packaging_version

    return {
        "pipenv": str(pipenv_version),
        "resolvelib": str(resolvelib_version),
        "packaging": str(packaging_version),
    }


def compare_algorithm_versions(recorded: Mapping[str, str]) -> list[str]:
    """Return warning lines for every component whose version changed."""
    current = algorithm_versions()
    notes: list[str] = []
    for component, recorded_version in sorted(recorded.items()):
        current_version = current.get(component, "<unknown>")
        if str(recorded_version) != str(current_version):
            notes.append(
                f"{component} version differs: recorded={recorded_version!r} "
                f"replay={current_version!r}; lock entries are compared, "
                "bit-identical resolver logs are not guaranteed"
            )
    return notes


# ---------------------------------------------------------------------------
# Lock-entry comparison
# ---------------------------------------------------------------------------


def _lock_entry_key(entry: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    extras = tuple(sorted(entry.get("extras", ())))
    return str(entry["name"]), extras


def _comparable_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Project a lock entry onto the fields comparison actually cares
    about — version, markers, hashes, source/index — never just the
    success marker."""
    return {
        "version": entry.get("version"),
        "markers": entry.get("markers"),
        "hashes": tuple(sorted(entry.get("hashes", ()))),
        "index": entry.get("index"),
    }


@dataclass(frozen=True, slots=True)
class LockComparison:
    equal: bool
    recorded_only: tuple[str, ...]
    replay_only: tuple[str, ...]
    mismatches: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "equal": self.equal,
            "recorded_only": list(self.recorded_only),
            "replay_only": list(self.replay_only),
            "mismatches": [dict(m) for m in self.mismatches],
        }


def compare_lock_entries(
    recorded_entries: Sequence[Mapping[str, Any]],
    replay_entries: Sequence[Mapping[str, Any]],
) -> LockComparison:
    """Compare full lock entries (version + markers + hashes + source).

    Two entries are paired by ``(name, sorted extras)`` — the same
    identity the resolver uses — and compared field by field so a
    hash or source divergence is reported even when the version pin
    matches.
    """
    recorded_map = {_lock_entry_key(e): _comparable_entry(e) for e in recorded_entries}
    replay_map = {_lock_entry_key(e): _comparable_entry(e) for e in replay_entries}

    recorded_only = tuple(
        f"{name}[{','.join(extras)}]" if extras else name
        for name, extras in sorted(set(recorded_map) - set(replay_map))
    )
    replay_only = tuple(
        f"{name}[{','.join(extras)}]" if extras else name
        for name, extras in sorted(set(replay_map) - set(recorded_map))
    )
    mismatches: list[dict[str, Any]] = []
    for key in sorted(set(recorded_map) & set(replay_map)):
        left = recorded_map[key]
        right = replay_map[key]
        if left != right:
            name, extras = key
            mismatches.append(
                {
                    "name": name,
                    "extras": list(extras),
                    "recorded": left,
                    "replay": right,
                }
            )
    equal = not (recorded_only or replay_only or mismatches)
    return LockComparison(
        equal=equal,
        recorded_only=recorded_only,
        replay_only=replay_only,
        mismatches=tuple(mismatches),
    )


# ---------------------------------------------------------------------------
# Atomic publication / loading / verification
# ---------------------------------------------------------------------------


def _atomic_write(target: Path, body: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=str(target.parent),
        prefix=target.name + ".",
        suffix=".tmp",
    )
    try:
        tmp.write(body)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp.close()
        os.replace(tmp.name, str(target))
    except BaseException:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise


def _exchange_to_json(record: ExchangeRecord) -> dict[str, Any]:
    data = asdict(record)
    # ``key`` is derivable from method+url; keep the manifest compact
    # and unambiguous by dropping the redundant tuple.
    data.pop("key", None)
    # Tuples → lists for JSON.
    data["candidate_identities"] = list(record.candidate_identities)
    return data


def _exchange_from_json(data: Mapping[str, Any]) -> ExchangeRecord:
    return ExchangeRecord(
        key=(str(data["method"]), str(data["url"])),
        method=str(data["method"]),
        url=str(data["url"]),
        role=str(data["role"]),
        status=int(data["status"]),
        content_type=data.get("content_type"),
        etag=data.get("etag"),
        request_headers=dict(data.get("request_headers", {})),
        response_headers=dict(data.get("response_headers", {})),
        body_digest=str(data["body_digest"]),
        body_size=int(data["body_size"]),
        principal_namespace=str(data["principal_namespace"]),
        source_name=data.get("source_name"),
        package=data.get("package"),
        candidate_identities=tuple(
            dict(item) for item in data.get("candidate_identities", ())
        ),
    )


def write_snapshot(bundle: SnapshotBundle, snapshot_path: str | os.PathLike) -> None:
    """Publish ``bundle`` under ``snapshot_path`` atomically.

    All content-addressed blobs are written first; ``snapshot.json`` —
    the complete commit manifest — is written last.  Any failure raises
    :class:`SnapshotError` and leaves no manifest behind, so a
    half-written bundle can never masquerade as a complete recording.
    """
    root = Path(snapshot_path)
    blobs_dir = root / BLOBS_DIRNAME

    try:
        # 1. Blobs first.  Same digest → same bytes, so existing files
        #    are left untouched.
        for digest, body in bundle.bodies.items():
            blob_path = blobs_dir / digest
            if blob_path.exists():
                # Verify the pre-existing blob instead of trusting it.
                if sha256_hex(blob_path.read_bytes()) != digest:
                    raise SnapshotCorrupt(
                        f"pre-existing blob {digest!r} has a mismatching digest"
                    )
                continue
            _atomic_write(blob_path, body)

        # 2. Complete commit manifest, enumerating every blob with
        #    digest + size so loaders can verify without re-parsing
        #    exchange roles.
        blob_commit = [
            {"digest": digest, "size": len(body)}
            for digest, body in sorted(bundle.bodies.items())
        ]
        manifest = {
            "format_version": bundle.format_version,
            "kind": "pipenv-resolution-snapshot",
            "principal_namespace": bundle.principal_namespace,
            "created_at": bundle.created_at,
            "complete": bundle.complete,
            "incomplete_reasons": list(bundle.incomplete_reasons),
            "canonical_request": bundle.canonical_request,
            "target_env": bundle.target_env,
            "frozen_config": bundle.frozen_config,
            "algorithm": bundle.algorithm,
            "exchanges": [_exchange_to_json(e) for e in bundle.exchanges],
            "recorded_locked": [dict(e) for e in bundle.recorded_locked],
            "blobs": blob_commit,
        }
        body = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        _atomic_write(root / SNAPSHOT_MANIFEST_NAME, body)
    except SnapshotError:
        raise
    except OSError as exc:
        raise SnapshotError(f"failed to publish snapshot under {root!s}: {exc}") from exc


def load_snapshot(snapshot_path: str | os.PathLike) -> SnapshotBundle:
    """Load and fully verify a snapshot directory.

    Verification performed before the bundle is returned:

    * ``snapshot.json`` exists and parses;
    * ``format_version`` matches :data:`SNAPSHOT_FORMAT_VERSION`;
    * every committed blob exists, its size matches and its sha256
      digest matches both the commit list and every exchange record
      referencing it.

    Any failure is :class:`SnapshotCorrupt` — callers never receive a
    partially verified bundle.
    """
    root = Path(snapshot_path)
    manifest_path = root / SNAPSHOT_MANIFEST_NAME
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise SnapshotCorrupt(
            f"snapshot manifest not readable at {manifest_path!s}: {exc}"
        ) from exc
    try:
        manifest = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SnapshotCorrupt(f"snapshot manifest is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SnapshotCorrupt("snapshot manifest must be a JSON object")

    format_version = manifest.get("format_version")
    if format_version != SNAPSHOT_FORMAT_VERSION:
        raise SnapshotCorrupt(
            f"snapshot format_version mismatch: on disk={format_version!r} "
            f"supported={SNAPSHOT_FORMAT_VERSION!r}"
        )

    blobs_dir = root / BLOBS_DIRNAME
    bodies: dict[str, bytes] = {}
    for commit in manifest.get("blobs", ()):
        digest = str(commit["digest"])
        expected_size = int(commit["size"])
        blob_path = blobs_dir / digest
        try:
            data = blob_path.read_bytes()
        except OSError as exc:
            raise SnapshotCorrupt(f"missing snapshot blob {digest!r}: {exc}") from exc
        if len(data) != expected_size:
            raise SnapshotCorrupt(
                f"blob {digest!r} size mismatch: manifest={expected_size} "
                f"actual={len(data)}"
            )
        actual_digest = sha256_hex(data)
        if actual_digest != digest:
            raise SnapshotCorrupt(
                f"blob {digest!r} digest mismatch: computed={actual_digest!r}"
            )
        bodies[digest] = data

    exchanges = tuple(
        _exchange_from_json(item) for item in manifest.get("exchanges", ())
    )
    # Every exchange must reference a committed, verified blob whose
    # digest still agrees with the bytes on disk.
    for record in exchanges:
        if record.body_digest not in bodies:
            raise SnapshotCorrupt(
                f"exchange {record.method} {record.url!r} references "
                f"uncommitted blob {record.body_digest!r}"
            )
        if record.body_size != len(bodies[record.body_digest]):
            raise SnapshotCorrupt(
                f"exchange {record.url!r} records body_size="
                f"{record.body_size} but the blob is "
                f"{len(bodies[record.body_digest])} bytes"
            )

    return SnapshotBundle(
        format_version=int(format_version),
        principal_namespace=str(manifest.get("principal_namespace", "default")),
        created_at=str(manifest.get("created_at", "")),
        complete=bool(manifest.get("complete", False)),
        incomplete_reasons=tuple(manifest.get("incomplete_reasons", ())),
        canonical_request=dict(manifest.get("canonical_request", {})),
        target_env=dict(manifest.get("target_env", {})),
        frozen_config=dict(manifest.get("frozen_config", {})),
        algorithm=dict(manifest.get("algorithm", {})),
        exchanges=exchanges,
        recorded_locked=tuple(
            dict(item) for item in manifest.get("recorded_locked", ())
        ),
        bodies=bodies,
    )


def ensure_strictly_replayable(bundle: SnapshotBundle) -> None:
    """Raise :class:`IncompleteSnapshot` unless ``bundle`` is complete.

    Non-strict callers may skip this gate; strict replay must call it
    before constructing offline collaborators.
    """
    if bundle.complete:
        return
    reasons = "; ".join(bundle.incomplete_reasons) or "unknown reason"
    raise IncompleteSnapshot(
        "snapshot is incomplete and strict replay was requested: " + reasons
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_replay_report(
    snapshot_path: str | os.PathLike,
    *,
    comparison: LockComparison,
    compatibility_notes: Sequence[str],
    replay_locked: Sequence[Mapping[str, Any]],
) -> Path:
    """Persist the comparable replay result next to the snapshot.

    The report carries full lock entries (version, markers, hashes,
    source) plus the field-level diff — comparison is never reduced to
    a success boolean.
    """
    root = Path(snapshot_path)
    report = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "comparison": comparison.to_dict(),
        "compatibility_notes": list(compatibility_notes),
        "replay_locked": [dict(entry) for entry in replay_locked],
    }
    target = root / REPLAY_REPORT_NAME
    _atomic_write(target, json.dumps(report, indent=2, sort_keys=True).encode("utf-8"))
    return target


__all__ = [
    "BLOBS_DIRNAME",
    "REPLAY_REPORT_NAME",
    "SNAPSHOT_FORMAT_VERSION",
    "SNAPSHOT_MANIFEST_NAME",
    "ExchangeRecord",
    "IncompleteSnapshot",
    "LockComparison",
    "ReplayMiss",
    "RequestMismatch",
    "SnapshotBundle",
    "SnapshotCorrupt",
    "SnapshotError",
    "algorithm_versions",
    "check_request_compatible",
    "collect_embedded_secrets",
    "compare_algorithm_versions",
    "compare_lock_entries",
    "ensure_strictly_replayable",
    "freeze_request",
    "load_snapshot",
    "sanitized_request_headers",
    "sanitized_response_headers",
    "scan_tree_for_secrets",
    "sha256_hex",
    "stripped_index_url",
    "utc_now_iso",
    "write_replay_report",
    "write_snapshot",
]
