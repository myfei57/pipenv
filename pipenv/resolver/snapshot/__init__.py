"""Record/replay resolution snapshots for the pure-Python resolver.

The public surface splits into two halves:

* :mod:`pipenv.resolver.snapshot.bundle` — the on-disk, content-addressed
  snapshot format: commit manifest, digest verification, request
  compatibility checks and lock-entry comparison.
* :mod:`pipenv.resolver.snapshot.transport` — the recording wrapper and
  the strictly-offline replay transport plugged in at the resolver's
  only network boundary.

A snapshot captures the exact candidate world (simple-index manifests +
wheel PEP 658 metadata), the canonical request, the marker environment
and the selection-affecting configuration of one resolve, then replays
the same world with no network and no host-cache access.  sdist builds,
VCS checkouts and other un-captured dynamic sources mark the recording
incomplete and are refused by strict replay.
"""
from __future__ import annotations

from typing import Any

from pipenv.resolver.snapshot.bundle import (
    SNAPSHOT_FORMAT_VERSION,
    ExchangeRecord,
    IncompleteSnapshot,
    LockComparison,
    ReplayMiss,
    RequestMismatch,
    SnapshotBundle,
    SnapshotCorrupt,
    SnapshotError,
    algorithm_versions,
    check_request_compatible,
    collect_embedded_secrets,
    compare_algorithm_versions,
    compare_lock_entries,
    ensure_strictly_replayable,
    freeze_request,
    load_snapshot,
    scan_tree_for_secrets,
    sha256_hex,
    stripped_index_url,
    utc_now_iso,
    write_replay_report,
    write_snapshot,
)
from pipenv.resolver.snapshot.transport import (
    NetworkBlockedSession,
    RecordingSession,
    ReplayFetcher,
    ReplaySession,
    build_page_url,
    classify_exchange,
    enrich_manifest_identities,
    make_replay_metadata_fetcher,
)

#: Spec-prefix shapes the pure-Python resolver cannot capture (active
#: VCS checkouts and local/direct file sources).  A top-level spec
#: matching one of these makes the recording incomplete up front.
_UNRECORDABLE_SPEC_PREFIXES = (
    "git+",
    "hg+",
    "svn+",
    "bzr+",
    "file://",
)


def detect_unrecordable_top_level_specs(request: Any) -> tuple[str, ...]:
    """Return human-readable incompleteness reasons for top-level specs
    that point at VCS or direct file sources.

    The wheel-metadata snapshot scope cannot represent an active VCS
    checkout or a local-path build, so such a resolve is marked
    incomplete before any transport evidence is even considered.
    """
    reasons: list[str] = []
    for name, spec in sorted(request.packages.specs.items()):
        line = str(spec).strip().lower()
        first_token = line.split(None, 1)[0] if line else ""
        for prefix in _UNRECORDABLE_SPEC_PREFIXES:
            if first_token.startswith(prefix):
                reasons.append(
                    f"top-level spec for {name!r} uses un-recordable source "
                    f"{prefix!r}: {spec!r}"
                )
                break
    return tuple(reasons)


__all__ = [
    "SNAPSHOT_FORMAT_VERSION",
    "ExchangeRecord",
    "IncompleteSnapshot",
    "LockComparison",
    "NetworkBlockedSession",
    "RecordingSession",
    "ReplayFetcher",
    "ReplayMiss",
    "ReplaySession",
    "RequestMismatch",
    "SnapshotBundle",
    "SnapshotCorrupt",
    "SnapshotError",
    "algorithm_versions",
    "build_page_url",
    "check_request_compatible",
    "classify_exchange",
    "collect_embedded_secrets",
    "compare_algorithm_versions",
    "compare_lock_entries",
    "detect_unrecordable_top_level_specs",
    "enrich_manifest_identities",
    "ensure_strictly_replayable",
    "freeze_request",
    "load_snapshot",
    "make_replay_metadata_fetcher",
    "scan_tree_for_secrets",
    "sha256_hex",
    "stripped_index_url",
    "utc_now_iso",
    "write_replay_report",
    "write_snapshot",
]
