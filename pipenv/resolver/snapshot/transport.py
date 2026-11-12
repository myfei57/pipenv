"""Recording and replay HTTP transports for resolution snapshots.

The transport layer is the ONLY network boundary the pure-Python
resolver has:

* :class:`PEP691Client` issues simple-index GETs through
  ``session.request``;
* :func:`pipenv.resolver.pure_python_metadata.fetch_metadata` issues
  PEP 658 ``.metadata`` GETs (and, on indexes without sidecars, wheel
  HEAD/range requests) the same way.

Wrapping that one boundary gives faithful, byte-level recording without
touching the resolver algorithm and a strictly offline replay that
raises :class:`ReplayMiss` instead of ever reaching for the network or
the host's mutable manifest cache.

Roles
-----

Every exchange is classified at record time:

``simple-manifest``
    A PEP 691/503 package index page.
``wheel-metadata``
    A PEP 658 ``*.metadata`` sidecar — the only fully recordable
    metadata source.
``sdist-build`` / ``wheel-binary`` / ``wheel-range`` / ``vcs`` /
``other``
    Anything outside the recordable scope.  Observing one of these
    marks the bundle incomplete with an explicit reason; strict replay
    refuses the bundle.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from pipenv.resolver.auth import extract_url_credentials
from pipenv.resolver.snapshot.bundle import (
    ExchangeRecord,
    ReplayMiss,
    SnapshotBundle,
    SnapshotError,
    sanitized_request_headers,
    sanitized_response_headers,
    sha256_hex,
    stripped_index_url,
)
from pipenv.vendor.packaging.utils import canonicalize_name

#: Exchanges carrying one of these roles are outside the fully
#: recordable wheel-metadata scope; seeing one makes the bundle
#: incomplete.
_INCOMPLETE_ROLES = frozenset(
    {"sdist-build", "wheel-binary", "wheel-range", "vcs", "other"}
)

_SDIST_SUFFIXES = (".tar.gz", ".tar.bz2", ".zip", ".tgz")
_VCS_SCHEME_PREFIXES = ("git+", "hg+", "svn+", "bzr+")


# ---------------------------------------------------------------------------
# Role classification
# ---------------------------------------------------------------------------


def classify_exchange(method: str, url: str, headers: Mapping[str, Any] | None) -> str:
    """Classify one HTTP exchange into a snapshot role.

    The classification is deliberately conservative: anything not
    recognisable as a simple-index page or a ``.metadata`` sidecar is
    an un-captured dynamic source and makes the recording incomplete.
    """
    method = method.upper()
    lowered = url.lower()
    header_map = {
        str(k).lower(): str(v) for k, v in (headers or {}).items()
    }

    # PEP 658 sidecar — checked before suffix rules because the URL
    # ends in ``.whl.metadata`` and would otherwise look wheel-ish.
    if lowered.endswith(".metadata"):
        return "wheel-metadata"

    if any(lowered.startswith(prefix) for prefix in _VCS_SCHEME_PREFIXES):
        return "vcs"

    # Wheel-head fallback path (HEAD probe or ranged GET) means the
    # metadata cannot be replayed from a single immutable sidecar.
    if method == "HEAD" or "range" in header_map:
        return "wheel-range"

    if lowered.endswith(".whl"):
        return "wheel-binary"

    if any(lowered.endswith(suffix) for suffix in _SDIST_SUFFIXES):
        return "sdist-build"

    # Simple-index pages carry the PEP 691 Accept header (PEP691Client
    # sends it on every fetch).  Fall back to the PEP 503 URL shape
    # ``.../<canonical-name>/`` for non-conformant sessions.
    accept = header_map.get("accept", "")
    if "pypi.simple" in accept.lower():
        return "simple-manifest"
    path = urlsplit(url).path.rstrip("/")
    last_segment = path.rsplit("/", 1)[-1] if path else ""
    if last_segment and canonicalize_name(last_segment) == last_segment.replace("_", "-"):
        return "simple-manifest"

    return "other"


# ---------------------------------------------------------------------------
# Recording transport
# ---------------------------------------------------------------------------


class RecordingSession:
    """Session wrapper that records every exchange while delegating the
    actual HTTP work to the wrapped production session.

    Responses are returned untouched, so recording cannot change
    resolver behaviour.  Bodies are held in memory until
    :meth:`build_bundle` finalizes the content-addressed bundle.
    """

    def __init__(
        self,
        inner: Any,
        *,
        principal_namespace: str,
        source_by_url: Mapping[str, str] | None = None,
    ) -> None:
        self._inner = inner
        self._principal_namespace = str(principal_namespace)
        # Stripped source URL -> source name, for manifest attribution.
        self._source_by_url: dict[str, str] = dict(source_by_url or {})
        self._bodies: dict[str, bytes] = {}
        self._records: list[ExchangeRecord] = []
        self._incomplete_reasons: list[str] = []
        self._exchange_roles: dict[tuple[str, str], str] = {}

    # -- introspection used by the backend finalizer ------------------

    @property
    def incomplete_reasons(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._incomplete_reasons))

    @property
    def records(self) -> tuple[ExchangeRecord, ...]:
        return tuple(self._records)

    @property
    def bodies(self) -> dict[str, bytes]:
        return dict(self._bodies)

    # -- session interface --------------------------------------------

    def request(self, method: str, url: str, headers: Any = None, **kwargs: Any) -> Any:
        # Defence in depth: nothing credential-bearing may reach the
        # recorded surface.  The PEP 691 client strips userinfo before
        # requesting, so a hit here is a wiring bug worth failing on.
        stripped_url, creds = extract_url_credentials(url)
        if creds is not None:
            raise SnapshotError(
                "refusing to record a credential-bearing URL; strip "
                "userinfo before the request and use principal_namespace"
            )

        response = self._inner.request(method, url, headers=headers, **kwargs)
        self._record_response(method.upper(), stripped_url, headers or {}, response)
        return response

    # -- internals -----------------------------------------------------

    def _record_response(
        self,
        method: str,
        url: str,
        request_headers: Mapping[str, Any],
        response: Any,
    ) -> None:
        body = self._response_body(response)
        digest = sha256_hex(body)
        role = classify_exchange(method, url, request_headers)
        response_headers = sanitized_response_headers(
            getattr(response, "headers", None)
        )
        package = self._package_for(url, role)
        source_name = self._source_name_for(url, role)

        key = (method, url)
        previous_role = self._exchange_roles.get(key)
        if previous_role is not None:
            previous = next(r for r in self._records if r.key == key)
            if previous.body_digest != digest:
                # Non-deterministic response for the same request: the
                # candidate world is not replayable as-is.
                reason = (
                    f"non-deterministic response for {method} {url!r}; "
                    "digests differ between observations"
                )
                if reason not in self._incomplete_reasons:
                    self._incomplete_reasons.append(reason)
            return

        self._exchange_roles[key] = role
        self._bodies.setdefault(digest, body)
        record = ExchangeRecord(
            key=key,
            method=method,
            url=url,
            role=role,
            status=int(self._response_status(response)),
            content_type=response_headers.get("content-type"),
            etag=response_headers.get("etag"),
            request_headers=sanitized_request_headers(request_headers),
            response_headers=response_headers,
            body_digest=digest,
            body_size=len(body),
            principal_namespace=self._principal_namespace,
            source_name=source_name,
            package=package,
        )
        self._records.append(record)

        if role in _INCOMPLETE_ROLES:
            reason = f"{role} exchange is outside the recordable wheel-metadata scope: {url}"
            if reason not in self._incomplete_reasons:
                self._incomplete_reasons.append(reason)

    def _package_for(self, url: str, role: str) -> str | None:
        if role != "simple-manifest":
            return None
        path = urlsplit(url).path.rstrip("/")
        segment = path.rsplit("/", 1)[-1] if path else ""
        return canonicalize_name(segment) if segment else None

    def _source_name_for(self, url: str, role: str) -> str | None:
        if role != "simple-manifest":
            return None
        for source_url, name in self._source_by_url.items():
            prefix = stripped_index_url(source_url).rstrip("/") + "/"
            if url.startswith(prefix):
                return name
        return None

    @staticmethod
    def _response_status(response: Any) -> int:
        status = getattr(response, "status", None)
        if status is None:
            status = getattr(response, "status_code", 0)
        return int(status) if status is not None else 0

    @staticmethod
    def _response_body(response: Any) -> bytes:
        body = getattr(response, "data", None)
        if body is None:
            body = getattr(response, "content", b"")
        return bytes(body or b"")


# ---------------------------------------------------------------------------
# Replay transport
# ---------------------------------------------------------------------------


class _ReplayResponse:
    """Minimal urllib3/requests-duck-typed response served from bytes."""

    def __init__(self, *, status: int, body: bytes, headers: Mapping[str, str]):
        self.status = status
        self.status_code = status
        self.data = body
        self.content = body
        self.headers = dict(headers)

    def release_conn(self) -> None:  # pragma: no cover — parity with real responses
        return None


class ReplaySession:
    """Offline session: serves recorded exchanges only.

    Every served body is re-verified against its recorded sha256
    digest.  Any request absent from the bundle raises
    :class:`ReplayMiss` — there is no network fallback and no host
    cache fallback by construction.
    """

    def __init__(self, bundle: SnapshotBundle) -> None:
        self._bundle = bundle
        self.served: list[tuple[str, str]] = []

    def request(self, method: str, url: str, headers: Any = None, **kwargs: Any) -> Any:
        method = method.upper()
        stripped_url, _creds = extract_url_credentials(url)
        record = self._bundle.exchange_for(method, stripped_url)
        if record is None:
            raise ReplayMiss(method, stripped_url)
        body = self._bundle.bodies.get(record.body_digest)
        if body is None:  # pragma: no cover — load_snapshot already verifies
            raise ReplayMiss(
                method,
                stripped_url,
                detail=f"blob {record.body_digest!r} missing from verified bundle",
            )
        actual = sha256_hex(body)
        if actual != record.body_digest:
            from pipenv.resolver.snapshot.bundle import SnapshotCorrupt

            raise SnapshotCorrupt(
                f"digest mismatch while replaying {method} {stripped_url!r}: "
                f"recorded={record.body_digest!r} computed={actual!r}"
            )
        self.served.append((method, stripped_url))
        return _ReplayResponse(
            status=record.status,
            body=body,
            headers=record.response_headers,
        )


class NetworkBlockedSession:
    """Session whose every request is a hard :class:`ReplayMiss`.

    Used as a defensive default so an unexpected I/O path in replay
    fails loudly instead of opening a socket.
    """

    def request(self, method: str, url: str, headers: Any = None, **kwargs: Any) -> Any:
        raise ReplayMiss(str(method).upper(), url, "network is disabled in replay mode")


def build_page_url(index_url: str, package_name: str) -> str:
    """Reproduce :meth:`PEP691Client.fetch`'s target-URL construction.

    Kept in lock-step with the client so the replay fetcher looks up
    exactly the URL the recording client requested.
    """
    stripped_url, _creds = extract_url_credentials(index_url)
    canonical_name = canonicalize_name(package_name)
    return stripped_url.rstrip("/") + "/" + canonical_name + "/"


class ReplayFetcher:
    """Offline stand-in for :class:`ParallelFetcher`.

    Populates a caller-owned (fresh, non-host)
    :class:`ParsedManifestCache` exclusively from verified snapshot
    blobs, reusing the production PEP 691/503 parsers so the candidate
    tuples are identical to an online fetch.  A target without a
    recorded manifest exchange raises :class:`ReplayMiss`.
    """

    def __init__(self, bundle: SnapshotBundle, cache: Any, session: ReplaySession) -> None:
        self._bundle = bundle
        self._cache = cache
        self._session = session

    def populate(self, targets: Sequence[tuple[str, str]]) -> dict:
        # Long TTL: replay cache entries must never expire mid-resolve
        # — wall-clock time is not an input to candidate selection and
        # hiding an entry via TTL would manufacture a spurious ReplayMiss.
        ttl_seconds = 100 * 365 * 24 * 3600
        for index_url, package_name in targets:
            page_url = build_page_url(index_url, package_name)
            record = self._bundle.exchange_for("GET", page_url)
            if record is None:
                raise ReplayMiss(
                    "GET",
                    page_url,
                    detail="simple-index manifest was not recorded",
                )
            if record.role != "simple-manifest":
                raise ReplayMiss(
                    "GET",
                    page_url,
                    detail=f"recorded exchange has role {record.role!r}",
                )
            response = self._session.request("GET", page_url)
            candidates = self._parse_manifest(
                response.data, page_url, response.headers.get("content-type", "")
            )
            self._cache.put(
                index_url,
                package_name,
                candidates,
                etag=record.etag,
                ttl_seconds=ttl_seconds,
            )
        return {}

    @staticmethod
    def _parse_manifest(body: bytes, page_url: str, content_type: str) -> tuple:
        from pipenv.resolver.pep691 import _parse_pep503_html, _parse_pep691_json

        ct_lower = (content_type or "").lower()
        if ct_lower.startswith("application/vnd.pypi.simple.v1+json"):
            return _parse_pep691_json(body, page_url)
        if ct_lower.startswith(
            ("application/vnd.pypi.simple.v1+html", "text/html")
        ):
            return _parse_pep503_html(body, page_url)
        # No usable content type on the recorded exchange — sniff JSON,
        # matching nothing else about the online client's behaviour.
        try:
            return _parse_pep691_json(body, page_url)
        except ValueError:
            return _parse_pep503_html(body, page_url)


def make_replay_metadata_fetcher(bundle: SnapshotBundle, session: ReplaySession):
    """Build the one-argument ``metadata_fetcher`` the provider expects.

    Wheel metadata is served solely from recorded PEP 658 sidecar
    bodies and re-parsed with the production parser.  Sdist candidates
    have no recordable sidecar and raise :class:`ReplayMiss` (a strict
    replay would already have refused the incomplete bundle).
    """
    from pipenv.resolver.pure_python_metadata import (
        CoreMetadata,
        _parse_metadata_text,
    )

    def fetch(candidate: Any) -> CoreMetadata:
        if not getattr(candidate, "is_wheel", True):
            raise ReplayMiss(
                "GET",
                str(getattr(candidate, "url", "<unknown>")),
                detail="sdist metadata cannot be replayed (build-time source)",
            )
        parts = urlsplit(candidate.url)
        metadata_url = parts._replace(path=parts.path + ".metadata").geturl()
        response = session.request("GET", metadata_url)
        try:
            text = response.data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SnapshotError(
                f"recorded metadata at {metadata_url!r} is not UTF-8: {exc}"
            ) from exc
        return _parse_metadata_text(text)

    return fetch


# ---------------------------------------------------------------------------
# Bundle finalization
# ---------------------------------------------------------------------------


def _candidate_identity(candidate: Any) -> dict[str, Any]:
    return {
        "filename": candidate.filename,
        "url": candidate.url,
        "version": candidate.version,
        "is_wheel": bool(candidate.is_wheel),
        "requires_python": candidate.requires_python,
        "yanked": bool(candidate.yanked),
        "hashes": sorted([h.algo, h.value] for h in candidate.hashes),
    }


def enrich_manifest_identities(
    recorder: RecordingSession,
    fetcher_cache: Any,
    sources: Sequence[Any],
) -> tuple[ExchangeRecord, ...]:
    """Attach parsed candidate-file identities to every recorded
    simple-manifest exchange.

    The authoritative blob stays the raw response body; the identity
    list makes source logical identity, canonical package name and
    per-file hashes auditable from ``snapshot.json`` without
    re-parsing anything.

    The manifest cache is keyed by the *original* source URL (which may
    carry a trailing-slash or userinfo variant) while exchanges store
    the stripped URL, so the request's source list is used to map a
    recorded page back to its cache key.
    """
    # Pre-compute (prefix, original_url) once: the prefix is what the
    # PEP 691 client builds page URLs from.
    source_prefixes: list[tuple[str, str]] = []
    for source in sources:
        stripped, _creds = extract_url_credentials(source.url)
        source_prefixes.append((stripped.rstrip("/") + "/", source.url))

    enriched: list[ExchangeRecord] = []
    for record in recorder.records:
        if record.role != "simple-manifest" or record.package is None:
            enriched.append(record)
            continue
        identities: tuple[dict[str, Any], ...] = ()
        # The production cache holds the parsed candidate tuple for
        # every manifest the resolve consumed; read it back rather than
        # re-deriving parser rules here.
        for prefix, original_url in source_prefixes:
            if not record.url.startswith(prefix):
                continue
            manifest = fetcher_cache.get(original_url, record.package)
            if manifest is None:
                continue
            identities = tuple(
                _candidate_identity(c) for c in manifest.candidates
            )
            break
        enriched.append(
            ExchangeRecord(
                key=record.key,
                method=record.method,
                url=record.url,
                role=record.role,
                status=record.status,
                content_type=record.content_type,
                etag=record.etag,
                request_headers=record.request_headers,
                response_headers=record.response_headers,
                body_digest=record.body_digest,
                body_size=record.body_size,
                principal_namespace=record.principal_namespace,
                source_name=record.source_name,
                package=record.package,
                candidate_identities=identities,
            )
        )
    return tuple(enriched)


__all__ = [
    "NetworkBlockedSession",
    "RecordingSession",
    "ReplayFetcher",
    "ReplaySession",
    "build_page_url",
    "classify_exchange",
    "enrich_manifest_identities",
    "make_replay_metadata_fetcher",
]
