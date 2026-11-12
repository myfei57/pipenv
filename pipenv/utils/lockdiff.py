"""Target-aware comparison of two ``Pipfile.lock`` or ``pylock.toml`` files.

The diff is driven by *explicit* installation targets: each target supplies the
marker variables, dependency groups and extras of one concrete installation
matrix.  Both lock formats are normalized into one comparison model that keeps
every package entry (a ``pylock.toml`` package may legitimately appear more
than once, once per environment) together with its origin, artifacts and
visibility conditions.

Design constraints honored here:

* No network access and no virtualenv inspection.  Markers are evaluated
  against the target's variables only; a variable the target does not provide
  makes the entry *unknown* rather than silently falling back to the host.
* Source files are read directly and never rewritten or "normalized" back to
  disk.
* Key ordering (JSON/TOML) and marker spellings that are equivalent on the
  requested targets produce no noise; equivalence is never claimed outside the
  finite target matrix supplied by the caller.
* URL credentials participate in identity comparison (so two different
  credential configurations never compare equal) but are redacted from every
  emitted report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from pipenv.patched.pip._internal.utils.misc import redact_auth_from_url
from pipenv.patched.pip._vendor.packaging.markers import (
    InvalidMarker,
    Marker,
    UndefinedComparison,
    UndefinedEnvironmentName,
    _evaluate_markers,
)
from pipenv.patched.pip._vendor.packaging.specifiers import InvalidSpecifier, SpecifierSet
from pipenv.patched.pip._vendor.packaging.utils import canonicalize_name

SCHEMA_VERSION = "1.0"

#: Overall / per-target outcomes.
STATUS_UNCHANGED = "unchanged"
STATUS_CHANGED = "changed"
STATUS_INCOMPLETE = "incomplete"

#: Per-name selection outcomes for one target.
OUTCOME_ACTIVE = "active"
OUTCOME_INACTIVE = "inactive"
OUTCOME_UNKNOWN = "unknown"
OUTCOME_AMBIGUOUS = "ambiguous"

#: Fixed, documentable ordering for difference categories.
CATEGORY_ORDER = ("added", "changed", "removed")

#: Marker variables defined by PEP 508.  ``extras`` and ``dependency_groups``
#: are lock-file-only variables supplied from the target's group/extras sets.
_KNOWN_MARKER_VARS = frozenset(
    {
        "implementation_name",
        "implementation_version",
        "os_name",
        "platform_machine",
        "platform_release",
        "platform_system",
        "platform_version",
        "platform_python_implementation",
        "python_full_version",
        "python_version",
        "sys_platform",
    }
)

#: Formats that can express per-file artifact and attestation metadata.
_FORMAT_HAS_ARTIFACTS = frozenset({"pylock-toml"})


class LockDiffError(Exception):
    """Raised when the diff cannot be performed at all (bad input)."""


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Target:
    """One explicit installation matrix entry."""

    name: str
    markers: Mapping[str, str]
    groups: frozenset[str]
    extras: frozenset[str]
    raw_groups: Tuple[str, ...]
    raw_extras: Tuple[str, ...]

    def evaluation_environment(self) -> Dict[str, Any]:
        """Build the restricted marker environment for this target.

        Only the explicitly supplied marker variables plus the group/extras
        sets are present; the host environment is never consulted.
        """
        environment: Dict[str, Any] = {
            key: str(value) for key, value in self.markers.items()
        }
        environment["extras"] = self.extras
        environment["dependency_groups"] = self.groups
        return environment


def _load_json_document(text: str, source: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LockDiffError(f"Could not parse JSON from {source}: {exc}") from exc


def _coerce_target_objects(document: Any, source: str) -> List[Dict[str, Any]]:
    objects = document if isinstance(document, list) else [document]
    for obj in objects:
        if not isinstance(obj, dict):
            raise LockDiffError(f"Each target in {source} must be a JSON object")
    return objects


def _read_target_document(value: str) -> List[Dict[str, Any]]:
    if value.startswith("@"):
        path = Path(value[1:].strip())
        if not path.is_file():
            raise LockDiffError(f"Targets file not found: {path}")
        text = path.read_text(encoding="utf-8-sig")
        return _coerce_target_objects(_load_json_document(text, str(path)), str(path))
    return _coerce_target_objects(
        _load_json_document(value, "<inline targets>"), "<inline>"
    )


def parse_target_specs(
    specs: Optional[Sequence[str]] = None,
    targets_file: Optional[Union[str, Path]] = None,
) -> List[Target]:
    """Parse ``--targets`` values and/or a ``--targets-file`` document.

    A target object accepts ``name``, ``markers`` (object), ``groups``
    (array, default ``["default"]``) and ``extras`` (array, default ``[]``).
    Inline JSON or ``@path`` references are both supported; documents may
    contain one object or an array of objects.
    """
    raw_objects: List[Dict[str, Any]] = []
    for spec in specs or []:
        if not isinstance(spec, str) or not spec.strip():
            raise LockDiffError("Invalid --targets value")
        raw_objects.extend(_read_target_document(spec))
    if targets_file:
        path = Path(str(targets_file))
        if not path.is_file():
            raise LockDiffError(f"Targets file not found: {path}")
        raw_objects.extend(
            _coerce_target_objects(
                _load_json_document(path.read_text(encoding="utf-8-sig"), str(path)),
                str(path),
            )
        )
    if not raw_objects:
        raise LockDiffError(
            "At least one target is required (use --targets or --targets-file)"
        )

    targets: List[Target] = []
    names: Dict[str, int] = {}
    for index, obj in enumerate(raw_objects, start=1):
        name = obj.get("name") or f"target-{index}"
        if not isinstance(name, str) or not name:
            raise LockDiffError(f"Target {index} has an invalid name")
        if name in names:
            raise LockDiffError(f"Duplicate target name: {name!r}")
        names[name] = index

        markers = obj.get("markers") or {}
        if not isinstance(markers, dict):
            raise LockDiffError(f"Target {name!r}: markers must be a JSON object")
        normalized_markers: Dict[str, str] = {}
        for key, value in markers.items():
            if key not in _KNOWN_MARKER_VARS:
                raise LockDiffError(
                    f"Target {name!r}: unknown marker variable {key!r}; "
                    f"supported variables: {', '.join(sorted(_KNOWN_MARKER_VARS))}"
                )
            if not isinstance(value, str):
                raise LockDiffError(f"Target {name!r}: marker {key!r} must be a string")
            normalized_markers[key] = value

        groups = obj.get("groups") or ["default"]
        extras = obj.get("extras") or []
        if not isinstance(groups, list) or not all(isinstance(g, str) for g in groups):
            raise LockDiffError(f"Target {name!r}: groups must be an array of strings")
        if not isinstance(extras, list) or not all(isinstance(e, str) for e in extras):
            raise LockDiffError(f"Target {name!r}: extras must be an array of strings")
        if not groups:
            raise LockDiffError(f"Target {name!r}: groups must not be empty")

        targets.append(
            Target(
                name=name,
                markers=dict(sorted(normalized_markers.items())),
                groups=frozenset(canonicalize_name(g) for g in groups),
                extras=frozenset(canonicalize_name(e) for e in extras),
                raw_groups=tuple(sorted(groups)),
                raw_extras=tuple(sorted(extras)),
            )
        )
    return targets


# ---------------------------------------------------------------------------
# Normalized comparison model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """Where an installed distribution comes from."""

    kind: str  # "index" | "vcs" | "directory" | "archive" | "unknown"
    url: Optional[str] = None
    name: Optional[str] = None
    recognized: bool = True
    path: Optional[str] = None
    vcs: Optional[str] = None
    commit_id: Optional[str] = None
    requested_revision: Optional[str] = None
    editable: bool = False
    subdirectory: Optional[str] = None
    raw: Optional[str] = None

    def identity(self) -> Tuple[Any, ...]:
        """Comparable identity.

        Index origins are identified by their normalized URL (trailing-slash
        spelling is ignored); the human-readable name is only consulted when
        the URL could not be resolved, which also flags the source unknown.
        """
        if self.kind == "index":
            normalized_url = self.url.rstrip("/") if self.url is not None else None
            return (
                "index",
                normalized_url,
                self.name if normalized_url is None else None,
                self.recognized,
            )
        return (
            self.kind,
            self.url,
            self.name,
            self.recognized,
            self.path,
            self.vcs,
            self.commit_id,
            self.requested_revision,
            self.editable,
            self.subdirectory,
            self.raw,
        )

    def public_dict(self) -> Dict[str, Any]:
        data = {
            "kind": self.kind,
            "url": _redact(self.url),
            "name": self.name,
            "recognized": self.recognized,
            "path": self.path,
            "vcs": self.vcs,
            "commit_id": self.commit_id,
            "requested_revision": self.requested_revision,
            "editable": self.editable,
            "subdirectory": self.subdirectory,
            "raw": _redact(self.raw),
        }
        return {key: value for key, value in data.items() if value not in (None, False)}


@dataclass(frozen=True)
class Artifact:
    """A wheel or sdist file recorded by a PEP 751 lock."""

    kind: str  # "wheel" | "sdist"
    filename: str
    url: Optional[str]
    hashes: frozenset[str]

    def identity(self) -> Tuple[Any, ...]:
        return (self.kind, self.filename, _redact(self.url), tuple(sorted(self.hashes)))

    def public_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "filename": self.filename,
            "url": _redact(self.url),
            "hashes": sorted(self.hashes),
        }


@dataclass(frozen=True)
class NormalizedEntry:
    """One lock entry in the format-neutral comparison model."""

    name: str
    raw_name: str
    version: Optional[str]
    marker: Optional[str]
    requires_python: Optional[str]
    source: Source
    hashes: frozenset[str]
    artifacts: frozenset[Artifact]
    attestations: frozenset[str]
    section: Optional[str] = None

    def public_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "raw_name": self.raw_name,
            "version": self.version,
            "marker": self.marker,
            "requires_python": self.requires_python,
            "source": self.source.public_dict(),
            "hashes": sorted(self.hashes),
            "artifacts": [
                artifact.public_dict()
                for artifact in sorted(self.artifacts, key=lambda item: item.identity())
            ],
            "attestations": [json.loads(item) for item in sorted(self.attestations)],
            "section": self.section,
        }

    def brief_dict(self) -> Dict[str, Any]:
        """Compact entry description for issue output."""
        return {
            "name": self.name,
            "version": self.version,
            "marker": self.marker,
            "source": self.source.public_dict(),
            "section": self.section,
        }


@dataclass(frozen=True)
class NormalizedLock:
    path: str
    lock_format: str  # "pipfile-lock" | "pylock-toml"
    entries: Tuple[NormalizedEntry, ...]
    declared_groups: Tuple[str, ...]
    declared_extras: Tuple[str, ...]


def _redact(value: Optional[str]) -> Optional[str]:
    """Redact credentials while preserving non-secret URL identity."""
    if not isinstance(value, str):
        return value
    if "://" not in value:
        return value
    try:
        return redact_auth_from_url(value)
    except Exception:
        return value


def _hash_set(
    values: Iterable[Mapping[str, str]] = (), extra: Iterable[str] = ()
) -> frozenset[str]:
    hashes = set()
    for mapping in values:
        for algorithm, digest in mapping.items():
            hashes.add(f"{algorithm}:{digest}".lower())
    for value in extra:
        if isinstance(value, str) and ":" in value:
            algorithm, _, digest = value.partition(":")
            hashes.add(f"{algorithm.lower()}:{digest.lower()}")
    return frozenset(hashes)


def _specifier_key(value: Optional[str]) -> Tuple[str, ...]:
    if not value:
        return ()
    try:
        return tuple(sorted(str(specifier) for specifier in SpecifierSet(value)))
    except InvalidSpecifier:
        return (value.strip(),)


# -- Pipfile.lock normalization ---------------------------------------------


def _pipfile_group(section: str) -> str:
    if section == "default":
        return "default"
    if section == "develop":
        return "dev"
    return canonicalize_name(section)


def _pipfile_source(
    package_data: Mapping[str, Any],
    sources_by_name: Mapping[str, Mapping[str, Any]],
    default_source: Optional[Mapping[str, Any]],
) -> Source:
    # Reuse the existing lock-entry -> PEP 751 field translation so VCS,
    # directory and archive handling stays identical to the pylock exporter.
    from pipenv.utils.pylock import _pylock_source_from_lockfile_entry

    editable = bool(package_data.get("editable"))
    translated = _pylock_source_from_lockfile_entry(dict(package_data))
    if "vcs" in translated:
        vcs = translated["vcs"]
        return Source(
            kind="vcs",
            vcs=vcs.get("type"),
            url=vcs.get("url"),
            path=vcs.get("path"),
            commit_id=vcs.get("commit-id"),
            requested_revision=vcs.get("requested-revision"),
            subdirectory=vcs.get("subdirectory"),
            editable=editable,
        )
    if "directory" in translated:
        directory = translated["directory"]
        return Source(
            kind="directory",
            path=directory.get("path"),
            editable=bool(directory.get("editable")),
            subdirectory=directory.get("subdirectory"),
        )
    if "archive" in translated:
        archive = translated["archive"]
        return Source(
            kind="archive",
            url=archive.get("url"),
            path=archive.get("path"),
            subdirectory=archive.get("subdirectory"),
        )

    index_name = package_data.get("index")
    if index_name is not None:
        source = sources_by_name.get(index_name)
        if source is None:
            return Source(kind="index", name=str(index_name), recognized=False)
        return Source(
            kind="index", name=str(index_name), url=source.get("url"), recognized=True
        )
    if default_source is not None:
        return Source(
            kind="index",
            name=default_source.get("name"),
            url=default_source.get("url"),
            recognized=True,
        )
    return Source(
        kind="unknown",
        raw=json.dumps(dict(package_data), sort_keys=True, default=str),
        recognized=False,
    )


def _normalize_pipfile_lock(path: Path, data: Mapping[str, Any]) -> NormalizedLock:
    if not isinstance(data, dict):
        raise LockDiffError(f"{path}: Pipfile.lock must contain a JSON object")
    meta = data.get("_meta") or {}
    sources = meta.get("sources") or []
    sources_by_name: Dict[str, Mapping[str, Any]] = {}
    for source in sources:
        if isinstance(source, dict) and source.get("name"):
            sources_by_name[source["name"]] = source
    default_source = sources[0] if sources else None

    entries: List[NormalizedEntry] = []
    sections = [section for section in data if section != "_meta"]
    declared_groups = sorted({_pipfile_group(section) for section in sections})
    for section in sections:
        packages = data.get(section) or {}
        if not isinstance(packages, dict):
            raise LockDiffError(f"{path}: section {section!r} must be an object")
        group = _pipfile_group(section)
        group_marker = f"'{group}' in dependency_groups"
        for raw_name, package_data in packages.items():
            if not isinstance(package_data, dict):
                raise LockDiffError(f"{path}: entry {raw_name!r} must be an object")
            env_marker = package_data.get("markers")
            marker = (
                f"({group_marker}) and ({env_marker})" if env_marker else group_marker
            )
            raw_version = package_data.get("version")
            if raw_version in (None, "*"):
                version = None
            elif isinstance(raw_version, str) and raw_version.startswith("=="):
                version = raw_version[2:]
            else:
                version = raw_version
            source = _pipfile_source(package_data, sources_by_name, default_source)
            entries.append(
                NormalizedEntry(
                    name=canonicalize_name(raw_name),
                    raw_name=raw_name,
                    version=str(version) if version is not None else None,
                    marker=marker,
                    requires_python=None,
                    source=source,
                    hashes=_hash_set(extra=package_data.get("hashes") or []),
                    artifacts=frozenset(),
                    attestations=frozenset(),
                    section=section,
                )
            )
    return NormalizedLock(
        path=str(path),
        lock_format="pipfile-lock",
        entries=tuple(entries),
        declared_groups=tuple(declared_groups),
        declared_extras=(),
    )


# -- pylock.toml normalization ----------------------------------------------


def _pylock_source(package: Mapping[str, Any]) -> Source:
    if "vcs" in package:
        vcs = package["vcs"]
        return Source(
            kind="vcs",
            vcs=vcs.get("type"),
            url=vcs.get("url"),
            path=vcs.get("path"),
            commit_id=vcs.get("commit-id"),
            requested_revision=vcs.get("requested-revision"),
            subdirectory=vcs.get("subdirectory"),
            editable=bool(vcs.get("editable", False)),
        )
    if "directory" in package:
        directory = package["directory"]
        return Source(
            kind="directory",
            path=directory.get("path"),
            editable=bool(directory.get("editable", False)),
            subdirectory=directory.get("subdirectory"),
        )
    if "archive" in package:
        archive = package["archive"]
        return Source(
            kind="archive",
            url=archive.get("url"),
            path=archive.get("path"),
            subdirectory=archive.get("subdirectory"),
        )
    if package.get("index"):
        return Source(kind="index", url=package["index"], recognized=True)
    return Source(
        kind="unknown",
        raw=json.dumps(dict(package), sort_keys=True, default=str),
        recognized=False,
    )


def _pylock_artifacts(
    package: Mapping[str, Any]
) -> Tuple[frozenset[Artifact], frozenset[str]]:
    artifacts: List[Artifact] = []
    hash_maps: List[Mapping[str, str]] = []
    for wheel in package.get("wheels") or []:
        hashes = wheel.get("hashes") or {}
        hash_maps.append(hashes)
        artifacts.append(
            Artifact(
                kind="wheel",
                filename=wheel.get("name", ""),
                url=wheel.get("url") or wheel.get("path"),
                hashes=_hash_set([hashes]),
            )
        )
    sdist = package.get("sdist")
    if sdist:
        hashes = sdist.get("hashes") or {}
        hash_maps.append(hashes)
        artifacts.append(
            Artifact(
                kind="sdist",
                filename=sdist.get("name", ""),
                url=sdist.get("url") or sdist.get("path"),
                hashes=_hash_set([hashes]),
            )
        )
    archive = package.get("archive")
    if archive and archive.get("hashes"):
        hash_maps.append(archive["hashes"])
    tool_hashes = []
    tool = package.get("tool") or {}
    if isinstance(tool, dict):
        pipenv_tool = tool.get("pipenv") or {}
        if isinstance(pipenv_tool, dict):
            tool_hashes = list(pipenv_tool.get("hashes") or [])
    return frozenset(artifacts), _hash_set(hash_maps, extra=tool_hashes)


def _attestation_set(package: Mapping[str, Any]) -> frozenset[str]:
    identities = package.get("attestation-identities") or []
    return frozenset(
        json.dumps(dict(identity), sort_keys=True, default=str) for identity in identities
    )


def _normalize_pylock(path: Path) -> NormalizedLock:
    from pipenv.utils.pylock import PylockError, PylockFile

    try:
        pylock = PylockFile.from_path(path)
    except PylockError as exc:
        raise LockDiffError(f"{path}: {exc}") from exc
    data = pylock.data

    sources = data.get("sources") or []
    index_names: Dict[str, str] = {}
    for source in sources:
        if isinstance(source, dict) and source.get("url") and source.get("name"):
            index_names[str(source["url"]).rstrip("/")] = str(source["name"])

    entries: List[NormalizedEntry] = []
    for package in pylock.packages:
        source = _pylock_source(package)
        if source.kind == "index" and source.url:
            name = index_names.get(source.url.rstrip("/"))
            if name:
                source = Source(kind="index", url=source.url, name=name, recognized=True)
        artifacts, hashes = _pylock_artifacts(package)
        raw_name = package.get("name", "")
        entries.append(
            NormalizedEntry(
                name=canonicalize_name(raw_name),
                raw_name=raw_name,
                version=package.get("version"),
                marker=package.get("marker"),
                requires_python=package.get("requires-python"),
                source=source,
                hashes=hashes,
                artifacts=artifacts,
                attestations=_attestation_set(package),
                section=None,
            )
        )
    return NormalizedLock(
        path=str(path),
        lock_format="pylock-toml",
        entries=tuple(entries),
        declared_groups=tuple(sorted(pylock.dependency_groups)),
        declared_extras=tuple(sorted(pylock.extras)),
    )


def load_lock_file(path: Union[str, Path]) -> NormalizedLock:
    """Load and normalize a lock file without touching anything else.

    The file is parsed directly (JSON or TOML) and is never written back.
    """
    lock_path = Path(str(path))
    if not lock_path.is_file():
        raise LockDiffError(f"Lock file not found: {lock_path}")
    text = lock_path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    suffix = lock_path.suffix.lower()
    is_json_hint = suffix in {".lock", ".json"} or stripped.startswith("{")
    is_toml_hint = suffix == ".toml" or "lock-version" in text
    if is_json_hint and not is_toml_hint:
        data = _load_json_document(text, str(lock_path))
        return _normalize_pipfile_lock(lock_path, data)
    if is_toml_hint and not stripped.startswith("{"):
        return _normalize_pylock(lock_path)
    # Ambiguous content: try JSON first, then TOML.
    if stripped.startswith("{"):
        try:
            return _normalize_pipfile_lock(
                lock_path, _load_json_document(text, str(lock_path))
            )
        except LockDiffError:
            raise
    raise LockDiffError(
        f"{lock_path}: unrecognized lock format (expected Pipfile.lock JSON or "
        "pylock.toml with a lock-version table)"
    )


# ---------------------------------------------------------------------------
# Marker evaluation against the restricted target environment
# ---------------------------------------------------------------------------


def evaluate_marker(marker_text: Optional[str], target: Target) -> str:
    """Return ``active``/``inactive``/``unknown`` for one entry marker."""
    if not marker_text:
        return OUTCOME_ACTIVE
    try:
        marker = Marker(marker_text)
    except InvalidMarker:
        return OUTCOME_UNKNOWN
    try:
        matched = _evaluate_markers(marker._markers, target.evaluation_environment())
    except UndefinedEnvironmentName:
        return OUTCOME_UNKNOWN
    except UndefinedComparison:
        return OUTCOME_UNKNOWN
    return OUTCOME_ACTIVE if matched else OUTCOME_INACTIVE


def missing_marker_variable(marker_text: Optional[str], target: Target) -> Optional[str]:
    """Name the variable a target lacks, when evaluation is otherwise unknown."""
    if not marker_text:
        return None
    try:
        marker = Marker(marker_text)
    except InvalidMarker:
        return None
    try:
        _evaluate_markers(marker._markers, target.evaluation_environment())
    except UndefinedEnvironmentName as exc:
        return str(exc.args[0]) if exc.args else None
    except UndefinedComparison:
        return None
    return None


@dataclass
class NameSelection:
    outcome: str
    entries: List[NormalizedEntry] = field(default_factory=list)
    unknown_entries: List[NormalizedEntry] = field(default_factory=list)
    missing_variable: Optional[str] = None


def _entry_identity_key(entry: NormalizedEntry) -> Tuple[Any, ...]:
    """Concrete artifact identity used to collapse equivalent duplicate entries.

    A package legitimately appears in several Pipfile.lock categories (or
    several pylock entries whose markers overlap on a given target); those
    duplicates only conflict when they pin different concrete artifacts.
    """
    return (
        entry.version,
        entry.source.identity(),
        entry.hashes,
        tuple(sorted(artifact.identity() for artifact in entry.artifacts)),
        entry.attestations,
        _specifier_key(entry.requires_python),
    )


def _dedupe_identical(entries: List[NormalizedEntry]) -> List[NormalizedEntry]:
    unique: Dict[Tuple[Any, ...], NormalizedEntry] = {}
    for entry in entries:
        unique.setdefault(_entry_identity_key(entry), entry)
    return list(unique.values())


def select_for_target(lock: NormalizedLock, target: Target) -> Dict[str, NameSelection]:
    """Select entries per canonical name for one explicit target."""
    grouped: Dict[str, List[NormalizedEntry]] = {}
    for entry in lock.entries:
        grouped.setdefault(entry.name, []).append(entry)

    selection: Dict[str, NameSelection] = {}
    for name, entries in grouped.items():
        active: List[NormalizedEntry] = []
        unknown: List[NormalizedEntry] = []
        missing_variable: Optional[str] = None
        for entry in entries:
            outcome = evaluate_marker(entry.marker, target)
            if outcome == OUTCOME_ACTIVE:
                active.append(entry)
            elif outcome == OUTCOME_UNKNOWN:
                unknown.append(entry)
                if missing_variable is None:
                    missing_variable = missing_marker_variable(entry.marker, target)
        distinct_active = _dedupe_identical(active)
        if len(distinct_active) > 1:
            selection[name] = NameSelection(
                OUTCOME_AMBIGUOUS,
                entries=distinct_active,
                unknown_entries=unknown,
                missing_variable=missing_variable,
            )
        elif len(distinct_active) == 1:
            # A definite selection is kept; any unresolvable sibling entry is
            # still surfaced as an issue because it could hide a conflict.
            selection[name] = NameSelection(
                OUTCOME_ACTIVE,
                entries=distinct_active,
                unknown_entries=unknown,
                missing_variable=missing_variable,
            )
        elif unknown:
            selection[name] = NameSelection(
                OUTCOME_UNKNOWN,
                unknown_entries=unknown,
                missing_variable=missing_variable,
            )
        else:
            selection[name] = NameSelection(OUTCOME_INACTIVE)
    return selection


# ---------------------------------------------------------------------------
# Difference engine
# ---------------------------------------------------------------------------


def _issue(code: str, side: str, package: str, **detail: Any) -> Dict[str, Any]:
    issue = {"code": code, "side": side, "package": package}
    issue.update(detail)
    return issue


def _selection_summary(entry: NormalizedEntry) -> Dict[str, Any]:
    return {
        "name": entry.name,
        "version": entry.version,
        "source_kind": entry.source.kind,
    }


def _source_identity_changed(left: Source, right: Source) -> bool:
    return left.identity() != right.identity()


def _compare_selected_entries(
    left: NormalizedEntry,
    right: NormalizedEntry,
    left_format: str,
    right_format: str,
    visibility_details: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    fields: List[str] = []
    details: Dict[str, Any] = {}

    if left.version != right.version:
        fields.append("version")
        details["version"] = {"before": left.version, "after": right.version}

    if _source_identity_changed(left.source, right.source):
        fields.append("source")
        details["source"] = {
            "before": left.source.public_dict(),
            "after": right.source.public_dict(),
        }

    added_hashes = sorted(right.hashes - left.hashes)
    removed_hashes = sorted(left.hashes - right.hashes)
    if added_hashes or removed_hashes:
        if added_hashes:
            fields.append("hashes.added")
        if removed_hashes:
            fields.append("hashes.removed")
        details["hashes"] = {"added": added_hashes, "removed": removed_hashes}

    # Artifact/attestation metadata only exists in pylock.toml; comparing it
    # against a format that cannot express it would manufacture format noise.
    if left_format in _FORMAT_HAS_ARTIFACTS and right_format in _FORMAT_HAS_ARTIFACTS:
        left_artifacts = {artifact.identity(): artifact for artifact in left.artifacts}
        right_artifacts = {artifact.identity(): artifact for artifact in right.artifacts}
        added_artifacts = [
            right_artifacts[key].public_dict()
            for key in sorted(right_artifacts.keys() - left_artifacts.keys())
        ]
        removed_artifacts = [
            left_artifacts[key].public_dict()
            for key in sorted(left_artifacts.keys() - right_artifacts.keys())
        ]
        if added_artifacts:
            fields.append("artifacts.added")
        if removed_artifacts:
            fields.append("artifacts.removed")
        if added_artifacts or removed_artifacts:
            details["artifacts"] = {
                "added": added_artifacts,
                "removed": removed_artifacts,
            }

        added_attestations = sorted(right.attestations - left.attestations)
        removed_attestations = sorted(left.attestations - right.attestations)
        if added_attestations:
            fields.append("attestations.added")
        if removed_attestations:
            fields.append("attestations.removed")
        if added_attestations or removed_attestations:
            details["attestations"] = {
                "added": [json.loads(item) for item in added_attestations],
                "removed": [json.loads(item) for item in removed_attestations],
            }

    if _specifier_key(left.requires_python) != _specifier_key(right.requires_python):
        fields.append("requires_python")
        details["requires_python"] = {
            "before": left.requires_python,
            "after": right.requires_python,
        }

    if visibility_details:
        fields.append("visibility")
        details["visibility"] = visibility_details

    if not fields:
        return None
    return {
        "category": "changed",
        "package": left.name,
        "fields": fields,
        "before": left.public_dict(),
        "after": right.public_dict(),
        "details": details,
    }


def _build_issues(
    side: str,
    selection: Mapping[str, NameSelection],
) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    for name in sorted(selection):
        chosen = selection[name]
        if chosen.outcome == OUTCOME_AMBIGUOUS:
            issues.append(
                _issue(
                    "ambiguous-entries",
                    side,
                    name,
                    detail=f"{len(chosen.entries)} distinct entries match this target",
                    entries=[entry.brief_dict() for entry in chosen.entries],
                )
            )
        # Report unresolvable entries even when a sibling gives a definite
        # selection: they could conceal a conflict on the real target.
        if chosen.unknown_entries:
            if chosen.missing_variable:
                issues.append(
                    _issue(
                        "unknown-marker-variable",
                        side,
                        name,
                        variable=chosen.missing_variable,
                        marker=chosen.unknown_entries[0].marker,
                    )
                )
            else:
                issues.append(
                    _issue(
                        "invalid-marker",
                        side,
                        name,
                        marker=chosen.unknown_entries[0].marker,
                    )
                )
        if chosen.outcome in {OUTCOME_ACTIVE, OUTCOME_AMBIGUOUS}:
            issues.extend(
                _issue(
                    "unknown-source",
                    side,
                    name,
                    source=entry.source.public_dict(),
                )
                for entry in chosen.entries
                if not entry.source.recognized or entry.source.kind == "unknown"
            )
    return issues


def diff_locks(
    left: NormalizedLock,
    right: NormalizedLock,
    targets: Sequence[Target],
) -> Dict[str, Any]:
    """Compute the full, deterministic diff report."""
    all_names = sorted({entry.name for lock in (left, right) for entry in lock.entries})
    target_reports: List[Dict[str, Any]] = []
    any_incomplete = False
    totals = {"added": 0, "removed": 0, "changed": 0, "issues": 0}

    # Matrix-level visibility: outcome of every name on every target/side.
    visibility: Dict[str, Dict[str, Tuple[str, str]]] = {name: {} for name in all_names}
    selections: Dict[str, Tuple[Dict[str, NameSelection], Dict[str, NameSelection]]] = {}
    for target in targets:
        left_selection = select_for_target(left, target)
        right_selection = select_for_target(right, target)
        selections[target.name] = (left_selection, right_selection)
        for name in all_names:
            left_outcome = left_selection.get(
                name, NameSelection(OUTCOME_INACTIVE)
            ).outcome
            right_outcome = right_selection.get(
                name, NameSelection(OUTCOME_INACTIVE)
            ).outcome
            visibility[name][target.name] = (left_outcome, right_outcome)

    for target in targets:
        left_selection, right_selection = selections[target.name]
        issues = _build_issues("left", left_selection) + _build_issues(
            "right", right_selection
        )
        differences: List[Dict[str, Any]] = []

        for name in all_names:
            left_chosen = left_selection.get(name)
            right_chosen = right_selection.get(name)
            left_state = left_chosen.outcome if left_chosen else OUTCOME_INACTIVE
            right_state = right_chosen.outcome if right_chosen else OUTCOME_INACTIVE

            # Anything not unambiguously selected cannot be claimed as a
            # concrete add/remove; it is surfaced through issues instead.
            left_blocked = left_state in {OUTCOME_UNKNOWN, OUTCOME_AMBIGUOUS}
            right_blocked = right_state in {OUTCOME_UNKNOWN, OUTCOME_AMBIGUOUS}
            if left_blocked or right_blocked:
                continue

            left_entry = left_chosen.entries[0] if left_state == OUTCOME_ACTIVE else None
            right_entry = (
                right_chosen.entries[0] if right_state == OUTCOME_ACTIVE else None
            )

            if left_entry is None and right_entry is not None:
                differences.append(
                    {
                        "category": "added",
                        "package": name,
                        "fields": [],
                        "before": None,
                        "after": right_entry.public_dict(),
                        "details": {},
                    }
                )
            elif left_entry is not None and right_entry is None:
                differences.append(
                    {
                        "category": "removed",
                        "package": name,
                        "fields": [],
                        "before": left_entry.public_dict(),
                        "after": None,
                        "details": {},
                    }
                )
            elif left_entry is not None and right_entry is not None:
                visibility_details = [
                    {"target": target_name, "before": outcomes[0], "after": outcomes[1]}
                    for target_name, outcomes in visibility[name].items()
                    if outcomes[0] != outcomes[1]
                ]
                changed = _compare_selected_entries(
                    left_entry,
                    right_entry,
                    left.lock_format,
                    right.lock_format,
                    visibility_details,
                )
                if changed is not None:
                    differences.append(changed)

        order = {category: index for index, category in enumerate(CATEGORY_ORDER)}
        differences.sort(key=lambda item: (order[item["category"]], item["package"]))

        for difference in differences:
            totals[difference["category"]] += 1
        totals["issues"] += len(issues)
        target_incomplete = bool(issues)
        any_incomplete = any_incomplete or target_incomplete

        target_reports.append(
            {
                "name": target.name,
                "markers": dict(target.markers),
                "groups": list(target.raw_groups),
                "extras": list(target.raw_extras),
                "status": STATUS_INCOMPLETE if target_incomplete else "complete",
                "issues": issues,
                "selection": {
                    "left": sorted(
                        (
                            _selection_summary(chosen.entries[0])
                            for chosen in left_selection.values()
                            if chosen.outcome == OUTCOME_ACTIVE
                        ),
                        key=lambda item: item["name"],
                    ),
                    "right": sorted(
                        (
                            _selection_summary(chosen.entries[0])
                            for chosen in right_selection.values()
                            if chosen.outcome == OUTCOME_ACTIVE
                        ),
                        key=lambda item: item["name"],
                    ),
                },
                "differences": differences,
            }
        )

    has_changes = any(report["differences"] for report in target_reports)
    if any_incomplete:
        status = STATUS_INCOMPLETE
    elif has_changes:
        status = STATUS_CHANGED
    else:
        status = STATUS_UNCHANGED

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "left": {"path": left.path, "format": left.lock_format},
        "right": {"path": right.path, "format": right.lock_format},
        "targets": target_reports,
        "summary": {
            "targets": len(target_reports),
            "incomplete_targets": sum(
                1 for report in target_reports if report["status"] == STATUS_INCOMPLETE
            ),
            "added": totals["added"],
            "removed": totals["removed"],
            "changed": totals["changed"],
            "issues": totals["issues"],
        },
    }


def diff_paths(
    left_path: Union[str, Path],
    right_path: Union[str, Path],
    targets: Sequence[Target],
) -> Dict[str, Any]:
    """Load both lock files and compute their target-aware diff."""
    left = load_lock_file(left_path)
    right = load_lock_file(right_path)
    return diff_locks(left, right, targets)


# ---------------------------------------------------------------------------
# Human-readable rendering
# ---------------------------------------------------------------------------


def _render_entry_line(entry: Optional[Dict[str, Any]]) -> str:
    if entry is None:
        return ""
    source = entry.get("source") or {}
    origin = (
        source.get("url")
        or source.get("path")
        or source.get("name")
        or source.get("kind")
    )
    version = entry.get("version") or "(unversioned)"
    line = f"{entry['name']} {version}"
    if origin:
        line += f" [{origin}]"
    return line


def render_text_report(report: Mapping[str, Any]) -> str:
    """Render the stable report dict as a readable summary."""
    lines: List[str] = []
    lines.append("pipenv lock diff")
    lines.append(f"  left:  {report['left']['path']} ({report['left']['format']})")
    lines.append(f"  right: {report['right']['path']} ({report['right']['format']})")
    lines.append("")

    for target in report["targets"]:
        marker_text = (
            ", ".join(
                f"{key}={value}" for key, value in sorted(target["markers"].items())
            )
            or "-"
        )
        groups = ", ".join(target["groups"]) or "-"
        extras = ", ".join(target["extras"]) or "-"
        lines.append(
            f"Target {target['name']} (markers: {marker_text}; groups: {groups}; "
            f"extras: {extras})"
        )
        lines.append(f"  status: {target['status']}")

        for category, symbol in (("added", "+"), ("removed", "-"), ("changed", "~")):
            items = [d for d in target["differences"] if d["category"] == category]
            if not items:
                continue
            lines.append(f"  {category}:")
            for item in items:
                if category == "added":
                    lines.append(f"    {symbol} {_render_entry_line(item['after'])}")
                elif category == "removed":
                    lines.append(f"    {symbol} {_render_entry_line(item['before'])}")
                else:
                    before = item["before"]
                    after = item["after"]
                    lines.append(
                        f"    {symbol} {item['package']} "
                        f"{before.get('version') or '(unversioned)'} -> "
                        f"{after.get('version') or '(unversioned)'} "
                        f"[fields: {', '.join(item['fields'])}]"
                    )
                    detail = item.get("details") or {}
                    if "hashes" in detail:
                        hash_detail = detail["hashes"]
                        lines.append(
                            f"        hashes: +{len(hash_detail['added'])} added, "
                            f"-{len(hash_detail['removed'])} removed"
                        )
                    if "visibility" in detail:
                        lines.extend(
                            f"        visibility on {change['target']}: "
                            f"{change['before']} -> {change['after']}"
                            for change in detail["visibility"]
                        )

        if target["issues"]:
            lines.append("  issues:")
            lines.extend(
                f"    ? [{issue['code']}] {issue['side']}:{issue['package']} "
                + (issue.get("detail") or issue.get("variable") or "")
                for issue in target["issues"]
            )
        lines.append("")

    summary = report["summary"]
    lines.append(
        "Summary: {targets} target(s), {incomplete_targets} incomplete; "
        "added={added} removed={removed} changed={changed} issues={issues}".format(
            **summary
        )
    )
    lines.append(
        "Equivalence is only asserted for the explicit target matrix above; "
        "no claim is made about other platforms."
    )
    lines.append(f"Overall status: {report['status']}")
    return "\n".join(lines)
