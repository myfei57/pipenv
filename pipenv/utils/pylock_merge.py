"""Three-way merge for PEP 751 ``pylock.toml`` files.

The merger combines a common ``base`` lock with two same-origin branches
(``left``/``right``) without writing files, contacting an index, or
re-resolving dependencies.  Marker semantics are evaluated exclusively against
an explicit, finite target matrix supplied by the caller: coverage is proven
only for those targets, never claimed for arbitrary unknown platforms.

Merge rules (per matched package entry):

* a change made on one side is kept;
* the same change made on both sides is merged once;
* a deletion opposed by a modification on the other side conflicts;
* different versions, sources, or security-related artifact hash sets for the
  same environment coverage conflict -- ``version``, the source identity and
  the artifact hashes are always handled as one indivisible unit, so a version
  is never combined with the other side's stale hashes and hash sets are never
  unioned (which would widen the trust set);
* reordering semantically unordered data (the ``packages`` array, the
  ``wheels`` array, group/extra/environment sets) never causes a conflict.

The result is either a complete merged lock (``MergeResult.ok is True``) or a
structured conflict report (``ok is False``, ``merged is None``).  Dependency
closure cannot be proven from lock contents alone, so a successful merge only
claims structural consistency over the target matrix; the integrity status is
recorded explicitly instead of fabricating resolution success.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from pipenv.patched.pip._vendor.packaging.markers import (
    InvalidMarker,
    Marker,
    UndefinedComparison,
    UndefinedEnvironmentName,
    Value,
    Variable,
)
from pipenv.patched.pip._vendor.packaging.specifiers import (
    InvalidSpecifier,
    SpecifierSet,
)
from pipenv.patched.pip._vendor.packaging.utils import canonicalize_name
from pipenv.utils.pylock import PylockError, PylockFile

# PEP 751 lock-version 1.x top-level keys understood by this merger.
STANDARD_TOP_LEVEL_KEYS = frozenset(
    {
        "lock-version",
        "environments",
        "requires-python",
        "extras",
        "dependency-groups",
        "default-groups",
        "created-by",
        "sources",
        "packages",
        "tool",
    }
)

# PEP 751 package-entry keys understood by this merger.
STANDARD_PACKAGE_KEYS = frozenset(
    {
        "name",
        "version",
        "requires-python",
        "marker",
        "index",
        "wheels",
        "sdist",
        "vcs",
        "directory",
        "archive",
        "tool",
    }
)

# Tool tables whose contents have a declared, supported merge semantics.
DEFAULT_SUPPORTED_TOOL_TABLES = ("pipenv",)

# Sentinel for "key absent" (distinct from None).
_MISSING = object()

# Conflict kind constants, kept as plain strings so reports serialize easily.
KIND_LOCK_VERSION = "lock_version_divergence"
KIND_INVALID_MARKER = "invalid_marker"
KIND_COVERAGE_UNDETERMINED = "coverage_undetermined"
KIND_DUPLICATE_SIGNATURE = "duplicate_signature"
KIND_DELETE_MODIFY = "delete_modified"
KIND_BOTH_ADDED_DIVERGENT = "both_added_divergent"
KIND_DIVERGENT_CHANGE = "divergent_change"
KIND_VERSION_DIVERGENCE = "version_divergence"
KIND_ARTIFACT_DIVERGENCE = "artifact_divergence"
KIND_REQUIRES_PYTHON_DIVERGENCE = "requires_python_divergence"
KIND_MARKER_TEXT_DIVERGENCE = "marker_text_divergence"
KIND_FIELD_DIVERGENCE = "field_divergence"
KIND_OVERLAP = "overlapping_entries"
KIND_UNDECLARED_GROUP = "undeclared_dependency_group"
KIND_UNDECLARED_EXTRA = "undeclared_extra"
KIND_DEFAULT_GROUP = "default_group_undeclared"
KIND_TARGET_REQUIRES_PYTHON = "target_excluded_by_requires_python"
KIND_TARGET_ENVIRONMENTS = "target_not_in_environments"
KIND_UNKNOWN_EXTENSION = "unknown_extension"
KIND_INVALID_INPUT = "invalid_input"

# Package fields that pin the actual downloadable artifact / its trust set.
_SECURITY_FIELDS = ("wheels", "sdist", "archive", "vcs")


class PylockMergeError(PylockError):
    """Raised for caller errors (bad arguments), not for merge conflicts."""


class UnknownExtensionPolicy:
    """How to treat keys this merger does not know about.

    * ``ERROR`` (default): refuse the merge and demand an explicit decision;
    * ``KEEP``: preserve unknown data, only conflicting when both sides change
      the same unknown leaf in incompatible ways;
    * ``DROP``: explicitly discard unknown data.
    """

    ERROR = "error"
    KEEP = "keep"
    DROP = "drop"

    ALL = frozenset({ERROR, KEEP, DROP})


@dataclass(frozen=True)
class TargetEnvironment:
    """One point of the explicit finite marker evaluation matrix.

    ``variables`` holds the PEP 508 marker variables (``sys_platform``,
    ``python_full_version`` ...).  Every variable referenced by any marker in
    the inputs must be provided here, otherwise the merge is refused.
    ``extras`` and ``dependency_groups`` are the lock-file-context sets used
    when evaluating ``... in extras`` / ``... in dependency_groups`` markers.
    """

    name: str
    variables: Mapping[str, str]
    extras: frozenset[str] = frozenset()
    dependency_groups: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Conflict:
    """A single structured reason the merge cannot complete automatically."""

    path: str
    kind: str
    message: str
    base: Any = None
    left: Any = None
    right: Any = None


@dataclass
class TargetCoverage:
    """Coverage proof for one target of the merged lock."""

    target: str
    variables: dict[str, str]
    extras: frozenset[str]
    dependency_groups: frozenset[str]
    # Matched environments markers, aligned with the merged environments list.
    environments_matches: list[bool] = field(default_factory=list)
    # Every retained package entry with its marker evaluation for this target.
    entries: list[dict[str, Any]] = field(default_factory=list)
    # canonical package name -> the unique entry selected on this target.
    selected: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class MergeResult:
    """Outcome of :func:`three_way_merge`."""

    ok: bool
    merged: Optional[dict[str, Any]]
    conflicts: list[Conflict]
    coverage: dict[str, TargetCoverage]
    # Always explicit: a structural merge never proves dependency closure.
    integrity: str
    integrity_notes: list[str] = field(default_factory=list)

    @property
    def conflict_paths(self) -> list[str]:
        return [conflict.path for conflict in self.conflicts]


# --------------------------------------------------------------------------- #
# Canonicalization / equality helpers
# --------------------------------------------------------------------------- #


def _canonical(value: Any) -> Any:
    """Hashable, order-insensitive representation for structural equality."""
    if isinstance(value, Mapping):
        return (
            "dict",
            tuple(sorted((key, _canonical(item)) for key, item in value.items())),
        )
    if isinstance(value, (list, tuple)):
        return ("list", tuple(_canonical(item) for item in value))
    if isinstance(value, frozenset):
        return ("set", tuple(sorted(_canonical(item) for item in value)))
    if isinstance(value, set):
        return ("set", tuple(sorted(_canonical(item) for item in value)))
    return (type(value).__name__, value)


def _canonical_wheels(wheels: Any) -> Any:
    # The wheels array is unordered per PEP 751; compare it as a multiset.
    return ("set", tuple(sorted(_canonical(wheel) for wheel in wheels)))


def _entry_body(entry: Mapping[str, Any]) -> Any:
    """Comparable body of a package entry, excluding identity fields.

    ``name`` and ``marker`` identify/partition the entry (marker text is
    compared separately); ``wheels`` is treated as an unordered multiset.
    """
    parts = []
    for key, value in entry.items():
        if key in ("name", "marker"):
            continue
        if key == "wheels":
            parts.append((key, _canonical_wheels(value)))
        else:
            parts.append((key, _canonical(value)))
    return ("body", tuple(sorted(parts)))


def _canonical_marker(marker: Optional[str]) -> Optional[str]:
    """Normalize cosmetic marker differences (whitespace, quote style)."""
    if not marker:
        return None
    return str(Marker(marker))


def _source_identity(entry: Mapping[str, Any]) -> tuple:
    """The source identity component of an entry's match signature."""
    if "index" in entry:
        return ("index", str(entry["index"]).rstrip("/"))
    if "vcs" in entry:
        vcs = entry["vcs"]
        return (
            "vcs",
            vcs.get("type"),
            vcs.get("url") or vcs.get("path"),
            vcs.get("subdirectory"),
        )
    if "directory" in entry:
        directory = entry["directory"]
        return ("directory", directory.get("path"), directory.get("subdirectory"))
    if "archive" in entry:
        archive = entry["archive"]
        return ("archive", archive.get("url") or archive.get("path"))
    return ("unsourced",)


def _source_label(identity: tuple) -> str:
    return "/".join(str(part) for part in identity if part is not None)


# --------------------------------------------------------------------------- #
# Marker introspection and matrix evaluation
# --------------------------------------------------------------------------- #


def _walk_marker_nodes(nodes: Any):
    for node in nodes:
        if isinstance(node, list):
            yield from _walk_marker_nodes(node)
        elif isinstance(node, tuple):
            yield node


def _marker_variables(marker: str) -> set[str]:
    """All marker variable names referenced by a marker string."""
    parsed = Marker(marker)
    variables = set()
    for node in _walk_marker_nodes(parsed._markers):
        for term in node:
            if isinstance(term, Variable):
                variables.add(term.value)
    return variables


def _marker_value_names(marker: Optional[str], variable_name: str) -> set[str]:
    """All values compared against a set-valued variable (extras/groups)."""
    if not marker:
        return set()
    parsed = Marker(marker)
    names = set()
    for node in _walk_marker_nodes(parsed._markers):
        if any(
            isinstance(term, Variable) and term.value == variable_name
            for term in node
        ):
            for term in node:
                if isinstance(term, Value):
                    names.add(canonicalize_name(term.value))
    return names


def _evaluate(marker: Optional[str], target: TargetEnvironment) -> bool:
    """Evaluate a marker against one matrix point, hermetically."""
    if not marker:
        return True
    environment: dict[str, Any] = dict(target.variables)
    environment["extras"] = target.extras
    environment["dependency_groups"] = target.dependency_groups
    return Marker(marker).evaluate(environment=environment, context="lock_file")


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #


def _as_data(lock: Any) -> dict[str, Any]:
    if isinstance(lock, PylockFile):
        lock = lock.data
    if not isinstance(lock, Mapping):
        raise PylockMergeError(
            "each of base/left/right must be a mapping or a PylockFile instance"
        )
    return copy.deepcopy(dict(lock))


# --------------------------------------------------------------------------- #
# Generic three-way primitives
# --------------------------------------------------------------------------- #


def _three_way_scalar(path: str, b: Any, lv: Any, rv: Any, kind: str) -> tuple[
    bool, Any, Optional[Conflict]
]:
    """Three-way merge for an atomic value.

    Returns ``(changed, value, conflict)``.  ``_MISSING`` stands for an absent
    key.  Equality is structural, so dict/list value changes compare deeply.
    """
    b_missing = b is _MISSING
    if b_missing:
        if lv is _MISSING and rv is _MISSING:
            return False, _MISSING, None
        if lv is _MISSING:
            return True, rv, None
        if rv is _MISSING:
            return True, lv, None
        if _canonical(lv) == _canonical(rv):
            return True, lv, None
        return (
            False,
            None,
            Conflict(path, KIND_BOTH_ADDED_DIVERGENT,
                     f"{path} was added differently on each side",
                     None, lv, rv),
        )
    if lv is _MISSING and rv is _MISSING:
        return True, _MISSING, None
    if lv is _MISSING:
        if _canonical(rv) == _canonical(b):
            return True, _MISSING, None  # left-only deletion
        return (
            False,
            None,
            Conflict(path, KIND_DELETE_MODIFY,
                     f"{path} was deleted on the left but modified on the right",
                     b, None, rv),
        )
    if rv is _MISSING:
        if _canonical(lv) == _canonical(b):
            return True, _MISSING, None  # right-only deletion
        return (
            False,
            None,
            Conflict(path, KIND_DELETE_MODIFY,
                     f"{path} was deleted on the right but modified on the left",
                     b, lv, None),
        )
    if _canonical(lv) == _canonical(b):
        if _canonical(rv) == _canonical(b):
            return False, b, None
        return True, rv, None  # right-only change
    if _canonical(rv) == _canonical(b):
        return True, lv, None  # left-only change
    if _canonical(lv) == _canonical(rv):
        return True, lv, None  # identical change on both sides
    return (
        False,
        None,
        Conflict(path, kind,
                 f"{path} was modified differently on each side", b, lv, rv),
    )


def _three_way_set(path: str, b: Any, lv: Any, rv: Any) -> set[Any]:
    """Element-wise three-way merge for semantically unordered sets.

    Additions on either side are kept, deletions on either side win, and there
    is no modify/delete ambiguity for bare set elements.
    """
    base = set(b or ())
    left = set(lv or ())
    right = set(rv or ())
    added = (left - base) | (right - base)
    removed = (base - left) | (base - right)
    return (base | added) - removed


def _three_way_mapping(
    path: str,
    b: Any,
    lv: Any,
    rv: Any,
    conflicts: list[Conflict],
    *,
    order: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Deep structural three-way merge for mappings (tool tables, unknown keys).

    Nested mappings recurse; every other value is an atomic leaf.  Leaf
    divergence produces a conflict carrying the full dotted path.
    """
    b = b or {}
    lv = lv or {}
    rv = rv or {}
    merged: dict[str, Any] = {}
    keys = list(dict.fromkeys([*b.keys(), *lv.keys(), *rv.keys()]))
    if order is not None:
        keys.sort(key=lambda key: (order.index(key) if key in order else len(order), key))
    for key in keys:
        child_path = f"{path}.{key}" if path else key
        bv = b.get(key, _MISSING)
        left_value = lv.get(key, _MISSING)
        right_value = rv.get(key, _MISSING)
        if isinstance(bv, Mapping) or isinstance(left_value, Mapping) or \
                isinstance(right_value, Mapping):
            merged[key] = _three_way_mapping(
                child_path,
                bv if isinstance(bv, Mapping) else {},
                left_value if isinstance(left_value, Mapping) else {},
                right_value if isinstance(right_value, Mapping) else {},
                conflicts,
            )
            # A type change mapping <-> scalar is a divergence at this leaf.
            kinds = {
                isinstance(value, Mapping)
                for value in (bv, left_value, right_value)
                if value is not _MISSING
            }
            if len(kinds) > 1:
                conflicts.append(
                    Conflict(child_path, KIND_FIELD_DIVERGENCE,
                             f"{child_path} changed type differently on each side",
                             None if bv is _MISSING else bv,
                             None if left_value is _MISSING else left_value,
                             None if right_value is _MISSING else right_value)
                )
            continue
        _, value, conflict = _three_way_scalar(
            child_path, bv, left_value, right_value, KIND_FIELD_DIVERGENCE
        )
        if conflict is not None:
            conflicts.append(conflict)
        elif value is not _MISSING:
            merged[key] = value
    return merged


# --------------------------------------------------------------------------- #
# Package entry records
# --------------------------------------------------------------------------- #


@dataclass
class _EntryRecord:
    side: str
    index: int
    name: str
    canonical_name: str
    source: tuple
    marker_text: Optional[str]
    coverage: frozenset[str]
    entry: dict[str, Any]
    body: Any

    @property
    def signature(self) -> tuple:
        return (self.canonical_name, self.source, self.coverage)

    @property
    def path(self) -> str:
        marker = self.marker_text or "<unconditional>"
        return (
            f"packages.{self.canonical_name}"
            f"[source={_source_label(self.source)}][marker={marker}]"
        )


def _build_records(
    side: str,
    data: Mapping[str, Any],
    targets: Sequence[TargetEnvironment],
    conflicts: list[Conflict],
) -> list[_EntryRecord]:
    records: list[_EntryRecord] = []
    packages = data.get("packages", [])
    if not isinstance(packages, list):
        conflicts.append(
            Conflict("packages", KIND_INVALID_INPUT,
                     f"{side}: 'packages' must be an array of tables")
        )
        return records
    for index, entry in enumerate(packages):
        if not isinstance(entry, Mapping):
            conflicts.append(
                Conflict(f"{side}.packages[{index}]", KIND_INVALID_INPUT,
                         f"{side}.packages[{index}] is not a table")
            )
            continue
        name = str(entry.get("name", ""))
        canonical = canonicalize_name(name) if name else ""
        if not canonical:
            conflicts.append(
                Conflict(f"{side}.packages[{index}]", KIND_INVALID_INPUT,
                         f"{side}.packages[{index}] is missing a name")
            )
            continue
        raw_marker = entry.get("marker")
        try:
            marker_text = _canonical_marker(raw_marker)
        except InvalidMarker as exc:
            conflicts.append(
                Conflict(f"{side}.packages[{index}].marker", KIND_INVALID_MARKER,
                         f"{side} package {name!r} has an invalid marker: {exc}")
            )
            continue
        coverage = set()
        for target in targets:
            try:
                if _evaluate(raw_marker, target):
                    coverage.add(target.name)
            except (InvalidMarker, UndefinedComparison, UndefinedEnvironmentName) as exc:
                conflicts.append(
                    Conflict(f"{side}.packages[{index}].marker",
                            KIND_INVALID_MARKER,
                            f"{side} package {name!r} marker cannot be evaluated "
                            f"for target {target.name!r}: {exc}")
                )
        records.append(
            _EntryRecord(
                side=side,
                index=index,
                name=name,
                canonical_name=canonical,
                source=_source_identity(entry),
                marker_text=marker_text,
                coverage=frozenset(coverage),
                entry=dict(entry),
                body=_entry_body(entry),
            )
        )
    return records


def _changed_fields(b: Mapping[str, Any], other: Mapping[str, Any]) -> set[str]:
    """Fields (including the canonical marker text) changed vs. base."""
    changed = set()
    keys = set(b) | set(other)
    for key in keys:
        if key == "marker":
            if _canonical_marker(b.get("marker")) != \
                    _canonical_marker(other.get("marker")):
                changed.add("marker")
            continue
        if key == "wheels":
            if _canonical_wheels(b.get("wheels", [])) != \
                    _canonical_wheels(other.get("wheels", [])):
                changed.add("wheels")
            continue
        if _canonical(b.get(key, _MISSING)) != _canonical(other.get(key, _MISSING)):
            changed.add(key)
    return changed


def _classify_divergence(path: str, b: Mapping[str, Any],
                         left_entry: Mapping[str, Any],
                         right_entry: Mapping[str, Any]) -> Conflict:
    """Give a divergent entry conflict a precise, security-aware kind.

    Only fields changed on *both* sides in incompatible ways get a specific
    kind; the atomic artifact unit always wins classification precedence.
    """
    left_changed = _changed_fields(b, left_entry)
    right_changed = _changed_fields(b, right_entry)
    hot = sorted(
        (left_changed & right_changed),
        key=lambda item: (
            # version first, then security fields, then python, marker, rest
            0 if item == "version"
            else 1 if item in _SECURITY_FIELDS
            else 2 if item == "requires-python"
            else 3 if item == "marker"
            else 4,
            item,
        ),
    )
    for field_name in hot:
        bv, lv, rv = (
            entry.get(field_name, _MISSING)
            for entry in (b, left_entry, right_entry)
        )
        if field_name == "version":
            return Conflict(
                f"{path}.version", KIND_VERSION_DIVERGENCE,
                f"{path}: version was modified differently on each side; the "
                "version, source and artifact hashes must stay one consistent unit",
                None if bv is _MISSING else bv,
                None if lv is _MISSING else lv,
                None if rv is _MISSING else rv,
            )
        if field_name in _SECURITY_FIELDS:
            return Conflict(
                f"{path}.{field_name}", KIND_ARTIFACT_DIVERGENCE,
                f"{path}: security-related {field_name} data (artifact hashes or "
                "source revision) were modified differently; hash sets are never "
                "merged automatically",
                None if bv is _MISSING else bv,
                None if lv is _MISSING else lv,
                None if rv is _MISSING else rv,
            )
        if field_name == "requires-python":
            return Conflict(
                f"{path}.requires-python", KIND_REQUIRES_PYTHON_DIVERGENCE,
                f"{path}: requires-python was modified differently on each side",
                None if bv is _MISSING else bv,
                None if lv is _MISSING else lv,
                None if rv is _MISSING else rv,
            )
        if field_name == "marker":
            return Conflict(
                f"{path}.marker", KIND_MARKER_TEXT_DIVERGENCE,
                f"{path}: marker text differs even though coverage over the target "
                "matrix is identical; equivalence outside the matrix is not assumed",
                b.get("marker"), left_entry.get("marker"),
                right_entry.get("marker"),
            )
    return Conflict(
        path, KIND_DIVERGENT_CHANGE,
        f"{path} was modified differently on each side "
        f"(left changed {sorted(left_changed)}, right changed "
        f"{sorted(right_changed)}); refusing to combine independent entry edits",
        b, left_entry, right_entry,
    )


def _merge_package_entries(
    base_records: list[_EntryRecord],
    left_records: list[_EntryRecord],
    right_records: list[_EntryRecord],
    conflicts: list[Conflict],
) -> list[dict[str, Any]]:
    by_side = []
    for records in (base_records, left_records, right_records):
        mapped: dict[tuple, _EntryRecord] = {}
        for record in records:
            if record.signature in mapped:
                conflicts.append(
                    Conflict(record.path, KIND_DUPLICATE_SIGNATURE,
                             f"{record.side} lists two entries for package "
                             f"{record.canonical_name!r} with the same source and "
                             "the same target-matrix coverage; they do not narrow "
                             "to one entry")
                )
                continue
            mapped[record.signature] = record
        by_side.append(mapped)
    base_map, left_map, right_map = by_side

    merged_entries: list[dict[str, Any]] = []
    signatures = set(base_map) | set(left_map) | set(right_map)
    for signature in sorted(signatures, key=lambda item: (item[0], _source_label(item[1]),
                                                          sorted(item[2]))):
        b_rec = base_map.get(signature)
        l_rec = left_map.get(signature)
        r_rec = right_map.get(signature)
        canonical_name, source, coverage = signature
        path = (
            f"packages.{canonical_name}[source={_source_label(source)}]"
            f"[targets={sorted(coverage) or 'none'}]"
        )

        if b_rec is None:
            if l_rec is not None and r_rec is not None:
                if l_rec.body == r_rec.body and l_rec.marker_text == r_rec.marker_text:
                    merged_entries.append(l_rec.entry)
                else:
                    conflicts.append(
                        Conflict(path, KIND_BOTH_ADDED_DIVERGENT,
                                 f"{path} was added differently on each side",
                                 None, l_rec.entry, r_rec.entry)
                    )
                continue
            one_sided = l_rec if l_rec is not None else r_rec
            assert one_sided is not None  # signature comes from one of the maps
            merged_entries.append(one_sided.entry)
            continue

        if l_rec is None and r_rec is None:
            continue  # deleted on both sides
        if l_rec is None:
            assert r_rec is not None  # both-None handled above
            if r_rec.body == b_rec.body and r_rec.marker_text == b_rec.marker_text:
                continue  # left-only deletion, right untouched
            conflicts.append(
                Conflict(path, KIND_DELETE_MODIFY,
                         f"{path} was deleted on the left but modified on the right",
                         b_rec.entry, None, r_rec.entry)
            )
            continue
        if r_rec is None:
            if l_rec.body == b_rec.body and l_rec.marker_text == b_rec.marker_text:
                continue  # right-only deletion, left untouched
            conflicts.append(
                Conflict(path, KIND_DELETE_MODIFY,
                         f"{path} was deleted on the right but modified on the left",
                         b_rec.entry, l_rec.entry, None)
            )
            continue

        # A marker text change that leaves coverage over the finite matrix
        # untouched could still change behavior on platforms outside the
        # matrix.  Refuse unless both sides independently made the exact same
        # textual edit; equivalence for unknown platforms is never assumed.
        marker_edited = (
            l_rec.marker_text != b_rec.marker_text
            or r_rec.marker_text != b_rec.marker_text
        )
        same_marker_edit_on_both = (
            l_rec.marker_text == r_rec.marker_text
            and l_rec.marker_text != b_rec.marker_text
        )
        if marker_edited and not same_marker_edit_on_both:
            conflicts.append(
                Conflict(f"{path}.marker", KIND_MARKER_TEXT_DIVERGENCE,
                         f"{path}: marker text changed without changing coverage "
                         "over the target matrix; behavior outside the matrix "
                         "cannot be determined, refusing automatic merge",
                         b_rec.entry.get("marker"),
                         l_rec.entry.get("marker"),
                         r_rec.entry.get("marker"))
            )
            continue

        l_same = l_rec.body == b_rec.body and l_rec.marker_text == b_rec.marker_text
        r_same = r_rec.body == b_rec.body and r_rec.marker_text == b_rec.marker_text
        if l_same and r_same:
            merged_entries.append(b_rec.entry)
        elif l_same:
            merged_entries.append(r_rec.entry)  # right-only change
        elif r_same:
            merged_entries.append(l_rec.entry)  # left-only change
        elif l_rec.body == r_rec.body and l_rec.marker_text == r_rec.marker_text:
            merged_entries.append(l_rec.entry)  # identical change on both sides
        else:
            conflicts.append(
                _classify_divergence(path, b_rec.entry, l_rec.entry, r_rec.entry)
            )
    return merged_entries


# --------------------------------------------------------------------------- #
# Deterministic output rendering
# --------------------------------------------------------------------------- #

_ENTRY_KEY_ORDER = (
    "name",
    "version",
    "requires-python",
    "marker",
    "index",
    "vcs",
    "directory",
    "archive",
    "wheels",
    "sdist",
    "tool",
)

_TOP_LEVEL_ORDER = (
    "lock-version",
    "environments",
    "requires-python",
    "extras",
    "dependency-groups",
    "default-groups",
    "created-by",
    "sources",
    "packages",
    "tool",
)


def _ordered_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key in _ENTRY_KEY_ORDER:
        if key in entry:
            value = entry[key]
            if key == "marker":
                # Emit the canonical spelling so quote/whitespace differences
                # between the inputs cannot make the output nondeterministic.
                value = _canonical_marker(value)
                if value is None:
                    continue
            elif key == "wheels" and isinstance(value, list):
                value = sorted(
                    (dict(wheel) for wheel in value),
                    key=_canonical,
                )
            elif isinstance(value, Mapping):
                value = dict(value)
            result[key] = value
    for key in sorted(set(entry) - set(_ENTRY_KEY_ORDER)):
        result[key] = entry[key]
    return result


def _sort_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(entry: Mapping[str, Any]):
        identity = _source_identity(entry)
        marker = _canonical_marker(entry.get("marker")) or ""
        return (canonicalize_name(str(entry.get("name", ""))),
                _source_label(identity), marker)

    return [_ordered_entry(entry) for entry in sorted(entries, key=key)]


# --------------------------------------------------------------------------- #
# Post-merge validation over the finite target matrix
# --------------------------------------------------------------------------- #


def _validate_merged(
    merged: dict[str, Any],
    targets: Sequence[TargetEnvironment],
    conflicts: list[Conflict],
    coverage: dict[str, TargetCoverage],
) -> None:
    environments = merged.get("environments", [])
    requires_python = merged.get("requires-python")
    extras = {canonicalize_name(name) for name in merged.get("extras", [])}
    groups = {
        canonicalize_name(name) for name in merged.get("dependency-groups", [])
    }
    default_groups = {
        canonicalize_name(name) for name in merged.get("default-groups", [])
    }

    conflicts.extend(
        Conflict("default-groups", KIND_DEFAULT_GROUP,
                 f"default-groups references {group!r} which is not "
                 "declared in dependency-groups")
        for group in sorted(default_groups)
        if group != "default" and group not in groups
    )

    # Every group/extra referenced by a retained marker must stay declared.
    for entry in merged.get("packages", []):
        name = canonicalize_name(str(entry.get("name", "")))
        marker = entry.get("marker")
        path = f"packages.{name}.marker"
        conflicts.extend(
            Conflict(path, KIND_UNDECLARED_GROUP,
                     f"packages.{name!r} marker selects dependency group "
                     f"{group!r}, which is not declared in the merged lock")
            for group in sorted(_marker_value_names(marker, "dependency_groups"))
            if group not in groups and group not in default_groups
        )
        conflicts.extend(
            Conflict(path, KIND_UNDECLARED_EXTRA,
                     f"packages.{name!r} marker selects extra {extra!r}, "
                     "which is not declared in the merged lock")
            for extra in sorted(_marker_value_names(marker, "extras"))
            if extra not in extras
        )

    target_names = [target.name for target in targets]
    for target in targets:
        proof = TargetCoverage(
            target=target.name,
            variables=dict(target.variables),
            extras=target.extras,
            dependency_groups=target.dependency_groups,
        )
        coverage[target.name] = proof

        if environments:
            matches = []
            for marker in environments:
                try:
                    matches.append(_evaluate(marker, target))
                except (InvalidMarker, UndefinedComparison,
                        UndefinedEnvironmentName) as exc:
                    conflicts.append(
                        Conflict("environments", KIND_INVALID_MARKER,
                                 f"environments marker {marker!r} cannot be "
                                 f"evaluated for target {target.name!r}: {exc}")
                    )
                    matches.append(False)
            proof.environments_matches = matches
            if not any(matches):
                conflicts.append(
                    Conflict("environments", KIND_TARGET_ENVIRONMENTS,
                             f"target {target.name!r} matches none of the merged "
                             "environments markers")
                )

        python_version = target.variables.get("python_full_version") or \
            target.variables.get("python_version")
        specifiers: list[tuple[str, str]] = []
        if requires_python:
            specifiers.append(("requires-python", requires_python))

        seen: dict[str, list[dict[str, Any]]] = {}
        for entry in merged.get("packages", []):
            name = canonicalize_name(str(entry.get("name", "")))
            marker = entry.get("marker")
            try:
                matched = _evaluate(marker, target)
            except (InvalidMarker, UndefinedComparison,
                    UndefinedEnvironmentName) as exc:
                conflicts.append(
                    Conflict(f"packages.{name}.marker", KIND_INVALID_MARKER,
                             f"marker for {name!r} cannot be evaluated for target "
                             f"{target.name!r}: {exc}")
                )
                matched = False
            proof.entries.append(
                {
                    "name": name,
                    "source": _source_label(_source_identity(entry)),
                    "marker": _canonical_marker(marker),
                    "matched": matched,
                    "version": entry.get("version"),
                }
            )
            if not matched:
                continue
            seen.setdefault(name, []).append(entry)
            if entry.get("requires-python"):
                specifiers.append(
                    (f"packages.{name}.requires-python", entry["requires-python"])
                )

        for name, matching in sorted(seen.items()):
            if len(matching) > 1:
                descriptions = [
                    f"source={_source_label(_source_identity(item))}, "
                    f"marker={item.get('marker') or '<unconditional>'}"
                    for item in matching
                ]
                conflicts.append(
                    Conflict(f"packages.{name}", KIND_OVERLAP,
                             f"target {target.name!r} has {len(matching)} overlapping "
                             f"entries for {name!r} that do not narrow to one "
                             f"installation choice: {descriptions}")
                )
            else:
                chosen = matching[0]
                proof.selected[name] = {
                    "version": chosen.get("version"),
                    "source": _source_label(_source_identity(chosen)),
                    "marker": _canonical_marker(chosen.get("marker")),
                    "has_artifact_hashes": _entry_has_artifact_hashes(chosen),
                }

        for path, specifier_text in specifiers:
            if python_version is None:
                conflicts.append(
                    Conflict(path, KIND_COVERAGE_UNDETERMINED,
                             f"cannot check {path} for target {target.name!r}: "
                             "the target provides neither python_full_version nor "
                             "python_version")
                )
                continue
            try:
                specifier = SpecifierSet(specifier_text)
            except InvalidSpecifier:
                conflicts.append(
                    Conflict(path, KIND_INVALID_INPUT,
                             f"invalid requires-python specifier {specifier_text!r}")
                )
                continue
            if not specifier.contains(python_version, prereleases=True):
                conflicts.append(
                    Conflict(path, KIND_TARGET_REQUIRES_PYTHON,
                             f"target {target.name!r} Python {python_version} does "
                             f"not satisfy {path} = {specifier_text!r}")
                )

    # Defensive: coverage must exist for every declared target, and only those.
    if set(coverage) != set(target_names):
        missing = sorted(set(target_names) - set(coverage))
        raise PylockMergeError(f"internal coverage gap for targets: {missing}")


def _entry_has_artifact_hashes(entry: Mapping[str, Any]) -> bool:
    wheels = entry.get("wheels") or []
    if any(wheel.get("hashes") for wheel in wheels):
        return True
    sdist = entry.get("sdist")
    if isinstance(sdist, Mapping) and sdist.get("hashes"):
        return True
    archive = entry.get("archive")
    if isinstance(archive, Mapping) and archive.get("hashes"):
        return True
    vcs = entry.get("vcs")
    if isinstance(vcs, Mapping) and vcs.get("commit-id"):
        return True
    return False


def _integrity_notes(merged: Mapping[str, Any],
                     coverage: Mapping[str, TargetCoverage]) -> list[str]:
    notes = [
        (
            "structural merge only: dependency closure was not re-resolved and "
            "cannot be proven from lock contents alone"
        ),
    ]
    unverified = set()
    for proof in coverage.values():
        for name, selection in proof.selected.items():
            if not selection["has_artifact_hashes"]:
                unverified.add(name)
    notes.extend(
        (
            f"package {name!r} is selected on at least one target without "
            "recorded artifact hashes or a pinned VCS commit-id; artifact "
            "integrity for that choice is not provable from the lock"
        )
        for name in sorted(unverified)
    )
    return notes


# --------------------------------------------------------------------------- #
# Top-level field merging
# --------------------------------------------------------------------------- #


def _merge_sources(
    base: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    conflicts: list[Conflict],
) -> list[dict[str, Any]]:
    def identity(source: Mapping[str, Any]) -> Any:
        return source.get("name") or source.get("url")

    def index_map(data: Mapping[str, Any]):
        mapped = {}
        for source in data.get("sources", []) or []:
            mapped[identity(source)] = source
        return mapped

    b_map, l_map, r_map = index_map(base), index_map(left), index_map(right)
    merged = []
    for key in sorted(set(b_map) | set(l_map) | set(r_map), key=str):
        path = f"sources[{key!r}]"
        _, value, conflict = _three_way_scalar(
            path,
            b_map.get(key, _MISSING),
            l_map.get(key, _MISSING),
            r_map.get(key, _MISSING),
            KIND_FIELD_DIVERGENCE,
        )
        if conflict is not None:
            conflicts.append(conflict)
        elif value is not _MISSING:
            merged.append(value)
    return merged


def _unknown_key_value(
    key: str,
    base: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    policy: str,
    conflicts: list[Conflict],
) -> Any:
    """Apply the explicit unknown-extension policy to one unknown key."""
    bv, lv, rv = (data.get(key, _MISSING) for data in (base, left, right))
    if policy == UnknownExtensionPolicy.ERROR:
        conflicts.append(
            Conflict(key, KIND_UNKNOWN_EXTENSION,
                     f"unknown extension {key!r} has no declared merge policy; "
                     "pass unknown_extension_policy='keep' or 'drop' explicitly")
        )
        return _MISSING
    if policy == UnknownExtensionPolicy.DROP:
        return _MISSING
    # KEEP: generic structural preservation, conflicts only on divergence.
    if any(isinstance(value, Mapping) for value in (bv, lv, rv)
           if value is not _MISSING):
        return _three_way_mapping(
            key,
            bv if isinstance(bv, Mapping) else {},
            lv if isinstance(lv, Mapping) else {},
            rv if isinstance(rv, Mapping) else {},
            conflicts,
        )
    _, value, conflict = _three_way_scalar(
        key, bv, lv, rv, KIND_UNKNOWN_EXTENSION
    )
    if conflict is not None:
        conflicts.append(conflict)
    return value


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def _input_validation_conflicts(
    inputs: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[Conflict]:
    conflicts: list[Conflict] = []
    for side, data in inputs:
        if not data.get("lock-version"):
            conflicts.append(
                Conflict("lock-version", KIND_INVALID_INPUT,
                         f"{side} input is missing required 'lock-version'")
            )
        if "packages" not in data:
            conflicts.append(
                Conflict("packages", KIND_INVALID_INPUT,
                         f"{side} input is missing required 'packages' array")
            )
    versions = {side: str(data.get("lock-version")) for side, data in inputs}
    if len(set(versions.values())) > 1:
        conflicts.append(
            Conflict("lock-version", KIND_LOCK_VERSION,
                     "the three locks use different lock-versions",
                     versions["base"], versions["left"], versions["right"])
        )
    return conflicts


def _marker_coverage_conflicts(
    inputs: Sequence[tuple[str, Mapping[str, Any]]],
    targets: Sequence[TargetEnvironment],
) -> list[Conflict]:
    """Refuse the merge when matrix coverage cannot be proven for any marker."""
    all_markers: list[tuple[str, str]] = []
    for side, data in inputs:
        all_markers.extend(
            (f"{side}.environments", marker)
            for marker in data.get("environments", []) or []
        )
        for index, package in enumerate(data.get("packages", []) or []):
            if isinstance(package, Mapping) and package.get("marker"):
                all_markers.append(
                    (f"{side}.packages[{index}].marker", package["marker"])
                )
    conflicts: list[Conflict] = []
    referenced_variables: set[str] = set()
    for path, marker in all_markers:
        try:
            referenced_variables.update(_marker_variables(marker))
        except InvalidMarker as exc:
            conflicts.append(
                Conflict(path, KIND_INVALID_MARKER,
                         f"invalid marker {marker!r}: {exc}")
            )
    needed = referenced_variables - {"extras", "dependency_groups"}
    for target in targets:
        missing = sorted(needed - set(target.variables))
        if missing:
            conflicts.append(
                Conflict(
                    f"targets.{target.name}",
                    KIND_COVERAGE_UNDETERMINED,
                    f"target {target.name!r} does not define marker variable(s) "
                    f"{missing}; coverage outside the explicitly supplied matrix "
                    "cannot be determined, refusing automatic merge",
                )
            )
    return conflicts


def _unknown_package_key_conflicts(
    inputs: Sequence[tuple[str, Mapping[str, Any]]],
    policy: str,
) -> list[Conflict]:
    if policy != UnknownExtensionPolicy.ERROR:
        return []
    unknown_keys: set[str] = set()
    for _side, data in inputs:
        for package in data.get("packages", []) or []:
            if isinstance(package, Mapping):
                unknown_keys.update(
                    key for key in package if key not in STANDARD_PACKAGE_KEYS
                )
    return [
        Conflict(f"packages.*.{key}", KIND_UNKNOWN_EXTENSION,
                 f"unknown package extension {key!r} has no declared merge "
                 "policy; pass unknown_extension_policy='keep' or 'drop' "
                 "explicitly")
        for key in sorted(unknown_keys)
    ]


def _merge_standard_top_level(
    base_data: Mapping[str, Any],
    left_data: Mapping[str, Any],
    right_data: Mapping[str, Any],
    conflicts: list[Conflict],
) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "lock-version": base_data.get("lock-version"),
    }
    environments = _three_way_set(
        "environments",
        base_data.get("environments"),
        left_data.get("environments"),
        right_data.get("environments"),
    )
    canonical_environments = {
        _canonical_marker(marker) for marker in environments if marker
    }
    sorted_environments = sorted(
        marker for marker in canonical_environments if marker is not None
    )
    if sorted_environments:
        merged["environments"] = sorted_environments

    for key in ("requires-python", "created-by"):
        _, value, conflict = _three_way_scalar(
            key,
            base_data.get(key, _MISSING),
            left_data.get(key, _MISSING),
            right_data.get(key, _MISSING),
            KIND_FIELD_DIVERGENCE,
        )
        if conflict is not None:
            conflicts.append(conflict)
        elif value is not _MISSING:
            merged[key] = value

    extras = _three_way_set(
        "extras", base_data.get("extras"),
        left_data.get("extras"), right_data.get("extras"),
    )
    if extras:
        merged["extras"] = sorted(extras)
    groups = _three_way_set(
        "dependency-groups", base_data.get("dependency-groups"),
        left_data.get("dependency-groups"),
        right_data.get("dependency-groups"),
    )
    if groups:
        merged["dependency-groups"] = sorted(groups)
    _, default_groups, conflict = _three_way_scalar(
        "default-groups",
        base_data.get("default-groups", _MISSING),
        left_data.get("default-groups", _MISSING),
        right_data.get("default-groups", _MISSING),
        KIND_FIELD_DIVERGENCE,
    )
    if conflict is not None:
        conflicts.append(conflict)
    elif default_groups is not _MISSING:
        merged["default-groups"] = list(default_groups)

    if any("sources" in data for data in (base_data, left_data, right_data)):
        sources = _merge_sources(base_data, left_data, right_data, conflicts)
        if sources:
            merged["sources"] = sources
    return merged


def _merge_tool_section(
    base_data: Mapping[str, Any],
    left_data: Mapping[str, Any],
    right_data: Mapping[str, Any],
    policy: str,
    supported_tools: frozenset[str],
    conflicts: list[Conflict],
) -> dict[str, Any]:
    tool_conflicts: list[Conflict] = []
    merged_tool: dict[str, Any] = {}
    tool_keys: set[str] = set()
    for data in (base_data, left_data, right_data):
        tool = data.get("tool")
        if isinstance(tool, Mapping):
            tool_keys.update(tool.keys())
    for table in sorted(tool_keys):
        if table not in supported_tools:
            value = _unknown_key_value(
                f"tool.{table}", base_data, left_data, right_data,
                policy, conflicts,
            )
            if value is not _MISSING and isinstance(value, Mapping) and value:
                merged_tool[table] = value
            continue
        merged_table = _three_way_mapping(
            f"tool.{table}",
            base_data.get("tool", {}).get(table, {}),
            left_data.get("tool", {}).get(table, {}),
            right_data.get("tool", {}).get(table, {}),
            tool_conflicts,
        )
        if merged_table:
            merged_tool[table] = merged_table
    conflicts.extend(tool_conflicts)
    return merged_tool


def _merge_unknown_top_level(
    base_data: Mapping[str, Any],
    left_data: Mapping[str, Any],
    right_data: Mapping[str, Any],
    policy: str,
    conflicts: list[Conflict],
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in sorted(set(base_data) | set(left_data) | set(right_data)):
        if key in STANDARD_TOP_LEVEL_KEYS:
            continue
        value = _unknown_key_value(
            key, base_data, left_data, right_data, policy, conflicts,
        )
        if value is not _MISSING:
            merged[key] = value
    return merged


def _matrix_basis(targets: Sequence[TargetEnvironment]) -> dict[str, TargetCoverage]:
    return {
        target.name: TargetCoverage(
            target=target.name,
            variables=dict(target.variables),
            extras=target.extras,
            dependency_groups=target.dependency_groups,
        )
        for target in targets
    }


def three_way_merge(
    base: Mapping[str, Any] | PylockFile,
    left: Mapping[str, Any] | PylockFile,
    right: Mapping[str, Any] | PylockFile,
    targets: Sequence[TargetEnvironment],
    *,
    unknown_extension_policy: str = UnknownExtensionPolicy.ERROR,
    supported_tool_tables: Sequence[str] = DEFAULT_SUPPORTED_TOOL_TABLES,
) -> MergeResult:
    """Merge three same-origin pylock locks over an explicit target matrix.

    The three inputs are never mutated and nothing is written to disk.  On
    success the result's ``merged`` mapping is a complete pylock document and
    ``coverage`` proves marker behavior for every requested target; on failure
    ``merged`` is ``None`` and ``conflicts`` explains every blocking path.
    """
    if unknown_extension_policy not in UnknownExtensionPolicy.ALL:
        raise PylockMergeError(
            f"unknown_extension_policy must be one of "
            f"{sorted(UnknownExtensionPolicy.ALL)}"
        )
    if not targets:
        raise PylockMergeError(
            "at least one TargetEnvironment is required: marker coverage is "
            "proven only over the explicit target matrix"
        )
    names = [target.name for target in targets]
    if len(set(names)) != len(names):
        raise PylockMergeError("target environment names must be unique")

    base_data, left_data, right_data = (
        _as_data(item) for item in (base, left, right)
    )
    inputs = (("base", base_data), ("left", left_data), ("right", right_data))
    conflicts: list[Conflict] = []
    conflicts.extend(_input_validation_conflicts(inputs))
    conflicts.extend(_marker_coverage_conflicts(inputs, targets))
    conflicts.extend(
        _unknown_package_key_conflicts(inputs, unknown_extension_policy)
    )

    # --- per-side entry records and the entry-level three-way merge --------- #
    base_records = _build_records("base", base_data, targets, conflicts)
    left_records = _build_records("left", left_data, targets, conflicts)
    right_records = _build_records("right", right_data, targets, conflicts)
    merged_packages = _merge_package_entries(
        base_records, left_records, right_records, conflicts
    )
    if unknown_extension_policy == UnknownExtensionPolicy.DROP:
        merged_packages = [
            {key: value for key, value in entry.items()
             if key in STANDARD_PACKAGE_KEYS}
            for entry in merged_packages
        ]

    # --- top-level fields, tool tables and unknown extensions --------------- #
    merged = _merge_standard_top_level(
        base_data, left_data, right_data, conflicts
    )
    supported_tools = frozenset(supported_tool_tables)
    merged_tool = _merge_tool_section(
        base_data, left_data, right_data,
        unknown_extension_policy, supported_tools, conflicts,
    )
    if merged_tool:
        merged["tool"] = merged_tool
    merged.update(
        _merge_unknown_top_level(
            base_data, left_data, right_data,
            unknown_extension_policy, conflicts,
        )
    )
    merged["packages"] = _sort_entries(merged_packages)

    # --- coverage proof + whole-matrix validation --------------------------- #
    coverage: dict[str, TargetCoverage] = {}
    if not conflicts:
        _validate_merged(merged, targets, conflicts, coverage)
    else:
        # Still record the raw matrix so a failed run carries its proof basis.
        coverage = _matrix_basis(targets)

    if conflicts:
        # Deterministic conflict ordering for stable reports and tests.
        conflicts.sort(key=lambda item: (item.path, item.kind))
        return MergeResult(
            ok=False,
            merged=None,
            conflicts=conflicts,
            coverage=coverage,
            integrity="refused:conflicts",
            integrity_notes=[
                "automatic merge refused; no dependency resolution was attempted",
            ],
        )

    notes = _integrity_notes(merged, coverage)
    return MergeResult(
        ok=True,
        merged=merged,
        conflicts=[],
        coverage=coverage,
        integrity="unverified:structural-only",
        integrity_notes=notes,
    )
