"""Tests for the structural three-way PEP 751 pylock.toml merger.

The merger never writes files or re-resolves dependencies: every assertion
here works on in-memory lock dictionaries and the explicit target marker
matrix.  Conflict paths and per-target installation selections are checked
explicitly.
"""

import copy

import pytest

from pipenv.patched.pip._vendor.packaging.markers import Marker
from pipenv.utils.pylock_merge import (
    KIND_ARTIFACT_DIVERGENCE,
    KIND_COVERAGE_UNDETERMINED,
    KIND_DELETE_MODIFY,
    KIND_LOCK_VERSION,
    KIND_MARKER_TEXT_DIVERGENCE,
    KIND_OVERLAP,
    KIND_TARGET_ENVIRONMENTS,
    KIND_TARGET_REQUIRES_PYTHON,
    KIND_UNDECLARED_GROUP,
    KIND_UNKNOWN_EXTENSION,
    KIND_VERSION_DIVERGENCE,
    Conflict,
    PylockMergeError,
    TargetEnvironment,
    UnknownExtensionPolicy,
    three_way_merge,
)

PYPI = "https://pypi.org/simple/"

WIN_PY312 = TargetEnvironment(
    "win-py312",
    {
        "sys_platform": "win32",
        "platform_system": "Windows",
        "os_name": "nt",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "implementation_name": "cpython",
        "implementation_version": "3.12.4",
        "python_version": "3.12",
        "python_full_version": "3.12.4",
    },
    extras=frozenset({"crypto"}),
    dependency_groups=frozenset({"dev", "test"}),
)
LIN_PY310 = TargetEnvironment(
    "lin-py310",
    {
        "sys_platform": "linux",
        "platform_system": "Linux",
        "os_name": "posix",
        "platform_machine": "x86_64",
        "platform_python_implementation": "CPython",
        "implementation_name": "cpython",
        "implementation_version": "3.10.15",
        "python_version": "3.10",
        "python_full_version": "3.10.15",
    },
    extras=frozenset({"crypto"}),
    dependency_groups=frozenset({"dev", "test"}),
)
MAC_PY312 = TargetEnvironment(
    "mac-py312",
    {
        "sys_platform": "darwin",
        "platform_system": "Darwin",
        "os_name": "posix",
        "platform_machine": "arm64",
        "platform_python_implementation": "CPython",
        "implementation_name": "cpython",
        "implementation_version": "3.12.4",
        "python_version": "3.12",
        "python_full_version": "3.12.4",
    },
    extras=frozenset({"crypto"}),
    dependency_groups=frozenset({"dev", "test"}),
)
TARGETS = [WIN_PY312, LIN_PY310, MAC_PY312]


# --------------------------------------------------------------------------- #
# Lock / entry builders
# --------------------------------------------------------------------------- #


def wheel(name, digest):
    return {
        "name": name,
        "url": f"https://files.example.org/{name}",
        "hashes": {"sha256": digest},
    }


def pkg(name, version=None, marker=None, *, wheels=(), sdist=None,
        index=PYPI, requires_python=None, tool=None, **extra):
    entry = {"name": name, "index": index}
    if version is not None:
        entry["version"] = version
    if marker:
        entry["marker"] = marker
    if requires_python:
        entry["requires-python"] = requires_python
    if wheels:
        entry["wheels"] = list(wheels)
    if sdist:
        entry["sdist"] = sdist
    if tool:
        entry["tool"] = tool
    entry.update(extra)
    return entry


def vcs_pkg(name, commit, *, marker=None, url="https://github.com/example/repo.git"):
    vcs = {"type": "git", "url": url}
    if commit is not None:
        vcs["commit-id"] = commit
    entry = {"name": name, "vcs": vcs}
    if marker:
        entry["marker"] = marker
    return entry


def lockfile(*, packages=(), groups=(), extras=(), default_groups=("default",),
             requires_python=None, environments=(), tool=None,
             lock_version="1.0", created_by="pytest", **extra_top):
    data = {"lock-version": lock_version, "created-by": created_by}
    if environments:
        data["environments"] = list(environments)
    if requires_python:
        data["requires-python"] = requires_python
    if extras:
        data["extras"] = list(extras)
    if groups:
        data["dependency-groups"] = list(groups)
    if default_groups is not None:
        data["default-groups"] = list(default_groups)
    data["packages"] = list(packages)
    if tool:
        data["tool"] = tool
    data.update(extra_top)
    return data


def merge(base, left, right, targets=TARGETS, **kwargs):
    return three_way_merge(
        copy.deepcopy(base), copy.deepcopy(left), copy.deepcopy(right),
        targets, **kwargs,
    )


def find_conflict(result, *, kind=None, path_contains=None):
    for conflict in result.conflicts:
        if kind is not None and conflict.kind != kind:
            continue
        if path_contains is not None and path_contains not in conflict.path:
            continue
        return conflict
    raise AssertionError(
        f"no conflict matching kind={kind!r}, path_contains={path_contains!r}; "
        f"got {[(c.kind, c.path) for c in result.conflicts]}"
    )


def merged_names(result):
    return [package["name"] for package in result.merged["packages"]]


def entry_for(result, name, marker=None):
    matches = [
        package for package in result.merged["packages"]
        if package["name"].lower() == name.lower()
        and (marker is None or package.get("marker") == marker)
    ]
    assert len(matches) == 1, f"expected one {name} entry, got {matches}"
    return matches[0]


# --------------------------------------------------------------------------- #
# Baseline behaviour
# --------------------------------------------------------------------------- #


def test_identical_locks_merge_without_conflicts():
    base = lockfile(packages=[pkg("requests", "2.31.0", wheels=[
        wheel("requests-2.31.0-py3-none-any.whl", "a" * 64),
    ])])

    result = merge(base, copy.deepcopy(base), copy.deepcopy(base))

    assert result.ok
    assert result.conflicts == []
    assert merged_names(result) == ["requests"]
    assert result.integrity == "unverified:structural-only"
    assert result.coverage["win-py312"].selected["requests"]["version"] == "2.31.0"


def test_empty_target_matrix_is_rejected():
    base = lockfile()
    with pytest.raises(PylockMergeError, match="at least one TargetEnvironment"):
        three_way_merge(base, copy.deepcopy(base), copy.deepcopy(base), [])


def test_unknown_policy_value_is_rejected():
    base = lockfile()
    with pytest.raises(PylockMergeError, match="unknown_extension_policy"):
        merge(base, copy.deepcopy(base), copy.deepcopy(base),
              unknown_extension_policy="bogus")


def test_lock_version_mismatch_conflicts():
    base = lockfile(lock_version="1.0")
    left = lockfile(lock_version="1.1")
    right = lockfile(lock_version="1.0")

    result = merge(base, left, right)

    assert not result.ok
    assert result.merged is None
    assert find_conflict(result, kind=KIND_LOCK_VERSION)


# --------------------------------------------------------------------------- #
# Independent group / extra edits
# --------------------------------------------------------------------------- #


def test_independent_group_and_extra_changes_merge():
    base = lockfile(
        groups=["dev"],
        extras=["crypto"],
        packages=[
            pkg("requests", "2.31.0"),
            pkg("pytest", "7.2.0", marker="'dev' in dependency_groups"),
            pkg("cryptography", "40.0.0", marker="'crypto' in extras"),
        ],
    )
    left = copy.deepcopy(base)
    left["dependency-groups"].append("test")
    left["packages"].append(
        pkg("tox", "4.0.0", marker="'test' in dependency_groups")
    )
    right = copy.deepcopy(base)
    right["dependency-groups"].append("docs")
    right["packages"].append(
        pkg("sphinx", "7.1.0", marker="'docs' in dependency_groups")
    )

    result = merge(base, left, right)

    assert result.ok, result.conflicts
    assert result.merged["dependency-groups"] == ["dev", "docs", "test"]
    assert set(merged_names(result)) == {
        "requests", "pytest", "cryptography", "tox", "sphinx",
    }
    win_selected = result.coverage["win-py312"].selected
    # The target matrix selects dev+test and the crypto extra, but not docs.
    assert set(win_selected) == {"requests", "pytest", "cryptography", "tox"}
    assert win_selected["tox"]["version"] == "4.0.0"
    assert "sphinx" not in win_selected
    # Every target sees the same group visibility for these entries.
    for proof in result.coverage.values():
        assert "sphinx" not in proof.selected
        assert "requests" in proof.selected


def test_removing_a_group_while_its_package_remains_conflicts():
    base = lockfile(
        groups=["dev"],
        packages=[pkg("pytest", "7.2.0", marker="'dev' in dependency_groups")],
    )
    left = copy.deepcopy(base)
    left["dependency-groups"] = []  # group dropped, dev-only package retained
    right = copy.deepcopy(base)

    result = merge(base, left, right)

    assert not result.ok
    conflict = find_conflict(result, kind=KIND_UNDECLARED_GROUP)
    assert "pytest" in conflict.path
    assert "'dev'" in conflict.message


# --------------------------------------------------------------------------- #
# Same package, different environments
# --------------------------------------------------------------------------- #


def test_same_package_cross_environment_versions_merge_independently():
    old_marker = "python_version < '3.12'"
    new_marker = "python_version >= '3.12'"
    base = lockfile(packages=[
        pkg("numpy", "1.26.4", marker=old_marker),
        pkg("numpy", "2.3.0", marker=new_marker),
    ])
    left = copy.deepcopy(base)
    left["packages"][0]["version"] = "1.26.5"  # bump the old line only
    right = copy.deepcopy(base)
    right["packages"][1]["version"] = "2.3.1"  # bump the new line only

    result = merge(base, left, right)

    assert result.ok, result.conflicts
    lin = result.coverage["lin-py310"].selected["numpy"]
    win = result.coverage["win-py312"].selected["numpy"]
    assert lin == {"version": "1.26.5", "source": "index/https://pypi.org/simple",
                   "marker": 'python_version < "3.12"',
                   "has_artifact_hashes": False}
    assert win["version"] == "2.3.1"
    assert win["marker"] == 'python_version >= "3.12"'
    # The linux target must not be able to select the 3.12 entry.
    assert all(
        entry["matched"] is False
        for entry in result.coverage["lin-py310"].entries
        if entry["name"] == "numpy" and entry["version"] == "2.3.1"
    )


def test_same_environment_different_versions_conflicts():
    base = lockfile(packages=[pkg("requests", "2.31.0")])
    left = copy.deepcopy(base)
    left["packages"][0]["version"] = "2.32.0"
    right = copy.deepcopy(base)
    right["packages"][0]["version"] = "2.33.0"

    result = merge(base, left, right)

    assert not result.ok
    conflict = find_conflict(result, kind=KIND_VERSION_DIVERGENCE)
    assert "requests" in conflict.path
    assert conflict.left == "2.32.0"
    assert conflict.right == "2.33.0"


# --------------------------------------------------------------------------- #
# Delete / modify semantics
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("deleted_side", ["left", "right"])
def test_delete_vs_modify_conflicts(deleted_side):
    base = lockfile(packages=[pkg("requests", "2.31.0")])
    left = copy.deepcopy(base)
    right = copy.deepcopy(base)
    if deleted_side == "left":
        left["packages"] = []
        right["packages"][0]["version"] = "2.32.0"
    else:
        right["packages"] = []
        left["packages"][0]["version"] = "2.32.0"

    result = merge(base, left, right)

    assert not result.ok
    conflict = find_conflict(result, kind=KIND_DELETE_MODIFY)
    assert "requests" in conflict.path
    assert "deleted" in conflict.message


def test_one_sided_deletion_is_kept():
    base = lockfile(packages=[pkg("requests", "2.31.0"), pkg("click", "8.1.0")])
    left = copy.deepcopy(base)
    left["packages"] = [entry for entry in left["packages"]
                        if entry["name"] != "requests"]
    right = copy.deepcopy(base)

    result = merge(base, left, right)

    assert result.ok, result.conflicts
    assert merged_names(result) == ["click"]


def test_deletion_on_both_sides_removes_entry_once():
    base = lockfile(packages=[pkg("requests", "2.31.0"), pkg("click", "8.1.0")])
    left = copy.deepcopy(base)
    left["packages"] = [entry for entry in left["packages"]
                        if entry["name"] != "requests"]
    right = copy.deepcopy(base)
    right["packages"] = [entry for entry in right["packages"]
                         if entry["name"] != "requests"]

    result = merge(base, left, right)

    assert result.ok
    assert merged_names(result) == ["click"]


def test_same_change_on_both_sides_merges_once():
    base = lockfile(packages=[pkg("requests", "2.31.0")])
    left = copy.deepcopy(base)
    right = copy.deepcopy(base)
    left["packages"][0]["version"] = "2.32.0"
    right["packages"][0]["version"] = "2.32.0"

    result = merge(base, left, right)

    assert result.ok
    assert entry_for(result, "requests")["version"] == "2.32.0"
    assert len(result.merged["packages"]) == 1


# --------------------------------------------------------------------------- #
# VCS revisions and artifact hash handling
# --------------------------------------------------------------------------- #


def test_vcs_ref_changed_differently_conflicts_as_artifact_divergence():
    base = lockfile(packages=[vcs_pkg("gitpkg", "a" * 40)])
    left = lockfile(packages=[vcs_pkg("gitpkg", "b" * 40)])
    right = lockfile(packages=[vcs_pkg("gitpkg", "c" * 40)])

    result = merge(base, left, right)

    assert not result.ok
    conflict = find_conflict(result, kind=KIND_ARTIFACT_DIVERGENCE)
    assert "gitpkg" in conflict.path
    assert conflict.left["commit-id"] == "b" * 40
    assert conflict.right["commit-id"] == "c" * 40


def test_vcs_ref_same_change_merges_and_one_sided_change_kept():
    base = lockfile(packages=[vcs_pkg("gitpkg", "a" * 40)])
    left = lockfile(packages=[vcs_pkg("gitpkg", "b" * 40)])
    same_both = merge(base, left, lockfile(packages=[vcs_pkg("gitpkg", "b" * 40)]))
    assert same_both.ok
    assert entry_for(same_both, "gitpkg")["vcs"]["commit-id"] == "b" * 40

    right_untouched = merge(base, left, copy.deepcopy(base))
    assert right_untouched.ok
    assert entry_for(right_untouched, "gitpkg")["vcs"]["commit-id"] == "b" * 40


def test_version_and_hashes_travel_as_one_unit_no_trust_set_expansion():
    h1, h2 = "1" * 64, "2" * 64
    base = lockfile(packages=[
        pkg("requests", "1.0.0", wheels=[wheel("requests-1.0.0-py3-none-any.whl", h1)])
    ])
    left = lockfile(packages=[
        pkg("requests", "2.0.0", wheels=[wheel("requests-2.0.0-py3-none-any.whl", h2)])
    ])
    right = copy.deepcopy(base)

    result = merge(base, left, right)

    assert result.ok, result.conflicts
    merged_entry = entry_for(result, "requests")
    assert merged_entry["version"] == "2.0.0"
    assert merged_entry["wheels"] == [
        wheel("requests-2.0.0-py3-none-any.whl", h2)
    ]
    # The stale base hash must never survive next to the new version.
    serialized = str(result.merged)
    assert h1 not in serialized
    assert h2 in serialized


def test_hash_sets_changed_differently_conflict_and_are_never_unioned():
    h2, h3 = "2" * 64, "3" * 64
    base = lockfile(packages=[
        pkg("requests", "1.0.0", wheels=[wheel("requests-1.0.0-py3-none-any.whl", "1" * 64)])
    ])
    left = copy.deepcopy(base)
    left["packages"][0]["wheels"] = [
        wheel("requests-1.0.0-py3-none-any.whl", h2)
    ]
    right = copy.deepcopy(base)
    right["packages"][0]["wheels"] = [
        wheel("requests-1.0.0-py3-none-any.whl", h3)
    ]

    result = merge(base, left, right)

    assert not result.ok
    assert find_conflict(result, kind=KIND_ARTIFACT_DIVERGENCE)
    # No merged document means no chance of an enlarged trusted hash set.
    assert result.merged is None


def test_unpinned_vcs_revision_is_flagged_in_integrity_notes():
    base = lockfile(packages=[vcs_pkg("floatrepo", None)])

    result = merge(base, copy.deepcopy(base), copy.deepcopy(base))

    assert result.ok
    assert any("floatrepo" in note for note in result.integrity_notes)
    assert result.integrity == "unverified:structural-only"


# --------------------------------------------------------------------------- #
# Marker overlap / coverage
# --------------------------------------------------------------------------- #


def test_marker_overlap_on_a_target_conflicts_with_paths():
    base = lockfile(packages=[
        pkg("colorama", "0.4.6", marker="sys_platform == 'win32'")
    ])
    left = copy.deepcopy(base)
    left["packages"].append(pkg("colorama", "0.5.0"))  # unconditional entry
    right = copy.deepcopy(base)

    result = merge(base, left, right)

    assert not result.ok
    conflict = find_conflict(result, kind=KIND_OVERLAP)
    assert "colorama" in conflict.path
    assert "win-py312" in conflict.message


def test_marker_text_equivalent_over_matrix_canonicalizes_without_conflict():
    base = lockfile(packages=[
        pkg("requests", "2.31.0", marker='sys_platform == "win32"')
    ])
    left = copy.deepcopy(base)
    left["packages"][0]["marker"] = "sys_platform == 'win32'"  # cosmetic only
    right = copy.deepcopy(base)

    result = merge(base, left, right)

    assert result.ok, result.conflicts
    assert entry_for(result, "requests")["marker"] == 'sys_platform == "win32"'


def test_different_marker_text_with_equal_matrix_coverage_refused():
    base = lockfile(packages=[
        pkg("requests", "2.31.0", marker="sys_platform == 'win32'")
    ])
    new_marker = "sys_platform == 'win32' or python_full_version == '9.9.9'"
    left = copy.deepcopy(base)
    # Same truth value on every matrix target, different text and different
    # meaning on platforms outside the matrix: must not be auto-merged.
    left["packages"][0]["marker"] = new_marker
    right = copy.deepcopy(base)

    result = merge(base, left, right)

    assert not result.ok
    assert find_conflict(result, kind=KIND_MARKER_TEXT_DIVERGENCE)

    # The exact same textual edit on both sides is a shared decision and is
    # accepted; no equivalence for unknown platforms has to be invented.
    both = merge(base, left, copy.deepcopy(left))
    assert both.ok, both.conflicts
    assert entry_for(both, "requests")["marker"] == str(Marker(new_marker))


def test_missing_target_variable_refuses_automatic_merge():
    base = lockfile(packages=[
        pkg("cryptography", "40.0.0", marker="platform_machine == 'x86_64'")
    ])
    narrow_targets = [
        TargetEnvironment("win-only", {
            "sys_platform": "win32",
            "python_version": "3.12",
            "python_full_version": "3.12.4",
        })
    ]

    result = merge(base, copy.deepcopy(base), copy.deepcopy(base), narrow_targets)

    assert not result.ok
    conflict = find_conflict(result, kind=KIND_COVERAGE_UNDETERMINED)
    assert "platform_machine" in conflict.message
    assert "win-only" in conflict.message


def test_environments_must_cover_every_target():
    base = lockfile(
        environments=["sys_platform == 'win32'", "sys_platform == 'linux'"],
        packages=[pkg("requests", "2.31.0")],
    )

    result = merge(base, copy.deepcopy(base), copy.deepcopy(base))

    assert not result.ok
    conflict = find_conflict(result, kind=KIND_TARGET_ENVIRONMENTS)
    assert "mac-py312" in conflict.message
    win_proof = result.coverage["win-py312"]
    # The proof stays aligned with the environments markers in their merged
    # (canonicalized, deterministically sorted) order.
    merged_environments = sorted(
        str(Marker(m)) for m in base["environments"]
    )
    win_matches = dict(zip(
        merged_environments, win_proof.environments_matches, strict=True
    ))
    assert win_matches == {
        'sys_platform == "linux"': False,
        'sys_platform == "win32"': True,
    }


# --------------------------------------------------------------------------- #
# requires-python
# --------------------------------------------------------------------------- #


def test_requires_python_change_excluding_a_target_conflicts():
    base = lockfile(
        requires_python=">=3.10",
        packages=[pkg("requests", "2.31.0")],
    )
    left = copy.deepcopy(base)
    left["requires-python"] = ">=3.11"
    right = copy.deepcopy(base)

    result = merge(base, left, right)

    assert not result.ok
    conflict = find_conflict(result, kind=KIND_TARGET_REQUIRES_PYTHON)
    assert "lin-py310" in conflict.message
    assert ">=3.11" in conflict.message


def test_requires_python_changed_differently_conflicts():
    base = lockfile(requires_python=">=3.10", packages=[pkg("requests", "2.31.0")])
    left = copy.deepcopy(base)
    left["requires-python"] = ">=3.11"
    right = copy.deepcopy(base)
    right["requires-python"] = ">=3.12"

    result = merge(base, left, right)

    assert not result.ok
    assert any(conflict.path == "requires-python" for conflict in result.conflicts)


# --------------------------------------------------------------------------- #
# Unknown extension policy
# --------------------------------------------------------------------------- #


def test_unknown_top_level_extension_needs_explicit_policy():
    base = lockfile(x_custom={"fingerprint": "abc"})

    errored = merge(base, copy.deepcopy(base), copy.deepcopy(base))
    assert not errored.ok
    assert find_conflict(errored, kind=KIND_UNKNOWN_EXTENSION,
                         path_contains="x_custom")

    dropped = merge(base, copy.deepcopy(base), copy.deepcopy(base),
                    unknown_extension_policy=UnknownExtensionPolicy.DROP)
    assert dropped.ok
    assert "x_custom" not in dropped.merged

    kept = merge(base, copy.deepcopy(base), copy.deepcopy(base),
                 unknown_extension_policy=UnknownExtensionPolicy.KEEP)
    assert kept.ok
    assert kept.merged["x_custom"] == {"fingerprint": "abc"}


def test_unknown_package_extension_policy():
    base = lockfile(packages=[pkg("requests", "2.31.0", x_vendor={"note": "z"})])

    errored = merge(base, copy.deepcopy(base), copy.deepcopy(base))
    assert find_conflict(errored, kind=KIND_UNKNOWN_EXTENSION)

    dropped = merge(base, copy.deepcopy(base), copy.deepcopy(base),
                    unknown_extension_policy=UnknownExtensionPolicy.DROP)
    assert dropped.ok
    assert "x_vendor" not in entry_for(dropped, "requests")

    kept = merge(base, copy.deepcopy(base), copy.deepcopy(base),
                 unknown_extension_policy=UnknownExtensionPolicy.KEEP)
    assert entry_for(kept, "requests")["x_vendor"] == {"note": "z"}


def test_kept_unknown_extension_still_conflicts_on_divergence():
    base = lockfile(x_custom = 1)
    left = lockfile(x_custom=2)
    right = lockfile(x_custom=3)

    result = merge(base, left, right,
                   unknown_extension_policy=UnknownExtensionPolicy.KEEP)

    assert not result.ok
    assert find_conflict(result, path_contains="x_custom")


def test_unknown_tool_table_uses_policy_supported_table_deep_merges():
    base = lockfile(
        packages=[pkg("requests", "2.31.0")],
        tool={"pipenv": {"categories": {"a": 1}}, "othertool": {"x": 1}},
    )
    left = copy.deepcopy(base)
    left["tool"]["pipenv"]["categories"]["a"] = 2
    right = copy.deepcopy(base)
    right["tool"]["pipenv"]["categories"]["b"] = 3

    assert find_conflict(
        merge(base, left, right), kind=KIND_UNKNOWN_EXTENSION,
        path_contains="tool.othertool",
    )

    dropped = merge(base, left, right,
                    unknown_extension_policy=UnknownExtensionPolicy.DROP)
    assert dropped.ok
    assert set(dropped.merged["tool"]) == {"pipenv"}
    assert dropped.merged["tool"]["pipenv"]["categories"] == {"a": 2, "b": 3}

    divergent = copy.deepcopy(base)
    divergent["tool"]["pipenv"]["categories"]["a"] = 9
    conflict_result = merge(base, left, divergent)
    assert any(
        conflict.path == "tool.pipenv.categories.a"
        for conflict in conflict_result.conflicts
    )


# --------------------------------------------------------------------------- #
# Determinism, canonical name matching, immutability
# --------------------------------------------------------------------------- #


def test_reordering_packages_and_wheels_does_not_create_conflicts():
    h1, h2 = "1" * 64, "2" * 64
    ordered = lockfile(packages=[
        pkg("alpha", "1.0", wheels=[
            wheel("alpha-1.0-cp312.whl", h1),
            wheel("alpha-1.0-py3-none-any.whl", h2),
        ]),
        pkg("beta", "2.0"),
        pkg("gamma", "3.0", marker="sys_platform == 'linux'"),
    ])
    shuffled = lockfile(packages=[
        pkg("gamma", "3.0", marker="sys_platform == 'linux'"),
        pkg("beta", "2.0"),
        pkg("alpha", "1.0", wheels=[
            wheel("alpha-1.0-py3-none-any.whl", h2),
            wheel("alpha-1.0-cp312.whl", h1),
        ]),
    ])

    first = merge(ordered, copy.deepcopy(shuffled), copy.deepcopy(ordered))
    second = merge(shuffled, copy.deepcopy(ordered), copy.deepcopy(shuffled))

    assert first.ok and second.ok
    assert first.merged == second.merged
    assert merged_names(first) == ["alpha", "beta", "gamma"]
    assert [wheel_item["hashes"]["sha256"]
            for wheel_item in entry_for(first, "alpha")["wheels"]] == [h1, h2]


def test_canonical_package_names_match_across_inputs():
    base = lockfile(packages=[pkg("django", "4.2.0")])
    left = lockfile(packages=[pkg("Django", "4.2.0")])  # spelling only
    right = copy.deepcopy(base)

    result = merge(base, left, right)

    assert result.ok, result.conflicts
    assert len(result.merged["packages"]) == 1


def test_inputs_are_never_mutated():
    base = lockfile(packages=[pkg("requests", "1.0.0", wheels=[
        wheel("requests-1.0.0-py3-none-any.whl", "1" * 64),
    ])])
    left = lockfile(packages=[pkg("requests", "2.0.0", wheels=[
        wheel("requests-2.0.0-py3-none-any.whl", "2" * 64),
    ])])
    right = copy.deepcopy(base)
    snapshots = copy.deepcopy((base, left, right))

    result = merge(base, left, right)
    assert result.ok

    assert (base, left, right) == snapshots


def test_conflicts_are_structured_objects_with_three_way_values():
    base = lockfile(packages=[pkg("requests", "2.31.0")])
    left = copy.deepcopy(base)
    left["packages"][0]["version"] = "2.32.0"
    right = copy.deepcopy(base)
    right["packages"][0]["version"] = "2.33.0"

    result = merge(base, left, right)

    conflict = find_conflict(result, kind=KIND_VERSION_DIVERGENCE)
    assert isinstance(conflict, Conflict)
    assert conflict.base == "2.31.0"
    assert conflict.left == "2.32.0"
    assert conflict.right == "2.33.0"
    assert result.conflict_paths == [conflict.path]


def test_coverage_proof_records_the_full_matrix_and_install_choices():
    base = lockfile(
        environments=["sys_platform == 'win32'", "sys_platform == 'linux'",
                      "sys_platform == 'darwin'"],
        packages=[
            pkg("requests", "2.31.0"),
            pkg("winattr", "1.0", marker="sys_platform == 'win32'"),
        ],
    )

    result = merge(base, copy.deepcopy(base), copy.deepcopy(base))

    assert result.ok
    assert set(result.coverage) == {"win-py312", "lin-py310", "mac-py312"}
    win = result.coverage["win-py312"]
    assert win.variables["sys_platform"] == "win32"
    assert win.dependency_groups == frozenset({"dev", "test"})
    env_matches = dict(zip(
        result.merged["environments"], win.environments_matches, strict=True
    ))
    assert env_matches == {
        'sys_platform == "darwin"': False,
        'sys_platform == "linux"': False,
        'sys_platform == "win32"': True,
    }
    assert set(win.selected) == {"requests", "winattr"}
    lin = result.coverage["lin-py310"]
    assert "winattr" not in lin.selected
    winattr_entry = next(item for item in lin.entries if item["name"] == "winattr")
    assert winattr_entry["matched"] is False


def test_same_package_different_sources_conflict_or_moves_as_unit():
    private = "https://example.org/simple/"
    mirror = "https://mirror.example.org/simple/"
    base = lockfile(packages=[pkg("requests", "2.31.0", index=PYPI)])

    # Both sides repoint the same package at different indexes: the old entry
    # is deleted on both sides and the two replacement entries coexist, which
    # the per-target overlap check must reject instead of picking silently.
    left = lockfile(packages=[pkg("requests", "2.31.0", index=private)])
    right = lockfile(packages=[pkg("requests", "2.31.0", index=mirror)])
    divergent = merge(base, left, right)
    assert not divergent.ok
    assert find_conflict(divergent, kind=KIND_OVERLAP, path_contains="requests")

    # A source switch opposed by a modification of the old entry is a
    # delete/modify conflict on the original signature.
    right_modified = lockfile(packages=[pkg("requests", "2.32.0", index=PYPI)])
    opposed = merge(base, left, right_modified)
    assert not opposed.ok
    assert find_conflict(opposed, kind=KIND_DELETE_MODIFY, path_contains="requests")

    # A one-sided source switch is kept, with the selection proof naming the
    # new source identity.
    one_sided = merge(base, left, copy.deepcopy(base))
    assert one_sided.ok, one_sided.conflicts
    merged_entry = entry_for(one_sided, "requests")
    assert merged_entry["index"] == private
    selected = one_sided.coverage["win-py312"].selected["requests"]
    assert selected["source"] == "index/https://example.org/simple"


def test_sources_merge_by_identity_with_divergence_conflict():
    base = lockfile(sources=[{"name": "pypi", "url": "https://pypi.org/simple"}])
    left = copy.deepcopy(base)
    left["sources"][0]["verify_ssl"] = False
    right = copy.deepcopy(base)
    right["sources"][0]["verify_ssl"] = True

    result = merge(base, left, right)

    assert not result.ok
    assert any("sources" in conflict.path for conflict in result.conflicts)

    one_sided = merge(base, left, copy.deepcopy(base))
    assert one_sided.ok
    assert one_sided.merged["sources"][0]["verify_ssl"] is False
