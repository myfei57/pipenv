"""Tests for the target-aware ``pipenv lock diff`` comparison engine.

The diff is cross-checked against an independent marker evaluation written in
the tests themselves: expected install sets are computed straight from
``packaging`` markers against the restricted target environment, then compared
with the engine's added/removed sets rather than only matching output text.
"""

import json
from types import SimpleNamespace

import pytest

from pipenv.patched.pip._vendor.packaging.markers import (
    Marker,
    UndefinedEnvironmentName,
    _evaluate_markers,
)
from pipenv.utils.lockdiff import (
    LockDiffError,
    diff_locks,
    diff_paths,
    load_lock_file,
    parse_target_specs,
    render_text_report,
    select_for_target,
)

PY310 = {"python_version": "3.10", "python_full_version": "3.10.15"}
PY312 = {"python_version": "3.12", "python_full_version": "3.12.4"}

H1 = "sha256:" + "a" * 64
H2 = "sha256:" + "b" * 64
H3 = "sha256:" + "c" * 64


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_json(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _write_text(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _pipfile_lock(sections, sources=None):
    sources = sources or [
        {"name": "pypi", "url": "https://pypi.org/simple", "verify_ssl": True}
    ]
    data = {"_meta": {"sources": sources, "requires": {}}}
    data.update(sections)
    return data


def _target(name, markers=None, groups=None, extras=None):
    spec = {"name": name, "markers": markers or {}}
    if groups is not None:
        spec["groups"] = groups
    if extras is not None:
        spec["extras"] = extras
    return parse_target_specs([json.dumps(spec)])[0]


def _diff_map(report):
    """Return {(target, category): [package names]} for concise assertions."""
    result = {}
    for target in report["targets"]:
        for difference in target["differences"]:
            result.setdefault((target["name"], difference["category"]), []).append(
                difference["package"]
            )
    return result


def _find_difference(report, target_name, package):
    for target in report["targets"]:
        if target["name"] != target_name:
            continue
        for difference in target["differences"]:
            if difference["package"] == package:
                return difference
    return None


def _independent_selection(lock, target):
    """Reference implementation: evaluate markers with packaging directly."""
    env = {
        **{key: str(value) for key, value in target.markers.items()},
        "extras": target.extras,
        "dependency_groups": target.groups,
    }
    selected = {}
    for entry in lock.entries:
        if entry.marker is None:
            active = True
        else:
            try:
                active = _evaluate_markers(Marker(entry.marker)._markers, env)
            except UndefinedEnvironmentName:
                continue  # unknown names are not part of the concrete install set
        if active:
            selected[entry.name] = entry
    return selected


# ---------------------------------------------------------------------------
# Target parsing
# ---------------------------------------------------------------------------


def test_target_defaults_groups_to_default_and_canonicalizes_names():
    target = parse_target_specs([json.dumps({"markers": PY312})])[0]
    assert target.name == "target-1"
    assert target.groups == frozenset({"default"})
    assert target.extras == frozenset()


def test_targets_accept_array_document_and_file_reference(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            [
                {"name": "a", "markers": PY310},
                {"name": "b", "markers": PY312, "groups": ["default", "dev"]},
            ]
        )
    )
    targets = parse_target_specs(["@" + str(path)])
    assert [target.name for target in targets] == ["a", "b"]
    assert targets[1].groups == frozenset({"default", "dev"})


def test_targets_file_is_merged_with_inline_specs(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps({"name": "from-file", "markers": PY310}))
    targets = parse_target_specs(
        [json.dumps({"name": "inline", "markers": PY312})], targets_file=path
    )
    assert [target.name for target in targets] == ["inline", "from-file"]


def test_unknown_marker_variable_is_rejected():
    with pytest.raises(LockDiffError, match="unknown marker variable"):
        parse_target_specs([json.dumps({"markers": {"bogus_var": "x"}})])


def test_duplicate_target_names_are_rejected():
    specs = [
        json.dumps({"name": "same", "markers": PY310}),
        json.dumps({"name": "same", "markers": PY312}),
    ]
    with pytest.raises(LockDiffError, match="Duplicate target name"):
        parse_target_specs(specs)


def test_at_least_one_target_is_required():
    with pytest.raises(LockDiffError, match="At least one target"):
        parse_target_specs()


# ---------------------------------------------------------------------------
# Noise suppression: ordering and equivalent marker spellings
# ---------------------------------------------------------------------------


def test_json_key_and_hash_ordering_produces_no_noise(tmp_path):
    h1, h2 = "sha256:111", "sha256:222"
    left = _pipfile_lock(
        {
            "default": {
                "requests": {"version": "==2.31.0", "hashes": [h1, h2], "index": "pypi"}
            }
        }
    )
    right = _pipfile_lock(
        {
            "default": {
                "requests": {"hashes": [h2, h1], "index": "pypi", "version": "==2.31.0"}
            }
        }
    )
    lp = _write_json(tmp_path, "a.lock", left)
    rp = _write_json(tmp_path, "b.lock", right)

    report = diff_paths(lp, rp, [_target("t", PY312)])

    assert report["status"] == "unchanged"
    assert report["targets"][0]["differences"] == []


def test_equivalent_marker_spellings_are_equivalent_on_the_matrix(tmp_path):
    # Quote style and AND-clause order differ; truth values on both targets do not.
    left = _write_text(
        tmp_path,
        "a.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'click'
version = '8.1.0'
marker = "sys_platform == 'linux' and python_version >= '3.10'"
index = 'https://pypi.org/simple/'
""",
    )
    right = _write_text(
        tmp_path,
        "b.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'click'
version = '8.1.0'
marker = 'python_version >= "3.10" and sys_platform == "linux"'
index = 'https://pypi.org/simple/'
""",
    )
    targets = [
        _target("linux310", {**PY310, "sys_platform": "linux"}),
        _target("win312", {**PY312, "sys_platform": "win32"}),
    ]
    report = diff_paths(left, right, targets)

    assert report["status"] == "unchanged"
    # The report must not claim equivalence beyond the supplied matrix.
    text = render_text_report(report)
    assert "only asserted for the explicit target matrix" in text


def test_marker_change_that_flips_selection_is_reported(tmp_path):
    left = _write_text(
        tmp_path,
        "a.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'click'
version = '8.1.0'
marker = "sys_platform == 'linux'"
index = 'https://pypi.org/simple/'
""",
    )
    right = _write_text(
        tmp_path,
        "b.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'click'
version = '8.1.0'
marker = "sys_platform == 'win32'"
index = 'https://pypi.org/simple/'
""",
    )
    targets = [_target("linux", {**PY312, "sys_platform": "linux"})]
    report = diff_paths(left, right, targets)
    diffs = _diff_map(report)

    assert diffs[("linux", "removed")] == ["click"]


# ---------------------------------------------------------------------------
# Field-level changes
# ---------------------------------------------------------------------------


def test_same_version_with_index_change_is_a_source_change(tmp_path):
    left = _pipfile_lock(
        {"default": {"pkg": {"version": "==1.0", "index": "pypi"}}},
        sources=[
            {"name": "pypi", "url": "https://pypi.org/simple"},
            {"name": "mirror", "url": "https://mirror.example/simple"},
        ],
    )
    right = _pipfile_lock(
        {"default": {"pkg": {"version": "==1.0", "index": "mirror"}}},
        sources=[
            {"name": "pypi", "url": "https://pypi.org/simple"},
            {"name": "mirror", "url": "https://mirror.example/simple"},
        ],
    )
    report = diff_paths(
        _write_json(tmp_path, "a.lock", left),
        _write_json(tmp_path, "b.lock", right),
        [_target("t", PY312)],
    )
    difference = _find_difference(report, "t", "pkg")
    assert difference["category"] == "changed"
    assert difference["fields"] == ["source"]
    assert difference["details"]["source"]["before"]["url"] == "https://pypi.org/simple"
    assert (
        difference["details"]["source"]["after"]["url"] == "https://mirror.example/simple"
    )


def test_hash_narrowing_and_growth_are_not_format_changes(tmp_path):
    left = _pipfile_lock({"default": {"pkg": {"version": "==1.0", "hashes": [H1, H2]}}})
    right = _pipfile_lock({"default": {"pkg": {"version": "==1.0", "hashes": [H2, H3]}}})
    report = diff_paths(
        _write_json(tmp_path, "a.lock", left),
        _write_json(tmp_path, "b.lock", right),
        [_target("t", PY312)],
    )
    difference = _find_difference(report, "t", "pkg")
    assert set(difference["fields"]) == {"hashes.added", "hashes.removed"}
    assert difference["details"]["hashes"]["added"] == [H3]
    assert difference["details"]["hashes"]["removed"] == [H1]


def test_pure_hash_removal_is_reported(tmp_path):
    left = _pipfile_lock({"default": {"pkg": {"version": "==1.0", "hashes": [H1, H2]}}})
    right = _pipfile_lock({"default": {"pkg": {"version": "==1.0", "hashes": [H2]}}})
    report = diff_paths(
        _write_json(tmp_path, "a.lock", left),
        _write_json(tmp_path, "b.lock", right),
        [_target("t", PY312)],
    )
    difference = _find_difference(report, "t", "pkg")
    assert difference["fields"] == ["hashes.removed"]


def test_vcs_ref_editable_and_direct_url_changes_are_separate(tmp_path):
    left = _pipfile_lock(
        {
            "default": {
                "demo": {
                    "git": "https://git.example.com/demo.git",
                    "ref": "abc123",
                    "editable": True,
                }
            }
        }
    )
    right = _pipfile_lock(
        {
            "default": {
                "demo": {
                    "git": "https://git.example.com/demo.git",
                    "ref": "def456",
                    "editable": False,
                }
            }
        }
    )
    report = diff_paths(
        _write_json(tmp_path, "a.lock", left),
        _write_json(tmp_path, "b.lock", right),
        [_target("t", PY312)],
    )
    difference = _find_difference(report, "t", "demo")
    assert difference["fields"] == ["source"]
    before = difference["details"]["source"]["before"]
    after = difference["details"]["source"]["after"]
    assert before["commit_id"] == "abc123"
    assert after["commit_id"] == "def456"
    assert before["editable"] is True
    assert "editable" not in after


def test_archive_url_change_is_a_source_change(tmp_path):
    left = _pipfile_lock(
        {
            "default": {
                "blob": {
                    "file": "https://files.example.com/blob-1.tar.gz",
                    "hashes": [H1],
                }
            }
        }
    )
    right = _pipfile_lock(
        {
            "default": {
                "blob": {
                    "file": "https://files.example.com/blob-2.tar.gz",
                    "hashes": [H1],
                }
            }
        }
    )
    report = diff_paths(
        _write_json(tmp_path, "a.lock", left),
        _write_json(tmp_path, "b.lock", right),
        [_target("t", PY312)],
    )
    difference = _find_difference(report, "t", "blob")
    assert difference["fields"] == ["source"]
    assert difference["details"]["source"]["after"]["url"].endswith("blob-2.tar.gz")


def test_attestation_identity_change_is_a_trust_change(tmp_path):
    left = _write_text(
        tmp_path,
        "a.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'pkg'
version = '1.0'
index = 'https://pypi.org/simple/'
attestation-identities = [{kind = 'pypi', 'url' = 'https://pypi.org/attestations/pkg/1.0'}]
""",
    )
    right = _write_text(
        tmp_path,
        "b.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'pkg'
version = '1.0'
index = 'https://pypi.org/simple/'
attestation-identities = [{kind = 'pypi', 'url' = 'https://pypi.org/attestations/pkg/1.1'}]
""",
    )
    report = diff_paths(left, right, [_target("t", PY312)])
    difference = _find_difference(report, "t", "pkg")
    assert "attestations.added" in difference["fields"]
    assert "attestations.removed" in difference["fields"]


def test_artifact_file_set_change_is_reported_for_pylock(tmp_path):
    def pylock(wheel_name):
        return f"""
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'pkg'
version = '1.0'
index = 'https://pypi.org/simple/'
[[packages.wheels]]
name = '{wheel_name}'
url = 'https://files.example.com/{wheel_name}'
hashes = {{sha256 = '{'a' * 64}'}}
"""

    left = _write_text(tmp_path, "a.toml", pylock("pkg-1.0-py3-none-any.whl"))
    right = _write_text(tmp_path, "b.toml", pylock("pkg-1.0-py3-none-manylinux.whl"))
    report = diff_paths(left, right, [_target("t", PY312)])
    difference = _find_difference(report, "t", "pkg")
    assert "artifacts.added" in difference["fields"]
    assert "artifacts.removed" in difference["fields"]


# ---------------------------------------------------------------------------
# Groups, extras and visibility
# ---------------------------------------------------------------------------


def test_cross_group_migration_is_removed_and_added_per_target(tmp_path):
    left = _pipfile_lock(
        {
            "default": {"tool": {"version": "==1.0"}},
            "develop": {},
        }
    )
    right = _pipfile_lock(
        {
            "default": {},
            "develop": {"tool": {"version": "==1.0"}},
        }
    )
    targets = [
        _target("default-only", PY312, groups=["default"]),
        _target("dev-only", PY312, groups=["dev"]),
        _target("both", PY312, groups=["default", "dev"]),
    ]
    report = diff_paths(
        _write_json(tmp_path, "a.lock", left),
        _write_json(tmp_path, "b.lock", right),
        targets,
    )
    diffs = _diff_map(report)
    assert diffs[("default-only", "removed")] == ["tool"]
    assert diffs[("dev-only", "added")] == ["tool"]

    both = _find_difference(report, "both", "tool")
    assert both["category"] == "changed"
    assert both["fields"] == ["visibility"]
    visibility = {
        item["target"]: (item["before"], item["after"])
        for item in both["details"]["visibility"]
    }
    assert visibility["default-only"] == ("active", "inactive")
    assert visibility["dev-only"] == ("inactive", "active")


def test_extras_marker_selects_entries_within_target(tmp_path):
    left = _write_text(
        tmp_path,
        "a.toml",
        """
lock-version = '1.0'
created-by = 'test'
extras = ['crypto']
[[packages]]
name = 'cryptography'
version = '40.0.0'
marker = "'crypto' in extras"
index = 'https://pypi.org/simple/'
""",
    )
    right = _write_text(
        tmp_path,
        "b.toml",
        """
lock-version = '1.0'
created-by = 'test'
extras = ['crypto']
[[packages]]
name = 'cryptography'
version = '41.0.0'
marker = "'crypto' in extras"
index = 'https://pypi.org/simple/'
""",
    )
    without_extra = diff_paths(left, right, [_target("plain", PY312, extras=[])])
    with_extra = diff_paths(left, right, [_target("crypto", PY312, extras=["crypto"])])
    assert without_extra["status"] == "unchanged"
    difference = _find_difference(with_extra, "crypto", "cryptography")
    assert difference["details"]["version"] == {"before": "40.0.0", "after": "41.0.0"}


# ---------------------------------------------------------------------------
# Unknown / ambiguous handling
# ---------------------------------------------------------------------------


def test_missing_marker_variable_is_unknown_not_host_substituted(tmp_path):
    lock = _write_text(
        tmp_path,
        "a.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'plat'
version = '1.0'
marker = "platform_machine == 'aarch64'"
index = 'https://pypi.org/simple/'
""",
    )
    normalized = load_lock_file(lock)
    # The target deliberately omits platform_machine; the host value must not leak.
    target = _target("t", {"python_version": "3.12"})
    selection = select_for_target(normalized, target)
    assert selection["plat"].outcome == "unknown"

    report = diff_paths(lock, lock, [target])
    assert report["status"] == "incomplete"
    issue = report["targets"][0]["issues"][0]
    assert issue["code"] == "unknown-marker-variable"
    assert issue["variable"] == "platform_machine"
    # An unknown name must not be claimed as added/removed/changed.
    assert report["targets"][0]["differences"] == []


def test_explicit_target_variables_drive_selection_without_host(tmp_path):
    lock = _write_text(
        tmp_path,
        "a.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'pkg'
version = '1.0'
marker = "sys_platform == 'linux'"
index = 'https://pypi.org/simple/'
""",
    )
    normalized = load_lock_file(lock)
    linux = select_for_target(
        normalized, _target("linux", {**PY312, "sys_platform": "linux"})
    )
    windows = select_for_target(
        normalized, _target("win", {**PY312, "sys_platform": "win32"})
    )
    assert linux["pkg"].outcome == "active"
    assert windows["pkg"].outcome == "inactive"


def test_identical_entries_shared_across_groups_are_not_ambiguous(tmp_path):
    # The same concrete artifact pinned in two categories must collapse to a
    # single selection when both groups are enabled (Pipfile.lock round-trip
    # exports these as two pylock entries with OR-ed group markers).
    left = _pipfile_lock(
        {
            "default": {"shared": {"version": "==1.0", "hashes": [H1]}},
            "develop": {"shared": {"version": "==1.0", "hashes": [H1]}},
        }
    )
    right = _pipfile_lock(
        {
            "default": {"shared": {"version": "==1.0", "hashes": [H1]}},
            "develop": {"shared": {"version": "==1.0", "hashes": [H1]}},
        }
    )
    report = diff_paths(
        _write_json(tmp_path, "a.lock", left),
        _write_json(tmp_path, "b.lock", right),
        [_target("both", PY312, groups=["default", "dev"])],
    )
    assert report["status"] == "unchanged"
    assert report["targets"][0]["issues"] == []


def test_unknown_sibling_entry_flags_incomplete_but_keeps_selection(tmp_path):
    lock = _write_text(
        tmp_path,
        "a.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'dual'
version = '1.0'
index = 'https://pypi.org/simple/'
[[packages]]
name = 'dual'
version = '2.0'
marker = "platform_machine == 'aarch64'"
index = 'https://pypi.org/simple/'
""",
    )
    normalized = load_lock_file(lock)
    target = _target("t", {"python_version": "3.12"})
    selection = select_for_target(normalized, target)
    # The unconditional entry still gives a concrete install choice.
    assert selection["dual"].outcome == "active"
    assert selection["dual"].entries[0].version == "1.0"
    assert len(selection["dual"].unknown_entries) == 1

    report = diff_paths(lock, lock, [target])
    assert report["status"] == "incomplete"
    issue = report["targets"][0]["issues"][0]
    assert issue["code"] == "unknown-marker-variable"
    assert issue["variable"] == "platform_machine"
    # The definite selection remains visible for diffing.
    names = [item["name"] for item in report["targets"][0]["selection"]["left"]]
    assert names == ["dual"]


def test_conflicting_active_entries_are_ambiguous(tmp_path):
    lock = _write_text(
        tmp_path,
        "a.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'demo'
version = '1.0'
marker = "'dev' in dependency_groups"
index = 'https://pypi.org/simple/'
[[packages]]
name = 'demo'
version = '2.0'
marker = "'dev' in dependency_groups"
index = 'https://pypi.org/simple/'
""",
    )
    report = diff_paths(lock, lock, [_target("dev", PY312, groups=["dev"])])
    target_report = report["targets"][0]
    assert target_report["status"] == "incomplete"
    issue = next(
        item for item in target_report["issues"] if item["code"] == "ambiguous-entries"
    )
    assert issue["package"] == "demo"
    assert len(issue["entries"]) == 2
    assert report["status"] == "incomplete"


def test_unrecognized_index_source_is_unknown(tmp_path):
    left = _pipfile_lock({"default": {"pkg": {"version": "==1.0", "index": "pypi"}}})
    right = _pipfile_lock(
        {"default": {"pkg": {"version": "==1.0", "index": "gone"}}},
        sources=[{"name": "pypi", "url": "https://pypi.org/simple"}],
    )
    report = diff_paths(
        _write_json(tmp_path, "a.lock", left),
        _write_json(tmp_path, "b.lock", right),
        [_target("t", PY312)],
    )
    target_report = report["targets"][0]
    assert any(issue["code"] == "unknown-source" for issue in target_report["issues"])
    assert report["status"] == "incomplete"


def test_pylock_multi_entries_are_not_flattened(tmp_path):
    lock = _write_text(
        tmp_path,
        "a.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'numpy'
version = '1.26.4'
marker = "python_version < '3.12'"
index = 'https://pypi.org/simple/'
[[packages]]
name = 'numpy'
version = '2.3.0'
marker = "python_version >= '3.12'"
index = 'https://pypi.org/simple/'
""",
    )
    normalized = load_lock_file(lock)
    assert len([entry for entry in normalized.entries if entry.name == "numpy"]) == 2

    py310 = select_for_target(normalized, _target("py310", PY310))
    py312 = select_for_target(normalized, _target("py312", PY312))
    assert py310["numpy"].entries[0].version == "1.26.4"
    assert py312["numpy"].entries[0].version == "2.3.0"


# ---------------------------------------------------------------------------
# Cross-format comparison
# ---------------------------------------------------------------------------


def test_cross_format_equivalent_locks_compare_unchanged(tmp_path):
    pipfile = _pipfile_lock(
        {
            "default": {
                "requests": {"version": "==2.31.0", "hashes": [H1, H2], "index": "pypi"}
            }
        }
    )
    lp = _write_json(tmp_path, "Pipfile.lock", pipfile)
    rp = _write_text(
        tmp_path,
        "pylock.toml",
        f"""
lock-version = '1.0'
created-by = 'pipenv'
default-groups = ['default']
[[packages]]
name = 'requests'
version = '2.31.0'
marker = "'default' in dependency_groups"
index = 'https://pypi.org/simple/'
[[packages.wheels]]
name = 'requests-2.31.0-py3-none-any.whl'
url = 'https://files.example.com/requests-2.31.0-py3-none-any.whl'
hashes = {{sha256 = '{'a' * 64}'}}
[packages.tool.pipenv]
hashes = ['{H2}']
""",
    )
    report = diff_paths(lp, rp, [_target("t", PY312)])
    assert report["left"]["format"] == "pipfile-lock"
    assert report["right"]["format"] == "pylock-toml"
    assert report["status"] == "unchanged"


def test_cross_format_hash_narrowing_still_detected(tmp_path):
    pipfile = _pipfile_lock(
        {
            "default": {
                "requests": {"version": "==2.31.0", "hashes": [H1, H2], "index": "pypi"}
            }
        }
    )
    lp = _write_json(tmp_path, "Pipfile.lock", pipfile)
    rp = _write_text(
        tmp_path,
        "pylock.toml",
        f"""
lock-version = '1.0'
created-by = 'pipenv'
default-groups = ['default']
[[packages]]
name = 'requests'
version = '2.31.0'
marker = "'default' in dependency_groups"
index = 'https://pypi.org/simple/'
[packages.tool.pipenv]
hashes = ['{H2}']
""",
    )
    report = diff_paths(lp, rp, [_target("t", PY312)])
    difference = _find_difference(report, "t", "requests")
    assert difference["fields"] == ["hashes.removed"]


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_url_credentials_are_redacted_but_keep_identity(tmp_path):
    left = _write_text(
        tmp_path,
        "a.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'private'
version = '1.0'
index = 'https://user:secret@index.example.com/simple/'
""",
    )
    right = _write_text(
        tmp_path,
        "b.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'private'
version = '1.0'
index = 'https://user:rotated@index.example.com/simple/'
""",
    )
    report = diff_paths(left, right, [_target("t", PY312)])
    serialized = json.dumps(report)
    assert "secret" not in serialized
    assert "rotated" not in serialized
    assert "user:****@index.example.com" in serialized

    difference = _find_difference(report, "t", "private")
    assert difference["fields"] == ["source"]


def test_placeholder_credentials_are_output_redacted(tmp_path):
    lock = _pipfile_lock(
        {"default": {"pkg": {"version": "==1.0"}}},
        sources=[
            {
                "name": "private",
                "url": "https://${USER}:${TOKEN}@index.example.com/simple",
            }
        ],
    )
    path = _write_json(tmp_path, "a.lock", lock)
    report = diff_paths(path, path, [_target("t", PY312)])
    serialized = json.dumps(report)
    assert "${TOKEN}" not in serialized
    assert report["status"] == "unchanged"


# ---------------------------------------------------------------------------
# Independent selection cross-check and determinism
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target",
    [
        _target("py310-default", PY310, groups=["default"]),
        _target("py312-both", PY312, groups=["default", "dev"]),
    ],
)
def test_added_removed_sets_match_independent_selection(tmp_path, target):
    left = _pipfile_lock(
        {
            "default": {
                "stays": {"version": "==1.0", "markers": "python_version >= '3.9'"},
                "leaves": {"version": "==1.0"},
            },
            "develop": {"pytest": {"version": "==7.0"}},
        }
    )
    right = _pipfile_lock(
        {
            "default": {
                "stays": {"version": "==1.0", "markers": "python_version >= '3.9'"},
                "arrives": {"version": "==2.0"},
            },
            "develop": {"pytest": {"version": "==8.0"}},
        }
    )
    lp = _write_json(tmp_path, "a.lock", left)
    rp = _write_json(tmp_path, "b.lock", right)
    left_lock = load_lock_file(lp)
    right_lock = load_lock_file(rp)

    report = diff_locks(left_lock, right_lock, [target])
    target_report = report["targets"][0]

    expected_left = set(_independent_selection(left_lock, target))
    expected_right = set(_independent_selection(right_lock, target))
    engine_left = {item["name"] for item in target_report["selection"]["left"]}
    engine_right = {item["name"] for item in target_report["selection"]["right"]}
    assert engine_left == expected_left
    assert engine_right == expected_right

    diffs = _diff_map(report)[(target.name, "added")], _diff_map(report).get(
        (target.name, "removed"), []
    )
    added, removed = diffs
    assert added == sorted(expected_right - expected_left)
    assert removed == sorted(expected_left - expected_right)


def test_report_is_deterministic_and_sorted(tmp_path):
    left = _pipfile_lock(
        {
            "default": {
                "zebra": {"version": "==1.0"},
                "apple": {"version": "==1.0"},
            }
        }
    )
    right = _pipfile_lock(
        {
            "default": {
                "zebra": {"version": "==2.0"},
                "mango": {"version": "==1.0"},
            }
        }
    )
    targets = [_target("t", PY312)]
    first = json.dumps(
        diff_paths(
            _write_json(tmp_path, "a.lock", left),
            _write_json(tmp_path, "b.lock", right),
            targets,
        ),
        sort_keys=True,
    )
    second = json.dumps(
        diff_paths(tmp_path / "a.lock", tmp_path / "b.lock", targets),
        sort_keys=True,
    )
    assert first == second
    report = json.loads(first)
    differences = report["targets"][0]["differences"]
    # Sorting is by (category, canonical package name), not package alone.
    order = {"added": 0, "changed": 1, "removed": 2}
    ranked = [(order[item["category"]], item["package"]) for item in differences]
    assert ranked == sorted(ranked)
    for category in ("added", "changed", "removed"):
        names = [item["package"] for item in differences if item["category"] == category]
        assert names == sorted(names)


def test_source_files_are_never_rewritten(tmp_path):
    left = _write_json(
        tmp_path, "a.lock", _pipfile_lock({"default": {"p": {"version": "==1.0"}}})
    )
    right = _write_text(
        tmp_path,
        "b.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'p'
version = '2.0'
index = 'https://pypi.org/simple/'
""",
    )
    before_left = left.read_bytes()
    before_right = right.read_bytes()
    diff_paths(left, right, [_target("t", PY312)])
    assert left.read_bytes() == before_left
    assert right.read_bytes() == before_right


def test_missing_file_raises_lockdiff_error(tmp_path):
    with pytest.raises(LockDiffError, match="Lock file not found"):
        load_lock_file(tmp_path / "nope.lock")


def test_undecodable_file_raises_lockdiff_error(tmp_path):
    path = _write_text(tmp_path, "weird.lock", "this is not json or toml")
    with pytest.raises(LockDiffError):
        load_lock_file(path)


# ---------------------------------------------------------------------------
# Status semantics and text rendering
# ---------------------------------------------------------------------------


def test_status_unchanged_changed_and_incomplete(tmp_path):
    base = _pipfile_lock({"default": {"p": {"version": "==1.0"}}})
    identical = _pipfile_lock({"default": {"p": {"version": "==1.0"}}})
    changed = _pipfile_lock({"default": {"p": {"version": "==2.0"}}})

    unchanged = diff_paths(
        _write_json(tmp_path, "u1.lock", base),
        _write_json(tmp_path, "u2.lock", identical),
        [_target("t", PY312)],
    )
    assert unchanged["status"] == "unchanged"

    changed_report = diff_paths(
        _write_json(tmp_path, "c1.lock", base),
        _write_json(tmp_path, "c2.lock", changed),
        [_target("t", PY312)],
    )
    assert changed_report["status"] == "changed"

    unknown_lock = _write_text(
        tmp_path,
        "x.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'p'
version = '1.0'
marker = "platform_machine == 'x86_64'"
index = 'https://pypi.org/simple/'
""",
    )
    incomplete = diff_paths(
        _write_json(tmp_path, "i1.lock", base), unknown_lock, [_target("t", PY312)]
    )
    assert incomplete["status"] == "incomplete"
    text = render_text_report(incomplete)
    assert "Overall status: incomplete" in text
    assert "unknown-marker-variable" in text


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def test_parser_wires_lock_diff_and_keeps_plain_lock():
    from pipenv.cli.options import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "lock",
            "diff",
            "a.lock",
            "b.toml",
            "--targets",
            json.dumps({"name": "t", "markers": PY312}),
            "--json",
        ]
    )
    assert args.command == "lock"
    assert args.lock_subcommand == "diff"
    assert args.left == "a.lock"
    assert args.right == "b.toml"
    assert args.as_json is True

    plain = parser.parse_args(["lock", "--dev"])
    assert getattr(plain, "lock_subcommand", None) is None
    assert plain.dev is True


def test_cli_handler_exit_codes_and_outputs(tmp_path, capsys):
    from pipenv.cli.command import cmd_lock_diff

    left = _write_json(
        tmp_path, "a.lock", _pipfile_lock({"default": {"p": {"version": "==1.0"}}})
    )
    right_same = _write_json(
        tmp_path, "same.lock", _pipfile_lock({"default": {"p": {"version": "==1.0"}}})
    )
    right_changed = _write_json(
        tmp_path, "changed.lock", _pipfile_lock({"default": {"p": {"version": "==2.0"}}})
    )
    spec = json.dumps({"name": "t", "markers": PY312})

    def args_for(right, **kwargs):
        values = {
            "targets": [spec],
            "targets_file": None,
            "as_json": False,
            "json_output": None,
        }
        values.update(kwargs)
        return SimpleNamespace(left=str(left), right=str(right), **values)

    assert cmd_lock_diff(args_for(right_same), state=None) == 0
    out = capsys.readouterr().out
    assert "Overall status: unchanged" in out

    with pytest.raises(SystemExit) as exc:
        cmd_lock_diff(args_for(right_changed), state=None)
    assert exc.value.code == 1
    capsys.readouterr()  # drain the text-mode output before the JSON call

    with pytest.raises(SystemExit):
        cmd_lock_diff(args_for(right_changed, as_json=True), state=None)
    json_out = capsys.readouterr().out
    parsed = json.loads(json_out)
    assert parsed["status"] == "changed"

    json_path = tmp_path / "report.json"
    with pytest.raises(SystemExit):
        cmd_lock_diff(args_for(right_changed, json_output=str(json_path)), state=None)
    assert json.loads(json_path.read_text())["status"] == "changed"


def test_cli_handler_incomplete_exits_two(tmp_path):
    from pipenv.cli.command import cmd_lock_diff

    left = _write_json(
        tmp_path, "a.lock", _pipfile_lock({"default": {"p": {"version": "==1.0"}}})
    )
    right = _write_text(
        tmp_path,
        "b.toml",
        """
lock-version = '1.0'
created-by = 'test'
[[packages]]
name = 'p'
version = '1.0'
marker = "platform_machine == 'x86_64'"
index = 'https://pypi.org/simple/'
""",
    )
    args = SimpleNamespace(
        left=str(left),
        right=str(right),
        targets=[json.dumps({"name": "t", "markers": PY312})],
        targets_file=None,
        as_json=True,
        json_output=None,
    )
    with pytest.raises(SystemExit) as exc:
        cmd_lock_diff(args, state=None)
    assert exc.value.code == 2


def test_cli_handler_bad_input_exits_two(tmp_path):
    from pipenv.cli.command import cmd_lock_diff

    args = SimpleNamespace(
        left=str(tmp_path / "missing.lock"),
        right=str(tmp_path / "missing2.lock"),
        targets=[json.dumps({"name": "t", "markers": PY312})],
        targets_file=None,
        as_json=False,
        json_output=None,
    )
    with pytest.raises(SystemExit) as exc:
        cmd_lock_diff(args, state=None)
    assert exc.value.code == 2


def test_cli_handler_missing_positional_exits_two():
    from pipenv.cli.command import cmd_lock_diff

    args = SimpleNamespace(
        left=None,
        right=None,
        targets=[json.dumps({"name": "t", "markers": PY312})],
        targets_file=None,
        as_json=False,
        json_output=None,
    )
    with pytest.raises(SystemExit) as exc:
        cmd_lock_diff(args, state=None)
    assert exc.value.code == 2
