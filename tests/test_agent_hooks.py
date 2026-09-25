"""Self-tests for the project agent hooks, including falsification cases.

Same discipline as `tests/test_check_conventions.py`: a guard that never fires
is decoration, not a gate. Each case either proves a dangerous operation is
denied or proves a legitimate read-only operation is allowed.

Hooks are imported and their `evaluate()` predicates called directly, and the
CLI layer is driven by calling `main()` with a substituted stdin. No child
process is spawned anywhere in this module: the behaviour under test is
"read stdin, decide, return an exit code", which this drives exactly.

Run:  pytest tests/test_agent_hooks.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / "tools" / "agent_hooks"

RAW = "data/" + "raw"
ARCHIVE = "legacy" + "_v531_full"
V6 = "legacy" + "_v6"
# A scratch repo path for the `git -C <dir> <subcmd>` case. Built from a
# temporary-directory base so this file carries no host-specific literal.
OTHER_REPO = str(Path(tempfile.gettempdir()) / "scratch-repo")


def _load(module_name: str):
    """Import a hook module from tools/agent_hooks by exact filename."""
    path = HOOKS / f"{module_name}.py"
    assert path.is_file(), f"hook script missing: {path}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def paths_guard():
    return _load("guard_protected_paths")


@pytest.fixture(scope="module")
def git_guard():
    return _load("guard_destructive_git")


@pytest.fixture(scope="module")
def digest_notice():
    return _load("check_model_digest_impact")


class _Stdin:
    """Minimal stdin stand-in for driving a hook's main() in-process."""

    def __init__(self, text: str):
        self._text = text

    def read(self) -> str:
        return self._text


def bash(command: str) -> dict:
    return {"tool_name": "Bash", "tool_input": {"command": command},
            "hook_event_name": "PreToolUse"}


def write_tool(path: str) -> dict:
    return {"tool_name": "Write", "tool_input": {"file_path": path, "content": "x"},
            "hook_event_name": "PreToolUse"}


def edit_tool(path: str) -> dict:
    return {"tool_name": "Edit",
            "tool_input": {"file_path": path, "old_string": "a", "new_string": "b"},
            "hook_event_name": "PreToolUse"}


# ------------------------------------------------------- guard_protected_paths

BLOCKED_PATH_CASES = [
    ("write into raw data", write_tool(f"{RAW}/source.csv")),
    ("edit into raw data", edit_tool(f"{RAW}/source.csv")),
    ("write into interim", write_tool("data/interim/x.json")),
    ("write into processed", write_tool("data/processed/x.json")),
    ("edit into v531 archive", edit_tool(f"{ARCHIVE}/utils/helper.py")),
    ("edit into v6 archive", edit_tool(f"{V6}/legacy_physical_consistency.py")),
    ("write into data archive", write_tool("data/legacy_v531/loader.py")),
    ("write into model archive", write_tool("model/legacy_v531/urban_pidit.py")),
    ("remove raw dir", bash(f"rm -rf {RAW}")),
    ("remove archive file", bash(f"rm {ARCHIVE}/a.py")),
    ("truncate raw file", bash(f"truncate -s 0 {RAW}/x.csv")),
    ("redirect into raw", bash(f"echo hi > {RAW}/out.txt")),
    ("append into archive", bash(f"echo hi >> {ARCHIVE}/out.txt")),
    ("copy over archive", bash(f"cp /tmp/f.py {V6}/f.py")),
    ("move out of archive", bash(f"mv {V6}/f.py /tmp/")),
    ("git rm in archive", bash(f"git rm {ARCHIVE}/x.py")),
    ("sed -i on archive", bash(f"sed -i s/a/b/ {ARCHIVE}/x.py")),
    ("chmod in raw", bash(f"chmod 644 {RAW}/x.csv")),
    ("tee into raw", bash(f"echo x | tee {RAW}/x.txt")),
    # Evasions measured as NOT caught before the fix; now regression tests.
    ("cd into protected then relative rm", bash(f"cd {RAW} && rm source.csv")),
    ("interior dot component", bash("rm data/./raw/x.csv")),
    ("duplicate slash", bash("rm data//raw/x.csv")),
    ("fd redirect 2>", bash(f"cmd 2>{RAW}/err.log")),
    ("fd redirect 1>", bash(f"echo hi 1>{RAW}/x.txt")),
    ("curl -o", bash(f"curl -o {RAW}/x.csv http://example.invalid")),
    ("sort -o", bash(f"sort -o {RAW}/x.txt /tmp/y")),
    ("dd of=", bash(f"dd if=/dev/zero of={RAW}/x bs=1 count=1")),
    ("find -delete", bash(f"find {RAW} -name '*.csv' -delete")),
    ("sed --in-place", bash(f"sed --in-place s/a/b/ {RAW}/x.csv")),
    ("uppercase component", bash("rm data/RAW/x.csv")),
    ("glob partial spelling", bash("rm -rf data/ra*")),
    ("git worktree add", bash(f"git worktree add {RAW}/wt HEAD")),
    ("install into protected tree", bash(f"install -m 644 /tmp/f {ARCHIVE}/f")),
    ("perl -i in place", bash(f"perl -i -pe s/a/b/ {RAW}/x.csv")),
    ("find -exec rm", bash(f"find {RAW} -name '*.csv' -exec rm {{}} +")),
    ("noclobber redirect", bash(f"echo hi >| {RAW}/x.txt")),
    ("wget -O output flag", bash(f"wget -O {RAW}/x.csv http://example.invalid")),
]


@pytest.mark.parametrize("label,payload", BLOCKED_PATH_CASES,
                         ids=[c[0].replace(" ", "-") for c in BLOCKED_PATH_CASES])
def test_protected_path_writes_are_denied(paths_guard, label, payload):
    verdict = paths_guard.evaluate(payload)
    assert verdict is not None, f"{label}: expected a deny verdict"
    prefix, what = verdict
    reason = paths_guard._reason(prefix, what)
    assert "BLOCKED" in reason, f"{label}: deny must state the rule"
    assert "R-00" in reason or "R-03" in reason, f"{label}: deny must cite a rule id"


ALLOWED_PATH_CASES = [
    ("read archive", {"tool_name": "Read", "tool_input": {"file_path": f"{ARCHIVE}/x.py"}}),
    ("grep archive", bash(f"grep -rn 'def ' {ARCHIVE}/")),
    ("grep mentioning rm", bash(f"grep -rn 'rm -rf' {ARCHIVE}/")),
    ("plain sed reads", bash(f"sed -n 1,5p {RAW}/x.csv")),
    ("ls archive", bash(f"ls -la {V6}/")),
    ("git status", bash("git status --porcelain")),
    ("git log", bash("git log --oneline -n 5")),
    ("write normal source", write_tool("training/r7_evaluate.py")),
    ("write outputs", write_tool("outputs/run/result.json")),
    ("write docs", write_tool("docs/rules/testing.md")),
    ("write data manifests", write_tool("data/manifests/real_smoke/provenance.json")),
    ("empty payload", {}),
    # A `>` in prose is not a shell redirect. This blocked a real commit:
    # the message described a file move as `a -> legacy_v6/b` and was denied.
    ("arrow in commit heredoc body",
     bash(f"git commit -F - <<MSG\nrename a -> {V6}/b\nMSG")),
    ("arrow in commit -m body",
     bash(f"git commit -m \"move a -> {V6}/b\"")),
    ("arrow near a protected path",
     bash(f"echo 'see: x -> {RAW}/y'")),
]


@pytest.mark.parametrize("label,payload", ALLOWED_PATH_CASES,
                         ids=[c[0].replace(" ", "-") for c in ALLOWED_PATH_CASES])
def test_legitimate_operations_are_allowed(paths_guard, label, payload):
    assert paths_guard.evaluate(payload) is None, f"{label}: must be allowed"


# The `->` fix must not have weakened real redirection detection.
REDIRECT_STILL_BLOCKED = [
    ("plain >", f"echo x > {RAW}/f"),
    ("append >>", f"echo x >> {ARCHIVE}/f"),
    ("fd 2>", f"cmd 2>{RAW}/e.log"),
    ("fd 1>>", f"echo x 1>>{RAW}/f"),
    ("noclobber >|", f"echo x >| {RAW}/f"),
    ("no space", f"echo x >{RAW}/f"),
]


@pytest.mark.parametrize("label,command", REDIRECT_STILL_BLOCKED,
                         ids=[c[0].replace(" ", "-") for c in REDIRECT_STILL_BLOCKED])
def test_redirection_detection_not_weakened(paths_guard, label, command):
    assert paths_guard.evaluate(bash(command)) is not None, (
        f"{label}: real redirects must still be denied")


def test_protected_list_matches_convention_checker(paths_guard):
    """The guard and the convention checker must not drift apart."""
    sys.path.insert(0, str(ROOT / "tools"))
    import check_conventions as cc
    for prefix in cc.ARCHIVAL_PREFIXES:
        assert prefix in paths_guard.PROTECTED_PREFIXES, f"{prefix} missing from the guard"
    for prefix in ("data/raw/", "data/interim/", "data/processed/"):
        assert prefix in paths_guard.PROTECTED_PREFIXES, f"{prefix} missing from the guard"


# ------------------------------------------------------- guard_destructive_git

BLOCKED_GIT_CASES = [
    ("force push", "git push --force origin r7/weather-reasoning"),
    ("force push short", "git push -f origin HEAD"),
    ("force with lease", "git push --force-with-lease origin HEAD"),
    ("force via refspec", "git push origin +r7/weather-reasoning:r7/weather-reasoning"),
    ("force push main", "git push --force origin main"),
    ("force via bare plus main", "git push origin +main"),
    ("reset hard", "git reset --hard HEAD~1"),
    ("reset hard global opt", "git -C /data/esw/UrbanPiDiT_R2 reset --hard"),
    ("clean -fd", "git clean -fd"),
    ("clean -fdx", "git clean -fdx"),
    ("clean --force", "git clean --force"),
    ("branch -D", "git branch -D experiment"),
    ("discard checkout", "git checkout ."),
    ("discard restore", "git restore ."),
    ("delete default branch", "git push origin :main"),
    ("delete default branch master", "git push origin :master"),
    ("delete default branch flag", "git push origin --delete main"),
    ("mirror push", "git push origin --mirror"),
    ("merge main", "git merge main"),
    ("merge origin main", "git merge origin/main"),
    ("gh pr merge", "gh pr merge --squash"),
    ("chained force push", "cd /tmp && git push --force"),
]


@pytest.mark.parametrize("label,command", BLOCKED_GIT_CASES,
                         ids=[c[0].replace(" ", "-") for c in BLOCKED_GIT_CASES])
def test_destructive_git_is_denied(git_guard, label, command):
    verdict = git_guard.evaluate(bash(command))
    assert verdict is not None, f"{label}: expected a deny verdict for {command!r}"
    _label, advice = verdict
    assert advice.strip(), f"{label}: deny must explain the supported alternative"


ALLOWED_GIT_CASES = [
    ("status", "git status --porcelain"),
    ("log", "git log --oneline -n 100"),
    ("diff", "git diff --stat"),
    ("diff cached", "git diff --cached"),
    ("show check", "git show --check --format= HEAD"),
    ("rev-parse", "git rev-parse HEAD"),
    ("ls-files", "git ls-files"),
    ("grep", "git grep -n 'def run'"),
    ("blame", "git blame -L 1,5 training/r7_evaluate.py"),
    ("stash list", "git stash list"),
    ("branch list", "git branch -a"),
    ("safe branch delete", "git branch -d merged-branch"),
    ("fetch", "git fetch origin"),
    ("plain push branch", "git push origin r7/weather-reasoning"),
    ("ff push to main via branch refspec",
     "git push origin r7/weather-reasoning:main"),
    ("ff push to main via HEAD refspec", "git push origin HEAD:main"),
    ("push main ref", "git push origin main"),
    ("push main with global opt", "git -C /data/esw/UrbanPiDiT_R2 push origin main"),
    ("add", "git add tools/agent_hooks/"),
    ("commit", "git commit -m 'feat(r7): add agent hooks (#59)'"),
    ("checkout new branch", "git checkout -b r7/new-work"),
    ("restore staged file", "git restore --staged training/r7_evaluate.py"),
    ("log grep mentioning reset", "git log --oneline --grep=reset-hard"),
    ("empty command", ""),
]


@pytest.mark.parametrize("label,command", ALLOWED_GIT_CASES,
                         ids=[c[0].replace(" ", "-") for c in ALLOWED_GIT_CASES])
def test_readonly_and_safe_git_is_allowed(git_guard, label, command):
    assert git_guard.evaluate(bash(command)) is None, (
        f"{label}: must be allowed ({command!r}) — read-only git is a primary "
        f"evidence source for this project")


def test_git_guard_ignores_non_bash_tools(git_guard):
    assert git_guard.evaluate({"tool_name": "Read", "tool_input": {"file_path": "x"}}) is None


# --------------------------------------------------- tests/ removal vs authoring
#
# Removing tests is what the hard constraints forbid. R-009 counts test
# functions and assertions in aggregate and is report-only, so a whole-file
# deletion would slip past it until the totals dropped below a baseline that is
# easy to bump. Adding, editing and running tests are NOT restricted.

TESTS_REMOVAL_CASES = [
    ("rm a test file", "rm tests/test_forward.py"),
    ("rm -rf the tests tree", "rm -rf tests"),
    ("unlink a test file", "unlink tests/test_forward.py"),
    ("git rm a test file", "git rm tests/test_forward.py"),
    ("find -delete over tests", "find tests -name 'test_*.py' -delete"),
]


@pytest.mark.parametrize("label,command", TESTS_REMOVAL_CASES,
                         ids=[c[0].replace(" ", "-") for c in TESTS_REMOVAL_CASES])
def test_test_removal_is_denied(paths_guard, label, command):
    verdict = paths_guard.evaluate(bash(command))
    assert verdict is not None, f"{label}: deleting tests must be denied"
    prefix, what = verdict
    assert "removal of test files" in paths_guard._reason(prefix, what)


TESTS_AUTHORING_CASES = [
    ("author a new test file", write_tool("tests/test_new_feature.py")),
    ("edit an existing test", edit_tool("tests/test_forward.py")),
    ("edit the hook tests themselves", edit_tool("tests/test_agent_hooks.py")),
    ("stage a test file", bash("git add tests/test_agent_hooks.py")),
    ("commit test changes", bash("git commit -m 'test(r7): extend coverage (#60)'")),
]


@pytest.mark.parametrize("label,payload", TESTS_AUTHORING_CASES,
                         ids=[c[0].replace(" ", "-") for c in TESTS_AUTHORING_CASES])
def test_test_authoring_is_not_blocked(paths_guard, label, payload):
    """Only removal is guarded; writing and editing tests stay free."""
    assert paths_guard.evaluate(payload) is None, f"{label}: must not be blocked"


# ---------------------------------------------------- project workflows allowed
#
# The guard must not break the repository's own documented commands. These write
# into data/ by design, so blocking them would be a self-inflicted wound.

WORKFLOW_CASES = [
    ("try_real_downloads", "python scripts/try_real_downloads.py"),
    ("prepare_real_smoke", "python scripts/prepare_real_smoke.py"),
    ("arco downloader module", "python -m data.download.arco_era5"),
    ("worldcover downloader module", "python -m data.download.worldcover_cog"),
    ("uci downloader module", "python -m data.download.uci_beijing"),
    ("run one test file", "python -m pytest tests/test_forward.py -q"),
    ("python -c without a write", "python -c \"import torch;print(1)\""),
    ("echo quoting a redirect", "echo \"see > " + RAW + "/x\""),
    ("grep quoting a redirect", f"grep -rn 'foo > {RAW}/x' docs/"),
]


@pytest.mark.parametrize("label,command", WORKFLOW_CASES,
                         ids=[c[0].replace(" ", "-") for c in WORKFLOW_CASES])
def test_project_workflows_are_not_blocked(paths_guard, label, command):
    assert paths_guard.evaluate(bash(command)) is None, (
        f"{label}: must not be blocked ({command!r})")


# ---------------------------------------------------- check_model_digest_impact

def test_digest_notice_fires_for_model_source(digest_notice):
    for path in ("model/cross_scale_adapter.py", "model/process_reasoner.py",
                 "model/layers/sdpa.py", "./model/urban_expert.py"):
        message = digest_notice.evaluate(write_tool(path))
        assert message, f"{path} must trigger the digest notice"
        assert "model_code_sha256" in message
        assert "[model-digest-change]" in message


def test_digest_notice_skips_archived_model_code(digest_notice):
    """model_code_digest() skips paths containing 'legacy'."""
    assert digest_notice.evaluate(write_tool("model/legacy_v531/urban_pidit.py")) is None


def test_digest_notice_skips_uncovered_scopes(digest_notice):
    """training/, scripts/, tests/ and data/ are outside model_code_digest."""
    for path in ("training/r7_evaluate.py", "scripts/train_r7.py",
                 "tests/test_forward.py", "data/multiscale_dataset.py",
                 "legacy_v6/legacy_physical_consistency.py",
                 "model/README.md"):
        assert digest_notice.evaluate(write_tool(path)) is None, f"{path} must not trigger"


def test_digest_notice_ignores_read_tools(digest_notice):
    assert digest_notice.evaluate(
        {"tool_name": "Read", "tool_input": {"file_path": "model/urban_expert.py"}}) is None
    assert digest_notice.evaluate({}) is None


# ------------------------------------------------------------------- CLI layer

CLI_CASES = [
    ("guard_protected_paths", bash(f"rm -rf {RAW}"), 2),
    ("guard_protected_paths", bash("git status"), 0),
    ("guard_destructive_git", bash("git push --force"), 2),
    ("guard_destructive_git", bash("git log --oneline"), 0),
    ("check_model_digest_impact", write_tool("model/coarse_encoder.py"), 0),
]


@pytest.mark.parametrize("module_name,payload,expected", CLI_CASES,
                         ids=[f"{c[0]}-rc{c[2]}" for c in CLI_CASES])
def test_cli_exit_codes(module_name, payload, expected, monkeypatch, capsys):
    module = _load(module_name)
    monkeypatch.setattr(sys, "stdin", _Stdin(json.dumps(payload)))
    code = module.main()
    capsys.readouterr()
    assert code == expected, f"{module_name}: rc={code}, expected {expected}"


@pytest.mark.parametrize("module_name", ["guard_protected_paths", "guard_destructive_git",
                                         "check_model_digest_impact"])
def test_every_hook_fails_open_on_garbage_input(module_name, monkeypatch, capsys):
    """A broken guard must not wedge the session."""
    module = _load(module_name)
    monkeypatch.setattr(sys, "stdin", _Stdin("not json at all"))
    code = module.main()
    capsys.readouterr()
    assert code == 0, f"{module_name} must fail open on garbage input"


@pytest.mark.parametrize("module_name", ["guard_protected_paths", "guard_destructive_git",
                                         "check_model_digest_impact"])
def test_every_hook_tolerates_empty_stdin(module_name, monkeypatch, capsys):
    """An empty payload (no tool info) must be a no-op, not a crash."""
    module = _load(module_name)
    monkeypatch.setattr(sys, "stdin", _Stdin(""))
    code = module.main()
    capsys.readouterr()
    assert code == 0, f"{module_name} must no-op on empty input"


def test_digest_notice_emits_only_a_system_message(digest_notice, monkeypatch, capsys):
    """PostToolUse output must be valid JSON with no extra keys (strict schema)."""
    monkeypatch.setattr(sys, "stdin", _Stdin(json.dumps(write_tool("model/coarse_encoder.py"))))
    code = digest_notice.main()
    out = capsys.readouterr().out.strip()
    assert code == 0
    assert out, "a digest-affecting edit must produce a message"
    parsed = json.loads(out)
    assert set(parsed) == {"systemMessage"}, f"unexpected keys: {set(parsed)}"


def test_digest_notice_is_silent_when_not_applicable(digest_notice, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", _Stdin(json.dumps(write_tool("training/lit_module.py"))))
    code = digest_notice.main()
    out = capsys.readouterr().out.strip()
    assert code == 0
    assert out == "", "an out-of-scope edit must emit nothing at all"
