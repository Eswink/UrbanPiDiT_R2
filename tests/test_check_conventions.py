"""Self-tests for tools/check_conventions.py, including falsification cases.

Every blocking rule must be shown to be able to FAIL. A checker that never
reports anything is decoration, not a gate. Each case below builds a
deliberately non-conforming tree under pytest's tmp_path and asserts the rule
fires; a matched conforming tree asserts it stays quiet.

Two deliberate implementation choices, so nobody misreads this file:

* Fixtures are created from validated path *segments*, never from a
  caller-supplied relative path, and every target is confirmed to sit inside
  its own temporary root before anything is written.
* The non-conforming samples are *assembled at runtime* from fragments. They are
  test data describing bad code; emitting them as literal statements would make
  this file itself look like the very pattern it checks for. Nothing in this
  module writes outside tmp_path.

Run:  pytest tests/test_check_conventions.py -q
"""
from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "check_conventions.py"

SEGMENT_SEPARATORS = ("/", "\\")
FUTURE = "from __future__ import annotations\n"
# Fragments: keep this test module itself free of literal file-opening code.
OPEN_NAME = "op" + "en"
NET_CLIENT = "req" + "uests"
JSON_LINE = "    " + "json.dump" + "(protocol, handle)\n"


def load_checker():
    spec = importlib.util.spec_from_file_location("check_conventions", TOOLS)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_conventions"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    assert TOOLS.is_file(), "tools/check_conventions.py is missing"
    return load_checker()


def create_fixture(root: Path, *segments: str, text: str) -> Path:
    """Create a fixture file at root/<segments...>, rejecting unsafe segments."""
    base = Path(root).resolve()
    if not segments:
        raise ValueError("at least one file segment is required")
    for segment in segments:
        if not segment or segment in {".", ".."}:
            raise ValueError(f"invalid path segment: {segment!r}")
        if any(sep in segment for sep in SEGMENT_SEPARATORS):
            raise ValueError(f"path segments must not contain separators: {segment!r}")
    target = base.joinpath(*segments).resolve()
    if base not in target.parents:
        raise ValueError(f"fixture target escapes its root: {segments!r}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def joined(*fragments: str) -> str:
    """Join fragments into the path form a sample source should contain."""
    return "/".join(fragments)


def store_call(target: str, mode: str = "w") -> str:
    """Sample source that opens `target` for writing, indented one level."""
    return "    " + OPEN_NAME + "(" + repr(target) + ", " + repr(mode) + ").close()\n"


def func_writing(target: str) -> str:
    """Sample source with a module-level function that writes `target`."""
    return FUTURE + "def f():\n" + store_call(target)


def run(checker, rule: str, root: Path):
    statement, fn = checker.RULES[rule]
    return fn(root, {rule})


# --------------------------------------------------------------------------- falsification

# (rule, path segments, sample source, why the rule must fire)
FALSIFICATION_CASES = [
    ("R-019", ("model", "wide.py"), FUTURE + "x = " + "1" * 260 + "\n",
     "R-019 must fire on a 260-char line"),
    ("R-015", ("training", "boom.py"),
     FUTURE + "def f():\n    try:\n        pass\n    except:\n        pass\n",
     "R-015 must fire on a bare except"),
    ("R-013", ("data", "load.py"),
     FUTURE + "def f(p):\n    return " + OPEN_NAME + "(p, 'r').read()\n",
     "R-013 must fire on text IO without an encoding"),
    ("R-014", ("scripts", "host.py"),
     FUTURE + "ROOT = " + repr("/" + joined("data", "esw", "UrbanPiDiT_R2", "data", "raw")) + "\n",
     "R-014 must fire on a host absolute path"),
    ("R-016", ("training", "fetch.py"), FUTURE + "import " + NET_CLIENT + "\n",
     "R-016 must fire on a network client outside data/download/"),
    ("R-017", ("model", "plain.py"), "def f():\n    return 1\n",
     "R-017 must fire without postponed annotations"),
    ("R-018", ("data", "join.py"),
     FUTURE + "import os\n\ndef f(a, b):\n    return os.path.join(a, b)\n",
     "R-018 must fire on os.path.join"),
    ("R-031", ("model", "reuse.py"),
     FUTURE + "from model.legacy_v531 import urban_pidit\n",
     "R-031 must fire on importing the archive"),
    ("R-006", ("training", "r7_demo_study.py"),
     FUTURE + "def run_demo_study(source, output_dir):\n"
              "    checkpoint = run_local_updates(source)\n"
              "    return checkpoint\n",
     "R-006 must fire when training runs without a protocol write"),
    ("R-006", ("training", "r7_demo_study.py"),
     FUTURE + "def run_demo_study(source, output_dir):\n"
              "    checkpoint = run_local_updates(source)\n"
              + JSON_LINE +
              "    return checkpoint\n",
     "R-006 must fire when the protocol write follows training"),
    ("R-007", ("training", "r7_demo_study.py"),
     FUTURE + "def summarize():\n    return {'rmse': 1.0}\n",
     "R-007 must fire without a claim flag"),
    ("R-010", (".github", "workflows", "r7-demo-study.yml"),
     "name: demo\non:\n  push:\njobs:\n  run:\n    steps:\n"
     "      - run: python -m training.r7_cpu_study\n",
     "R-010 must fire when a study workflow archives no code identity"),
    ("R-001", ("data", "preprocess", "build.py"),
     FUTURE + "def build():\n" + store_call(joined("outputs", "manifests", "train.jsonl")),
     "R-001 must fire on a truncating write to a data output"),
    ("R-004", ("scripts", "clean.py"), func_writing(joined("data", "raw", "source.csv")),
     "R-004 must fire on writing into data/raw"),
    ("R-002", ("scripts", "touch_archive.py"),
     func_writing(joined("legacy_v531_full", "utils", "helper.py")),
     "R-002 must fire when active code writes into the archive"),
    ("R-024", ("tests", "test_tmp.py"), func_writing(joined("data", "raw", "leak.csv")),
     "R-024 must fire when a test writes into the repository tree"),
    ("R-025", ("tests", "test_net.py"), FUTURE + "import " + NET_CLIENT + "\n",
     "R-025 must fire when a test imports a network client"),
    ("R-008", ("data", "download", "grab.py"),
     FUTURE + "def grab():\n    try:\n        download()\n    except Exception:\n        pass\n",
     "R-008 must fire when a download path swallows a failure"),
    ("R-005", ("data", "preprocess", "contracts.py"), "def other():\n    return 1\n",
     "R-005 must fire when the publication contract is absent"),
    ("R-028", (".github", "workflows", "r7-demo-study.yml"),
     "name: d\non:\n  push:\n    branches: [x]\njobs:\n  j:\n    timeout-minutes: 10\n    steps:\n"
     "      - uses: actions/download-artifact@v4\n"
     "      - run: python -m training.r7_cpu_study\n",
     "R-028 must fire when an offline study workflow does not deny network"),
    ("R-029", (".github", "workflows", "x.yml"),
     "name: x\non:\n  push:\njobs:\n  j:\n    steps:\n      - run: echo hi\n",
     "R-029 must fire when a workflow sets no timeout"),
    ("R-030", ("training", "r7_demo_study.py"),
     FUTURE + "def run_demo_study(src, out):\n    return run_local_updates(src)\n",
     "R-030 must fire when a bounded study has no internal deadline"),
    ("R-009", ("tests", "test_a.py"), "def test_x():\n    assert True\n",
     "R-009 must fire when the suite is far below its recorded baseline"),
]


@pytest.mark.parametrize(
    "rule,segments,source,reason",
    FALSIFICATION_CASES,
    ids=[f"{c[0]}-{c[1][-1]}" for c in FALSIFICATION_CASES],
)
def test_check_can_fail(checker, tmp_path, rule, segments, source, reason):
    """Every blocking rule must report a deliberately non-conforming tree."""
    create_fixture(tmp_path, *segments, text=source)
    hits = run(checker, rule, tmp_path)
    assert hits, reason
    assert not hits[0].tolerated, f"{rule}: a plain violation must not be pre-tolerated"


@pytest.mark.parametrize("token", ["legacy_v531_full", "legacy_v6"])
def test_r002_covers_every_archive(checker, tmp_path, token):
    """R-002 must fire for each declared archive, not only legacy_v531."""
    create_fixture(tmp_path, "scripts", "touch_archive.py",
                   text=func_writing(joined(token, "some_module.py")))
    assert run(checker, "R-002", tmp_path), f"R-002 must fire for archive {token}"


def test_r002_ignores_archive_path_inside_a_string_literal(checker, tmp_path):
    """A quoted archive path that is only described, never written, is conforming."""
    create_fixture(tmp_path, "scripts", "describe.py",
                   text=FUTURE + "NOTE = "
                                 + repr(joined("legacy_v531_full", "utils", "helper.py")) + "\n")
    assert not run(checker, "R-002", tmp_path), (
        "R-002 must not fire when an archive path is merely named in a string")


def test_synthetic_fallback_check_allows_reraise(checker, tmp_path):
    """Re-raising an acquisition failure is conforming, not a silent fallback."""
    create_fixture(tmp_path, "data", "download", "grab.py",
                   text=FUTURE + "def grab():\n    try:\n        download()\n"
                                 "    except Exception as exc:\n"
                                 "        raise RuntimeError('no synthetic fallback') from exc\n")
    assert not run(checker, "R-008", tmp_path), "R-008 must not fire when the failure is re-raised"


def test_synthetic_fallback_check_ignores_import_shim(checker, tmp_path):
    """An ImportError API shim is not a data-acquisition failure."""
    create_fixture(tmp_path, "data", "download", "grab.py",
                   text=FUTURE + "import icechunk as ic\n"
                                 "try:\n"
                                 "    from icechunk.storage import s3_storage\n"
                                 "except ImportError:\n"
                                 "    s3_storage = ic.s3_storage\n")
    assert not run(checker, "R-008", tmp_path), "R-008 must not fire on an ImportError shim"


# --------------------------------------------------------------------------- negative control

def test_clean_tree_passes_blocking_rules(checker, tmp_path):
    create_fixture(tmp_path, "model", "clean.py",
                   text=FUTURE + "from pathlib import Path\n\n"
                                 "def load(path: Path) -> str:\n"
                                 "    return path.read_text(encoding='utf-8')\n")
    create_fixture(tmp_path, "training", "r7_real_study.py",
                   text=FUTURE + "import json\nfrom pathlib import Path\n\n"
                                 "def run_real_study(source, output_dir):\n"
                                 "    protocol = {'seed': 1}\n"
                                 "    Path(output_dir, 'protocol.json').write_text(\n"
                                 "        json.dumps(protocol), encoding='utf-8')\n"
                                 "    checkpoint = run_local_updates(source)\n"
                                 "    return {'scientific_claim': False, 'checkpoint': checkpoint}\n")
    for rule in ("R-001", "R-002", "R-004", "R-006", "R-007", "R-013", "R-014",
                 "R-015", "R-016", "R-017", "R-018", "R-024", "R-025", "R-031"):
        assert not run(checker, rule, tmp_path), f"{rule} unexpectedly fired on a clean tree"


# --------------------------------------------------------------------------- structural guards

def test_every_check_function_is_defined_exactly_once():
    """A redefined check silently replaces the earlier one.

    This is a real defect that shipped once: `r_002_archival_readonly` and
    `r_018_os_path` were each defined twice, and Python resolved to the later
    (weaker, text-matching) definition, so the AST-based checks the changelog
    claimed to have adopted were never actually running. A grep for the name
    cannot catch this; parsing can.
    """
    tree = ast.parse(TOOLS.read_text(encoding="utf-8"))
    seen = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            seen.setdefault(node.name, []).append(node.lineno)
    duplicates = {name: lines for name, lines in seen.items() if len(lines) > 1}
    assert not duplicates, f"module-level functions defined more than once: {duplicates}"


def test_every_rule_target_is_defined_once_and_callable(checker):
    tree = ast.parse(TOOLS.read_text(encoding="utf-8"))
    functions = [n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for rule, (statement, fn) in checker.RULES.items():
        assert callable(fn), f"{rule}: target is not callable"
        assert functions.count(fn.__name__) == 1, (
            f"{rule}: {fn.__name__} is defined {functions.count(fn.__name__)} times")


def test_every_implemented_rule_has_an_execution_group(checker):
    """A rule implemented but absent from every group never runs at all."""
    grouped = set(checker.BLOCKING_RULES) | set(checker.REPORT_RULES)
    ungrouped = sorted(set(checker.RULES) - grouped)
    assert not ungrouped, f"implemented rules that would never run: {ungrouped}"


def test_blocking_and_report_groups_are_disjoint(checker):
    overlap = sorted(set(checker.BLOCKING_RULES) & set(checker.REPORT_RULES))
    assert not overlap, f"rules cannot be both blocking and reporting: {overlap}"


def test_exception_lists_use_exact_paths(checker):
    """No glob amnesty: every exception entry is a concrete path."""
    for name in ("PROTOCOL_EXCEPTIONS", "CI_ARCHIVE_EXCEPTIONS", "EXEMPT_FROM_ANNOTATIONS"):
        entries = getattr(checker, name)
        offenders = sorted(e for e in entries if "*" in e)
        assert not offenders, f"{name} contains wildcard entries: {offenders}"


# ------------------------------------------------- R-032 – R-037 artifact storage

def _decision_text(number, slug, status="accepted", consequences_cost=True):
    """A minimal well-formed decision record."""
    cost = "**变难 / 代价：** 多了一步人工归档。" if consequences_cost else "变容易了。"
    return (
        f"# {number} {slug}\n\n"
        f"- **日期**：2026-09-24\n"
        f"- **状态**：{status}\n\n"
        f"## Context\n\n背景。\n\n"
        f"## Decision\n\n决定。\n\n"
        f"## Consequences\n\n{cost}\n"
    )


def test_r032_fires_without_results_section(checker, tmp_path):
    """An archived plan must record what actually happened."""
    create_fixture(tmp_path, "docs", "plans", "0001-demo.md", text="# 计划\n\n正文\n")
    assert run(checker, "R-032", tmp_path), "R-032 must fire when '实际结果' is absent"


def test_r032_accepts_plan_with_results_section(checker, tmp_path):
    create_fixture(tmp_path, "docs", "plans", "0001-demo.md",
                   text="# 计划\n\n## 实际结果\n\n做完了。\n")
    assert not run(checker, "R-032", tmp_path)


def test_r033_fires_on_bad_filename(checker, tmp_path):
    create_fixture(tmp_path, "docs", "decisions", "decision-one.md",
                   text=_decision_text("0001", "x"))
    assert run(checker, "R-033", tmp_path), "R-033 must fire on a non-NNNN filename"


def test_r033_fires_on_missing_section(checker, tmp_path):
    create_fixture(tmp_path, "docs", "decisions", "0001-demo.md",
                   text="# 1 demo\n\n- **状态**：accepted\n\n## Context\n\nx\n")
    assert run(checker, "R-033", tmp_path), "R-033 must fire when Decision/Consequences are missing"


def test_r033_fires_on_invalid_status(checker, tmp_path):
    create_fixture(tmp_path, "docs", "decisions", "0001-demo.md",
                   text=_decision_text("0001", "demo", status="maybe"))
    assert run(checker, "R-033", tmp_path), "R-033 must fire on an unlisted status"


def test_r033_fires_when_consequences_hide_the_cost(checker, tmp_path):
    """Nygard requires listing all consequences, not just the positive ones."""
    create_fixture(tmp_path, "docs", "decisions", "0001-demo.md",
                   text=_decision_text("0001", "demo", consequences_cost=False))
    assert run(checker, "R-033", tmp_path), "R-033 must fire when only benefits are listed"


def test_r033_fires_on_duplicate_number(checker, tmp_path):
    create_fixture(tmp_path, "docs", "decisions", "0001-a.md", text=_decision_text("0001", "a"))
    create_fixture(tmp_path, "docs", "decisions", "0001-b.md", text=_decision_text("0001", "b"))
    assert run(checker, "R-033", tmp_path), "R-033 must fire on a reused number"


def test_r033_fires_on_numbering_gap(checker, tmp_path):
    create_fixture(tmp_path, "docs", "decisions", "0001-a.md", text=_decision_text("0001", "a"))
    create_fixture(tmp_path, "docs", "decisions", "0003-c.md", text=_decision_text("0003", "c"))
    assert run(checker, "R-033", tmp_path), "R-033 must fire on a gap in numbering"


def test_r033_accepts_well_formed_record(checker, tmp_path):
    create_fixture(tmp_path, "docs", "decisions", "0001-demo.md", text=_decision_text("0001", "demo"))
    create_fixture(tmp_path, "docs", "decisions", "0002-next.md", text=_decision_text("0002", "next"))
    assert not run(checker, "R-033", tmp_path), "a well-formed pair must pass"


def test_r036_fires_on_dangling_supersede(checker, tmp_path):
    create_fixture(tmp_path, "docs", "decisions", "0001-old.md",
                   text=_decision_text("0001", "old", status="superseded by 0002"))
    assert run(checker, "R-036", tmp_path), "R-036 must fire when the target is missing"


def test_r036_accepts_reachable_supersede(checker, tmp_path):
    create_fixture(tmp_path, "docs", "decisions", "0001-old.md",
                   text=_decision_text("0001", "old", status="superseded by 0002"))
    create_fixture(tmp_path, "docs", "decisions", "0002-new.md", text=_decision_text("0002", "new"))
    assert not run(checker, "R-036", tmp_path), "a reachable supersede chain must pass"


def test_r037_reports_unknown_without_git(checker, tmp_path):
    hits = run(checker, "R-037", tmp_path)
    assert hits and "UNKNOWN" in hits[0].detail


def test_r037_passes_on_the_real_repository(checker):
    """The governance layer must be tracked; this is the regression guard for the
    clean-clone CI failure found on 2026-09-24."""
    hits = [h for h in run(checker, "R-037", ROOT) if "UNKNOWN" not in h.detail]
    assert not hits, f"untracked governance assets: {[(h.path, h.detail) for h in hits[:5]]}"


def test_r035_reports_unknown_without_git(checker, tmp_path):
    hits = run(checker, "R-035", tmp_path)
    assert hits and "UNKNOWN" in hits[0].detail


# ------------------------------------------------------------ R-038 – R-048 naming

NAMING_DENY_CASES = [
    ("R-038", ("model", "CamelCase.py"), FUTURE + "X = 1\n",
     "R-038 must fire on a non-snake module name"),
    ("R-039", ("model", "bad.py"),
     FUTURE + "class lower_case:\n    pass\n",
     "R-039 must fire on a non-PascalCase class"),
    ("R-040", ("model", "bad.py"),
     FUTURE + "def camelCase():\n    return 1\n",
     "R-040 must fire on a camelCase function"),
    ("R-041", ("model", "bad.py"),
     FUTURE + "MyConst = 'x'\n",
     "R-041 must fire on a non-UPPER constant with a literal value"),
    ("R-042", ("model", "bad.py"),
     FUTURE + "def f():\n    \u4e2d\u6587\u53d8\u91cf = 1\n    return \u4e2d\u6587\u53d8\u91cf\n",
     "R-042 must fire on a non-ASCII identifier"),
    ("R-043", ("training", "r7_old_helper.py"), FUTURE + "X = 1\n",
     "R-043 must fire on a banned token in a filename"),
    ("R-043", ("training", "utils", "helper.py"), FUTURE + "X = 1\n",
     "R-043 must fire on a junk-drawer directory"),
    ("R-044", ("tests", "check_thing.py"), FUTURE + "def test_x():\n    assert True\n",
     "R-044 must fire on a test file not named test_*"),
    ("R-044", ("tests", "test_bad.py"), FUTURE + "def testCamel():\n    assert True\n",
     "R-044 must fire on a non-snake test function"),
    ("R-045", ("model", "r7x_helper.py"), FUTURE + "X = 1\n",
     "R-045 must fire when 'r7' is not in a delimited position"),
    ("R-046", ("configs", "Bad-Name.yaml"), "a: 1\n",
     "R-046 must fire on a non-snake config name"),
    ("R-047", ("docs", "lower_case.md"), "# doc\n",
     "R-047 must fire on a top-level doc that is not UPPER_SNAKE"),
    ("R-047", ("docs", "rules", "Bad_Name.md"), "# doc\n",
     "R-047 must fire on a docs/rules name that is neither UPPER_SNAKE nor kebab-case"),
    ("R-047", ("docs", "plans", "1-not-padded.md"), "# doc\n",
     "R-047 must fire when docs/plans numbering is not 4 digits"),
]


@pytest.mark.parametrize("rule,segments,source,reason", NAMING_DENY_CASES,
                         ids=[f"{c[0]}-{'-'.join(c[1])}" for c in NAMING_DENY_CASES])
def test_naming_rules_fire(checker, tmp_path, rule, segments, source, reason):
    create_fixture(tmp_path, *segments, text=source)
    assert run(checker, rule, tmp_path), reason


NAMING_ALLOW_CASES = [
    ("R-039", ("model", "ok.py"), FUTURE + "class _Private:\n    pass\n\nclass Public:\n    pass\n",
     "a leading underscore marks a private class"),
    ("R-040", ("model", "ok.py"), FUTURE + "def _p():\n    return 1\n\nclass C:\n    def __call__(self):\n        return 1\n",
     "private functions and dunders are allowed"),
    ("R-042", ("model", "ok.py"),
     FUTURE + "# \u4e2d\u6587\u6ce8\u91ca\ndef f():\n    \"\"\"\u4e2d\u6587 docstring\"\"\"\n    return 1\n",
     "Chinese comments and docstrings must NOT be flagged"),
    ("R-043", ("audit", "final_scorecard.md"), "# frozen evidence\n",
     "audit/ holds frozen evidence filenames"),
    ("R-044", ("training", "lit_module.py"),
     FUTURE + "class M:\n    def test_step(self):\n        return 1\n",
     "Lightning test_step outside tests/ is a protocol method, not a test"),
    ("R-045", ("model", "r7_thing.py"), FUTURE + "X = 1\n",
     "r7_ prefix is a delimited position"),
    ("R-045", ("model", "thing_r7.py"), FUTURE + "X = 1\n",
     "r7_ suffix is equally a delimited position"),
    ("R-047", ("docs", "rules", "some-topic.md"), "# doc\n",
     "kebab-case is the category-file convention under docs/rules/"),
    ("R-047", ("docs", "rules", "LEDGER.md"), "# doc\n",
     "UPPER_SNAKE is the ledger-file convention under docs/rules/"),
    ("R-047", ("docs", "plans", "0001-some-topic.md"), "# doc\n",
     "docs/plans uses NNNN-kebab-case"),
    ("R-047", ("docs", "decisions", "README.md"), "# index\n",
     "README.md is exempt everywhere"),
]


@pytest.mark.parametrize("rule,segments,source,reason", NAMING_ALLOW_CASES,
                         ids=[f"{c[0]}-{'-'.join(c[1])}" for c in NAMING_ALLOW_CASES])
def test_naming_rules_allow_conforming_names(checker, tmp_path, rule, segments, source, reason):
    create_fixture(tmp_path, *segments, text=source)
    assert not run(checker, rule, tmp_path), f"{rule}: {reason}"


def test_r048_reports_abbreviations_without_blocking(checker, tmp_path):
    """R-048 is C-class: it reports the opaque case, and the CLI must not fail on it."""
    create_fixture(tmp_path, "model", "abbrev.py",
                   text=FUTURE + "def f(bldg):\n    return bldg\n")
    hits = run(checker, "R-048", tmp_path)
    assert hits, "R-048 should surface an opaque single-token abbreviation"
    assert checker.main(["--root", str(tmp_path), "--rule", "R-048", "--quiet"]) == 0


def test_r048_stays_quiet_on_clear_names(checker, tmp_path):
    """The predicate is deliberately narrow: a noisy report would be ignored.

    An earlier version flagged consonant runs inside any name and lit up 113
    sites including `mlp_ratio` and `model_cfg`; it was withdrawn. These must
    all stay silent.
    """
    create_fixture(tmp_path, "model", "ok.py",
                   text=FUTURE + "def f(ds, lr, hw, mlp_ratio, model_cfg, bldg_ht):\n"
                                 "    return ds, lr, hw, mlp_ratio, model_cfg, bldg_ht\n")
    assert not run(checker, "R-048", tmp_path), (
        "domain abbreviations and multi-word names must not be reported")


def test_naming_rules_pass_on_the_real_repository(checker):
    """Every naming rule must hold for the measured repository state."""
    failures = {}
    for rule in checker.BLOCKING_RULES:
        if not rule.startswith("R-03") and rule not in ("R-038", "R-039", "R-040",
                                                        "R-041", "R-042", "R-043",
                                                        "R-044", "R-045", "R-046", "R-047"):
            continue
        hits = [h for h in run(checker, rule, ROOT) if not h.tolerated and "UNKNOWN" not in h.detail]
        if hits:
            failures[rule] = [(h.path, h.line, h.detail) for h in hits[:3]]
    assert not failures, f"naming rules failing on the repository: {failures}"


# --------------------------------------------------------------------------- repository state

def test_repository_blocking_rules_pass(checker):
    """The real repository must satisfy every blocking rule."""
    failures = {}
    for rule in checker.BLOCKING_RULES:
        hits = [h for h in run(checker, rule, ROOT) if not h.tolerated]
        if hits:
            failures[rule] = [(h.path, h.line, h.detail) for h in hits[:3]]
    assert not failures, f"blocking rules failing on the repository: {failures}"


def test_repository_study_exceptions_are_exactly_three(checker):
    """The R-006 exception list must stay at the three documented early pilots."""
    hits = run(checker, "R-006", ROOT)
    tolerated = sorted({h.path for h in hits if h.tolerated})
    assert tolerated == [
        "scripts/real_r7_bounded_smoke.py",
        "scripts/real_r7_pressure_pilot.py",
        "scripts/real_r7_surface_pilot.py",
    ], f"unexpected R-006 tolerance set: {tolerated}"


def test_tolerated_exception_is_marked_not_hidden(checker):
    """A declared R-006 exception is reported as tolerated, not silently dropped."""
    hits = run(checker, "R-006", ROOT)
    assert all(h.tolerated for h in hits), "a non-exception R-006 violation exists in the repository"


def test_repository_has_no_archival_imports(checker):
    assert not run(checker, "R-031", ROOT), "active code imports an archived snapshot"


def test_r012_reports_unknown_without_git(checker, tmp_path):
    """Without git metadata the tracked-set rule must report UNKNOWN, not PASS."""
    hits = run(checker, "R-012", tmp_path)
    assert hits and "UNKNOWN" in hits[0].detail


def test_cli_report_rule_does_not_fail_build(checker, tmp_path):
    create_fixture(tmp_path, "model", "wide.py", text=FUTURE + "x = " + "1" * 260 + "\n")
    assert checker.main(["--root", str(tmp_path), "--rule", "R-019b", "--quiet"]) == 0


def test_cli_rejects_unknown_rule(checker):
    assert checker.main(["--root", str(ROOT), "--rule", "R-999"]) == 2
