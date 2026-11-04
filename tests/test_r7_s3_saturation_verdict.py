"""CPU counterproofs for the Stage B saturation verdict tool; no GPU, no real dataset.

The rule this tool applies was frozen in the goal document §3 before any of the four points
ran, so the tests that matter are the ones that pin the frozen boundaries exactly and the ones
that prove the tool refuses input it cannot verify -- a verdict computed from an unverified
reading is the specific failure this campaign exists to avoid.

Fixtures are synthetic dictionaries written to tmp_path; nothing here is weather.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import r7_s3_saturation_verdict as tool


def _write(tmp_path, point, updates, skill, *, batch=4, verdict=None, protocol_mismatch=False):
    """A minimal but self-consistent attempt directory."""
    protocol = {"protocol_sha256": None, "curve": {"point": point, "updates": updates}}
    from training.r7_experiment import canonical_digest
    protocol.pop("protocol_sha256")
    protocol["protocol_sha256"] = canonical_digest(protocol)
    folder = tmp_path / f"attempt_{point}"
    folder.mkdir()
    (folder / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    variables = ("t2m", "u10")
    leads = {str(lead): {"rmse": {v: 1.0 for v in variables},
                         "skill": {v: (skill if v == "t2m" else None) for v in variables}}
             for lead in (6, 12, 24, 48, 72)}
    readings = {"scientific_claim": False, "test_read": False,
                "protocol_sha256": ("0" * 64 if protocol_mismatch else protocol["protocol_sha256"]),
                "curve_point": point, "updates": updates, "batch_size": batch,
                "saturation_verdict": verdict, "leads": leads}
    (folder / "readings.json").write_text(json.dumps(readings), encoding="utf-8")
    return folder


def test_the_two_boundaries_are_the_frozen_numbers():
    """0.01 and 0.005 are the goal document's thresholds; this tool must not retune them."""
    assert tool.UNSATURATED_DELTA == 0.01
    assert tool.SATURATED_DELTA == 0.005
    assert (tool.LOW_POINT, tool.LOW_UPDATES) == ("p4000", 32132)
    assert (tool.HIGH_POINT, tool.HIGH_UPDATES) == ("p1600", 128528)


def _pair(low_skill, high_skill):
    return ({"point": "p4000", "updates": 32132, "skill": low_skill, "rmse": 1.0},
            {"point": "p1600", "updates": 128528, "skill": high_skill, "rmse": 1.0})


@pytest.mark.parametrize("low,high,expected", [
    # delta strictly above 0.01 -> unsaturated
    (0.10, 0.115, "unsaturated"),
    (0.10, 0.1101, "unsaturated"),
    # delta strictly below 0.005 AND high below low -> saturated
    (0.10, 0.096, "saturated"),
    (0.10, 0.0999, "saturated"),
    # A decline of ANY size is saturated: because a negative delta is always < 0.005, the frozen
    # rule's saturated branch reduces to "16 epoch is below 4 epoch". This asymmetry is real and
    # worth seeing plainly: a 0.0005 decline and a 0.06 decline are both saturated, while a
    # 0.004 RISE is record-only. The tool applies the rule as written rather than repairing it.
    (0.10, 0.0995, "saturated"),
    (0.10, 0.094, "saturated"),
    (0.10, 0.04, "saturated"),
    # Exactly zero delta is NOT saturated: 0 < 0.005 holds but high < low does not, and the rule
    # requires BOTH clauses. It is also not unsaturated, so a flat curve lands in record-only.
    (0.10, 0.10, "record-only"),
    # exactly at a boundary is NOT above/below it: 0.01 is not > 0.01
    (0.10, 0.110, "record-only"),
    (0.10, 0.105, "record-only"),
    (0.10, 0.1049, "record-only"),
])
def test_the_three_outcomes_follow_the_frozen_rule(low, high, expected):
    body = tool.verdict(*_pair(low, high))
    assert body["outcome"] == expected
    assert body["delta"] == pytest.approx(high - low)


def test_saturation_requires_both_a_small_delta_and_a_decline():
    """A big drop is not automatically 'saturated' by size alone and a small rise never is."""
    # Large decline, definitely below the threshold, and below the low point -> saturated.
    assert tool.verdict(*_pair(0.20, 0.05))["outcome"] == "saturated"
    # Small positive delta: cannot be saturated even though it is below 0.005.
    assert tool.verdict(*_pair(0.10, 0.101))["outcome"] == "record-only"


def test_a_saturated_verdict_is_returned_verbatim():
    """The rule's 'saturated' branch contradicts the campaign's premise; it must still be returned."""
    body = tool.verdict(*_pair(0.15, 0.10))
    assert body["outcome"] == "saturated"
    assert "saturation point" in body["action"]
    assert "no extrapolation" in body["action"]


def test_unsaturated_names_the_range_but_not_a_configuration():
    """Picking the final configuration is Stage F's menu and the user's choice, not this tool's."""
    body = tool.verdict(*_pair(0.10, 0.20))
    assert body["outcome"] == "unsaturated"
    assert "16-64 epoch" in body["action"]
    assert "user" in body["action"]


def test_a_verified_pair_produces_a_complete_record(tmp_path):
    low = _write(tmp_path, "p4000", 32132, 0.10)
    high = _write(tmp_path, "p1600", 128528, 0.20)
    body = tool.evaluate(low, high)
    assert body["format"] == "r7-s3-saturation-verdict-v1"
    assert body["outcome"] == "unsaturated"
    assert body["scientific_claim"] is False and body["test_read"] is False
    assert body["readings"]["low"]["protocol_sha256"] and body["readings"]["high"]["protocol_sha256"]
    assert any("n=1" in item for item in body["limitations"])


@pytest.mark.parametrize("mutation,match", [
    (lambda f: (f / "readings.json").unlink(), "missing"),
    (lambda f: (f / "protocol.json").unlink(), "missing"),
    (lambda f: _rewrite(f, protocol_sha256="f" * 64), "different protocol"),
    (lambda f: _rewrite(f, curve_point="p0025"), "expected p4000"),
    (lambda f: _rewrite(f, updates=2008), "expected p4000"),
    (lambda f: _rewrite(f, batch_size=8), "batch is 8"),
    (lambda f: _rewrite(f, test_read=True), "test_read must be False"),
    (lambda f: _rewrite(f, scientific_claim=True), "no scientific claim"),
    (lambda f: _rewrite(f, saturation_verdict="saturated"), "must not carry"),
    (lambda f: _set_skill(f, float("nan")), "not finite"),
    (lambda f: _set_skill(f, None), "not finite"),
])
def test_unverifiable_input_is_refused(tmp_path, mutation, match):
    """Every one of these would otherwise yield a verdict built on a bad reading."""
    low = _write(tmp_path, "p4000", 32132, 0.10)
    high = _write(tmp_path, "p1600", 128528, 0.20)
    mutation(low)
    with pytest.raises((RuntimeError, FileNotFoundError), match=match):
        tool.evaluate(low, high)


def _rewrite(folder, **changes):
    path = folder / "readings.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body.update(changes)
    path.write_text(json.dumps(body), encoding="utf-8")


def _set_skill(folder, value):
    path = folder / "readings.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["leads"]["72"]["skill"]["t2m"] = value
    path.write_text(json.dumps(body), encoding="utf-8")


def test_the_tool_writes_exclusively_and_never_truncates(tmp_path):
    """R-001: an existing artifact is never overwritten."""
    low = _write(tmp_path, "p4000", 32132, 0.10)
    high = _write(tmp_path, "p1600", 128528, 0.20)
    out = tmp_path / "verdict.json"
    assert tool.main(["--low", str(low), "--high", str(high), "--out", str(out)]) is None
    assert json.loads(out.read_text(encoding="utf-8"))["outcome"] == "unsaturated"
    with pytest.raises(FileExistsError):
        tool.main(["--low", str(low), "--high", str(high), "--out", str(out)])


def test_the_tool_is_runnable_as_a_script(tmp_path):
    """A missing __main__ guard made point 1's prepare report success having written nothing."""
    import ast
    tree = ast.parse(Path(tool.__file__).read_text(encoding="utf-8"))
    guards = [node for node in tree.body
              if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
              and getattr(node.test.left, "id", None) == "__name__"]
    assert guards and "main" in ast.dump(guards[0])
