"""The readings auditor must accept a sound file and reject every way one can be wrong.

R1's verdict comes only from ``readings.json``'s ``exit_conditions``, so the file has to be
readable as evidence before a human reads a verdict out of it. The auditor in
``tools/audit_r1_readings.py`` checks structure and internal consistency and explicitly
does not decide the science.

Every invariant gets a counter-proof here, because an auditor that only ever says "fine" is
worse than none: a missing seed, a hand-flipped `all`, a bridge whose `passed` disagrees
with its own difference and band, and two arms that trained on the same manifest (which
would mean the round's factor is absent from that seed) must all be refused.

Fixtures are synthetic: the real readings do not exist until seed43 finishes, and this must
run on CI where ``outputs/`` is absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import audit_r1_readings as auditor  # noqa: E402

LEADS = ("6", "12", "24", "48", "72")
BAND = {lead: 0.15 for lead in LEADS}


def _arm(identity: str, rmse: float) -> dict:
    return {
        "train_data_identity": identity,
        "full_t2m_rmse": {lead: rmse + index * 0.4 for index, lead in enumerate(LEADS)},
        "full_t2m_skill": {lead: 0.2 - index * 0.05 for index, lead in enumerate(LEADS)},
    }


def _seed(index: int) -> dict:
    control, treatment = _arm(f"c{index}" * 8, 2.0 + index * 0.01), _arm(f"t{index}" * 8, 1.9 + index * 0.01)
    return {
        "seed": 41 + index,
        "arms": {"control": control, "treatment": treatment},
        "delta_treatment_minus_control": {
            lead: treatment["full_t2m_rmse"][lead] - control["full_t2m_rmse"][lead]
            for lead in LEADS
        },
    }


def _readings() -> dict:
    return {
        "scientific_claim": False,
        "test_read": False,
        "protocol_sha256": "a" * 64,
        "seeds": {str(41 + index): _seed(index) for index in range(3)},
        "exit_conditions": {
            "a": True, "b": True, "c": True, "d": False,
            "seed_mean_delta_6_12_24": {"6": 0.001, "12": 0.002, "24": -0.003},
            "all": False,
            "verdict": "close-training-year-coverage-branch-at-matched-compute",
        },
        "bridge_e": {
            "scientific": False,
            "record_id": "s3-fy-annual-coverage-result",
            "leads": {lead: {"control_mean": 2.0, "reference_mean": 2.05,
                             "difference": -0.05, "band": BAND[lead], "passed": True}
                      for lead in LEADS},
            "passed": True,
        },
    }


def test_a_sound_file_is_accepted():
    assert auditor.audit(_readings()) == []


def test_main_reports_readable_and_exits_zero(tmp_path):
    path = tmp_path / "readings.json"
    path.write_text(json.dumps(_readings()), encoding="utf-8")
    assert auditor.main(["--readings", str(path)]) == 0


def test_main_exits_two_without_a_file(tmp_path):
    assert auditor.main(["--readings", str(tmp_path / "absent.json")]) == 2


@pytest.mark.parametrize("mutation, needle", [
    (lambda r: r["seeds"].pop("43"), "seeds missing"),
    (lambda r: r["seeds"].__setitem__("44", _seed(0)), "unexpected entries"),
    (lambda r: r["exit_conditions"].__setitem__("all", True), "conjunction"),
    (lambda r: r["exit_conditions"].__setitem__("a", "yes"), "not a bool"),
    (lambda r: r["bridge_e"].__setitem__("scientific", True), "must be false"),
    (lambda r: r["bridge_e"]["leads"]["24"].__setitem__("passed", False), "is True"),
    (lambda r: r["seeds"]["42"]["arms"]["treatment"].__setitem__(
        "train_data_identity", r["seeds"]["42"]["arms"]["control"]["train_data_identity"]),
     "share train_data_identity"),
    (lambda r: r["seeds"]["41"]["arms"]["control"]["full_t2m_rmse"].pop("72"), "lacks leads"),
    (lambda r: r.__setitem__("scientific_claim", True), "scientific_claim"),
    (lambda r: r["seeds"]["41"]["arms"]["control"]["full_t2m_rmse"].__setitem__("6", "x"),
     "not a number"),
])
def test_each_invariant_rejects_its_violation(mutation, needle):
    payload = _readings()
    mutation(payload)
    faults = auditor.audit(payload)
    assert any(needle in fault for fault in faults), (needle, faults)


def test_the_auditor_never_reports_a_verdict_of_its_own(capsys, tmp_path):
    """It may echo the file's verdict but must not compute one; the science stays human."""
    path = tmp_path / "readings.json"
    payload = _readings()
    payload["exit_conditions"]["all"] = True   # inconsistent on purpose: it must complain
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert auditor.main(["--readings", str(path)]) == 1
    printed = capsys.readouterr().out
    assert "UNREADABLE AS EVIDENCE" in printed
    assert "advance-to-independent-review" not in printed
