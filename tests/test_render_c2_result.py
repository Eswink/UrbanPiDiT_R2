"""The C2 result page is rendered from readings.json, never typed.

The page is a registration artifact: its numbers must come from the driver's own
``evaluate_exit``. These tests pin that nothing is transcribed and that a page cannot be
produced from a file that is not a readings file.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RENDER = ROOT / "tools/render_c2_result.py"
#: The renderer only reads code_commit, model_code_sha256, training_code_sha256 and
#: protocol_sha256 out of the protocol, so the fixture is built here rather than read from
#: ``outputs/`` -- that directory is git-ignored (.gitignore:31) and absent in CI, where a
#: dependency on it would turn the suite red.
PROTOCOL_FIELDS = {
    "code_commit": "0" * 40,
    "model_code_sha256": "1" * 64,
    "training_code_sha256": "2" * 64,
    "protocol_sha256": "3" * 64,
}
LEADS = (6, 12, 24, 48, 72)


def _readings(bump: float) -> dict:
    """A readings file shaped exactly like the driver's, with a chosen on-arm shift."""
    seeds = {}
    for seed in (41, 42, 43):
        entry = {}
        for arm in ("off", "on"):
            base = {6: 0.67, 12: 0.50, 24: 0.59, 48: 0.34, 72: 0.16}
            entry[arm] = {"rmse": {str(lead): 1.0 for lead in LEADS},
                          "skill": {str(lead): base[lead] + (bump if arm == "on" else 0.0)
                                    for lead in LEADS}}
            if arm == "on":
                entry[arm]["constraint_terms_recorded"] = True
        seeds[str(seed)] = entry
    return {"scientific_claim": False, "test_read": False,
            "protocol_sha256": "0" * 64,
            "preflight": {"rows": [{"total": 0.53}, {"total": 0.47}]},
            "seeds": seeds,
            "exit_conditions": {"a_72h_skill_mean_delta": bump > 0, "b_matching_bit": True,
                                "c_off_path_unchanged": True, "d_no_criterion_changed": True,
                                "all": bump > 0, "delta_72h": {str(s): bump for s in (41, 42, 43)},
                                "verdict": ("advance-to-independent-review" if bump > 0
                                            else "permanently-close-derivative-matching-branch")}}


def _protocol(tmp_path: Path) -> Path:
    """Write the minimal protocol the renderer needs, inside the test's own tmp_path."""
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(PROTOCOL_FIELDS), encoding="utf-8")
    return path


def _render(tmp_path: Path, readings: dict) -> tuple[int, str]:
    source = tmp_path / "readings.json"
    source.write_text(json.dumps(readings), encoding="utf-8")
    result = tmp_path / "RESULT.md"
    done = subprocess.run([sys.executable, str(RENDER), str(source), str(_protocol(tmp_path)), str(result)],
                          capture_output=True, text=True, timeout=120)
    return done.returncode, (result.read_text(encoding="utf-8") if result.exists() else "")


def test_the_page_reports_the_verdict_the_readings_carry(tmp_path):
    """Both directions, because a renderer that always prints 'advance' would pass one."""
    code, page = _render(tmp_path, _readings(0.005))
    assert code == 0
    assert "advance-to-independent-review" in page and "永久关闭" not in page

    code, page = _render(tmp_path, _readings(-0.005))
    assert code == 0
    assert "permanently-close-derivative-matching-branch" in page
    assert "永久关闭" in page


def test_the_numbers_come_from_the_file_not_from_the_renderer(tmp_path):
    """Change a skill in the input and the page must change with it."""
    readings = _readings(0.005)
    readings["seeds"]["41"]["off"]["skill"]["72"] = 0.987654
    code, page = _render(tmp_path, readings)
    assert code == 0
    assert "0.987654" in page


def test_a_non_readings_file_is_not_rendered(tmp_path):
    """A wrong or truncated input must fail loudly instead of producing a plausible page."""
    source = tmp_path / "readings.json"
    source.write_text("{\"exit_conditions\": {}}", encoding="utf-8")
    result = tmp_path / "RESULT.md"
    done = subprocess.run([sys.executable, str(RENDER), str(source), str(_protocol(tmp_path)), str(result)],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode != 0
    assert not result.exists()


def test_the_rendered_page_has_no_duplicated_section_heading(tmp_path):
    """The assembled page must not repeat a section.

    The renderer once emitted the "## §2" heading and its three per-seed rows twice: once
    inlined in main() and once inside _risk_tables(). Every existing test passed, because they
    checked individual pieces and the verdict string rather than the page's structure. This test
    looks at the whole assembled document, which is what a reader actually receives.
    """
    code, page = _render(tmp_path, _readings(0.005))
    assert code == 0
    headings = [line for line in page.splitlines() if line.startswith("## ")]
    assert len(headings) == len(set(headings)), (
        f"duplicate section headings in the rendered page: {headings}")
    # The per-seed 72 h table must appear exactly once: one header row.
    assert page.count("| seed | off | on | Δ(on−off) |") == 1
