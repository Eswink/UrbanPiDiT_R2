"""The evidence-page row reader must reproduce the page and must never invent a value.

`scripts/read_r1_seed_rows.py` exists because the 17-variable x 5-lead table was once
hand-transcribed and came out wrong against the CSV. A reader is only worth having if
it is checked two ways: it agrees with readings that are already on disk, and it refuses
to print anything at all when a lead has not finished yet -- a silent default would put
an invented number under a real seed label.

The fixtures below are synthetic and tiny on purpose: they exercise the CSV contract and
the refusal paths, not the round's real 65x65 results (which live in ``outputs/`` and are
absent on CI).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import read_r1_seed_rows as rows  # noqa: E402

HEADER = ["region", "margin_cells", "n_grid_points", "full_area_fraction",
          "lead_hours", "variable", "unit", "rmse", "n_initializations"]


def _write_table(root: Path, *, seed: int, arm: str, lead: int,
                 values: dict[str, float]) -> Path:
    directory = root / f"seed{seed}/evaluation/{arm}/lead_{lead:03d}h"
    directory.mkdir(parents=True)
    path = directory / "boundary_rmse.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for variable, rmse in values.items():
            writer.writerow(["full", 0, 4225, 1.0, lead, variable, "K", rmse, 1451])
    return path


def _full_round(root: Path, *, seed: int, arm: str, lead: int, base: float) -> None:
    _write_table(root, seed=seed, arm=arm, lead=lead,
                 values={variable: base for variable in rows.VARIABLES})


def test_reader_formats_every_variable_and_lead(tmp_path):
    for lead in rows.LEADS:
        _full_round(tmp_path, seed=7, arm="treatment", lead=lead, base=float(lead) / 10)
    values = rows.seed_rows(7, "treatment", tmp_path)
    assert set(values) == set(rows.VARIABLES)
    assert all(set(leads) == set(rows.LEADS) for leads in values.values())
    assert values["t2m"][72] == pytest.approx(7.2)

    lines = rows.markdown_rows(values, 7)
    assert len(lines) == len(rows.VARIABLES) * len(rows.LEADS)
    assert lines[0] == "| t2m | 6 h | 0.6000 | `seed 7` |"


def test_reader_reads_only_the_named_region(tmp_path):
    """A margin row for the same variable must not be mistaken for the full-area one."""
    directory = tmp_path / "seed3/evaluation/treatment/lead_006h"
    directory.mkdir(parents=True)
    path = directory / "boundary_rmse.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerow(["interior_32", 32, 4225, 0.84, 6, "t2m", "K", 1.0, 1451])
        writer.writerow(["full", 0, 4225, 1.0, 6, "t2m", "K", 2.5, 1451])
    assert rows.region_variable_rmse(path, "full", "t2m") == pytest.approx(2.5)
    assert rows.region_variable_rmse(path, "interior_32", "t2m") == pytest.approx(1.0)


def test_reader_refuses_a_missing_lead(tmp_path, capsys):
    """The counter-proof: an unfinished seed must fail loudly, not print zeros."""
    for lead in rows.LEADS[:-1]:
        _full_round(tmp_path, seed=9, arm="treatment", lead=lead, base=1.0)
    assert rows.main(["--seed", "9", "--arm", "treatment", "--out", str(tmp_path)]) == 2
    assert "not on disk" in capsys.readouterr().err

    with pytest.raises(FileNotFoundError):
        rows.seed_rows(9, "treatment", tmp_path)


def test_reader_refuses_a_missing_variable(tmp_path):
    """A table missing one variable is an error, not a partially filled column."""
    for lead in rows.LEADS:
        _write_table(tmp_path, seed=5, arm="treatment", lead=lead,
                     values={variable: 1.0 for variable in rows.VARIABLES[:-1]})
    with pytest.raises(RuntimeError, match="missing from"):
        rows.seed_rows(5, "treatment", tmp_path)
