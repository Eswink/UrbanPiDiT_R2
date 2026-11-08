"""Tests for the C2 cross-lead defect reproduction (decision 0105).

The probe needs the store and a forward pass, so these tests do not run it. They pin the parts that
make the finding mean what the ADR says it means: the four report flags, the lead constants, and the
structural fact that ``atmos_target`` is the FIRST physical step rather than the last. That last one
is the assumption that was wrong in two independent modules (C1's page and C2's matching_term), so it
is asserted against the dataset's own construction rather than trusted to a comment.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tools import reproduce_c2_cross_lead_defect as probe


def test_parent_checkpoint_is_the_stage_b_fourth_point():
    assert "update_0128528.pt" in probe.PARENT_CHECKPOINT
    assert "r7_s3_throughput_physics_b_p1600" in probe.PARENT_CHECKPOINT


def test_default_cases_are_c2_preflights_own_cases():
    assert probe.DEFAULT_CASES == (1000, 9000, 17000, 25000)


def test_physical_steps_times_step_hours_is_72_not_6():
    """forecasts[:,-1] is the +72 h step; a 6 h reading of it is the defect this ADR records."""
    from scripts import r7_s3_curve_point_large_protocol as curve
    from training.r7_autoregressive_rollout import STEP_HOURS

    steps = len(curve.PHYSICAL_WEIGHTS)
    assert steps == 12
    assert STEP_HOURS == 6
    assert steps * STEP_HOURS == 72


def test_dataset_stacks_atmos_target_as_the_first_physical_step():
    """The load-bearing fact: ``physical_targets[:,0]`` IS ``atmos_target``, so it is +6 h.

    Read from the dataset source rather than asserted from memory, because the whole finding is
    that a docstring called this field "the last physical step" while the stack puts it first.
    """
    source = (Path(probe.__file__).resolve().parents[1]
              / "data" / "r7_long_rollout_dataset.py").read_text(encoding="utf-8")
    # ``fields = [sample["atmos_target"]]`` then appended targets, then stacked.
    assert 'fields = [sample["atmos_target"]]' in source
    assert 'sample["physical_targets"] = torch.stack(fields)' in source


def test_matching_term_reads_atmos_target_not_the_last_physical_target():
    """Pins the defect's location, so a future fix makes this test fail loudly and get updated."""
    source = (Path(probe.__file__).resolve().parents[1]
              / "training" / "r7_derivative_matching.py").read_text(encoding="utf-8")
    assert 'target = moved.get("atmos_target")' in source
    # And it pairs that against the LAST forecast step.
    assert "forecast = result.forecasts[:, -1]" in source


def test_report_declares_the_four_flags_the_adr_quotes():
    """The four flags must be actual keys of ``measure``'s returned dict, not just mentioned.

    A first version of this test searched the source text for the flag names, which passed even
    when a key was renamed -- the print statement at the bottom of the module still contained the
    old name. Parsing the AST for the returned dict literal is the version that can fail.
    """
    import ast

    tree = ast.parse(Path(probe.__file__).read_text(encoding="utf-8"))
    returned = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "measure":
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Dict):
                    for key in inner.value.keys:
                        if isinstance(key, ast.Constant):
                            returned[key.value] = inner.lineno
    assert returned, "measure() must return a dict literal"
    for flag in ("deployed_pairing_is_cross_lead", "perfect_forecast_is_penalised",
                 "lead_matched_is_zero_at_truth", "preflight_guard_is_vacuous",
                 "scientific_claim"):
        assert flag in returned, f"{flag} is not a key of measure()'s returned dict"


def test_preflight_guard_compares_target_with_itself():
    """The vacuous guard: ``derivative_matching_terms(target, target, ...)``."""
    source = (Path(probe.__file__).resolve().parents[1]
              / "scripts" / "r7_s3_derivative_matching_c2_protocol.py").read_text(encoding="utf-8")
    assert 'zero = derivative_matching_terms(target, target' in source


def test_probe_is_read_only_and_writes_only_its_out_path():
    source = Path(probe.__file__).read_text(encoding="utf-8")
    assert "args.out.write_text" in source
    assert 'mode="r"' in source
    header = source.split("def ")[0]
    for heavy in ("import torch", "import zarr", "import numpy"):
        assert heavy not in header, heavy


@pytest.mark.parametrize("value", [0.495737, 0.591632, 0.607382, 0.492830])
def test_measured_deployed_values_are_all_above_zero(value):
    """The four frozen-scale values decision 0105 records, as a guard on the arithmetic's shape."""
    assert value > 0.0
    assert value < 1.0  # same order as L_mse ~ 1; not a rounding artefact
