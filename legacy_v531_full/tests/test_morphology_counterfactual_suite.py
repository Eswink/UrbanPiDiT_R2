from __future__ import annotations

from tests.test_v531_reviewer_evidence_checks import CONFIG_PATH, _small_cfg
from experiments.run_morphology_counterfactual_suite import build_counterfactual_report, dry_run_report
from urbanpidit_version import VERSION


def test_counterfactual_dry_run_lists_cases() -> None:
    report = dry_run_report(["zero_static"])

    assert report["version"] == VERSION
    assert report["intervention_names"] == ["original_static", "zero_static"]


def test_counterfactual_report_contains_delta_fields() -> None:
    report = build_counterfactual_report(
        _small_cfg(),
        config_path=CONFIG_PATH,
        batch_size=1,
        case_names=["zero_static"],
    )

    assert report["version"] == VERSION
    assert report["trained_checkpoint"] is False
    assert report["case_count"] == 2
    assert report["forward_report"]["prediction_finite"] is True
    baseline, zero_case = report["cases"]
    assert baseline["name"] == "original_static"
    assert zero_case["name"] == "zero_static"
    for key in ("prediction_delta_norm", "proxy_delta_norm", "process_graph_delta_norm"):
        assert key in zero_case
        assert isinstance(zero_case[key], float)
    assert zero_case["proxy_delta"]["available"] is True
    assert zero_case["process_graph_delta"]["available"] is True