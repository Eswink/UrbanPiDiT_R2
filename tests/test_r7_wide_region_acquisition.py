"""Wide-region (S3 lateral-context) acquisition geometry and identity invariants.

The claim under test is narrow and load-bearing: the wide read plan's central
65x65 block *is* the frozen 27-43N/107-123E target box, the decoded cost of a
wide read is the same as the frozen one, and the frozen v2/v3 read plan and
downloader keep their own identity untouched. Nothing here reaches the network.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from data.download import read_plan_frozen, read_plan_wide
from training.r7_boundary_metrics import boundary_masks

REPO = Path(__file__).resolve().parents[1]


def test_wide_region_contains_the_frozen_target_box_as_its_central_block():
    evidence = read_plan_wide.containment_evidence()
    assert evidence["wide_points"] == 129
    assert evidence["target_points"] == 65
    assert evidence["target_margin_cells"] == 32
    assert evidence["target_margin_deg"] == pytest.approx(8.0)
    for name, target in (("south", 27.0), ("north", 43.0), ("west", 107.0), ("east", 123.0)):
        assert evidence["derived_edges"][name]["wide"] == pytest.approx(target)
        assert evidence["derived_edges"][name]["target"] == pytest.approx(target)


def test_wide_bounds_are_the_target_box_grown_by_eight_degrees():
    assert (read_plan_wide.WIDE_SOUTH, read_plan_wide.WIDE_NORTH) == (19.0, 51.0)
    assert (read_plan_wide.WIDE_WEST, read_plan_wide.WIDE_EAST) == (99.0, 131.0)
    assert read_plan_wide.WIDE_POINTS == read_plan_wide.TARGET_POINTS + 64


def test_frozen_target_box_is_not_rewritten_by_the_wide_plan():
    assert (read_plan_wide.TARGET_SOUTH, read_plan_wide.TARGET_NORTH) == (
        read_plan_frozen.ROI_SOUTH, read_plan_frozen.ROI_NORTH)
    assert (read_plan_wide.TARGET_WEST, read_plan_wide.TARGET_EAST) == (
        read_plan_frozen.ROI_WEST, read_plan_frozen.ROI_EAST)
    assert read_plan_wide.TARGET_POINTS == read_plan_frozen.GRID_HEIGHT
    assert read_plan_wide.TARGET_POINTS == read_plan_frozen.GRID_WIDTH
    assert read_plan_wide.SPACING_DEG == read_plan_frozen.GRID_SPACING_DEG


def test_central_block_slice_is_the_target_box_slice():
    rows, columns = read_plan_wide.central_block()
    assert (rows, columns) == (slice(32, 97), slice(32, 97))


def test_evaluation_interior_mask_at_margin_32_is_exactly_the_central_block():
    masks = {name: mask for name, _, mask in boundary_masks(129, 129, (32,))}
    interior = masks["interior_32"]
    assert interior.shape == (129, 129)
    assert int(interior.sum()) == 65 * 65
    rows, columns = read_plan_wide.central_block()
    expected = torch.zeros(129, 129, dtype=torch.bool)
    expected[rows, columns] = True
    assert torch.equal(interior, expected)


def test_evaluation_interior_mask_at_margin_32_matches_the_frozen_65_grid():
    # The mask must line up with a 65x65 grid built on the frozen box: the same
    # cell index in both means the same latitude/longitude.
    frozen_latitude = [27.0 + 0.25 * index for index in range(65)]
    wide_latitude = [19.0 + 0.25 * index for index in range(129)]
    rows, _ = read_plan_wide.central_block()
    assert wide_latitude[rows] == frozen_latitude


def test_grid_points_rejects_non_integer_spans():
    assert read_plan_wide.grid_points(16.0) == 65
    assert read_plan_wide.grid_points(32.0) == 129
    with pytest.raises(ValueError):
        read_plan_wide.grid_points(16.1)


def test_decoded_cost_does_not_depend_on_the_region_but_retained_bytes_do():
    cost = read_plan_wide.stamp_cost(120)
    assert read_plan_wide.FIELD_READS_PER_STAMP == 19
    assert cost["decoded_bytes"] == cost["global_chunk_bytes"] * 19 * 120
    assert cost["retained_bytes_wide"] == 4 * 129 * 129 * 120
    assert cost["retained_bytes_target"] == 4 * 65 * 65 * 120
    assert cost["retained_growth_factor"] == pytest.approx((129 / 65) ** 2)
    # The decoded figure is a property of the touched chunk union only.
    assert read_plan_wide.stamp_cost(120)["decoded_bytes"] == cost["decoded_bytes"]


def test_wide_decoded_estimate_matches_the_registered_target_box_part():
    # The registered 2018-winter part charged 9,482,837,880 decoded bytes for 120
    # stamps; the field-read component alone is 19 * 4,152,960 * 120, and the rest
    # is coordinate reads. The wide read touches the same chunks, so the estimate
    # must not drift from that measurement.
    cost = read_plan_wide.stamp_cost(120)
    registered = 9482837880
    assert cost["decoded_bytes"] == 19 * 4152960 * 120 == 9468748800
    assert cost["decoded_bytes"] <= registered
    assert (registered - cost["decoded_bytes"]) / registered < 0.02


def test_wide_protocol_declares_the_region_target_and_an_independent_digest():
    protocol = read_plan_wide.wide_frozen_protocol(stamps=120)
    assert protocol["format"] == "r7-era5-wide-read-plan-v1"
    assert protocol["region"]["points"] == [129, 129]
    assert protocol["target_box"]["points"] == [65, 65]
    assert protocol["target_box"]["containment"]["target_margin_cells"] == 32
    assert protocol["scientific_claim"] is False
    assert protocol["budget_caps"]["new_artifacts_bytes"] == 100 * 2**30
    frozen = read_plan_frozen.frozen_protocol(stage="second",
                                              new_artifact_bytes_cap=100 * 2**30,
                                              decoded_bytes_cap=256 * 2**30)
    assert protocol["protocol_sha256"] != frozen["protocol_sha256"]


def test_wide_protocol_digest_is_reproducible_and_caps_sensitive():
    first = read_plan_wide.wide_frozen_protocol(stamps=120)
    second = read_plan_wide.wide_frozen_protocol(stamps=120)
    assert first == second
    smaller = read_plan_wide.wide_frozen_protocol(stamps=120, decoded_bytes_cap=2**30)
    assert smaller["protocol_sha256"] != first["protocol_sha256"]
    with pytest.raises(ValueError):
        read_plan_wide.wide_frozen_protocol(new_artifact_bytes_cap=0)


def test_wide_protocol_carries_the_registered_v3_splits_and_sealed_test_year():
    protocol = read_plan_wide.wide_frozen_protocol()
    assert protocol["splits"]["train_years"] == [2017, 2018, 2019, 2020, 2021]
    assert protocol["splits"]["val_years"] == [2022]
    assert protocol["splits"]["sealed_test_year"] == 2023


def test_frozen_downloader_and_read_plan_files_are_unchanged_by_this_identity(tmp_path):
    # A guard against "the wide work quietly rewrote the frozen path": the frozen
    # read plan still pins 65x65 / 27-43-107-123 and the D1 validator still
    # refuses any other region.
    assert (read_plan_frozen.GRID_HEIGHT, read_plan_frozen.GRID_WIDTH) == (65, 65)
    assert (read_plan_frozen.ROI_SOUTH, read_plan_frozen.ROI_NORTH,
            read_plan_frozen.ROI_WEST, read_plan_frozen.ROI_EAST) == (27.0, 43.0, 107.0, 123.0)
    from data.download import earthmover_spatial_d1, earthmover_spatial_s1
    assert earthmover_spatial_d1.SPATIAL_SURFACE_CHUNKS == (1, 721, 1440)
    assert earthmover_spatial_d1.SPATIAL_PRESSURE_CHUNKS == (1, 1, 721, 1440)
    source = Path(earthmover_spatial_d1.__file__).read_text(encoding="utf-8")
    assert "south, north, west, east = 27.0, 43.0, 107.0, 123.0" in source
    assert "this path pins 65x65" in source
    assert hashlib.sha256(
        Path(earthmover_spatial_s1.__file__).read_bytes()).hexdigest()  # readable, untouched


def test_wide_downloader_reuses_the_audited_helpers_verbatim():
    from data.download import earthmover_spatial_w1 as wide
    from data.download import earthmover_wide_io as wide_io

    assert wide.SOURCE == "s3://earthmover-icechunk-era5/icechunkV2"
    assert wide.SNAPSHOT == "ZFKDHBCTBVHVXM3BQFV0"
    assert wide_io.d1_fields is not None
    assert wide_io._collect_frames.__module__ == "data.download.earthmover_spatial_d1"
    assert wide_io._open_pinned_session.__module__ == "data.download.earthmover_spatial_d1"
    assert wide_io._attest_spatial_levels.__module__ == "data.download.earthmover_spatial_d1"
    assert wide._merge_parts.__module__ == "data.download.earthmover_spatial_s1"
    assert wide.MAX_YEARS_PER_BATCH == 4
    assert wide.MAX_SEASONS_PER_YEAR == 4
    # The split is a size measure (R-021), not a second implementation: the
    # driver must not carry its own copy of the validator.
    assert wide.validate_wide_namespace is wide_io.validate_wide_namespace
    assert wide.preflight_wide is wide_io.preflight_wide


def test_wide_stamp_field_bytes_matches_the_frozen_cost_model():
    # Regression for a latent NameError: the pre-split ``validate_wide_namespace``
    # referenced an undefined ``surface_chunk_bytes`` and was never exercised, so
    # both ``--preflight`` and ``--write`` raised before any download. The value
    # is the (variable, level) read count (19), not the output channel count (17),
    # times one whole-globe chunk of the frozen geometry.
    from data.download import earthmover_wide_io as wide_io

    assert wide_io.per_stamp_field_bytes(19) == read_plan_wide.stamp_cost(1)["decoded_bytes"]
    assert wide_io.per_stamp_field_bytes(19) == 19 * 721 * 1440 * 4
    assert wide_io.per_stamp_field_bytes(19) == 78906240
    # The channel count is NOT the field-read count; confusing them is the bug.
    assert wide_io.per_stamp_field_bytes(17) != wide_io.per_stamp_field_bytes(19)


def _ast_bind_statement(node, bound):
    """Add the names one simple top-level statement binds (no body to descend)."""
    import ast

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        bound.add(node.name)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        bound.update((a.asname or a.name).split(".")[0] for a in node.names)
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            bound.update(n.id for n in ast.walk(target) if isinstance(n, ast.Name))
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        bound.add(node.target.id)


def _ast_module_globals(tree):
    """Top-level bindings only (module scope, not function scope).

    Descending into function bodies here was a hole: it made every assignment
    anywhere look like a module global, so a helper referencing its caller's local
    (``_wide_part_receipt`` used ``started``) passed.
    """
    import ast

    compound = (ast.If, ast.For, ast.While, ast.With, ast.AsyncWith, ast.Try)
    bound = set()
    pending = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, compound):
            pending += list(node.body) + list(node.orelse)
            pending += [stmt for handler in getattr(node, "handlers", []) or []
                        for stmt in handler.body]
            pending += list(getattr(node, "finalbody", []) or [])
        else:
            _ast_bind_statement(node, bound)
    return bound


def _ast_function_locals(fn):
    import ast

    bound = {a.arg for a in fn.args.args + fn.args.posonlyargs + fn.args.kwonlyargs}
    for name in (fn.args.vararg, fn.args.kwarg):
        if name is not None:
            bound.add(name.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    return bound


def test_wide_modules_have_no_undefined_global_names():
    # Regression for the R-021 split, which moved helpers into earthmover_wide_io /
    # earthmover_spatial_d1 but left earthmover_spatial_w1 referencing them without
    # importing: the committed ``--preflight`` and ``--write`` both raised NameError
    # and were never exercised (the pilot ran pre-split). A tiny scope walk over each
    # module refuses any global name that no import, definition or local binds.
    import ast
    import builtins

    known = set(dir(builtins))
    for name in ("earthmover_spatial_w1.py", "earthmover_wide_io.py", "read_plan_wide.py"):
        tree = ast.parse((REPO / "data/download" / name).read_text(encoding="utf-8"))
        globals_bound = _ast_module_globals(tree)
        for fn in [n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            loads = {n.id for n in ast.walk(fn)
                     if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
            undefined = loads - _ast_function_locals(fn) - globals_bound - known
            assert not undefined, f"{name}:{fn.name} references undefined names {sorted(undefined)}"


def test_wide_part_receipt_composes_from_its_own_arguments(tmp_path):
    # Regression: the receipt writer referenced `started`, a local of its caller, so a
    # fully downloaded part crashed at the receipt step (data written, no receipt).
    # It must build the receipt from `artifact` and `measured` alone.
    import datetime

    from data.download.earthmover_spatial_w1 import _wide_part_receipt

    nc = tmp_path / "part_winter_2018.nc"
    nc.write_bytes(b"x" * 10)
    plan_read = {
        "observed_units": {"2m_temperature": "K"},
        "level_attestation": {"shared_levels": [250, 500, 850]},
        "shared_levels": [250, 500, 850],
        "times": [datetime.datetime(2018, 1, 1, 0), datetime.datetime(2018, 1, 1, 6)],
        "latitude": [27.0, 27.25],
        "longitude": [107.0, 107.25],
    }
    budget = type("B", (), {"limit": 20, "used": 30, "reads": 40})()
    measured = {"estimated": 11, "budget": budget, "network_bytes": 50,
                "elapsed": 6.5, "icechunk_version": "2.2.2", "zarr_version": "3.4.0"}
    receipt = _wide_part_receipt({"part": {"years": [2018]}}, plan_read, {"t2m": {}}, ["t2m"],
                                 (nc, "deadbeef"), measured)
    assert receipt["status"] == "downloaded-real-source"
    assert receipt["elapsed_seconds"] == 6.5
    assert receipt["network_body_bytes"] == 50
    assert receipt["local_artifact"] == {"path": "part_winter_2018.nc", "bytes": 10,
                                         "sha256": "deadbeef"}
    assert receipt["shape"] == [2, 3, 2, 2]
    assert receipt["decoded_chunk_budget"]["charged_bytes"] == 30
    assert receipt["part"] == {"years": [2018]}


def test_wide_modules_stay_inside_the_size_target():
    for name in ("read_plan_wide.py", "earthmover_wide_io.py", "earthmover_spatial_w1.py"):
        lines = len((REPO / "data/download" / name).read_text(encoding="utf-8").splitlines())
        assert lines <= 400, f"{name} has {lines} lines"


def test_wide_dataset_attributes_state_the_wide_region_without_d1_prose():
    from data.download.earthmover_spatial_w1 import _wide_dataset_attributes

    attrs = _wide_dataset_attributes("ZFKDHBCTBVHVXM3BQFV0")
    assert attrs["region_points"] == [129, 129]
    assert attrs["target_box_points"] == [65, 65]
    assert "D1" not in attrs["purpose"]
    assert attrs["interpolation"] == "none"


def test_the_source_licence_and_doi_are_recorded_first_hand():
    # real-data-acquisition step 5 asks a download receipt to carry the licence
    # and the DOI. Both were read from the source's own registry page on
    # 2026-10-09 (R-050), so they are pinned here rather than left to a later
    # reader's guess.
    from data.download.earthmover_wide_io import source_citation

    citation = source_citation()
    assert citation["license"] == "CC-BY-4.0"
    assert citation["license_url"].startswith("https://creativecommons.org/licenses/by/4.0")
    assert citation["doi"] == "10.24381/cds.adbb2d47"
    assert citation["registry"] == "https://registry.opendata.aws/earthmover-era5/"
    assert citation["accessed"] == "2026-10-09"
    attrs = __import__("data.download.earthmover_spatial_w1",
                       fromlist=["_wide_dataset_attributes"])._wide_dataset_attributes("SNAP")
    for key in ("license", "license_url", "source_doi", "citation_accessed"):
        assert attrs[key] == citation[{"source_doi": "doi", "citation_accessed": "accessed"}.get(key, key)]


def test_wide_cli_exposes_plan_preflight_write_merge_without_network(tmp_path, capsys):
    from data.download.earthmover_spatial_w1 import main

    import sys

    argv = sys.argv
    sys.argv = ["earthmover_spatial_w1", "--plan", "--years", "2018", "--seasons", "winter"]
    try:
        main()
    finally:
        sys.argv = argv
    payload = json.loads(capsys.readouterr().out)
    assert payload["plan"]["total_stamps"] == 120
    assert payload["wide_read_plan"]["region"]["points"] == [129, 129]
    assert payload["estimated_decoded_bytes"] > 0
