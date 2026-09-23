from pathlib import Path
import json


def test_physicsnemo_corrdiff_source_snapshot_present():
    root = Path(__file__).resolve().parents[1]
    src = root / "baselines" / "external_sources" / "physicsnemo_weather_corrdiff"
    assert (src / "README.md").exists()
    assert (src / "train.py").exists()
    assert (src / "generate.py").exists()
    assert (src / "score_samples.py").exists()
    assert (src / "conf" / "config_training_hrrr_mini_diffusion.yaml").exists()
    assert (src / "conf" / "base" / "model" / "diffusion.yaml").exists()
    assert (src / "conf" / "base" / "model" / "regression.yaml").exists()
    assert not (src / "tests").exists()


def test_physicsnemo_corrdiff_manifest_matches_snapshot():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "baselines" / "external_sources" / "PHYSICSNEMO_CORRDIFF_SOURCE_MANIFEST.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["upstream_project"] == "NVIDIA PhysicsNeMo"
    assert manifest["extracted_subdir"].endswith("examples/weather/corrdiff")
    assert manifest["file_count"] >= 50
    listed = {item["path"] for item in manifest["files"]}
    for required in ["README.md", "train.py", "generate.py", "score_samples.py", "requirements.txt"]:
        assert required in listed