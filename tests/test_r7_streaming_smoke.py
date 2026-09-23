import json
from pathlib import Path
import subprocess
import sys


def test_streaming_cli_executes_real_optimizer_updates():
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run([sys.executable, str(root / "scripts/smoke_r7_streaming.py"),
                           "--steps", "3", "--updates", "2"], cwd=root,
                          capture_output=True, text=True, check=True, timeout=60)
    result = json.loads(proc.stdout)
    assert result["synthetic_only"] is True
    assert result["device"] == "cpu"
    assert result["optimizer_updates"] == 2
    assert len(result["results"]) == 2
    assert all(row["logical_saved_tensor_peak_bytes"] > 0 for row in result["results"])
