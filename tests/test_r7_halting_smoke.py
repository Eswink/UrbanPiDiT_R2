def test_halting_smoke_entrypoint():
    import json
    from pathlib import Path
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / 'scripts/smoke_r7_halting.py'), '--steps', '3', '--updates', '1'],
        cwd=root, capture_output=True, text=True, check=True, timeout=90)
    report = json.loads(completed.stdout)
    assert report['scientific_validation'] is False
    assert report['forecaster_unchanged'] is True
    assert report['controller_updates'] == 1
