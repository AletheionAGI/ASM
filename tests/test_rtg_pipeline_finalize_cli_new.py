import subprocess
import sys


def test_allowed_cli_exposes_finalize_stage():
    result = subprocess.run(
        [sys.executable, "scripts/run_attr_rtg_allowed.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "finalize" in result.stdout
