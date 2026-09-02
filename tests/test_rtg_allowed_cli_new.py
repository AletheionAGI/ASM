import subprocess
import sys


def test_smoke_is_separate_nonsealable_and_writes_nothing(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/run_attr_rtg_allowed.py", "smoke", "--output", str(tmp_path)],
        check=True, capture_output=True, text=True,
    )
    assert "NON-SEALABLE" in result.stdout
    assert list(tmp_path.iterdir()) == []
