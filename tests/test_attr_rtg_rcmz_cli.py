from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest

from attr_rtg_rcmz.cli import main, run_dry, verify_official_lock


def test_dry_run_creates_complete_portable_artifacts(tmp_path):
    output = tmp_path / "run"
    paths = run_dry(output, 10.0, stream=io.StringIO())
    expected_charts = {
        f"{name}.{suffix}"
        for name in ("architecture_quality", "g_vs_c", "governance", "seed_differences")
        for suffix in ("png", "svg")
    }
    assert {path.name for path in paths} == {
        "summary.png",
        "summary.svg",
        "summary.html",
        "manifest.json",
        "manifest.csv",
        *expected_charts,
    }
    assert (output / "summary.png").read_bytes().startswith(b"\x89PNG")
    from PIL import Image

    with Image.open(output / "summary.png") as image:
        assert image.size == (900, 480)
    assert "<svg" in (output / "summary.svg").read_text()
    html_text = (output / "summary.html").read_text()
    assert "LOCAL-ONLY" in html_text
    assert all(
        name in html_text
        for name in ("architecture_quality", "g_vs_c", "governance", "seed_differences")
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["rows"]) == 20
    status = json.loads((output / "status.json").read_text())
    assert status["phase"] == "completed"
    assert status["update"] == 2000
    assert "phase=completed" in (output / "run.log").read_text()


def test_official_mode_fails_closed_without_lock(tmp_path):
    assert main(["--official", "--output-dir", str(tmp_path)]) == 2
    assert not (tmp_path / "manifest.json").exists()


def test_verified_anchor_result_is_forwarded_without_running_backend(
    tmp_path, monkeypatch
):
    digest = "a" * 64
    monkeypatch.setattr(
        "attr_rtg_rcmz.cli.verify_official_lock", lambda path, expected: digest
    )
    captured = {}

    def fake_run(output_dir, heartbeat_seconds, *, lock, smoke=False, stream=None):
        captured.update(lock)
        return []

    monkeypatch.setattr("attr_rtg_rcmz.cli.run_engine", fake_run)
    output = tmp_path / "official"
    assert (
        main(
            [
                "--official",
                "--output-dir",
                str(output),
                "--lock-file",
                str(tmp_path / "lock"),
                "--lock-sha256",
                digest,
            ]
        )
        == 0
    )
    assert captured == {
        "verified": True,
        "state": "LOCAL PROTOCOL LOCK",
        "sha256": digest,
    }


def test_arbitrary_lock_path_is_rejected(tmp_path):
    lock = tmp_path / "draft.json"
    lock.write_text('{"state":"LOCAL PROTOCOL LOCK"}')
    digest = hashlib.sha256(lock.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="canonical receipt"):
        verify_official_lock(lock, digest)


def test_smoke_official_cli_renders_synthetic_engine_rows(tmp_path):
    output = tmp_path / "smoke"
    assert main(["--smoke-official", "--output-dir", str(output)]) == 0
    assert (output / "smoke_official_rows.json").exists()
    assert json.loads((output / "manifest.json").read_text())["synthetic"] is True
    assert json.loads((output / "status.json").read_text())["phase"] == "completed"


def test_full_canonical_receipt_shape_binds_all_hash_classes():
    from attr_rtg_rcmz.lock_anchor import (
        EXPECTED_AUTHORIZATION_SHA256,
        EXPECTED_CANDIDATE_CONTENT_SHA256,
        EXPECTED_CANDIDATE_MANIFEST_SHA256,
        EXPECTED_PROTOCOL_SHA256,
    )
    from attr_rtg_rcmz.lock_guard import canonical_receipt_bytes, validate_receipt_bytes

    receipt = {
        "schema_version": 1,
        "state": "LOCAL PROTOCOL LOCK",
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "authorization_sha256": EXPECTED_AUTHORIZATION_SHA256,
        "manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "content_sha256": EXPECTED_CANDIDATE_CONTENT_SHA256,
    }
    raw = canonical_receipt_bytes(receipt)
    assert validate_receipt_bytes(raw, hashlib.sha256(raw).hexdigest()) == receipt
    minimal = canonical_receipt_bytes({"state": "LOCAL PROTOCOL LOCK"})
    with pytest.raises(PermissionError, match="missing or unexpected"):
        validate_receipt_bytes(minimal, hashlib.sha256(minimal).hexdigest())


def test_direct_official_data_and_training_calls_fail_before_work():
    from attr_rtg_rcmz.official_data import generate_registered_origins
    from attr_rtg_rcmz.official_training import train_and_score

    with pytest.raises(PermissionError, match="verified canonical lock"):
        generate_registered_origins(miniature=False)
    with pytest.raises(PermissionError, match="verified canonical lock"):
        train_and_score({}, Path("unused"), lambda event: None)


def test_direct_official_orchestrator_rejects_self_pinned_mapping(tmp_path):
    from attr_rtg_rcmz.official import run_official

    fake = {"verified": True, "state": "LOCAL PROTOCOL LOCK", "sha256": "a" * 64}
    with pytest.raises((PermissionError, FileNotFoundError)):
        run_official(tmp_path, None, fake)


def test_anchor_path_is_explicitly_excluded_from_candidate_manifest():
    from attr_rtg_rcmz.lock_anchor import ANCHOR_SOURCE_RELATIVE_PATH
    from attr_rtg_rcmz.manifests import CANDIDATE_MANIFEST_EXCLUDED_PATHS

    assert ANCHOR_SOURCE_RELATIVE_PATH in CANDIDATE_MANIFEST_EXCLUDED_PATHS


def test_candidate_manifest_rehashes_all_anchored_artifacts(tmp_path, monkeypatch):
    from attr_rtg_rcmz import lock_anchor
    from attr_rtg_rcmz.lock_guard import verify_candidate_manifest

    artifacts = []
    for index in range(lock_anchor.EXPECTED_ARTIFACT_COUNT):
        relative = f"artifacts/{index:03d}.txt"
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(index))
        raw = target.read_bytes()
        artifacts.append(
            {
                "path": relative,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    manifest = {"artifacts": artifacts, "content_sha256": ""}
    content = dict(manifest)
    content.pop("content_sha256")
    content_hash = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["content_sha256"] = content_hash
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    monkeypatch.setattr(
        lock_anchor, "CANDIDATE_MANIFEST_RELATIVE_PATH", "candidate.json"
    )
    monkeypatch.setattr(
        lock_anchor,
        "EXPECTED_CANDIDATE_MANIFEST_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(lock_anchor, "EXPECTED_CANDIDATE_CONTENT_SHA256", content_hash)
    assert (
        len(verify_candidate_manifest(tmp_path)["artifacts"])
        == lock_anchor.EXPECTED_ARTIFACT_COUNT
    )
    (tmp_path / artifacts[-1]["path"]).write_text("tampered")
    with pytest.raises(PermissionError, match="wrong size|digest differs"):
        verify_candidate_manifest(tmp_path)
