"""Canonical source-anchored local protocol lock verification."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import lock_anchor

_LOCK_STATE = "LOCAL PROTOCOL LOCK"
_LOCK_RELATIVE = Path("locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json")
_HEX = re.compile(r"[0-9a-f]{64}")
_KEYS = {
    "schema_version",
    "state",
    "protocol_sha256",
    "authorization_sha256",
    "manifest_sha256",
    "content_sha256",
}


class LockGuardError(PermissionError):
    """The canonical receipt or compiled source anchor is invalid."""


def canonical_lock_path() -> Path:
    return Path(__file__).resolve().parents[2] / _LOCK_RELATIVE


def canonical_receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        + b"\n"
    )


def validate_receipt_bytes(raw: bytes, trusted_sha256: str) -> dict[str, Any]:
    """Validate exact canonical bytes and every required bound digest."""
    digest = hashlib.sha256(raw).hexdigest()
    if _HEX.fullmatch(trusted_sha256) is None or digest != trusted_sha256:
        raise LockGuardError("receipt is not bound by the generated source anchor")
    try:
        receipt = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LockGuardError("lock receipt is not canonical JSON") from error
    if not isinstance(receipt, dict) or set(receipt) != _KEYS:
        raise LockGuardError("lock receipt has missing or unexpected fields")
    if raw != canonical_receipt_bytes(receipt):
        raise LockGuardError("lock receipt bytes are not canonical")
    if receipt["schema_version"] != 1 or receipt["state"] != _LOCK_STATE:
        raise LockGuardError("lock receipt does not declare canonical lock state")
    if receipt["protocol_sha256"] != lock_anchor.EXPECTED_PROTOCOL_SHA256:
        raise LockGuardError(
            "lock receipt protocol hash differs from the frozen protocol"
        )
    if receipt["authorization_sha256"] != lock_anchor.EXPECTED_AUTHORIZATION_SHA256:
        raise LockGuardError("lock receipt authorization hash is not approved")
    if receipt["manifest_sha256"] != lock_anchor.EXPECTED_CANDIDATE_MANIFEST_SHA256:
        raise LockGuardError("receipt candidate manifest file hash is not frozen")
    if receipt["content_sha256"] != lock_anchor.EXPECTED_CANDIDATE_CONTENT_SHA256:
        raise LockGuardError("receipt candidate manifest content hash is not frozen")
    return receipt


def verify_canonical_lock(
    path: Path | None = None, expected_sha256: str | None = None
) -> dict[str, object]:
    """Read only the repository's canonical receipt and check its source anchor."""
    canonical = canonical_lock_path().resolve()
    supplied = canonical if path is None else Path(path).resolve()
    if supplied != canonical:
        raise LockGuardError(
            f"official lock must be the canonical receipt: {canonical}"
        )
    verify_candidate_manifest()
    raw = canonical.read_bytes()
    trusted = lock_anchor.TRUSTED_RECEIPT_SHA256
    receipt = validate_receipt_bytes(raw, trusted)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise LockGuardError("caller-pinned lock hash differs from the source anchor")
    return {
        "verified": True,
        "state": _LOCK_STATE,
        "sha256": digest,
        "receipt": receipt,
        "path": str(canonical),
    }


def verify_runtime_lock(lock: Mapping[str, object] | None) -> dict[str, object]:
    """Re-read the anchor at each official data/training boundary."""
    if (
        lock is None
        or lock.get("verified") is not True
        or lock.get("state") != _LOCK_STATE
    ):
        raise LockGuardError("official operation requires a verified canonical lock")
    digest = lock.get("sha256")
    if not isinstance(digest, str):
        raise LockGuardError("official operation has no receipt digest")
    return verify_canonical_lock(expected_sha256=digest)


def verify_candidate_manifest(root: Path | None = None) -> dict[str, Any]:
    """Rehash the frozen manifest and every frozen artifact entry."""
    root = Path(__file__).resolve().parents[2] if root is None else Path(root).resolve()
    path = root / lock_anchor.CANDIDATE_MANIFEST_RELATIVE_PATH
    raw = path.read_bytes()
    if (
        hashlib.sha256(raw).hexdigest()
        != lock_anchor.EXPECTED_CANDIDATE_MANIFEST_SHA256
    ):
        raise LockGuardError(
            "candidate manifest file digest differs from the source anchor"
        )
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise LockGuardError("candidate manifest is not JSON") from error
    if manifest.get("content_sha256") != lock_anchor.EXPECTED_CANDIDATE_CONTENT_SHA256:
        raise LockGuardError("candidate manifest claims the wrong content digest")
    content = dict(manifest)
    content.pop("content_sha256", None)
    if (
        hashlib.sha256(canonical_receipt_bytes(content).rstrip(b"\n")).hexdigest()
        != lock_anchor.EXPECTED_CANDIDATE_CONTENT_SHA256
    ):
        raise LockGuardError("candidate manifest canonical content digest differs")
    entries = manifest.get("artifacts")
    if (
        not isinstance(entries, list)
        or len(entries) != lock_anchor.EXPECTED_ARTIFACT_COUNT
    ):
        raise LockGuardError(
            "candidate manifest must contain the exact anchored artifact count"
        )
    anchor_path = lock_anchor.ANCHOR_SOURCE_RELATIVE_PATH
    if any(
        isinstance(entry, dict) and entry.get("path") == anchor_path
        for entry in entries
    ):
        raise LockGuardError("source anchor must be excluded from candidate artifacts")
    for entry in entries:
        _verify_artifact(root, entry)
    return manifest


def _verify_artifact(root: Path, entry: object) -> None:
    if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
        raise LockGuardError("invalid candidate artifact entry")
    relative, size, digest = entry["path"], entry["bytes"], entry["sha256"]
    if (
        not isinstance(relative, str)
        or not isinstance(size, int)
        or not isinstance(digest, str)
        or _HEX.fullmatch(digest) is None
    ):
        raise LockGuardError("invalid candidate artifact fields")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise LockGuardError("candidate artifact escapes repository root") from error
    if not path.is_file() or path.stat().st_size != size:
        raise LockGuardError(f"candidate artifact missing or wrong size: {relative}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise LockGuardError(f"candidate artifact digest differs: {relative}")
