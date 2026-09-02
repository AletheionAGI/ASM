"""Opaque integrity evidence for the sealed ATTR-RTG evaluation flow."""

from __future__ import annotations

_TOKEN = object()


class IntegrityEvidence:
    """Proof that an exact sealed pipeline entered evaluation."""

    __slots__ = ("seal_sha256",)

    def __init__(self, seal_sha256: str, token: object) -> None:
        if token is not _TOKEN:
            raise TypeError("integrity evidence is issued only by the sealed pipeline")
        self.seal_sha256 = seal_sha256


def _issue_integrity_evidence(seal_sha256: str) -> IntegrityEvidence:
    return IntegrityEvidence(seal_sha256, _TOKEN)


def require_integrity_evidence(
    evidence: IntegrityEvidence, expected_sha256: str | None = None
) -> bool:
    """Validate opaque evidence and its optional expected seal identity."""
    if not isinstance(evidence, IntegrityEvidence):
        raise PermissionError("opaque pipeline integrity evidence required")
    digest = evidence.seal_sha256
    try:
        valid_digest = len(digest) == 64 and int(digest, 16) >= 0
    except (TypeError, ValueError):
        valid_digest = False
    if not valid_digest or (expected_sha256 is not None and digest != expected_sha256):
        raise PermissionError("integrity evidence seal SHA-256 mismatch")
    return True
