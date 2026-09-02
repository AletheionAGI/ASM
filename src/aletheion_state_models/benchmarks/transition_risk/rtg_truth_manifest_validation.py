"""Relational validation for truth and paired input/attached state ledgers."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .rtg_artifacts import canonical_json, canonical_sha256

SPLITS = ("train", "validation", "calibration")


def _canonical(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest JSON: {path}") from error
    if raw != canonical_json(payload) + b"\n" or not isinstance(payload, dict):
        raise ValueError(f"manifest is not canonical: {path}")
    return payload


def _verify_content(payload: dict[str, object], schema: str) -> None:
    if payload.get("schema") != schema or not isinstance(payload.get("content_sha256"), str):
        raise ValueError("manifest schema or content digest field differs")
    body = dict(payload)
    claimed = body.pop("content_sha256")
    if claimed != canonical_sha256(body):
        raise ValueError("manifest content digest differs")


def _split_counts(allowed: dict[str, object], truth: dict[str, object]) -> dict[str, int]:
    allowed_rows = allowed.get("splits")
    truth_rows = truth.get("splits")
    if not isinstance(allowed_rows, list) or not isinstance(truth_rows, list):
        raise TypeError("allowed/truth split tables are missing")
    if [row.get("name") for row in allowed_rows] != list(SPLITS) or [row.get("name") for row in truth_rows] != list(SPLITS):
        raise ValueError("allowed/truth split order differs")
    counts = {}
    for allowed_row, truth_row in zip(allowed_rows, truth_rows, strict=True):
        audit = allowed_row.get("audit")
        if not isinstance(audit, dict):
            raise TypeError("allowed split audit is missing")
        origin_count = audit.get("origin_count")
        candidate_count = audit.get("candidate_count")
        if origin_count != truth_row.get("origin_count") or candidate_count != truth_row.get("candidate_count"):
            raise ValueError("truth counts differ from allowed input counts")
        positives, negatives = truth_row.get("positive_labels"), truth_row.get("negative_labels")
        digest = truth_row.get("truth_sha256")
        if not all(type(value) is int and value >= 0 for value in (origin_count, candidate_count, positives, negatives)):
            raise ValueError("truth counts are invalid")
        minimum = 500 if truth_row["name"] == "train" else 100
        if (candidate_count != origin_count * 6
                or positives + negatives != candidate_count
                or origin_count < minimum or positives < 25 or negatives < 25):
            raise ValueError("truth class totals or frozen minima differ")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("truth split digest differs")
        counts[truth_row["name"]] = candidate_count
    return counts


def _ledger(path: Path, *, input_only: bool) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("state ledger is not a mapping")
    input_fields = {"identity", "pre_state", "next_state", "fixed_frame"}
    if input_only:
        if set(payload) != input_fields | {"schema"} or payload.get("schema") != "ATTR-RTG-STATE-INPUTS-V1":
            raise ValueError("input-only state ledger schema differs")
    else:
        truth_fields = {"physical_target", "persistence_target", "unsafe", "failure_delay"}
        if set(payload) != input_fields | truth_fields:
            raise ValueError("attached state ledger schema differs")
    count = len(payload["identity"])
    if any(len(payload[field]) != count for field in set(payload) - {"schema"}):
        raise ValueError("state ledger column counts differ")
    return payload



def _attached_truth_sha256(payload: dict[str, object]) -> str:
    rows = []
    identities = payload["identity"]
    for offset in range(0, len(identities), 6):
        group = identities[offset : offset + 6]
        origin = list(group[0][:-1])
        if len(group) != 6 or any(list(identity[:-1]) != origin or identity[-1] != index
                                  for index, identity in enumerate(group)):
            raise ValueError("attached ledger does not preserve six-candidate origins")
        persistence = [int(value) for value in payload["persistence_target"][offset]]
        failure_delay = int(payload["failure_delay"][offset])
        if any([int(value) for value in payload["persistence_target"][offset + index]] != persistence
               or int(payload["failure_delay"][offset + index]) != failure_delay
               for index in range(6)):
            raise ValueError("attached origin persistence truth differs by candidate")
        candidates = []
        for index in range(6):
            target = [int(value) for value in payload["physical_target"][offset + index]]
            candidates.append({
                "action_index": index,
                "target_sha256": canonical_sha256(target),
                "unsafe": bool(payload["unsafe"][offset + index]),
            })
        rows.append({
            "identity": origin,
            "persistence_sha256": canonical_sha256(persistence),
            "failure_delay": failure_delay,
            "candidates": candidates,
        })
    return canonical_sha256(rows)

def validate_truth_manifest_relations(output: str | Path) -> None:
    """Recompute truth content/count relations and all 30 paired ledger identities."""
    output = Path(output)
    allowed = _canonical(output / "allowed_manifest.json")
    truth = _canonical(output / "truth_manifest.json")
    _verify_content(allowed, "ATTR-RTG-ALLOWED-MANIFEST-V1")
    _verify_content(truth, "ATTR-RTG-TRUTH-MANIFEST-V1")
    counts = _split_counts(allowed, truth)
    truth_digests = {row["name"]: row["truth_sha256"] for row in truth["splits"]}
    arms = tuple(f"{kind}_seed{seed}" for kind in ("asm", "transformer")
                 for seed in (29, 43, 71, 89, 107))
    for arm in arms:
        for split in SPLITS:
            inputs = _ledger(output / arm / f"state_inputs_{split}.pt", input_only=True)
            attached = _ledger(output / arm / f"states_{split}.pt", input_only=False)
            if len(inputs["identity"]) != counts[split] or inputs["identity"] != attached["identity"]:
                raise ValueError("input/attached ledger identities or truth counts differ")
            if _attached_truth_sha256(attached) != truth_digests[split]:
                raise ValueError("attached ledger truth differs from truth manifest digest")
