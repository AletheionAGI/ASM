from __future__ import annotations

import json

import pytest

from aletheion_state_models.benchmarks.transition_risk.rtg_artifacts import (
    atomic_write_json,
    canonical_json,
    digest_files,
    verify_files,
)
from aletheion_state_models.benchmarks.transition_risk.rtg_records import (
    CandidateIdentity,
    CandidateRecord,
    read_records_jsonl,
    validate_candidate_records,
    write_records_jsonl,
)


def _records():
    return [
        CandidateRecord(CandidateIdentity(29, 1, "w", "e", 1, action), {"risk": action / 10})
        for action in range(6)
    ]


def test_canonical_json_and_create_only(tmp_path):
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(ValueError):
        canonical_json({"bad": float("nan")})
    path = atomic_write_json(tmp_path / "value.json", {"b": 2, "a": 1})
    assert path.read_bytes() == b'{"a":1,"b":2}\n'
    with pytest.raises(FileExistsError):
        atomic_write_json(path, {})


def test_digest_verification_detects_tamper(tmp_path):
    path = tmp_path / "source.py"
    path.write_text("original")
    records = digest_files({"source": path})
    verify_files(records, {"source": path})
    path.write_text("tampered")
    with pytest.raises(ValueError, match="mismatch"):
        verify_files(records, {"source": path})


def test_records_require_unique_complete_six_candidate_origins():
    rows = _records()
    assert len(validate_candidate_records(reversed(rows))) == 6
    with pytest.raises(ValueError, match="six"):
        validate_candidate_records(rows[:-1])
    with pytest.raises(ValueError, match="unique"):
        validate_candidate_records(rows + [rows[0]])
    with pytest.raises(ValueError, match="finite"):
        CandidateRecord(rows[0].identity, {"risk": float("inf")})


def test_records_jsonl_is_strict_ordered_and_create_only(tmp_path):
    path = write_records_jsonl(tmp_path / "records.jsonl", reversed(_records()))
    loaded = read_records_jsonl(path)
    assert [row.identity.action_index for row in loaded] == list(range(6))
    with pytest.raises(FileExistsError):
        write_records_jsonl(path, loaded)
    raw = json.loads(path.read_text().splitlines()[0])
    raw["extra"] = 1
    path.write_text(json.dumps(raw) + "\n")
    with pytest.raises(ValueError, match="unexpected"):
        read_records_jsonl(path)
