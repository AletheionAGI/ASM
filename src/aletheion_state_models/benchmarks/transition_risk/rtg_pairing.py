"""Canonical byte-equivalent pairing checks for registered RTG records."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

Record = Mapping[str, Any]
IDENTITY_FIELDS = ("seed", "world_id", "episode_id", "t", "action_index")
_PAIRED_FIELDS = (
    "fixed_frame",
    "persistence_target",
    "candidate_unsafe",
    "brake_unsafe",
    "failure_delay",
)


def identity(row: Record) -> tuple[int, str, str, int, int]:
    values = tuple(row[name] for name in IDENTITY_FIELDS)
    if (type(values[0]) is not int or not isinstance(values[1], str)
            or not isinstance(values[2], str) or type(values[3]) is not int
            or type(values[4]) is not int):
        raise ValueError("candidate identity types differ from the registered schema")
    return values  # type: ignore[return-value]


def _bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("paired field is not canonical finite JSON") from error


def _truth(row: Record) -> Any:
    has_y, has_group = "y_common" in row, "group_targets" in row
    if not has_y and not has_group:
        raise ValueError("paired record lacks y_common/group_targets")
    if has_y and has_group and _bytes(row["y_common"]) != _bytes(row["group_targets"]):
        raise ValueError("y_common and group_targets differ within a record")
    return row["group_targets"] if has_group else row["y_common"]


def canonical_records(records: Iterable[Record]) -> tuple[Record, ...]:
    """Require unique records in exact seed/world/episode/t/action order."""
    rows = tuple(records)
    identities = tuple(identity(row) for row in rows)
    if not rows or len(set(identities)) != len(rows) or identities != tuple(sorted(identities)):
        raise ValueError("records require unique lexicographic seed/world/episode/t/action order")
    return rows


def require_byte_equivalent(left: Iterable[Record], right: Iterable[Record]) -> None:
    """Authenticate CRN identities and all outcome-independent paired fields."""
    left_rows, right_rows = canonical_records(left), canonical_records(right)
    if len(left_rows) != len(right_rows):
        raise ValueError("paired record counts differ")
    for left_row, right_row in zip(left_rows, right_rows, strict=True):
        if identity(left_row) != identity(right_row):
            raise ValueError("paired candidate identities differ")
        for field in _PAIRED_FIELDS:
            if field not in left_row or field not in right_row:
                raise ValueError(f"paired records lack {field}")
            if _bytes(left_row[field]) != _bytes(right_row[field]):
                raise ValueError(f"paired field differs: {field}")
        if _bytes(_truth(left_row)) != _bytes(_truth(right_row)):
            raise ValueError("paired field differs: y_common/group_targets")
