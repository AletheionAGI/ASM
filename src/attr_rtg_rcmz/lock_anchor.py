"""Generated trust anchor for the ATTR-RTG-RCMZ local protocol lock.

The lock ceremony replaces ``UNLOCKED`` with the canonical receipt SHA-256.
This source file is deliberately outside all candidate/data manifests.
"""

EXPECTED_PROTOCOL_SHA256 = (
    "c37fa09bdad9715d82d5cb6b6108ce5d2147462c79738674abb75ca50dbc0f84"
)
EXPECTED_AUTHORIZATION_SHA256 = (
    "06c9fb584bb149c9f14c51ce3167959b75c0c4694c1f4777e32257f37d7a3cf5"
)
PRIOR_AUTHORIZATION_SHA256 = (
    "e9456e1ba5d8c57e939c8388c78917b3dcd3ec52dedda836dd147de91776f214"
)
CANDIDATE_MANIFEST_RELATIVE_PATH = (
    "docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v5.json"
)
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "0edca10830c63363bbfe5dd1253f488b25e961f784610caefa7b017c607551a6"
)
EXPECTED_CANDIDATE_CONTENT_SHA256 = (
    "ec51589784d8320ad19999eca929aa84f57e56ac5bc47b7de0c7e4e6730614af"
)
EXPECTED_ARTIFACT_COUNT = 107
ANCHOR_SOURCE_RELATIVE_PATH = "src/attr_rtg_rcmz/lock_anchor.py"
# Compatibility names for receipt tooling; verification uses EXPECTED_* above.
PROTOCOL_SHA256 = EXPECTED_PROTOCOL_SHA256
AUTHORIZATION_SHA256 = EXPECTED_AUTHORIZATION_SHA256
TRUSTED_RECEIPT_SHA256 = (
    "6ed6bd2b39460695a891c27c51cc417453e59eb10ad519660219f84a8cb8950e"
)
