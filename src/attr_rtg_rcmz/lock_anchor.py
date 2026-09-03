"""Generated trust anchor for the ATTR-RTG-RCMZ local protocol lock.

The lock ceremony replaces ``UNLOCKED`` with the canonical receipt SHA-256.
This source file is deliberately outside all candidate/data manifests.
"""

EXPECTED_PROTOCOL_SHA256 = (
    "c37fa09bdad9715d82d5cb6b6108ce5d2147462c79738674abb75ca50dbc0f84"
)
EXPECTED_AUTHORIZATION_SHA256 = (
    "cd2ff6ec04fd2b806970614cd7a706d0c3ca5688018ab208d2483920e9d02551"
)
PRIOR_AUTHORIZATION_SHA256 = (
    "2e0c73ba6d272a73bd3979265fff17767e56730248d3c95b87565e51e6ae7986"
)
CANDIDATE_MANIFEST_RELATIVE_PATH = (
    "docs/review/attr_rtg_rcmz_v1/local_lock_manifest_v9.json"
)
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "0fcee34c7d5e99d1a151614685817fa32d1136d26758b193f73090c09028ed39"
)
EXPECTED_CANDIDATE_CONTENT_SHA256 = (
    "34955553dfcf748971cd98ccbfbf5f35c47d2793f6ad2caa10bc9e160605a075"
)
EXPECTED_ARTIFACT_COUNT = 116
ANCHOR_SOURCE_RELATIVE_PATH = "src/attr_rtg_rcmz/lock_anchor.py"
# Compatibility names for receipt tooling; verification uses EXPECTED_* above.
PROTOCOL_SHA256 = EXPECTED_PROTOCOL_SHA256
AUTHORIZATION_SHA256 = EXPECTED_AUTHORIZATION_SHA256
TRUSTED_RECEIPT_SHA256 = (
    "3c48351f130b2134614c925a1c36bba2a90bdf377355d80cead3874cec434d42"
)
