# Commercial distribution policy

This document defines the repository's default commercial-delivery boundary.
It is an engineering and provenance control, not a grant of commercial rights.
Commercial rights to ASM require a separate signed agreement under
[LICENCE-COMMERCIAL.md](../LICENCE-COMMERCIAL.md).

## Default deliverable

A commercial source, wheel, container, or deployment should contain only the
reviewed ASM implementation, required project notices, and dependencies whose
licenses permit the intended delivery. Research data and generated artifacts
are not part of the default deliverable.

The repository-root [commercial-exclusions.txt](../commercial-exclusions.txt)
is the machine-readable denylist. `MANIFEST.in` applies the corresponding
exclusions to Python source distributions.

Excluded by default:

- all files under `data/`;
- generated benchmark material under `docs/benchmarks/`;
- local run output under `runs/`;
- checkpoints, weights, exported models, and binary model artifacts;
- PDF documents; and
- any artifact marked research-only, noncommercial, share-alike, or of
  unresolved provenance.

An excluded item may enter a particular commercial delivery only after a
documented review identifies its source, exact license, attribution duties,
redistribution rights, model-training implications, and compatibility with the
customer agreement. Approval applies only to the reviewed version and use.

## Build review

Before release:

1. construct the deliverable from an explicit allowlist;
2. compare its contents against `commercial-exclusions.txt`;
3. generate a dependency and artifact inventory;
4. preserve all required third-party notices;
5. record hashes for code, models, containers, and configuration;
6. verify that no research dataset or benchmark answer is embedded; and
7. archive the completed provenance review with the customer release record.

The checked-in Wikipedia sample remains available for reproducible research,
but it is separately licensed and is not part of the ASM commercial license or
default commercial deliverable.
