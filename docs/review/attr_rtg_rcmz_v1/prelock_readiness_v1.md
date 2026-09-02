# ATTR-RTG-RCMZ-V1 — prelock readiness

> **FOUR FINAL REVIEWS PENDING — NOT LOCKED**

- Protocol SHA-256: `0bb486cca2069580f1563ea0d03934f67f727849b88106d7e1cda85c205464b1`
- Candidate manifest file SHA-256: `8e73c9f5bb230dd1e33a6b3823771945e8988e5c1c01c1502bada79515ae3cf2`
- Candidate manifest content SHA-256: `866c51da817b479d6cb002a444f9dcd411f26c1a980141947c8aa00499d653af`

Closed candidate items:

1. 20 configs; every arm/seed is 50,000 trainable and graph-active parameters;
2. deterministic split/batch/candidate manifests;
3. exact R/CM/Z/T four-field adapters and strict ASM-Z tests;
4. common24/native and graph/padding tests;
5. exact H8/calibration/decision/bootstrap/gate implementation and goldens;
6. two clean 20-cell CUDA runs with identical `a1fd29e899a5f0508f687edbd9e1674c75feb956a29d58f766091a0924ee3f2f`, peak `2839544588` below 20 GiB and supervised `COMPLETED`;
7. terminal CLI with heartbeat/ETA/status/log and RTG-style grouped-bar PNG/SVG/HTML outputs.

Validation: 31 tests, Ruff, compileall, diff check, CLI smoke and source modularity passed.

The user conditionally authorizes lock creation only after four exact `READY TO LOCAL LOCK: YES` reviews. No official data, training, calibration, test opening or lock has occurred.
