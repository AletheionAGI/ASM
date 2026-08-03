# Contributing

Thank you for considering a contribution to ASM — Aletheion State Models.

## Scope

Useful contributions include:

- CPU-runnable tests;
- geometric diagnostics;
- training and ablation scripts;
- documentation that clarifies limitations;
- bug fixes that preserve the non-Transformer design.

Do not add Transformer blocks, self-attention, Q/K/V attention, or `nn.MultiheadAttention`.

## Development Setup

```bash
pip install -e .
python -m pytest -q
```

## Before Opening A PR

Run:

```bash
python -m pytest -q
python scripts/train_tiny.py --config configs/tiny.yaml --text data/tiny.txt --output-dir runs/contrib_smoke --steps 3 --batch-size 2
```

`runs/` is ignored and should not be committed.

## Documentation Standard

Documentation must be honest about experimental status. Do not claim:

- production readiness;
- safety certification;
- AGI;
- alignment;
- superiority over Transformers without evidence;
- exact geodesic solving unless implemented and tested.

## Licensing

The public repository remains AGPL-3.0-only. Commercial use of the public code
is permitted when the applicable AGPL obligations are satisfied.

By submitting a contribution, you represent that you have the right to submit
it and grant Felipe Maya Muniz a perpetual, worldwide, non-exclusive,
irrevocable, royalty-free copyright licence to use, reproduce, modify,
distribute, sublicense, and relicense the contribution, including as part of
commercially licensed versions of ASM.

You retain copyright in your contribution. Every accepted contribution remains
available to the public under AGPL-3.0-only in this repository.

Add this attestation to the pull-request description:

```text
I have the right to submit this contribution and agree to the contributor
licence grant in CONTRIBUTING.md.
```

Substantial contributions to the commercially licensed core may additionally
require an individually signed contributor licence agreement, confirmation of
employer authorization, or a copyright assignment before acceptance. The
maintainer may decline or postpone a contribution when its chain of title is
unclear.

This contributor policy is an initial project safeguard and has not been
presented as a substitute for jurisdiction-specific legal review.

Commercial licensing inquiries: contact@aletheionagi.com
