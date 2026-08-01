# ASM-C compact streaming validation through 32K

This directory versions the completed ASM-C validation suite run on 2026-08-01.
ASM-C reuses the promoted 100M-token ASM-R seed-1 checkpoint and changes only
the incremental inference state.

## Verdict

ASM-C validates the engineering claim that the implemented fixed-block
streaming state can remain bounded through 32K tokens. It does **not** validate
long-range associative recall: the required short MQAR control failed.

| Criterion | Result |
|---|---:|
| Cache bounded from 4K to 32K | PASS |
| Peak VRAM growth at most 10% | PASS |
| 32K throughput at least 90% of 4K | PASS |
| Short MQAR control accuracy at least 80% | FAIL (`32.25%`) |

ASM-C therefore remains experimental. The streaming mechanism is successful;
the current associative-memory mechanism is not demonstrated by this run.

## Streaming results

| Length | ASM-R tok/s | ASM-C tok/s | Speedup | ASM-R cache | ASM-C cache | ASM-R peak VRAM | ASM-C peak VRAM |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 446.1 | 453.6 | 1.02x | 10,240 B | 6,144 B | 387.8 MiB | 387.5 MiB |
| 1K | 489.3 | 497.9 | 1.02x | 14,336 B | 6,144 B | 388.0 MiB | 387.5 MiB |
| 2K | 474.4 | 500.1 | 1.05x | 22,528 B | 6,144 B | 388.5 MiB | 387.5 MiB |
| 4K | 441.4 | 505.4 | 1.15x | 38,912 B | 6,144 B | 432.3 MiB | 387.5 MiB |
| 8K | 366.1 | 505.8 | 1.38x | 71,680 B | 6,144 B | 522.3 MiB | 387.5 MiB |
| 16K | 259.1 | 504.3 | 1.95x | 137,216 B | 6,144 B | 702.4 MiB | 387.5 MiB |
| 32K | 169.8 | 503.4 | 2.97x | 268,288 B | 6,144 B | 1,062.7 MiB | 387.5 MiB |

From 4K to 32K, ASM-C retained `99.6%` of its measured throughput. Its retained
cache stayed exactly `6,144` bytes and measured peak allocation stayed
`387.53 MiB`. At 32K it was `2.97x` faster than the legacy ASM-R inference path,
used `97.71%` less persistent cache, and reduced measured peak allocation by
`63.53%`.

These are implementation-level CUDA measurements on one RTX 4090, not a claim
of hardware-independent asymptotic performance.

## BF16 reference parity

Across 512 decoded positions (batch 2, 256 decode tokens), compact inference
had mean absolute logit error `0.008416`, maximum absolute error `0.25`, and
three argmax differences (`0.586%`). This demonstrates close, but not bitwise,
BF16 agreement. The maximum error should remain disclosed when comparing
generated sequences because autoregressive sampling can amplify early token
differences.

## MQAR

The model was adapted for 5,000 steps and evaluated on 4,096 targets per
length. The required 40-token control reached only `32.25%` accuracy
(`95% CI 30.84%–33.70%`), below the `80%` gate. Long-distance accuracies of
roughly `1.3%–2.0%` are consequently **not interpretable as retention decay**:
the model had not learned the short task well enough first.

## Paired Transformer context

On the shared Wikipedia evaluation, the matched Transformer had lower CE at
every supported shared length from 64 through 512. ASM-C evaluated lengths of
1K and 2K, while this specific Transformer checkpoint could not exceed its
learned absolute-position limit of 512. That is a limitation of the paired
checkpoint configuration, not proof that Transformers in general are limited
to 512 tokens.

Incremental decode was comparable: ASM-C measured `386.8`, `502.1`, and
`502.2 tok/s` after prompts of 64, 128, and 256 tokens; the Transformer measured
`395.2`, `401.0`, and `390.7 tok/s`. Transformer prefill remained substantially
faster at the larger shared prompt sizes.

## Artifacts

- [`raw/results.json`](raw/results.json): ASM-C streaming and MQAR measurements;
- [`raw/comparison.json`](raw/comparison.json): ASM-R/ASM-C promotion criteria;
- [`raw/bf16_parity.json`](raw/bf16_parity.json): real-checkpoint parity;
- [`raw/asm_c_transformer_paired.json`](raw/asm_c_transformer_paired.json): paired context, speed, memory, and samples;
- [`charts/`](charts/): SVG figures and CSV tables.
- [`SHA256SUMS`](SHA256SUMS): integrity hashes for the raw JSON artifacts.

## Reproduction

```bash
./scripts/run_asm_c_validation_suite.sh

.venv/bin/python scripts/plot_asm_c_validation.py \
  --asm-c-results runs/asm_c_streaming_32k/results.json \
  --asm-r-results runs/asm_r_long_streaming_32k/results.json \
  --comparison runs/asm_c_streaming_32k/comparison.json \
  --paired runs/asm_c_streaming_32k/asm_c_transformer_paired.json \
  --output-root docs/benchmarks/asm_c_streaming_32k/charts
```
