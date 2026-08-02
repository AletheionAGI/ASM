# ASM-CM post-FP32 frozen revalidation

ASM-CM — Aletheion Compact Memory Model — is the promoted public name of the
technical `ASM-C2-FW-LM` lineage.

The three already-trained candidate checkpoints were rescored after correcting
the critical recurrent path to FP32. No weights were updated. Every final gate
passed.

| Seed | CE | PPL | 4K tok/s | 32K tok/s | Peak VRAM 32K | Cache 32K |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.329401 | 3.7788 | 80.86 | 80.92 | 363.66 MiB | 143,360 B |
| 2 | 1.327736 | 3.7725 | 80.76 | 81.05 | 363.66 MiB | 143,360 B |
| 3 | 1.328352 | 3.7748 | 80.38 | 80.07 | 363.66 MiB | 143,360 B |

Aggregate CE is `1.328496 ± 0.000687` (population standard deviation). Mean
throughput is `80.67 tok/s` at 4K and `80.68 tok/s` at 32K. Peak VRAM and cache
are invariant across the measured lengths and seeds.

BF16 comparison against full recomposition produced zero argmax mismatches in
all three seeds; mean absolute logit errors ranged from `6.97e-6` to `9.03e-6`.

## Scope

Promotion recognizes durable associative memory, bounded-state streaming, and
language compatibility relative to ASM-R. It does not claim general-language
superiority over the matched Transformer, which retains lower CE.

Source data: [`summary.json`](summary.json).
