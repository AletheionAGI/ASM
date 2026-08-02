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

## Measurement details

- hardware: NVIDIA GeForce RTX 4090;
- streaming batch size: 1;
- outer inference precision: BF16 autocast;
- numerically sensitive recurrence and fast-weight state: FP32;
- `363.66 MiB` is `torch.cuda.max_memory_allocated` after resetting peak stats
  for each streaming segment;
- peak allocated memory includes live model weights, retained inference state,
  PyTorch buffers and temporary tensors from the segment;
- it does not include CUDA allocator reserved memory, driver/runtime memory or
  allocations made by other processes;
- `143,360 bytes` is computed recursively over the retained `InferenceState`
  tensors as `numel × element_size`, including the completed state, bounded
  open block and fast-weight memory state;
- the retained tensor count does not include model parameters;
- measured streaming checkpoints are 512, 4K and 32K. Intermediate 1K, 2K, 8K
  and 16K values must not be presented as measured results.

The matched Transformer was evaluated on the same RTX 4090 and CUDA inference
environment in the paired suite, but its learned absolute-position limit is
512. It therefore has no valid 32K VRAM/cache measurement in this protocol.
The 32K plots establish the bounded-state property of ASM-CM; they are not a
same-length memory-efficiency comparison against that Transformer.

BF16 comparison against full recomposition produced zero argmax mismatches in
all three seeds; mean absolute logit errors ranged from `6.97e-6` to `9.03e-6`.

## Scope

Promotion recognizes durable associative memory, bounded-state streaming, and
language compatibility relative to ASM-R. It does not claim general-language
superiority over the matched Transformer, which retains lower CE.

Source data: [`summary.json`](summary.json).

## Charts

- [Retained state versus stream length](charts/cache_vs_context.svg)
- [Peak allocated VRAM versus stream length](charts/vram_vs_context.svg)
- [Decode throughput versus stream length](charts/throughput_vs_context.svg)
- [Frozen language CE by seed](charts/language_ce_by_seed.svg)
- [Streaming chart source data](charts/streaming_metrics.csv)

For a public post, the strongest primary figure is
`cache_vs_context.svg`. Pair it with `throughput_vs_context.svg` to show that
the constant state was not achieved through progressive slowdown. Use
`language_ce_by_seed.svg` only when discussing the explicit limitation that the
Transformer still has lower CE.
