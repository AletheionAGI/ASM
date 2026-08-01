# ASM-C architecture

ASM-C is the compact-streaming inference form of ASM-R. It reuses the same
trained parameters and full-forward language model; only its incremental state
and emission path change.

```text
completed state ─┐
open local block ├─> next block state ─> local emitter ─> next-token logits
tokens seen ─────┘
```

The cache contains no complete token prefix. At a block boundary it contains a
state tensor of shape `[batch, d_state]`, an empty local-token tensor, and scalar
metadata. Within a block, at most `block_size - 1` token IDs are retained.

ASM-C requires a fixed-block geometric transition. It raises an explicit error
instead of falling back to prefix recomputation when this invariant is absent.

The legacy ASM-R path remains the default for checkpoint compatibility and BF16
reference parity. ASM-C must be enabled with `compact_streaming_inference=true`
or through `build_compact_streaming`.

Promotion requires bounded cache, bounded peak memory, stable 4K–32K throughput,
measured BF16 parity, and a successful short MQAR control before long-retention
results are interpreted.

## Current validation status

The 2026-08-01 RTX 4090 run passed the bounded-cache, bounded-peak-VRAM, and
throughput-retention criteria through 32K. At 32K, ASM-C retained a 6,144-byte
cache, measured 387.53 MiB peak allocation, and ran 2.97x faster than the
legacy ASM-R inference path.

The short MQAR control reached only 32.25% accuracy against the 80% gate.
Long-range associative retention is therefore not interpretable, and ASM-C
remains experimental. See the
[versioned benchmark](benchmarks/asm_c_streaming_32k/README.md).
