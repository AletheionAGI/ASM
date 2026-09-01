# ASM-CM-VR fixed-32 — Phase 1

This artifact validates the first strict fixed-rank composition of durable ASM-CM memory and ASM Variable Rank.

- Dashboard: [`index.html`](index.html)
- Protocol: [`manifest.json`](manifest.json)
- Complete results: [`summary.json`](summary.json)

## Contract

ASM-CM-VR separates two planes:

- **control/address plane:** causal token embeddings, keys, gates, and previous-token control remain available to the associative mechanism;
- **state/value payload plane:** recurrent state and the fast-weight value axis follow the same prefix rank mask.

For fixed-32:

1. the input-only controller is frozen at the exact prefix `[0:32]`;
2. state is projected before memory read/write;
3. `matrix` and `consolidated` are projected on their value axis;
4. write candidates and read vectors are projected;
5. memory output is projected before it reaches the emitter or cache;
6. discarded payload cannot reappear after a `64→32→64` transport test.

The fast-weight key dimension remains an address space. It is not counted as logical state rank.

## Matched arms

| Arm | Total parameters | Trainable parameters | Logical rank |
|---|---:|---:|---:|
| ASM-CM-VR full-64 | 265,866 | 261,706 | 64 |
| ASM-CM-VR fixed-32 | 265,866 | 261,706 | 32 |

Both arms use state-aligned `d_value=64`, durable FP32 fast/slow memory, the same seed 17 initialization, and frozen controllers. The 4,160 controller parameters are instantiated but not trainable.

## MQAR-40 smoke

Both arms received the same 1,000-update short-MQAR stream. Selection did not use the held-out test stream.

| Arm | Held-out accuracy | CE | Update reaching 100% validation |
|---|---:|---:|---:|
| full-64 | 100% | 0.00747 | 500 |
| fixed-32 | 100% | 0.01627 | 750 |

Fixed-32 learned more slowly, but retained complete final short recall.

### Causal memory canaries

| Arm | Normal | No read | No write |
|---|---:|---:|---:|
| full-64 | 100% | 3.22% | 3.22% |
| fixed-32 | 100% | 1.56% | 1.56% |

Disabling either read or write collapsed accuracy to near chance. The result therefore exercises the associative memory path rather than only the recurrent core.

## Streaming

| Arm | Full/stream max error | 4K throughput | Retained state | 4K status |
|---|---:|---:|---:|---|
| full-64 | 2.86e-6 | 123.14 tok/s | 66,112 bytes | completed |
| fixed-32 | 2.86e-6 | 123.70 tok/s | 66,112 bytes | completed |

The equal physical state size and throughput are expected: fixed-32 remains a logical mask over dense storage and kernels.

## Structural validation

The dedicated tests cover:

- invalid configuration combinations;
- exact frozen rank 32;
- inactive state and fast-weight payload invariance;
- positive canary without masking;
- zero inactive Jacobians;
- exact zero inactive matrix/consolidated columns;
- `64→32→64` without payload resurrection;
- full-64 equivalence with a state-aligned ASM-CM control;
- full/stream parity across prompt and block boundaries 1/31/32/33/65;
- projected compact cache after replay and block closure.

## Decision

**Phase 1 passes.** ASM-CM-VR fixed-32 is structurally valid and retains short associative capacity in seed 17.

This is not yet a promotion. The next gate is the delayed-MQAR ladder through 4K, frozen extrapolation at 32K, and new seeds 29/43. The adaptive controller remains out of scope.
