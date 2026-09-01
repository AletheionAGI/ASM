# ASM-CM-VR Architecture

## 1. Identity

**ASM-CM-VR — Variable-Rank Compact Memory State Model** combines:

- the bounded durable fast-weight memory of ASM-CM;
- the causal hard projection contract of ASM Variable Rank;
- a fixed rank-32 policy in the first validated implementation.

The initial public builder is:

```python
build_compact_memory_variable_rank(base, fixed_rank=32)
```

It is experimental and is not the adaptive-rank controller.

## 2. Why a direct composition is unsafe

A plain ASM-CM + ASM-VR composition previously failed configuration validation. Removing that check alone would create two problems:

1. full forward applied memory before the final rank projection, while decode applied memory after it;
2. fast-weight memory could store discarded state payload and later mix it back into active coordinates.

The resulting model would violate forward/stream parity and the no-bypass contract.

## 3. Strict payload contract

ASM-CM-VR defines rank over the **state/value payload plane**.

Let `m ∈ {0,1}^D` be the prefix mask. At every memory boundary:

```text
state_in       <- m ⊙ state
matrix         <- matrix ⊙ m on value axis
consolidated   <- consolidated ⊙ m on value axis
candidate      <- m ⊙ candidate
read           <- m ⊙ read
state_out      <- m ⊙ (state_in + projected_read)
```

The fast-weight state has shape `[batch, key_dim, d_state]` in strict mode. Its last axis is aligned with the recurrent state.

Projection uses `torch.where`, not multiplication, so an invalid inactive sentinel cannot survive through `0 × NaN`.

## 4. Permitted control plane

Token embeddings, normalized keys, scalar gates, and the retained previous token remain causal control/address inputs. They are not state payload and are not counted as logical rank.

This distinction preserves the associative nature of ASM-CM while ensuring that a discarded **state/value coordinate** cannot be restored.

Claims of a total 32-dimensional information channel would require masking token/key control as well and are not made.

## 5. Unified execution order

Full forward and retained-state decode use the same adapter:

```text
observe fixed mask
  -> project carried state
  -> projected core/scaffold
  -> project memory input
  -> rank-aware fast-weight read/write
  -> project memory output
  -> emitter and compact cache
```

The existing committed-memory replay rule remains unchanged. Writes in an open block are recomputed from the last committed memory and promoted only when the block closes.

## 6. Configuration safety

`variable_rank_memory_policy="project_io"` is accepted only when:

- Variable Rank is enabled;
- the backend is `fast_weight`;
- the memory value dimension equals `d_state`;
- projected scaffold mode and compact block-cumsum inference are enabled.

All other VR + addressable-memory combinations remain rejected.

## 7. Fixed-rank implementation

The builder initializes an input-only prefix controller with:

- score weights exactly zero;
- biases `+20` for active coordinates;
- biases `-20` for inactive coordinates;
- every controller parameter frozen.

Full-64 and fixed-32 instantiate the same number of parameters and have identical trainable counts. Rank does not yet change tensor shapes or physical kernels.

## 8. Validated invariants

Phase 1 validates:

- exact inactive zeros in state, matrix, and consolidated memory;
- zero Jacobian from discarded state/memory payload to active future payload;
- positive unmasked-memory sensitivity canary;
- shrink→grow without resurrection;
- full-64 equivalence within `1e-6` to state-aligned ASM-CM;
- full/stream parity through multiple block boundaries;
- cache projection after prefill and decode;
- short MQAR dependence on both reads and writes.

## 9. Phase-1 result

Seed-17 MQAR-40 after 1,000 equal updates:

- full-64: 100%, CE 0.00747;
- fixed-32: 100%, CE 0.01627.

Both completed retained-state streaming through 4K with max full/stream error `2.86e-6`. Fixed-32 used the same 66,112 retained bytes and nearly identical throughput as full-64. Thus fixed-32 preserved short associative function without physical savings.

## 10. Long multiseed result

The long curriculum executed full-64, fixed-32, and exploratory adaptive-32 for seeds 17, 29, and 43. The fixed-32 promotion gate failed.

- full-64 and fixed-32 passed all long gates in 2/3 seeds;
- seed 29 collapsed beyond MQAR-40 in both arms, produced non-finite CE at 32K, and failed 32K streaming near token 14K;
- fixed-32 reached 100% at 32K in the two finite successful seeds, but multiseed robustness is required;
- all arms retained 66,112 bytes, so logical rank still provided no physical memory reduction;
- memory ablations remained near chance, supporting the causal memory-payload interpretation.

The exploratory adaptive controller received gradients in `7,725/7,725` checked updates and varied between ranks 16 and 64. Mean rank was 17.51 at MQAR-40 and 36.91 at 32K. Its 32K accuracy was 46.88%, 81.25%, and 90.63% across the three seeds. It completed finite streaming in every seed but passed the 80% quality threshold in only 2/3 seeds. It is not promoted and does not change the fixed-32 decision.

Full results, including numerical failures, are in `docs/benchmarks/asm_cm_vr_fixed32_long/`.

## 11. Decision and next gates

ASM-CM-VR fixed-32 remains experimental and is **not promoted**. The next work is:

1. diagnose the shared seed-29 numerical collapse without using test for checkpoint selection;
2. add numerical telemetry around effective coordinates, memory matrices, and solve boundaries;
3. repeat only under a newly registered train/validation-only stabilization protocol;
4. perform language-retention comparison after numerical robustness is established;
5. implement physical sparse/prefix kernels before making efficiency claims.

The ATTR line remains separate and still precedes Phase 3B/Transition Memory.
