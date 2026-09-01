# PMCS-64: ASM-CM versus ASM-VR-S

The **Parameter-Matched Capability Suite (PMCS-64)** compares the practical purposes of ASM-CM and ASM-VR-S without reusing unmatched historical scores.

- Offline dashboard: [`index.html`](index.html)
- Machine-readable protocol: [`manifest.json`](manifest.json)
- Complete measurements and gates: [`summary.json`](summary.json)

## Question

Under the same corpus, token budget, seeds, optimizer, state width, and nearly identical parameter count:

1. Which model is better for byte-level language modeling?
2. Which model learns durable associative recall?
3. Does fixed rank provide a physical efficiency gain in the current dense implementation?
4. Does retained-state streaming remain finite through 32K?

## Matched arms

| Arm | Total parameters | Trainable parameters | Logical rank |
|---|---:|---:|---:|
| ASM-CM | 274,058 | 274,058 | 64 |
| ASM-VR-S full | 274,135 | 269,975 | 64 |
| ASM-VR-S fixed-32 | 274,135 | 269,975 | 32 |

The total-count mismatch is 77 parameters (`0.028%`). The VR controllers contain 4,160 frozen parameters. Therefore this is a total-parameter match, not an exact trainable-parameter match.

## Protocol

### Language

- document-hash corpus splits inherited from Phase 3A;
- 489 updates, batch 16, sequence length 256;
- 2,002,944 tokens per run;
- AdamW, the same minibatch stream for paired seed `17`, `29`, or `43`;
- checkpoints selected on validation only;
- language test opened after every language checkpoint was frozen.

### Memory specialization

Every frozen language checkpoint received the same memory-heavy curriculum:

- 80% MQAR and 20% language replay;
- MQAR batches: 1,000 at 40; 500 at 80; 500 at 160; 300 at 320; 200 at 512; 100 at 1K; 25 at 4K;
- 2,625 MQAR batches and 3,281 optimizer updates;
- terminal checkpoint fixed without test-based selection;
- held-out evaluation at 40, 512, 4K, and 32K.

A preliminary 80%-language/20%-MQAR sensitivity failed the ASM-CM short control and is retained as `mqar_light_result.json` for seed 17. It is not the primary memory result.

### Streaming

- retained-state `prefill`/`decode_step`, batch 1, BF16 outer context;
- 512 and 4K in all seeds;
- 32K confirmation in seed 17;
- state bytes exclude model parameters and CUDA allocator residency;
- random-token streaming measures numerical stability and cost, not MQAR.

## Results

### Language

| Arm | Test CE, mean ± population SD | Train tokens/s | Peak CUDA MiB |
|---|---:|---:|---:|
| ASM-CM | 2.5489 ± 0.0288 | 9,070 | 1,653.8 |
| ASM-VR-S full | **2.5168 ± 0.0240** | **168,127** | **77.9** |
| ASM-VR-S fixed-32 | 2.5451 ± 0.0178 | 166,302 | 77.9 |

ASM-VR-S full won language quality. Fixed-32 remained within the preregistered `+0.03 nat` non-inferiority margin (`+0.0283 nat` versus full). It did not produce a physical speed or memory gain over full.

### MQAR accuracy after equal specialization

| Arm | 40 | 512 | 4K | 32K | Seeds ≥80% at 32K |
|---|---:|---:|---:|---:|---:|
| ASM-CM | **99.95%** | **36.85%** | **34.38%** | **33.33%** | 1/3 |
| ASM-VR-S full | 3.96% | 3.09% | 1.56% | 0.00% | 0/3 |
| ASM-VR-S fixed-32 | 2.29% | 2.54% | 2.60% | 0.00% | 0/3 |

ASM-CM alone learned the short associative task in all seeds. Long-distance generalization was seed-sensitive: seed 17 reached approximately 100% through 32K, while seeds 29 and 43 collapsed beyond the short control. Thus ASM-CM demonstrated the relevant mechanism, but did not pass the robust 32K promotion gate in this small matched setting.

Neither ASM-VR-S arm learned the short control. Its selective recurrent memory is not a substitute for explicit durable key/value association under this protocol.

### Streaming

| Arm | State bytes at 4K | Throughput at 4K | 32K seed-17 result | Full/stream max error |
|---|---:|---:|---|---:|
| ASM-CM | 131,584 | 162.3 tok/s | failed at token 15,200: singular metric solve | 6.08e-4 |
| ASM-VR-S full | **320** | **1,132.3 tok/s** | failed at token 30,335: non-finite state | **1.91e-6** |
| ASM-VR-S fixed-32 | **320** | **1,135.5 tok/s** | **completed, 1,139.8 tok/s** | **1.59e-6** |

The retained bytes were bounded through each successful measurement. However, bounded allocation does not imply numerical stability. ASM-CM and VR-S full failed before 32K. Fixed-32 was the only matched arm that completed the 32K stream.

ASM-CM also failed the strict `1e-4` short streaming-parity gate after language training. This does not invalidate its full-forward language/MQAR results, but it blocks a production streaming claim for this matched checkpoint.

## Practical conclusion

- Choose **ASM-VR-S full** for the best language quality in this parameter range.
- Choose **ASM-VR-S fixed-32** when a controlled logical bottleneck and the strongest observed streaming stability matter. Do not claim physical savings from rank alone.
- Choose **ASM-CM** as the research architecture for explicit durable associative memory. Require multiseed long-context confirmation and numerical streaming validation for each trained configuration.
- There is no overall winner. Language modeling, associative recall, and streaming stability form different axes.

## Non-claims

- Historical ASM-CM and ASM-VR-S scores were not mixed into this comparison.
- One successful ASM-CM 32K seed is not a robust promotion.
- Logical fixed-32 is not half the FLOPs, VRAM, state bytes, or latency.
- Full-forward MQAR is not evidence of retained-state streaming.
- Random-token streaming is not evidence of associative recall.
