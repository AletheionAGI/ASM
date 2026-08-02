# ASM-C2 architecture

ASM-C2 is the experimental addressable-memory generation of ASM-C. It keeps the
compact recurrent state and bounded open block, then adds a fixed number of
content-addressable key/value slots.

```text
token + recurrent state
        ├─> fixed-slot memory read ─> relational transition ─> emitter
        └─> selective memory write ─> next fixed-slot memory
```

For a fixed slot count `S` and memory width `d_memory`, persistent memory is
`O(S * d_memory)` and does not grow with total sequence length. Access costs
`O(S * d_memory)` per token. This is attention-like lookup over bounded memory
slots, not self-attention over the token prefix.

Available experimental forms are ASM-C2-16, ASM-C2-32, and ASM-C2-64. The
NOREAD, NOWRITE, and shuffled-read controls test whether retrieval and writing
cause any observed MQAR gain.

## ASM-C2-FW correction

ASM-C2-FW replaces learned slot routing with a fixed-size fast-weight matrix.
The same key encoder is used for a previous-token write key and a current-token
read query. With write gate $w_t$, retention $f_t$, key $k_t$, and candidate
value $v_t$, the update follows a delta rule:

$$
M_t = f_t M_{t-1} + w_t k_{t-1}
\left(v_t - k_{t-1}^{\mathsf T}M_{t-1}\right)^{\mathsf T}
$$

$$
r_t = q_t^{\mathsf T}M_t
$$

The persistent matrix has shape `addressable_memory_dim` squared and is
independent of sequence length. ASM-C2-FW remains experimental until short
MQAR, causal ablations, long streaming, parity, and language regression pass.

### Durable experimental form

The durable ASM-C2-FW form adds a second consolidated matrix, selective hard
write and consolidation gates, FP32 memory state, distance curriculum, and
language replay. Fast and slow matrices remain fixed in size; this provides
bounded neural working memory, not unlimited archival storage.

ASM-C2 reuses compatible ASM-R weights, while all `addressable_memory.*`
parameters are newly initialized. It is not a pretrained or promoted model.
Promotion is gated by short MQAR, causal ablations, 32K cache/VRAM/throughput,
incremental parity, and language-CE regression checks.

See [report 047](report/047_Plano_Implementacao_ASM_C2_Memoria_Enderecavel_2026_08_01.md).
