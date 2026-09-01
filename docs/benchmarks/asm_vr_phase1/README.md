# ASM-VR Phase 1 — colapso sem bypass

A Fase 1 integra ao núcleo ASM-R um rank hard por bloco, frame identidade fixo e
controller causal que observa somente o primeiro token do bloco. A configuração
é opt-in e mantém memória de transição desabilitada.

## Resultados integrados

O experimento `integration_summary.json` valida o caminho real de logits e cache:

- diferença máxima entre logits do par após colapso: `0.0`;
- diferença máxima entre estados do par: `0.0`;
- norma do Jacobiano no complemento descartado: `0.0`;
- maior coordenada inativa encontrada no cache: `0.0`;
- erro máximo entre forward e `prefill/decode_step`: `1.0430813e-07`;
- cache sem estado ambiente e sem prefixo completo;
- nenhuma memória de transição.

Execute:

```bash
.venv/bin/python scripts/eval_asm_vr_phase1_integration.py \
  --seed 2026 \
  --output docs/benchmarks/asm_vr_phase1/integration_summary.json
```

## Gate isolado

`summary.json` preserva o experimento anterior `8→3→5→8`, que testa forcing
pós-colapso, emissor efetivo e não recuperação antes da integração ASM-R:

```bash
.venv/bin/python scripts/eval_asm_vr_phase1.py \
  --seed 2026 \
  --output docs/benchmarks/asm_vr_phase1/summary.json
```

## Limite do resultado

A Fase 1 prova o contrato de inferência sem bypass. Ela não demonstra ainda
melhora de qualidade, rank treinável, economia de FLOPs ou vantagem sobre
ASM-R. Esses claims exigem as Fases 2–4 e validação multiseed.
