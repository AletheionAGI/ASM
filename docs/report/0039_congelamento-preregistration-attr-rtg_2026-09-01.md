# Request Report

- Status: completed
- Date: 2026-09-01

## User request

Revisar e congelar o preregistration ATTR-RTG ASM-X versus Transformer, incorporando as checagens finais de input candidato, erro de transição, selagem, consequência física, cinco seeds, calibração, contagens de parâmetros, failure_delay e safe_terminal, sem implementar, treinar ou abrir tests.

## Summary

## Resultado

O ATTR-RTG foi formalmente congelado como **FROZEN PREREGISTRATION** para a comparação ASM-X versus Transformer. Dois pareceres independentes finais retornaram `READY TO FREEZE`: auditabilidade/matemática e causalidade/semântica.

**PREREGISTRATION HASH:** `4db4e22029431c6544a3d8c032cd75d45308250b3b601c0fe437ace4c51ee7f0`

O hash é o SHA-256 de `docs/ATTR_RTG_PREREGISTRATION_MANIFEST.json`. O manifesto registra caminho, tamanho e SHA-256 do documento congelado e dos dez YAMLs completos por seed.

## Escopo congelado

- backbones: ASM-X e Transformer;
- parâmetros reais: ASM-X 30.122; Transformer 30.120;
- seeds: 29, 43, 71, 89 e 107;
- configs ASM: 156/156 campos literais por seed;
- configs Transformer: 10/10 campos literais por seed;
- nenhum override runtime de seed;
- `G+D` primário: 39.105 parâmetros;
- `C` primário: 39.123 parâmetros;
- mismatch: 18 parâmetros (0,0460%);
- Transformer nativo `G+D/C`: 39.877/39.865.

## Correções incorporadas antes do freeze

1. `G` e `C` recebem o frame candidato completo por codificação fixa dos quatro bytes, sem executar o backbone ou avançar o ambiente.
2. RTG1 foi separado em `RTG1-Z` (MSE/NMSE da transição interna contra persistência de estado) e `RTG1-Y` (NLL da consequência física contra persistência física).
3. O preregistration hash foi separado do futuro implementation seal.
4. `hidden mode` foi retirado de `y_common`, `D`, `P` e dos gates primários; o schema passou a 485 logits/11 grupos.
5. O estudo passou de três para cinco seeds, exigindo direção 5/5.
6. `q95` foi definido como banda residual empírica, não conformal nem intervalo de confiança. Temperatura e resíduos usam subconjuntos disjuntos de calibration, sem garantia de cobertura em ID/shift/OOD.
7. Os dez configs completos eliminam defaults implícitos e o conflito anterior de `seed: 1337`.
8. As 128 amostras de risco usam inverse CDF com CRN SHA-256 keyed e sem estado global de RNG.
9. RTG2 mede outcomes inseguros efetivamente executados; unsafe-ALLOW fica somente descritivo.
10. `failure_delay` foi fechado em 3 para baseline/shift e 1 para OOD, compatível com `low_energy_steps` clipped em 0…3.
11. `safe_terminal=1` somente para término seguro; valor zero sozinho não marca risco. A igualdade `P(y_common_true)==HazardTransition.unsafe` tornou-se auditoria obrigatória.
12. Cobertura decisória abaixo de 0,80 ou lowerCI abaixo de 0,75 bloqueia interpretação operacionalmente útil.

## Verificações

- os 10 YAMLs foram carregados pelas dataclasses reais e os modelos foram instanciados no ambiente do projeto;
- todas as contagens de backbone e MLP foram confirmadas por `sum(p.numel())`;
- `sha256sum -c docs/ATTR_RTG_PREREGISTRATION_MANIFEST.sha256` passou;
- todos os 11 registros internos do manifesto foram re-hashados e conferidos;
- `git diff --check` passou;
- nenhum processo ATTR-RTG, diretório de run, treino, dado de test ou test aberto foi encontrado.

## Próxima etapa autorizada

Pode começar a implementação literal de generator, dataset permitido, `G/D/C`, treinamento, validation e calibration. Os tests ID/shift/OOD permanecem proibidos até a emissão e validação do futuro **IMPLEMENTATION SEAL**.

## Modified files

- [configs/rtg_asm_30k.yaml](../../configs/rtg_asm_30k.yaml)
- [transformer/rtg_transformer_30k.yaml](../../transformer/rtg_transformer_30k.yaml)
- [docs/ATTR_RTG_PREREGISTRATION.md](../ATTR_RTG_PREREGISTRATION.md)
- [configs/rtg_asm_30k_seed29.yaml](../../configs/rtg_asm_30k_seed29.yaml)
- [configs/rtg_asm_30k_seed43.yaml](../../configs/rtg_asm_30k_seed43.yaml)
- [configs/rtg_asm_30k_seed71.yaml](../../configs/rtg_asm_30k_seed71.yaml)
- [configs/rtg_asm_30k_seed89.yaml](../../configs/rtg_asm_30k_seed89.yaml)
- [configs/rtg_asm_30k_seed107.yaml](../../configs/rtg_asm_30k_seed107.yaml)
- [transformer/rtg_transformer_30k_seed29.yaml](../../transformer/rtg_transformer_30k_seed29.yaml)
- [transformer/rtg_transformer_30k_seed43.yaml](../../transformer/rtg_transformer_30k_seed43.yaml)
- [transformer/rtg_transformer_30k_seed71.yaml](../../transformer/rtg_transformer_30k_seed71.yaml)
- [transformer/rtg_transformer_30k_seed89.yaml](../../transformer/rtg_transformer_30k_seed89.yaml)
- [transformer/rtg_transformer_30k_seed107.yaml](../../transformer/rtg_transformer_30k_seed107.yaml)
- [docs/ATTR_RTG_PREREGISTRATION_MANIFEST.json](../ATTR_RTG_PREREGISTRATION_MANIFEST.json)
- [docs/ATTR_RTG_PREREGISTRATION_MANIFEST.sha256](../ATTR_RTG_PREREGISTRATION_MANIFEST.sha256)
- [docs/report/0039_congelamento-preregistration-attr-rtg_2026-09-01.md](0039_congelamento-preregistration-attr-rtg_2026-09-01.md)

## Changes

- Substituí os configs genéricos por dez YAMLs completos e específicos por seed.
- Corrigi o protocolo causal, as métricas RTG1-Z/Y, a calibração empírica e os gates RTG2/RTG3.
- Congelei o documento após dois pareceres READY TO FREEZE.
- Criei e verifiquei o manifesto e o PREREGISTRATION HASH.

## Validation

- Parecer matemático/auditável — READY TO FREEZE
- Parecer causal/semântico — READY TO FREEZE
- Instanciação real dos dez configs — ASM 30.122 e Transformer 30.120 em todas as seeds
- Contagem PyTorch real das MLPs — 39.105 versus 39.123; nativo Transformer 39.877 versus 39.865
- sha256sum -c docs/ATTR_RTG_PREREGISTRATION_MANIFEST.sha256 — passou
- Re-hash dos 11 artefatos do manifesto — passou
- git diff --check — passou
- Busca de processos e diretórios ATTR-RTG — nenhum treino/run encontrado
