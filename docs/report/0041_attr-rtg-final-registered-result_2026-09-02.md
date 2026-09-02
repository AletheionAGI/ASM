# Request Report

- Status: completed
- Date: 2026-09-02

## User request

O usuário mostrou que a execução confirmatória one-shot ATTR-RTG terminou e, antes de continuar o novo ATTR-RTG-RCM, pediu implicitamente a validação, interpretação e publicação final do resultado atual. Preservar a decisão futura de registrar CM−R, CM−T e R−T como três contrastes confirmatórios separados, sem misturá-la ao estudo atual ASM-X versus Transformer.

## Summary

# ATTR-RTG — resultado confirmatório final

> **Resultado registrado e concluído.** Este estudo compara **ASM-X/ASM30122** com **Transformer/T30120**. Não representa ASM-R, ASM-CM ou ASM-VR.

## Conclusão em linguagem humana

- O único gate aprovado foi **`Transformer.RTG1-Z`**.
- O Transformer previu a próxima interface interna melhor que a persistência em ID, em 5/5 seeds.
- ASM-X ficou pior que a persistência em RTG1-Z.
- Os decoders físicos D dos dois braços falharam.
- Nenhum sistema passou governança RTG2-G, RTG2-C ou RTG2-V.
- Nenhum gate RTG3 passou em shift/OOD.
- O resultado não autoriza claims de safety, causalidade, governança eficaz, superioridade ASM ou vencedor universal.

## Proveniência e integridade

| Item | Valor |
|---|---|
| Resultado | `runs/attr_rtg/registered_test_result.json` |
| Bytes | 2.150.671.910 |
| SHA-256 do resultado | `3e936b983d0652f3048e3b9ccc90b435ab7e0f3555d95788f035c4b488f2e217` |
| Payload do implementation seal | `173e5be79a21b38341ca878511042cf8932510ff0e05e26cd27ced521a9e1784` |
| Arquivo do implementation seal | `f853d0de3e268618e408e2250d23a30928d0e9713795593af173b220d8940bea` |
| Preregistration | `4db4e22029431c6544a3d8c032cd75d45308250b3b601c0fe437ace4c51ee7f0` |
| Receipt de abertura | `9f66b0e6e8075d9ec4796c23b1cd3dbb568fe428449623725f9c8d5b12fb4564` |
| Receipt de conclusão | `9b8827eb51728728946f227d65f5a6fd8b359b66e5e79c19f699166a1d49465b` |
| Execução | 2026-09-01 22:32:03 −03 até 2026-09-02 07:22:19 −03; aproximadamente 8h50m16s |
| Matriz | 3 splits × 10 sistemas × G/C = 60 batches; 482.520 records |
| Bootstrap | 21 blocos; PCG64 seed 20260903; 1.000 réplicas; seed→world→episode |

A auditoria streaming encontrou zero truncamentos, nonfinite, erros canônicos, divergências de pairing ou inconsistências de gates. Verificou 384 artefatos do implementation seal e 11 artefatos do preregistration.

## Gates

- Passaram: **Transformer.RTG1-Z**.
- Falharam: os outros 32 gates. A tabela completa está em [`gates.csv`](../benchmarks/attr_rtg/gates.csv).

## Transição e consequência física

| Regime | Braço | NMSE RTG1-Z [IC95] | D macro-accuracy | Persistência accuracy | NLL D(G) | NLL persistência | ΔNLL | ECE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ID | ASM-X | 1.4308 [1.2670, 1.6126] | 0.3312 | 0.6628 | 3.1980 | 3.1083 | +0.0898 | 0.0049 |
| ID | Transformer | 0.7238 [0.7058, 0.7432] | 0.3308 | 0.6628 | 2.9476 | 3.1083 | -0.1607 | 0.0054 |
| Shift | ASM-X | 1.5156 [1.3615, 1.6707] | 0.3382 | 0.6856 | 3.2005 | 2.8991 | +0.3014 | 0.0143 |
| Shift | Transformer | 0.7546 [0.7256, 0.7840] | 0.3353 | 0.6856 | 2.8769 | 2.8991 | -0.0222 | 0.0151 |
| OOD | ASM-X | 1.1798 [1.0397, 1.3439] | 0.3389 | 0.6735 | 3.0290 | 3.0100 | +0.0189 | 0.0904 |
| OOD | Transformer | 0.7191 [0.6848, 0.7511] | 0.3357 | 0.6735 | 2.8511 | 3.0100 | -0.1590 | 0.0716 |

Em ID, o Transformer teve NMSE `0,7238 [0,7058; 0,7432]`, cumprindo `<=0,90`, upper CI `<1` e direção 5/5. ASM-X teve `1,4308 [1,2670; 1,6126]`.

O Transformer também reduziu NLL em ID (`Δ=-0,1607`), mas a razão foi aproximadamente `0,9483`, acima do limite registrado `0,90`. Além disso, D falhou: macro-accuracy ficou perto de `0,331`, contra `0,663` da persistência. Por isso `Transformer.RTG1-Y=false`.

## Governança absoluta

| Regime | Braço | Sistema | Unsafe | Redução relativa | Safe-service | Coverage |
|---|---|---|---:|---:|---:|---:|
| ID | ASM-X | G | 5.217% | 0.000% | 100.000% | 100.000% |
| ID | ASM-X | C | 5.143% | 1.421% | 99.825% | 99.571% |
| ID | Transformer | G | 5.217% | 0.000% | 100.000% | 100.000% |
| ID | Transformer | C | 5.186% | 0.593% | 99.945% | 99.845% |
| Shift | ASM-X | G | 6.524% | 0.000% | 100.000% | 100.000% |
| Shift | ASM-X | C | 6.450% | 1.133% | 99.303% | 98.652% |
| Shift | Transformer | G | 6.524% | 0.000% | 100.000% | 100.000% |
| Shift | Transformer | C | 6.380% | 2.211% | 99.484% | 98.752% |
| OOD | ASM-X | G | 16.417% | 0.000% | 100.000% | 100.000% |
| OOD | ASM-X | C | 16.382% | 0.215% | 99.709% | 99.407% |
| OOD | Transformer | G | 16.417% | 0.000% | 100.000% | 99.998% |
| OOD | Transformer | C | 16.380% | 0.229% | 99.743% | 99.473% |

- **G:** redução exatamente zero em todas as seeds, braços e regimes; todas as decisões efetivas mantiveram a taxa unsafe da baseline.
- **C:** reduções relativas agregadas entre aproximadamente `0,2%` e `2,2%`, muito abaixo dos `50%` registrados. Em shift/OOD, vários ICs incluem zero.
- Safe-service e coverage permaneceram altos, mas isso não compensa a falha da redução unsafe no gate composto.

## Comparação ASM-X versus Transformer

| Regime | ΔNMSE ASM−T [IC95] | ΔNLL ASM−T [IC95] | Δunsafe G |
|---|---:|---:|---:|
| ID | +0.7070 [+0.5468, +0.8834] | +0.2505 [+0.1976, +0.2998] | +0.0000 |
| Shift | +0.7610 [+0.6061, +0.9095] | +0.3236 [+0.2612, +0.3840] | +0.0000 |
| OOD | +0.4607 [+0.3207, +0.6145] | +0.1779 [+0.1218, +0.2350] | +0.0000 |

Como menor NMSE/NLL é melhor, os deltas positivos mostram que o Transformer foi descritivamente melhor em previsão nos três regimes. A direção foi a mesma em 5/5 seeds de cada regime. Isso não produz um claim de vencedor universal, e não houve diferença de governança G: ambos tiveram redução zero.

## RTG2-V: valor adicional de G sobre C

O sinal registrado favorável a G exigia `delta_safety >= 0,02`. Os deltas foram negativos e muito pequenos:

- ASM-X: ID `-0.000741`, Shift `-0.000739`, OOD `-0.000353`.
- Transformer: ID `-0.000310`, Shift `-0.001443`, OOD `-0.000376`.

G preservou ligeiramente mais safe-service/coverage, mas não entregou redução unsafe. Portanto RTG2-V e RTG3-V falharam.

## q95 e calibração

q95 é o `k=min(n,ceil((n+1)×0,95))`-ésimo menor resíduo absoluto `|p-y|` na metade `calibration_residual`. Ele define uma banda residual empírica. Não é intervalo conformal, IC bootstrap, coverage decisória ou garantia de safety. Os valores q95 não estão no resumo derivado e não foram inventados.

## Claims permitidos e bloqueados

**Permitido:** neste benchmark registrado, o Transformer passou RTG1-Z em ID e apresentou NMSE/NLL descritivamente menores que ASM-X nos regimes avaliados.

**Bloqueado:** safety real, garantia de cobertura, causal understanding, memória causal, governança eficaz, valor adicional de G, generalização RTG3, superioridade ASM, superioridade universal do Transformer ou extrapolação para ASM-R/ASM-CM/ASM-VR.

## Artefatos visuais

- [`index.html`](../benchmarks/attr_rtg/index.html) — dashboard final.
- [`architecture_quality.svg`](../benchmarks/attr_rtg/architecture_quality.svg) — NMSE/NLL.
- [`governance.svg`](../benchmarks/attr_rtg/governance.svg) — unsafe, safe-service e coverage.
- [`g_vs_c.svg`](../benchmarks/attr_rtg/g_vs_c.svg) — trade-off G/C.
- [`seed_differences.svg`](../benchmarks/attr_rtg/seed_differences.svg) — direção por seed.
- CSV/JSON derivados estão no mesmo diretório e têm `registered_summary.json` como única fonte.

## Validation and publication status

The one-shot run completed naturally. It was not restarted. The final result, dashboard, plots, CSV tables, reusable summary, and human interpretation were published. The separate ATTR-RTG-RCM-V1 draft remains NOT FROZEN. The user's later design decision is recorded for the next revision: three separate confirmatory contrasts CM−R, CM−T, and R−T, with no pooled winner. It was intentionally not mixed into this completed ASM-X versus Transformer result.

The full repository suite had one known workspace-level failure: `tests/test_rtg_source_inventory.py` detects the 15 new RCM YAMLs outside the already sealed ATTR-RTG literal inventory. The dashboard-targeted tests all passed. The sealed result itself passed independent streaming integrity verification.

## Modified files

- [runs/attr_rtg/registered_test_result.json](../../runs/attr_rtg/registered_test_result.json)
- [runs/attr_rtg/IMPLEMENTATION_SEAL.json.opened.completed](../../runs/attr_rtg/IMPLEMENTATION_SEAL.json.opened.completed)
- [docs/benchmarks/attr_rtg/README.md](../benchmarks/attr_rtg/README.md)
- [docs/benchmarks/attr_rtg/architecture.csv](../benchmarks/attr_rtg/architecture.csv)
- [docs/benchmarks/attr_rtg/architecture_quality.png](../benchmarks/attr_rtg/architecture_quality.png)
- [docs/benchmarks/attr_rtg/architecture_quality.svg](../benchmarks/attr_rtg/architecture_quality.svg)
- [docs/benchmarks/attr_rtg/dashboard_data.json](../benchmarks/attr_rtg/dashboard_data.json)
- [docs/benchmarks/attr_rtg/g_vs_c.csv](../benchmarks/attr_rtg/g_vs_c.csv)
- [docs/benchmarks/attr_rtg/g_vs_c.png](../benchmarks/attr_rtg/g_vs_c.png)
- [docs/benchmarks/attr_rtg/g_vs_c.svg](../benchmarks/attr_rtg/g_vs_c.svg)
- [docs/benchmarks/attr_rtg/gates.csv](../benchmarks/attr_rtg/gates.csv)
- [docs/benchmarks/attr_rtg/governance.csv](../benchmarks/attr_rtg/governance.csv)
- [docs/benchmarks/attr_rtg/governance.png](../benchmarks/attr_rtg/governance.png)
- [docs/benchmarks/attr_rtg/governance.svg](../benchmarks/attr_rtg/governance.svg)
- [docs/benchmarks/attr_rtg/index.html](../benchmarks/attr_rtg/index.html)
- [docs/benchmarks/attr_rtg/registered_summary.json](../benchmarks/attr_rtg/registered_summary.json)
- [docs/benchmarks/attr_rtg/seed_differences.png](../benchmarks/attr_rtg/seed_differences.png)
- [docs/benchmarks/attr_rtg/seed_differences.svg](../benchmarks/attr_rtg/seed_differences.svg)
- [docs/benchmarks/attr_rtg/seeds.csv](../benchmarks/attr_rtg/seeds.csv)
- [scripts/build_attr_rtg_dashboard.py](../../scripts/build_attr_rtg_dashboard.py)
- [scripts/attr_rtg_dashboard/__init__.py](../../scripts/attr_rtg_dashboard/__init__.py)
- [scripts/attr_rtg_dashboard/data.py](../../scripts/attr_rtg_dashboard/data.py)
- [scripts/attr_rtg_dashboard/charts.py](../../scripts/attr_rtg_dashboard/charts.py)
- [scripts/attr_rtg_dashboard/html.py](../../scripts/attr_rtg_dashboard/html.py)
- [tests/test_attr_rtg_dashboard.py](../../tests/test_attr_rtg_dashboard.py)
- [docs/report/0041_attr-rtg-final-registered-result_2026-09-02.md](0041_attr-rtg-final-registered-result_2026-09-02.md)

## Changes

- Validado o resultado one-shot de 2,15 GB e a cadeia completa preregistration → implementation seal → opened → result → completed.
- Extraído um resumo registrado pequeno sem alterar o run selado.
- Criados dashboard HTML, quatro gráficos em PNG/SVG, cinco tabelas CSV, JSON derivado e README humano.
- Adicionados builder modular e testes do dashboard, todos os novos módulos com até 97 linhas.
- Mantida separação explícita entre ATTR-RTG atual (ASM-X versus Transformer) e o futuro ATTR-RTG-RCM.

## Validation

- Auditoria streaming integral — passou; JSON canônico/finito, 482.520 records, pairing completo, hashes e gates coerentes.
- verify_pipeline_seal somente leitura — passou; 7 grupos e 384 artefatos.
- Preregistration audit — passou; 11 artefatos e hash 4db4e220...ee7f0.
- .venv/bin/python scripts/build_attr_rtg_dashboard.py — passou.
- .venv/bin/python -m pytest -q tests/test_rtg_registered_summary.py tests/test_attr_rtg_dashboard.py — 10 passaram.
- .venv/bin/python -m compileall -q scripts/build_attr_rtg_dashboard.py scripts/attr_rtg_dashboard tests/test_attr_rtg_dashboard.py — passou.
- git diff --check — passou.
- HTML/README link audit — passou; nenhum asset ausente.
- Inspeção visual de quatro PNGs — passou.
- SOLID source audit — 426 conformes; 8 exceções e 4 violações preexistentes; novos arquivos todos conformes.
- Suite completa — 437 passaram, 1 falhou em tests/test_rtg_source_inventory.py porque os 15 YAMLs RCM novos estão fora do inventário literal já selado; falha não pertence ao dashboard nem ao resultado registrado.
