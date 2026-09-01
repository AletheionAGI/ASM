# ATTR-TG1 — trajectory-grounded anticipation

## O que foi testado

A `HazardHead` foi removida. ASM-X Base e Transformer receberam o mesmo plano open-loop H8 e tiveram de prever distribuições de trajetórias físicas. O risco foi calculado somente pela fração de K=256 trajetórias previstas que satisfaziam o predicado unsafe fixo. Nenhum label de hazard, estado futuro realizado ou trap verdadeiro entrou no scorer.

Foram usados os mesmos cinco optimizer seeds do P2 (`29, 43, 71, 89, 107`), mas mundos, episódios, planos, targets, checkpoints e tests eram novos. O test só foi materializado depois de dez checkpoints terminais e dez avaliações K256 em validation serem congelados.

## Resultado principal H8

| Split | Prevalência | ASM-X Base AUPRC | Transformer AUPRC | Δ AUPRC ASM−T | IC95 | ASM Brier | Transformer Brier | Δ Brier (IC95) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ID | 0,5153 | 0,6279 | 0,6236 | +0,0043 | [-0,0115; +0,0206] | 0,2416 | 0,2417 | -0,00010 [-0,00251; +0,00250] |
| shift | 0,6483 | 0,7742 | 0,7732 | +0,0010 | [-0,0135; +0,0143] | 0,2061 | 0,2079 | -0,00188 [-0,00459; +0,00066] |
| OOD | 0,8586 | 0,9354 | 0,9301 | +0,0052 | [-0,00027; +0,01140] | 0,1173 | 0,1197 | -0,00242 [-0,00445; -0,00059] |

O gate registrado exigia ΔAUPRC ID ≥0,03, limite inferior do IC95 acima de zero e limite superior do ΔBrier ≤0,01. **TG2 falhou.** Todos os IC95 de ΔAUPRC H8 incluem zero. O resultado OOD favorece ASM em Brier, mas não confirma vantagem de ranking por AUPRC.

Como a prevalência é alta sob os planos open-loop, a AUPRC absoluta não deve ser comparada diretamente com o P2. Em ID, o lift sobre prevalência é `1,218×` para ASM e `1,210×` para Transformer; em OOD cai para `1,089×/1,083×`.

## Fidelidade da trajetória

| Split | ASM joint NLL H8 | Transformer joint NLL H8 | Δ NLL ASM−T | IC95 |
|---|---:|---:|---:|---:|
| ID | 20,5427 | 20,2058 | +0,3369 | [+0,2251; +0,4543] |
| shift | 19,9059 | 19,6089 | +0,2970 | [+0,1300; +0,4739] |
| OOD | 20,9931 | 20,7392 | +0,2539 | [+0,1144; +0,3998] |

Menor é melhor. O Transformer teve NLL conjunta significativamente menor nos três splits. Portanto, neste protocolo multi-horizonte categórico e privilegiadamente supervisionado, o ASM não mostrou trajetória global melhor.

## Interpretação humana

O experimento decisivo não revelou a separação mecanística esperada. Quando ambos foram obrigados a passar por `representação → trajetória prevista → predicado unsafe`, eles continuaram muito próximos na antecipação H8. O ASM tem pequenos deltas pontuais positivos de AUPRC, mas eles são muito menores que o efeito registrado e compatíveis com zero. Ao mesmo tempo, o Transformer prevê a trajetória física com NLL melhor.

Isso não prova que os mecanismos internos sejam iguais. Mostra que este teste não encontrou evidência de que o mecanismo do ASM produza antecipação trajectory-grounded superior. A tese forte `estado → trajetória → previsibilidade` não foi confirmada para ASM-X Base neste desenho. O último elo, intervenção, permanece não avaliado e não pode ser alegado.

Os resultados também mostram que a boa NLL one-step do ASM no P2 não se transferiu automaticamente para esta tarefa joint H8. As métricas medem objetos diferentes: observáveis Gaussianos no próximo passo versus estado físico categórico completo sob planos contrafactuais em oito passos.

## Integridade e desvio de reporting

A matriz contém 10/10 checkpoints, 10/10 arquivos validation e 30/30 arquivos test, todos hashados no prediction manifest. O scorer usa somente estados previstos e o predicado físico.

Após o test, o leitor de sumarização encontrou um mismatch de caminhos: esperava nomes planos, enquanto o runner selado gravou subdiretórios por split. Nenhum score ou prediction foi recalculado. O summarizer pré-selado e inalterado foi executado sobre links temporários read-only para os mesmos 40 arquivos; sua verificação de hashes passou. O desvio está registrado em `reporting_compatibility_manifest.json`. Isso preserva a integridade dos resultados preditivos, mas deve permanecer como limitação de execução do pipeline de reporting.

## Artefatos

- `trajectory_grounded_anticipation.png/.svg`: comparação direta dos dois braços;
- `quality_by_horizon.png/.svg`: H1/H4/H8 em ID;
- `paired_deltas.png/.svg`: bootstrap pareado ID;
- `summary.json`: métricas, gates e intervalos;
- `index.html`: dashboard separado.
