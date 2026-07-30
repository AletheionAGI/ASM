# Auditoria do `src/drm_language_emitter` e alinhamento do roadmap

Data: 2026-07-30  
Escopo: `src/drm_language_emitter`, testes associados, `roadmap.md` e estado
versionado do benchmark independente 125M.

## Conclusão

O núcleo está estável para o benchmark de linguagem em execução, mas o projeto
ainda não entrou na Fase 1 do roadmap matemático. Estamos no **Gate 0:
validação empírica independente e fidelidade de runtime**.

O benchmark não deve ser alterado durante os runs congelados. As correções de
geração e configuração identificadas abaixo devem ser preparadas e testadas
separadamente, sem mudar checkpoints já em treinamento.

## Achados

### Alto — geração não reproduz o forward do modelo 125M treinado

`src/drm_language_emitter/generation.py` sempre avança o estado com
`direction_field -> metric -> flow -> updater`. Ele não executa o caminho
`directional_block_cumsum` nem o `CausalLocalMixer`.

O benchmark atual não é invalidado: treino e avaliação CE chamam
`DRMEmitterModel.forward`. Entretanto, chat ou amostras geradas pelo checkpoint
local-mixer não representam fielmente a arquitetura que produziu sua CE.

Correção recomendada: criar uma API única de prefill/decode no modelo e fazer
`generate` reutilizá-la. Adicionar teste de paridade entre logits de prefixo
obtidos por `forward` e pelo estado de geração para cada sequence mode
suportado.

### Médio — flags públicas inertes

`DRMConfig` aceita `use_toroidal_state` e `tie_embeddings`, mas nenhuma delas é
consumida pela construção ou dinâmica do modelo.

Isso pode produzir experimentos rotulados como toroidais ou tied sem qualquer
mudança efetiva. Até a implementação formal, a opção mais segura é rejeitar
`true` com erro explícito ou marcá-las como experimentais/inativas na saída do
run.

### Médio — alvo implícito incorreto no relatório geométrico

Quando `targets` não é informado, `geometry_report` usa `targets = input_ids`.
Essa loss mede reconstrução do token atual, não previsão do próximo token.

Os diagnósticos geométricos continuam úteis, mas a CE default não deve ser
publicada como CE de linguagem. A API deve exigir targets ou deslocar
explicitamente entrada/alvo.

### Baixo — monólito no caminho crítico

`model.py` tem mais de 1.100 linhas e concentra dinâmica recorrente, cumsum,
Anderson, mixer, losses e diagnósticos. Isso aumenta o risco de divergência
entre treino, avaliação e geração — divergência já observada no primeiro
achado.

Após o benchmark, recomenda-se extrair engines de sequência e uma interface
comum de transição/prefill/decode antes de adicionar transporte e curvatura.

## Pontos positivos verificados

- configuração rejeita campos desconhecidos e valida limites/tipos;
- checkpoints usam `weights_only=True` e validam estrutura;
- manifests memmap verificam tamanho, SHA-256 e path traversal;
- métrica SPD e naturalização usam solve em precisão adequada;
- caminhos block/superblock possuem testes de causalidade e backward finito;
- avaliação preserva o modo anterior do modelo;
- benchmark externo possui separação documental e auditoria de contaminação;
- suíte atual: 73 testes aprovados e 1 teste CUDA ignorado no ambiente isolado.

## Posição no roadmap

| Etapa | Situação |
|---|---|
| Gate 0 — benchmark independente | Em andamento |
| Gate 0 — fidelidade de geração/configuração | Pendente |
| Fase 1 — rank, kernel e estratos | Não iniciada |
| Fases 2–6 formalmente dependentes | Não iniciadas |

O próximo avanço formal é a **Fase 1 — rank métrico, kernel efetivo e
estratificação**, mas somente depois de concluir o benchmark congelado e
corrigir a paridade de geração.

## Comparação com `docs/paper/drm_v6.tex`

### Mapeamento formal

| Construção do paper | Estado no código | Classificação |
|---|---|---|
| Estado/base `M` e bundle ambiente `E` | Tensor latente fixo `z ∈ R^d` | Analogia computacional |
| Métrica PSD `g` | `diag(softplus + eps) + UUᵀ` | Implementada como SPD, não PSD degenerável |
| `Ker(g)` e fibra quociente `E/Ker(g)` | Ausentes | Não implementado |
| `d_DRM = rank(g)` | `dimD = sum(gates)` | Não equivalente |
| Estratos de rank constante `Sα` | Ausentes | Não implementado |
| Anchor `rho:E→TM` e compatibilidade do kernel | Ausentes | Não implementado |
| Curvas admissíveis | Update neural do estado | Apenas analogia |
| Conexão métrica `nabla` | Ausente | Não implementado |
| Transporte paralelo intrastatum `P` | Ausente | Não implementado |
| Mapas de transição de rank `J` | Ausentes | Não implementado |
| Transporte híbrido ordenado | Ausente | Não implementado |
| Holonomia/histerese de transição | Proxies de recorrência/estabilidade | Não equivalente |
| Critério `JᵀG+J <= G-` | Ausente | Não implementado |
| Ação/energia relacional | `metric_energy` e regularização de action | Aproximação parcial |
| Redução Fisher–Rao | Ausente | Não implementado |
| Fechamento toroidal condicional | Flag inerte e helper seno/cosseno | Não implementado |

### Achado conceitual crítico

No paper, a dimensão efetiva surge do kernel de uma métrica positiva
semidefinida. No código, `metric_eps > 0` e `softplus` tornam toda entrada
diagonal estritamente positiva. Logo, `G` tem rank matemático completo em todo
estado, independentemente dos gates.

Consequências:

1. `dimD_mean` não pode ser descrito como dimensão DRM formal;
2. não existem eventos de mudança de estrato segundo a definição do paper;
3. sem mudança de rank não há `J` interstratum nem holonomia dimensional;
4. transporte por alinhamento de frames, como proposto no roadmap antigo,
   seria um diagnóstico neural útil, mas não implementaria sozinho o
   transporte do bundle quociente do paper.

### Ordem corrigida

O roadmap anterior colocava transporte e holonomia antes de rank/kernel. A
ordem foi corrigida para respeitar as dependências de `drm_v6.tex`:

1. estimar/representar rank, kernel, fibra efetiva e estratos;
2. identificar mapas de transição e defeitos energéticos;
3. introduzir anchor explícito e movimento admissível;
4. definir conexão e transporte dentro de cada estrato;
5. compor transporte híbrido e medir holonomia/histerese;
6. validar reduções clássicas, incluindo Fisher–Rao.

O fechamento toroidal fica como trilha opcional posterior. O próprio paper
afirma que boundedness e recorrência não implicam topologia toroidal.

## Evidência de validação

```text
python -m pytest
73 passed, 1 skipped
```

Também foram pesquisadas referências a transporte, holonomia, curvatura,
Fisher-Rao, toroidal e anchor explícito. Nenhum módulo formal das Fases 1–6
existe hoje; usos atuais de “anchor” pertencem aos solvers de proposta
geodésica/direcional e não implementam `rho_p`.
