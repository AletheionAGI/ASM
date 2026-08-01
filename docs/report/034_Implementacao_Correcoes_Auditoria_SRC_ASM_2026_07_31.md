# Implementação das correções da auditoria do `src/`

**Data:** 31 de julho de 2026
**Origem:** relatórios 031 e 032
**Estado:** implementação concluída, ainda sem commit

## Resultado

As falhas A-01 a A-07 foram tratadas sem modificar os checkpoints da scaling law e
sem duplicar o núcleo do modelo. A suíte aumentou de 98 para 107 testes aprovados.

```text
107 passed, 1 skipped, 1 warning esperado
```

O aviso restante é produzido intencionalmente pelo teste que simula falha do
`torch.compile` e confirma fallback para execução eager.

## 1. Builders ASM e invariantes

Os builders ASM-R, ASM-D e ASM-S agora transformam modos recorrentes incompatíveis
em um block scan causal compatível. Configurações que removem o campo direcional e
mantêm `local_step`, `geodesic_step` ou `directional_candidates` são rejeitadas
antecipadamente.

Foi adicionada cobertura para construir e executar as variantes diretamente sobre
`DRMConfig()` padrão. O erro tardio `'NoneType' object is not callable` deixou de
ser uma condição alcançável por esses builders.

## 2. Paridade entre treino e geração

O caminho legado `_advance()` foi removido. Ele implementava apenas:

```text
direction_field → metric → flow → naturalize → updater
```

e ignorava transições diretas, memória seletiva, mixer, residual, refinamentos e
composições métricas modernas.

O modelo agora expõe:

```python
state = model.init_inference_state(batch_size, device)
logits, state = model.prefill(prompt, state)
logits, state = model.decode_step(token, state)
```

`generate()` e o benchmark legado de world model utilizam essa API. A implementação
de referência preserva o prefixo e recompõe o mesmo forward usado no treinamento.
Ela é mais lenta do que um decode com cache, porém garante corretude para todas as
composições sem criar uma segunda dinâmica.

Os testes comparam logits completos e token a token em FP32 para ASM-X, ASM-U,
ASM-F, ASM-R, ASM-D, ASM-S e ASM-M com tolerância de 1e-6. Também verificam geração
em batch para todas essas variantes.

### Atualização: cache incremental

Após esta primeira implementação, foi adicionado um cache de blocos fixos para
ASM-R, ASM-S, ASM-F e demais variantes que usam o mesmo forward blockwise. Modos
sem fronteira fixa continuam na implementação de referência.

O resultado e a validação BF16 estão documentados no relatório 035.

## 3. Diagnósticos de gates

O forward em blocos passou a propagar os gates individuais. Os seguintes valores
agora são calculados sobre gates reais, antes da redução direcional:

- frações acima de 0,25, 0,50, 0,75 e 0,90;
- mínimo e máximo;
- quantis 10%, 25%, 50%, 75% e 90%.

Antes, parte desses campos comparava a soma dos gates com os limiares. Um teste com
hook captura os gates produzidos e compara os diagnósticos com cálculo manual.

CE, PPL e logits históricos não eram afetados por esse erro.

## 4. Cálculos auxiliares desativados

O forward em blocos agora evita cálculos quando seus pesos são zero e os
diagnósticos estão desligados:

- energia/action;
- chamada ao RiskField desativado;
- entropia dimensional;
- regularização métrica;
- condition proxy;
- limiar de gates em 0,50;
- norma de `U`.

Os operadores necessários para produzir a transição continuam sempre ativos. Um
teste confirma igualdade exata dos logits com diagnósticos habilitados e
desabilitados quando as losses auxiliares têm peso zero.

Essa mudança reduz trabalho dispensável, mas ainda deve receber benchmark CUDA
dedicado antes de qualquer afirmação quantitativa de throughput.

## 5. Configuração e checkpoint versionados

Foi introduzido `schema_version=2` na configuração e no payload de checkpoints.

O carregador agora:

- interpreta checkpoints históricos sem versão como schema 1;
- carrega schemas 1 e 2;
- rejeita versões futuras não suportadas;
- continua usando `weights_only=True`;
- mantém as mesmas chaves de pesos.

Um checkpoint real ASM-R da scaling law foi carregado após a mudança:

```text
DRMEmitterModel 1 83206400
```

Também foram adicionadas validações para `top_k`, `emitter_layers`,
`tokenizer_type` e combinações de transição/modo.

## 6. Núcleo neutro ASM

O namespace `aletheion_state_models` agora oferece:

- `ASMConfig`;
- `InferenceState`;
- `StateModel`;
- `StateModelProtocol`;
- `load_state_model()`.

Os builders públicos cobrem ASM-X, ASM-U, ASM-F, ASM-R, ASM-D, ASM-S e ASM-M.

O protocolo define forward, inicialização de inferência, prefill e decode. A
implementação continua única no pacote legado para preservar checkpoints durante a
promoção provisória de ASM-R.

Essa é uma consolidação por adapters, não uma cópia do código. A movimentação física
dos módulos poderá ocorrer posteriormente, com aliases de depreciação, quando a API
estiver estabilizada.

## 7. Cobertura ampliada

Foram adicionados testes para:

- builders ASM com configuração padrão;
- validação cruzada de configuração;
- campos públicos de geração e emitter;
- equivalência entre forward e decode causal;
- geração de todas as variantes públicas;
- diagnósticos blockwise calculados manualmente;
- igualdade de logits sem diagnósticos;
- migração de checkpoint schema 1;
- rejeição de schema futuro;
- protocolo público ASM;
- ausência do aviso de inicialização de projeção métrica com largura zero.

## Arquivos principais alterados

```text
src/drm_language_emitter/config.py
src/drm_language_emitter/checkpoint.py
src/drm_language_emitter/inference.py
src/drm_language_emitter/generation.py
src/drm_language_emitter/model.py
src/drm_language_emitter/directional_blocks.py
src/drm_language_emitter/directional_forward.py
src/drm_language_emitter/metric.py
src/aletheion_state_models/config.py
src/aletheion_state_models/checkpoint.py
src/aletheion_state_models/core/interfaces.py
```

## Verificações executadas

```bash
.venv/bin/python -m compileall -q src scripts
.venv/bin/python -m pytest
```

Resultado:

```text
107 passed, 1 skipped
```

Também foi carregado diretamente o checkpoint:

```text
runs/asm_scaling_law_100m_seed1/
variant_j_no_direction_seed_1/checkpoint_milestone_1000000.pt
```

## Próximos passos técnicos

1. Executar benchmark CUDA antes/depois para quantificar a remoção de auxiliares.
2. Confirmar geração qualitativa após definir tokenizer/checkpoint promovido.
3. Atualizar Architecture e README quando ASM-R deixar de ser uma promoção
   provisória.

## Conclusão

A base agora possui uma única semântica causal para treinamento e geração, builders
seguros, diagnósticos coerentes e checkpoints versionados. A implementação de
inferência privilegia corretude; desempenho incremental ficou isolado como uma fase
posterior mensurável. Nenhuma evidência da scaling law foi reescrita ou invalidada.
