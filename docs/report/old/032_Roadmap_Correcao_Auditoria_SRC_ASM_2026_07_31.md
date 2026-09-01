# Roadmap de correção da auditoria do `src/` — ASM

**Data:** 31 de julho de 2026
**Origem:** `031_Auditoria_SRC_ASM_2026_07_31.md`
**Estado:** implementado; validação registrada no relatório 034
**Escopo:** corrigir A-01 a A-07 sem invalidar checkpoints ou resultados existentes

## Objetivo

Transformar o código atual, que está em bom estado como plataforma experimental,
em uma base ASM consistente para treinamento, avaliação e inferência.

O requisito central é:

> O modelo usado para gerar deve executar a mesma arquitetura causal que foi usada
> para treinar e avaliar o checkpoint.

As correções preservarão compatibilidade com checkpoints existentes e serão
separadas da scaling law atualmente em execução.

## Atualização de execução

O resultado da scaling law foi congelado e publicado no relatório 033 antes das
alterações. As fases deste roadmap foram implementadas em 31 de julho de 2026.

A inferência incremental foi entregue inicialmente em modo de referência: preserva
o prefixo e recompõe o forward exato. Essa escolha resolve a divergência funcional
entre treino e geração. Caches incrementais de mixer, memória e blocos permanecem
como otimização futura, pois exigem um benchmark próprio e não são necessários para
corretude.

Resultados da validação:

```text
107 passed, 1 skipped
checkpoint real schema 1: carregado com 83.206.400 parâmetros
```

Detalhes: `034_Implementacao_Correcoes_Auditoria_SRC_ASM_2026_07_31.md`.

## Regra de isolamento experimental

Nenhum código de modelo deve ser alterado antes de:

1. o treino atual terminar;
2. todas as configurações resolvidas serem preservadas;
3. os checkpoints e respectivos hashes serem registrados;
4. o rescoring congelado ser concluído;
5. o relatório da scaling law ser produzido;
6. o commit exato usado no experimento ser registrado.

Isso separa três conjuntos de evidência:

- resultados da arquitetura originalmente treinada;
- correções de API e observabilidade;
- eventuais mudanças matemáticas posteriores.

## Fase 0 — Congelar a scaling law

### Entregas

- inventário dos checkpoints de 1M, 2M, 5M, 10M, 20M, 30M, 50M e 100M;
- hashes SHA-256 dos checkpoints avaliados;
- rescoring sobre a mesma sequência contínua de validação;
- curvas CE por tokens e CE por tempo;
- registro do commit, configuração, versões de Python, PyTorch e CUDA;
- relatório final da scaling law e do ponto de crossover.

### Critério de aceite

Todos os resultados necessários para reproduzir a comparação devem permanecer
disponíveis mesmo que o código seja posteriormente refatorado.

## Fase 1 — Corrigir os construtores ASM

### Problema atendido

A-02 e parte de A-06/A-07.

Os builders `build_relational_state()`, `build_direct_state()` e
`build_selective_state()` falham quando recebem `DRMConfig()` padrão, porque removem
módulos exigidos pelo modo `local_step` sem selecionar outro modo de execução.

### Implementação

1. Definir explicitamente um modo compatível em cada builder.
2. Centralizar invariantes cruzados entre:
   - modo de sequência;
   - campo direcional;
   - transição direta;
   - métrica;
   - naturalização;
   - memória seletiva.
3. Substituir chamadas a módulos `None` por erros de configuração antecipados e
   descritivos.
4. Introduzir, se necessário, um nome neutro como `block_state_scan` para que
   variantes sem direção não dependam semanticamente de `directional_*`.
5. Manter aliases para nomes legados usados nos checkpoints.

### Testes

- cada builder com `DRMConfig()`;
- cada builder com configuração mínima;
- cada builder com sua configuração real de benchmark;
- forward causal finito;
- serialização, recarga e igualdade de logits;
- teste parametrizado de combinações inválidas.

### Critério de aceite

O seguinte padrão deve funcionar para todas as variantes públicas:

```python
model = builder(DRMConfig())
output = model(tokens)
```

Combinações impossíveis devem falhar na construção da configuração, e não durante
o forward.

## Fase 2 — Unificar treinamento e inferência

### Problema atendido

A-01, o achado de maior severidade.

`generation.py` executa uma recorrência DRM antiga e não reproduz mixers, memória,
transições diretas, composição métrica ou modos em blocos usados pelas variantes
ASM atuais.

### Interface proposta

```python
state = model.init_inference_state(batch_size, device)
logits, state = model.prefill(input_ids, state)
logits, state = model.decode_step(next_token, state)
```

`InferenceState` deverá representar, quando aplicável:

- estado latente;
- estado da memória seletiva;
- buffers causais do mixer;
- posição absoluta;
- posição e estado do bloco;
- força de naturalização correspondente ao checkpoint;
- caches opcionais da geometria.

### Estratégia

1. Extrair um pipeline causal compartilhado por forward e inferência.
2. Implementar inicialmente um caminho de referência correto, mesmo que refaça o
   prefixo completo.
3. Validar equivalência contra o forward de treino.
4. Somente depois implementar caches incrementais de alto desempenho.
5. Remover a duplicação arquitetural de `_advance()`.

### Operadores compartilhados

```text
TokenEmbedding
→ StateTransition
→ GeometryOperator opcional
→ CausalMixer opcional
→ SelectiveMemory opcional
→ TokenEmitter
```

### Testes

- geração para ASM-X, ASM-U, ASM-F, ASM-R, ASM-D, ASM-S e ASM-M;
- prompt com um token;
- batch maior que um;
- cruzamento de fronteiras de bloco;
- geração determinística com amostragem desativada;
- salvamento e restauração do estado incremental;
- ausência de acesso a tokens futuros.

### Critério de aceite

Os logits do forward completo e da execução token a token devem obedecer:

```text
max_abs_error < 1e-5 em FP32
```

BF16 terá tolerância explícita adequada ao dtype. Toda variante promovida deve
gerar sem acessar módulos ausentes.

## Fase 3 — Corrigir diagnósticos de gates

### Problema atendido

A-03 e parte de A-07.

No forward em blocos, a soma dos gates está sendo interpretada como se contivesse
gates individuais nos limiares e quantis.

### Implementação

1. Calcular estatísticas antes de reduzir o eixo de direções.
2. Propagar por bloco:
   - `soft_active_fraction`;
   - `hard_active_fraction_025`;
   - `hard_active_fraction_050`;
   - `hard_active_fraction_075`;
   - `hard_active_fraction_090`;
   - mínimo, máximo e quantis dos gates.
3. Remover a reutilização de `dim_tensor` para métricas de gates.
4. Comparar caminhos recorrente e em blocos com tensores controlados.
5. Registrar na documentação que dashboards anteriores de gates podem estar
   afetados; CE e PPL não são afetados.

### Critério de aceite

Todas as estatísticas devem coincidir com um cálculo manual sobre gates conhecidos.

## Fase 4 — Remover cálculos auxiliares desnecessários

### Problema atendido

A-05.

### Implementação

Criar decisões explícitas antes do forward:

```text
need_action
need_gate_stats
need_metric_stats
need_condition
need_risk
need_recurrence
need_stability
need_consistency
```

Cada cálculo será executado apenas quando:

- sua loss possuir peso diferente de zero; ou
- `collect_diagnostics=True`; ou
- for estruturalmente necessário para a transição.

Deverão ser evitados quando dispensáveis:

- energia métrica;
- condition proxy;
- quantis de gates;
- chamadas ao RiskField desativado;
- armazenamento de trajetórias intermediárias;
- regularizações com peso zero.

### Validação

Comparar antes e depois:

- igualdade dos logits;
- CE congelado;
- tokens por segundo;
- memória máxima de GPU;
- tempo de forward e backward.

### Critério de aceite

Com losses auxiliares desativadas, a otimização não pode alterar os logits dentro da
tolerância numérica. O ganho de desempenho deve ser medido, não presumido.

## Fase 5 — Endurecer e versionar a configuração

### Problema atendido

A-06.

### Implementação

1. Introduzir `ASMConfig` sem quebrar `DRMConfig`.
2. Validar todos os campos públicos, incluindo:
   - `top_k`;
   - `emitter_layers`;
   - `tokenizer_type`;
   - compatibilidade completa entre módulos e modo de execução.
3. Impedir ou detectar mutações inválidas após a construção.
4. Criar nomes neutros para propriedades hoje chamadas `directional_*` quando
   também configuram variantes não direcionais.
5. Versionar o schema de checkpoint:

```text
schema_version 1: DRMConfig legado
schema_version 2: ASMConfig
```

6. Criar migração determinística de configurações antigas.
7. Preservar aliases com avisos de depreciação durante uma janela documentada.

### Critério de aceite

- checkpoints antigos carregam sem alteração dos pesos;
- checkpoints novos registram a versão do schema;
- campos desconhecidos ou combinações inválidas falham com mensagens claras;
- recarregar um checkpoint não altera seus logits.

## Fase 6 — Consolidar o núcleo ASM

### Problema atendido

A-04.

Esta fase só começa depois da decisão da scaling law, para que a organização final
seja orientada pela arquitetura promovida.

### Interfaces neutras

```text
StateModel
StateTransition
StateMemory
CausalMixer
GeometryOperator
DirectionalBasis
TokenEmitter
InferenceState
```

### Estratégia de migração

1. Criar protocolos e classes neutras no namespace ASM.
2. Mover uma única implementação por vez; não copiar arquivos.
3. Manter o pacote `drm_language_emitter` como fachada compatível.
4. Preservar chaves de `state_dict` ou fornecer migração explícita.
5. Emitir avisos de depreciação somente em APIs públicas substituídas.
6. Atualizar scripts e documentação depois da compatibilidade estar coberta por
   testes.

Exemplo final:

```python
from aletheion_state_models import StateModel

# Compatibilidade temporária:
DRMEmitterModel = StateModel
```

### Critério de aceite

O pacote ASM deve deixar de ser apenas uma coleção de aliases, sem criar duas
fontes de verdade e sem invalidar checkpoints históricos.

## Fase 7 — Ampliar a cobertura nas fronteiras públicas

### Problema atendido

A-07.

Adicionar uma matriz de testes para:

- builders com configuração padrão e real;
- geração de todas as variantes;
- equivalência forward/inferência incremental;
- causalidade e ausência de vazamento futuro;
- diagnósticos de gates;
- serialização pelo namespace ASM;
- migração de schemas;
- `metric_rank=0` sem inicialização inútil de camada vazia;
- FP32 e BF16 onde houver suporte;
- estado incremental atravessando blocos.

## Organização dos commits

As mudanças serão separadas em lotes reversíveis:

1. `fix ASM variant builders and configuration invariants`
2. `unify training and autoregressive inference paths`
3. `fix blockwise gate diagnostics`
4. `skip disabled auxiliary computations`
5. `introduce versioned ASM configuration compatibility`
6. `consolidate ASM core with legacy adapters`
7. `expand public API and inference regression coverage`

Commit e push serão realizados somente quando solicitados.

## Validação final

A correção completa exige:

- suíte anterior integralmente verde;
- novos testes de builders e variantes;
- equivalência causal entre forward e decode;
- geração funcional de todas as variantes;
- recarga de checkpoints históricos;
- migração de configuração versionada;
- teste de causalidade;
- teste FP32 e BF16;
- benchmark de throughput;
- rescoring de checkpoint existente sem regressão de CE.

## Ordem de execução recomendada

```text
Congelar scaling law
        ↓
Builders e invariantes
        ↓
Paridade treino–inferência
        ↓
Diagnósticos corretos
        ↓
Otimização do caminho crítico
        ↓
Configuração versionada
        ↓
Consolidação ASM
        ↓
Validação final
```

## Riscos e contenção

### Alteração involuntária dos resultados

Contenção: rescoring e hashes anteriores às mudanças; testes de igualdade de
logits; commits pequenos.

### Quebra de checkpoints

Contenção: adapters, schema versionado e testes com checkpoints reais.

### Otimização incorreta da inferência incremental

Contenção: primeiro implementar referência correta; caches somente após testes de
equivalência.

### Migração prematura para ASM

Contenção: consolidar o núcleo apenas depois que a scaling law definir quais
componentes sobreviveram.

## Conclusão

A execução começará após o congelamento do experimento atual. A prioridade será
corrigir os builders e, principalmente, garantir paridade entre treinamento e
geração. Depois serão corrigidos os diagnósticos e o desperdício computacional. A
migração estrutural para ASM será a última etapa, orientada pelos resultados da
scaling law e protegida por compatibilidade explícita com o histórico DRM.
