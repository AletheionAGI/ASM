# Auditoria do `src/` — ASM / Aletheion State Models

**Data:** 31 de julho de 2026
**Escopo:** `src/aletheion_state_models/` e `src/drm_language_emitter/`
**Natureza:** auditoria estática, execução da suíte e provas mínimas de comportamento
**Alterações de implementação nesta etapa:** nenhuma

## Resumo executivo

O núcleo de treinamento está funcional e bem protegido contra várias classes de
erro numérico. A suíte completa passou com **98 testes, 1 ignorado e 2 avisos**,
e todos os módulos de `src/` compilam. A divisão do antigo `model.py` também foi
bem-sucedida: nenhum arquivo Python ultrapassa 400 linhas; o maior possui 381.

Entretanto, foram encontrados **dois problemas de alta severidade na API pública**:

1. a geração autoregressiva não reproduz a arquitetura efetivamente treinada nas
   variantes ASM modernas;
2. três construtores públicos ASM falham quando recebem um `DRMConfig` padrão,
   apesar de esse ser um uso natural da API.

Também foram encontrados diagnósticos de gates incorretamente rotulados no caminho
`directional_cumsum`, computação dispensável no caminho crítico e uma migração ASM
ainda incompleta. Esses problemas não invalidam os resultados de CE obtidos pelos
runners de treinamento e rescoring, mas impedem considerar a API de inferência e o
novo namespace ASM prontos para uso externo.

## Metodologia

Foram realizados:

- inventário de todos os módulos e tamanhos;
- leitura dos caminhos de configuração, forward, geometria, memória, emissão,
  geração, checkpoint e dados;
- busca por exceções amplas, operações lineares sensíveis e marcadores de dívida;
- compilação de todo o código Python;
- execução da suíte completa;
- construção e forward das variantes públicas usando a configuração padrão;
- comparação entre o forward de treinamento e o caminho de geração.

Comandos principais:

```bash
.venv/bin/python -m compileall -q src
.venv/bin/python -m pytest
```

Resultado:

```text
98 passed, 1 skipped, 2 warnings
```

## Achados

### A-01 — Geração não representa as variantes ASM treinadas

**Severidade:** alta
**Arquivos:** `src/drm_language_emitter/generation.py`,
`src/drm_language_emitter/directional_blocks.py`

`generate()` mantém um caminho recorrente legado próprio. A função auxiliar
`_advance()` chama diretamente:

```text
direction_field -> metric -> flow -> naturalize -> updater
```

Esse caminho:

- ignora `sequence_mode` e o processamento em blocos;
- ignora `direct_transition`;
- ignora `directional_metric_composition`;
- ignora `local_mixer`;
- ignora `token_state_residual`;
- ignora `selective_memory`;
- ignora `refinement_layers`;
- não reproduz o warm-up de naturalização;
- falha quando `direction_field`, `metric` ou `flow` são `None`.

Consequentemente, ASM-R, ASM-D, ASM-S e ASM-M podem falhar na geração; ASM-X,
ASM-U e ASM-F podem gerar usando uma dinâmica diferente daquela otimizada no
treinamento. O teste existente cobre apenas a configuração DRM recorrente padrão.

**Impacto:** checkpoints podem apresentar CE correto no rescoring, mas produzir
inferência incoerente ou falhar ao gerar texto.

**Correção recomendada:** criar uma interface única de estado incremental, por
exemplo `prefill()` e `decode_step()`, implementada por cada composição. O forward
de treino e a geração devem compartilhar os mesmos operadores. Para mecanismos em
bloco, o estado de inferência deve preservar buffers causais do mixer e a memória
seletiva. Adicionar testes de equivalência entre logits do forward e logits obtidos
por prefill/decode token a token para todas as variantes promovidas.

### A-02 — Construtores públicos ASM falham com a configuração padrão

**Severidade:** alta
**Arquivos:** `src/aletheion_state_models/variants/relational_state.py`,
`direct_state.py`, `selective_state.py`, `src/drm_language_emitter/model.py`

Os construtores `build_relational_state()`, `build_direct_state()` e
`build_selective_state()` removem o campo direcional, mas não alteram o
`sequence_mode="local_step"` padrão. O forward local pressupõe que
`direction_field`, `metric` e `flow` existem e tenta chamar módulos `None`.

Reprodução observada:

```text
build_relational_state TypeError 'NoneType' object is not callable
build_direct_state TypeError 'NoneType' object is not callable
build_selective_state TypeError 'NoneType' object is not callable
```

Os testes atuais não detectam o problema porque sua configuração auxiliar já usa
`sequence_mode="directional_block_cumsum"`.

**Impacto:** a API pública parece aceitar qualquer configuração base válida, mas
somente funciona com uma combinação implícita usada pelos benchmarks.

**Correção recomendada:** os builders devem definir explicitamente um modo
compatível ou rejeitar a configuração com mensagem clara. A melhor solução é
desacoplar o nome do modo de execução das direções: um modo neutro como
`block_state_scan` evitaria que ASM-R e ASM-S dependessem semanticamente de uma
opção chamada `directional_*`. A validação deve verificar todas as combinações de
`sequence_mode`, direção, métrica e transição.

### A-03 — Diagnósticos de gates estão incorretos no forward em blocos

**Severidade:** média
**Arquivo:** `src/drm_language_emitter/directional_forward.py`

No caminho em blocos, `dim_tensor` contém a **soma dos gates por token**, mas é
usado para produzir:

- `hard_active_fraction_025/075/090`;
- `gate_min` e `gate_max`;
- quantis `gate_q10` a `gate_q90`.

Comparar `sum(gates) > 0.75` não equivale a calcular a fração de gates individuais
acima de 0,75. Os nomes sugerem a segunda interpretação. O valor
`hard_active_fraction_050` percorre outro caminho e é calculado corretamente antes
da redução, tornando o conjunto internamente inconsistente.

**Impacto:** dashboards e análises de fechamento dos gates podem chegar a
conclusões erradas. CE, logits e gradientes principais não são afetados, salvo se
algum consumidor externo reutilizar esses diagnósticos como controle.

**Correção recomendada:** propagar estatísticas ou gates individuais de cada bloco,
ou calcular todos os limiares dentro de `_directional_cumsum_block_base()` antes da
redução. Adicionar teste comparando diagnósticos do modelo com cálculo manual.

### A-04 — O pacote ASM ainda é uma fachada sobre o pacote legado

**Severidade:** média, arquitetural
**Arquivos:** todo `src/aletheion_state_models/`

O namespace ASM contém principalmente aliases:

- `StateModel = DRMEmitterModel`;
- `DirectionalBasis = DirectionField`;
- `ExplicitDirectionalTransition = DRMFlow`;
- componentes de memória, mixer, métrica e emissão são importados do pacote DRM;
- builders recebem `DRMConfig` e retornam `DRMEmitterModel`.

Essa decisão foi adequada para preservar checkpoints durante os experimentos, mas
o layout atual não representa ainda a modularização prometida pelos nomes das
pastas. O pacote novo não é independente e continua expondo terminologia DRM em
tipos, configurações e mensagens de erro.

**Impacto:** documentação e imports sugerem uma família neutra enquanto a
implementação continua centralizada no modelo legado. Uma futura remoção ou
renomeação de DRM terá grande raio de mudança.

**Correção recomendada:** manter a compatibilidade durante a scaling law, mas
definir uma migração em duas etapas:

1. criar protocolos/classes ASM neutros e adaptadores compatíveis com checkpoints;
2. mover implementações somente após escolher a arquitetura promovida, mantendo
   aliases legados por uma janela de depreciação.

Não se recomenda duplicar agora o código do modelo: isso criaria duas fontes de
verdade durante um experimento decisivo.

### A-05 — Cálculos auxiliares permanecem no caminho crítico mesmo desativados

**Severidade:** média, desempenho
**Arquivos:** `directional_blocks.py`, `directional_forward.py`, `model.py`

O forward em blocos calcula repetidamente energia, entropia, condição aproximada,
normas, regularização e RiskField mesmo quando `collect_diagnostics=False` e os
pesos auxiliares correspondentes são zero. O RiskField desativado retorna zeros,
mas ainda participa do encadeamento de chamadas; dependendo de
`instantiate_disabled_risk`, também pode manter parâmetros que nunca recebem
gradiente.

Parte dessa instrumentação é útil nos experimentos, porém ela está acoplada ao
caminho de produção. Isso é especialmente relevante porque ASM-S já demonstrou
throughput aproximadamente duas vezes maior que as variantes geométricas.

**Impacto:** custo de treino e inferência maior do que o necessário, dificultando
a separação entre custo essencial da arquitetura e custo de observabilidade.

**Correção recomendada:** construir flags `need_*` por loss/diagnóstico, como já é
feito parcialmente no forward recorrente, e evitar operações e armazenamento não
necessários. Medir o ganho com benchmark pareado antes e depois.

### A-06 — Validação de configuração possui lacunas e permite estados mutáveis inválidos

**Severidade:** baixa a média
**Arquivo:** `src/drm_language_emitter/config.py`

A validação é extensa e rejeita campos desconhecidos, o que é positivo. Porém:

- não valida integralmente a compatibilidade de `sequence_mode` com módulos
  opcionais, origem do A-02;
- campos como `top_k`, `emitter_layers` e `tokenizer_type` não recebem validação
  equivalente no construtor;
- a dataclass permanece mutável após `__post_init__`, permitindo que código altere
  valores para combinações inválidas sem chamar `validated_copy()`;
- nomes predominantemente `directional_*` configuram também variantes sem direção.

**Correção recomendada:** acrescentar invariantes cruzados, validar todos os campos
públicos e favorecer configuração imutável ou builders que sempre retornem cópias
validadas. A renomeação de campos deve ocorrer com aliases e avisos de depreciação,
não durante a execução da scaling law.

### A-07 — Cobertura não testa as fronteiras públicas mais arriscadas

**Severidade:** média
**Arquivos:** `tests/test_asm_package.py`, `tests/test_generation.py`

A cobertura unitária do núcleo é boa, mas os testes ASM usam uma única configuração
pré-adaptada. A geração cobre somente o DRM padrão. Faltam matrizes de testes para:

- cada builder com configuração padrão e com configuração de benchmark;
- geração de todas as variantes;
- equivalência treino/inferência causal;
- diagnósticos manuais de gates;
- serialização e recarga pelo namespace ASM;
- `metric_rank=0` sem o aviso de inicialização de tensor vazio.

## Pontos positivos

- Todos os arquivos permanecem abaixo do limite desejado de 400–500 linhas.
- Checkpoints usam `weights_only=True` na carga e validam estrutura/configuração.
- O dataset memmap valida tamanho, SHA-256 e impede escape do diretório do manifesto.
- A métrica SPD usa Woodbury e promove solves BF16/FP16 para FP32.
- A composição ortonormal possui `cholesky_ex` com fallback espectral, corrigindo o
  erro numérico observado anteriormente.
- A memória seletiva evita a identidade instável baseada em divisão por `cumprod`.
- A inicialização por componente é determinística e independente de módulos
  opcionais.
- Não foram encontrados `TODO`, `FIXME`, `HACK` ou `NotImplementedError` no `src/`.
- A compilação e os 98 testes confirmam uma base estável para continuar os
  experimentos em andamento.

## Priorização proposta

### Antes de publicar ou demonstrar geração

1. Corrigir A-01 com API incremental compartilhada entre treino e inferência.
2. Criar testes de equivalência para ASM-R, ASM-S e para a variante vencedora da
   scaling law.
3. Corrigir A-02 e testar builders com `DRMConfig()` padrão.

### Antes de interpretar novos diagnósticos geométricos

4. Corrigir A-03 e marcar dashboards anteriores de gates como potencialmente
   afetados.
5. Separar observabilidade opcional do caminho crítico conforme A-05.

### Depois da decisão da scaling law

6. Promover uma arquitetura e consolidar nomes ASM neutros.
7. Executar a migração do A-04 sem duplicar fontes de verdade.
8. Endurecer a configuração e instituir aliases de depreciação conforme A-06.

## Relação com o treinamento de 100M em andamento

Os achados A-01 e A-02 estão concentrados na API pública de construção/inferência.
O runner de scaling law usa configurações resolvidas compatíveis e o forward em
blocos já exercitado pelos testes e pelas rodadas anteriores. Portanto, não há
evidência nesta auditoria de que seja necessário interromper ou reiniciar o treino
de 100M.

O A-03 exige cautela apenas ao interpretar quantis e limiares de gates. A seleção
por **CE congelado**, PPL, tokens e tempo permanece válida.

## Conclusão

O `src/` está em bom estado como plataforma experimental, mas ainda não como API
ASM pronta para inferência geral. A prioridade não deve ser uma grande reescrita
durante a scaling law. O caminho mais seguro é concluir o experimento, corrigir a
paridade treino–geração e os builders públicos, e só então consolidar a variante
vencedora no núcleo neutro `aletheion_state_models`.
