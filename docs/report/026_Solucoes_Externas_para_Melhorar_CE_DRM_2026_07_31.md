# Soluções externas para melhorar o CE do DRM

Data: 2026-07-31  
Branch: `drm-fix`  
Escopo: alternativas posteriores à triagem A–I

## 1. Resultado que motiva o próximo ciclo

O rescoring contínuo e determinístico de toda a validação confirmou:

| Variante | CE | Perplexidade | Tokens |
|---|---:|---:|---:|
| F | **1,875146** | **6,5218** | 4.834.787 |
| I | 1,884072 | 6,5802 | 4.834.787 |
| H | 1,885751 | 6,5913 | 4.834.787 |

F confirmou que ampliar o campo receptivo causal ajuda, mas o ganho ainda é
pequeno diante da diferença observada para o GPT-2 corrigido. Além disso,
F–I ainda contém um confundidor de inicialização: remover fisicamente o
`RiskField` muda o ponto da sequência global de RNG em que emitter e mixer são
inicializados.

Decisão:

> F é a vencedora observada da seed 1, mas ainda não é uma vencedora
> estatisticamente confirmada. O mixer dilatado deve ser mantido como base do
> próximo ciclo, enquanto memória seletiva e associative recall recebem
> prioridade.

## 2. Evidência externa consultada

### 2.1 Mamba e estado seletivo

O Mamba identifica como fraqueza de SSMs anteriores a dificuldade de raciocínio
dependente de conteúdo. Sua mudança central é tornar parâmetros de retenção e
propagação funções do input.

Fonte primária:

- Gu e Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces*: <https://arxiv.org/abs/2312.00752>

Aplicação ao DRM:

```text
forget_t = sigmoid(F(z, token))
write_t  = sigmoid(W(z, token))
candidate_t = DRMFlow(z, token)
z_t = forget_t * z + write_t * candidate_t
```

Os gates atuais escolhem direções, mas não fornecem mecanismos explícitos e
independentes de preservar, apagar e escrever memória.

### 2.2 Gated DeltaNet e edição associativa

Gated DeltaNet combina esquecimento adaptativo com delta rule, permitindo
editar uma associação sem sobrescrever indiscriminadamente a memória
comprimida.

Fonte primária:

- Yang, Kautz e Hatamizadeh, *Gated Delta Networks: Improving Mamba2 with
  Delta Rule*: <https://arxiv.org/abs/2412.06464>

Aplicação ao DRM:

```text
read_t = M @ key_t
error_t = value_t - read_t
M_t = decay_t * M + write_t * outer(error_t, key_t)
```

Direções relacionais podem produzir `key`, `value`, `decay` e `write`.

### 2.3 Associative recall como causa da diferença

O estudo Zoology mede a diferença entre attention e mixers eficientes e
atribui grande parte dela a associative recall. Também propõe tarefas
sintéticas MQAR como instrumento de desenvolvimento arquitetural.

Fonte primária:

- Arora et al., *Zoology: Measuring and Improving Recall in Efficient
  Language Models*: <https://arxiv.org/abs/2312.04927>

Aplicação imediata:

- MQAR;
- key-value recall;
- induction/copy;
- recuperação entre blocos;
- permutação de ordem;
- distância variável entre chave e valor.

Esses testes são mais baratos e diagnósticos que novos treinos cegos de 30M.

### 2.4 Hyena e convolução longa controlada por dados

Hyena intercala filtros longos implícitos e gates dependentes dos dados. O
trabalho reporta qualidade competitiva sem attention em language modeling.

Fonte primária:

- Poli et al., *Hyena Hierarchy: Towards Larger Convolutional Language
  Models*: <https://arxiv.org/abs/2302.10866>

Aplicação ao DRM:

- substituir a pilha dilatada fixa por filtro longo parametrizado;
- cobrir todo o contexto de 512 tokens;
- manter modulação dependente do token;
- preservar residual e CE-only.

Ressalva: Zoology mostra que convolução longa isolada ainda pode falhar em
associative recall. Hyena deve ser comparada com uma solução seletiva/delta.

### 2.5 Byte-level seletivo e patches

O DRM opera em bytes, o que alonga dependências linguísticas. MambaByte mostra
que estado seletivo é viável nesse regime. BLT reduz o custo agrupando bytes em
patches dinâmicos definidos por entropia.

Fontes primárias:

- Wang et al., *MambaByte: Token-free Selective State Space Model*:
  <https://arxiv.org/abs/2401.13660>
- Pagnoni et al., *Byte Latent Transformer: Patches Scale Better Than Tokens*:
  <https://arxiv.org/abs/2412.09871>

Aplicação ao DRM:

```text
bytes -> encoder local -> patches -> dinâmica DRM -> decoder local de bytes
```

Essa é uma mudança de médio prazo. Primeiro deve-se verificar memória seletiva
na arquitetura byte-level atual.

### 2.6 Mega e memória local position-aware

Mega combina moving average exponencial, gating e interação local
position-aware.

Fonte primária:

- Ma et al., *Mega: Moving Average Equipped Gated Attention*:
  <https://arxiv.org/abs/2209.10655>

Aplicação ao DRM:

- adicionar múltiplas escalas de EMA ao estado;
- separar memória rápida e lenta;
- alimentar o campo direcional com ambas;
- evitar recalcular toda a geometria em block16.

### 2.7 Scaling e orçamento de tokens

150M tokens para 127M parâmetros correspondem a aproximadamente 1,18 token por
parâmetro. Isso é pouco para convergência absoluta, embora não explique a
diferença relativa quando o GPT-2 recebe o mesmo orçamento.

Fontes primárias:

- Hoffmann et al., *Training Compute-Optimal Large Language Models*:
  <https://arxiv.org/abs/2203.15556>
- Kaplan et al., *Scaling Laws for Neural Language Models*:
  <https://arxiv.org/abs/2001.08361>

Aplicação:

- medir curvas IsoFLOP;
- fazer sweep conjunto de learning rate e batch;
- usar warmup e cosine decay;
- separar eficiência amostral de convergência final.

## 3. Alternativas propostas

### J — DRM seletivo

Base: I/F parameter-matched.

Mudanças:

- forget gate por token;
- write gate por token;
- candidato produzido pela dinâmica DRM;
- mixer dilatado preservado;
- residual e normalização.

Objetivo: tornar memória e geometria dependentes do conteúdo.

### K — memória relacional delta

Adicionar uma memória matricial de baixa ordem com delta rule. O estado vetorial
DRM continua existindo; a memória guarda associações recuperáveis.

Objetivo: reduzir a diferença de associative recall.

### L — DRM híbrido com sliding attention

Adicionar uma janela pequena, por exemplo 64 ou 128 tokens, a cada alguns
blocos DRM.

Objetivo: medir o teto de CE obtido quando recuperação explícita está
disponível. Esta variante não será attention-free, mas serve como controle
arquitetural.

### M — mixer Hyena-like

Substituir convoluções dilatadas por filtro longo implícito, cobrindo 512
tokens.

Objetivo: testar se F está limitado pelo alcance e pela parametrização fixa do
filtro.

### N — DRM byte-patch

Codificar bytes localmente, executar DRM sobre patches e decodificar bytes.

Objetivo: reduzir o comprimento efetivo das dependências e concentrar compute
onde a entropia é maior.

## 4. Stack DRM recomendada

O segundo estágio simples de G não equivale a um backbone profundo moderno. Uma
stack adequada deve usar:

```text
x = x + selective_DRM(norm(x))
x = x + causal_mixer(norm(x))
x = x + SwiGLU(norm(x))
```

Recomendação:

- 6 a 12 blocos estreitos;
- pre-norm;
- residual em todos os subblocos;
- SwiGLU;
- estado seletivo;
- mixer dilatado ou Hyena;
- orçamento total próximo de 127M.

## 5. Otimização independente da arquitetura

Executar após identificar um bloco promissor:

- learning rate `1e-4`, `2e-4`, `3e-4`;
- warmup de 1–2%;
- cosine decay;
- AdamW `beta2` entre 0,95 e 0,99;
- batch efetivo maior;
- clipping sweep;
- weight tying;
- emitter residual/SwiGLU;
- checkpoint averaging.

Esses testes não substituem memória seletiva. Eles refinam uma arquitetura que
já demonstrou capacidade.

## 6. Ordem experimental

1. Corrigir a dependência da inicialização em módulos opcionais.
2. Repetir F/I em seeds pareadas e validação contínua.
3. Implementar MQAR e testes de recuperação.
4. Implementar J, mantendo o mixer de F.
5. Comparar J com K em 5M tokens e tarefas sintéticas.
6. Usar L como controle do quanto attention local fecha a diferença.
7. Testar M apenas se associative recall não for o gargalo dominante.
8. Promover a vencedora para 30M.
9. Fazer sweep de otimização.
10. Só então executar 150M e múltiplas seeds.

PG-19 deve permanecer congelado durante todo esse ciclo.

## 7. Implementação realizada

A infraestrutura do primeiro ciclo proposto foi materializada:

- inicialização por componente, independente de módulos opcionais;
- orquestração e rescoring contínuo pareados para F/I em três seeds;
- benchmark MQAR com CE e acurácia nas consultas;
- J com memória seletiva forget/write;
- J_DILATED combinando essa memória com o mixer causal dilatado;
- gate automático que bloqueia 30M sem ganho médio e maioria das seeds.

Os scripts principais são `run_drm_fix_paired_5m.sh`,
`run_mqar_architecture_probe.py`, `rescore_drm_fix_validation.py` e
`check_drm_fix_promotion.py`.

Isso não representa ainda um resultado experimental: os treinos pareados e o
MQAR completo precisam ser executados antes de qualquer promoção.

## 8. Conclusão

O resultado de F é consistente com a literatura: ampliar contexto causal ajuda,
mas convolução maior não resolve necessariamente recuperação associativa.

A prioridade técnica passa a ser:

```text
memória seletiva por token
+ escrita/remoção controlada
+ mixer causal de longo alcance
+ stack residual estreita
```

As fases formais do roadmap continuam relevantes para fidelidade matemática,
mas as evidências externas indicam que seleção de memória e associative recall
são os caminhos mais diretos para melhorar CE.

## 9. Estado implementado e comandos para execução

### Principais entregas

- Inicialização determinística por componente, independente de módulos
  opcionais.
- Protocolo F/I com seeds pareadas 1, 2 e 3 e validação contínua idêntica.
- Benchmark MQAR/associative recall com CE e acurácia.
- Variante J com memória seletiva forget/write.
- Variante J_DILATED combinando memória seletiva e mixer dilatado.
- Gate automático para impedir promoção prematura para 30M.

### Contagem de parâmetros

- J: 126.080.896.
- J_DILATED: 127.191.968.
- GPT-2 de referência: 126.080.640.

A suíte passou com 85 testes e 1 teste ignorado. Os forward smokes de J e
J_DILATED também passaram.

### Execução recomendada

Para executar apenas F/I com três seeds pareadas:

```bash
./scripts/run_drm_fix_paired_5m.sh
```

Para executar J e J_DILATED separadamente:

```bash
VARIANTS=J,J_DILATED \
OUTPUT_ROOT=runs/drm_fix_selective_paired_5m \
./scripts/run_drm_fix_paired_5m.sh
```

Para executar MQAR:

```bash
.venv/bin/python scripts/run_mqar_architecture_probe.py \
  --variants F,I,J,J_DILATED \
  --steps 1000 \
  --device cuda \
  --output runs/drm_fix_mqar/results.json
```

Para que o gate compare baseline e candidata, ambos precisam constar no mesmo
`paired_validation_summary.json`. A execução completa recomendada é:

```bash
VARIANTS=F,I,J,J_DILATED \
OUTPUT_ROOT=runs/drm_fix_all_paired_5m \
./scripts/run_drm_fix_paired_5m.sh
```

Após o término:

```bash
.venv/bin/python scripts/check_drm_fix_promotion.py \
  --summary runs/drm_fix_all_paired_5m/paired_validation_summary.json \
  --candidate J_DILATED \
  --baseline F
```

A promoção exige redução média de pelo menos 0,005 CE, vitória em pelo menos
duas das três seeds pareadas e desvio-padrão máximo de 0,03. Se os critérios
não forem atendidos, o script retorna código 2 e a variante permanece em 5M.

Nenhum commit ou push foi feito nesta etapa.

## 10. Controle sem geometria DRM

Foi implementada a variante `SSM_CONTROL` para testar diretamente se o ganho
de J vem da geometria DRM ou da memória seletiva. Seu caminho é:

```text
token embedding
→ mixer causal curto
→ residual token→estado
→ memória seletiva forget/write
→ emitter
```

O controle não instancia `DirectionField`, `RelationalMetric`, `DRMFlow` nem
`RiskField`. A largura interna da memória seletiva foi ajustada para manter o
orçamento:

| Variante | Parâmetros |
|---|---:|
| J | 126.080.896 |
| SSM_CONTROL | 126.076.000 |
| Diferença | 4.896 (0,0039%) |

O controle possui testes que verificam ausência dos módulos geométricos,
causalidade, gradientes finitos e equivalência do orçamento. A execução pareada
é:

```bash
VARIANTS=J,SSM_CONTROL \
OUTPUT_ROOT=runs/drm_vs_ssm_control_paired_5m \
./scripts/run_drm_fix_paired_5m.sh
```

Como os parâmetros removidos da geometria foram transferidos para a memória
seletiva, esse é um controle de orçamento, não de compute. Throughput, uso de
VRAM e CE devem ser reportados conjuntamente.
