# Aletheion State Models: família arquitetural e critérios de nomenclatura

## 1. Objetivo deste documento

O DRM Language Emitter nasceu como uma tradução computacional parcial da
teoria Directional Relational Manifolds. As ablações recentes produziram
arquiteturas que preservam partes diferentes dessa proposta.

Este documento separa três níveis de identidade:

```text
Directional Relational Manifolds
        teoria
          ↓
Aletheion State Models
 programa de pesquisa
          ↓
ASM-X, ASM-R, ASM-S, ASM-F...
 variantes arquiteturais
```

A arquitetura principal será escolhida por evidência reproduzível, não pelo
nome histórico do repositório.

## 2. Os três níveis de identidade

### 2.1 Teoria: Directional Relational Manifolds

DRM permanece como a teoria matemática e filosófica que investiga:

- estados e relações locais;
- direções efetivamente acessíveis;
- métricas relacionais;
- transporte e dependência do caminho;
- transições de posto e dimensão efetiva;
- transformação do conjunto de futuros possíveis.

O software atual não realiza toda a teoria formal. Em particular, sua métrica
SPD possui posto exato fixo. Atividade de gates e posto numérico não devem ser
apresentados como o posto formal variável da teoria.

### 2.2 Programa de pesquisa: Aletheion State Models

**Aletheion State Models**, abreviado como **ASM**, é o guarda-chuva para
modelos causais que comprimem o histórico em um estado persistente e aprendem
como esse estado deve evoluir.

ASM pode conter variantes geométricas, direcionais, seletivas ou puramente
neurais. Dessa forma, uma hipótese DRM pode ser removida sem invalidar o
programa de pesquisa inteiro.

### 2.3 Arquitetura promovida

A arquitetura promovida é a variante que demonstra a melhor combinação de:

- entropia cruzada;
- scaling com dados;
- qualidade por hora de GPU;
- throughput;
- estabilidade entre seeds;
- memória e associative recall;
- contexto longo;
- controlabilidade e observabilidade.

Ela poderá receber um nome próprio depois da scaling law e da confirmação com
seeds adicionais.

## 3. Taxonomia proposta

| Código | Nome | Núcleo arquitetural |
|---|---|---|
| ASM-X | Explicit DRM State Model | direção explícita + métrica + movimento + memória |
| ASM-U | Metric Subspace State Model | movimento naturalizado dentro do subespaço |
| ASM-F | Relational Frame State Model | frame direcional normalizado pela métrica |
| ASM-R | Relational State Model | transição direta condicionada pela métrica |
| ASM-D | Direct State Model | transição neural direta sem geometria |
| ASM-S | Selective State Model | capacidade concentrada em memória seletiva |
| ASM-M | Causal Memory State Model | mixer, residual e memória seletiva estreita |

As letras são identificadores de mecanismos, não uma classificação de
qualidade.

## 4. ASM-X — Explicit DRM State Model

`ASM-X` representa a arquitetura DRM explícita. A letra **X** identifica a
fatoração explícita da dinâmica em direções, gates, coeficientes e geometria.

```text
token
  → estado causal
  → campo direcional
  → gates e coeficientes
  → movimento direcional
  → naturalização métrica
  → memória seletiva e mixer
  → emitter
```

Em notação compacta:

$$
v_{\mathrm{raw}}=Vc,
\qquad
v=G^{-1}Vc.
$$

`ASM-X` é o sucessor taxonômico de J e a variante mais próxima da identidade
original DRM Language Emitter.

O nome público **DRM Language Emitter** só deve permanecer como arquitetura
principal se direção e métrica demonstrarem contribuição positiva em escala ou
em capacidades relevantes que justifiquem seu custo.

## 5. ASM-U — Metric Subspace State Model

`ASM-U` preserva o campo direcional, mas faz a métrica atuar dentro do
subespaço de possibilidades.

Assumindo direções como colunas de $V$:

$$
\hat c=(V^\top GV+\lambda I)^{-1}c,
$$

$$
v=V\hat c.
$$

Por construção:

$$
v\in\mathrm{span}(V).
$$

Nome público sugerido caso seja promovida:

> DRM Subspace Emitter

Essa variante sustenta uma segunda geração legítima do DRM: a geometria
organiza a ação sem retirar o movimento do espaço direcional declarado.

## 6. ASM-F — Relational Frame State Model

`ASM-F` transforma as direções numa frame local condicionada pela métrica.

Com direções como colunas de $V$:

$$
V^\top GV+\lambda I=LL^\top,
$$

$$
Q=VL^{-\top}.
$$

Para damping pequeno:

$$
Q^\top GQ\approx I.
$$

O código armazena direções por linhas e executa a operação transposta
equivalente. Essa convenção deve ser informada para evitar a fórmula
dimensionalmente incorreta $L^{-1}V$ quando $V$ é definido por colunas.

Nome público sugerido:

> Relational Frame Emitter

Descrição:

> Um modelo causal no qual as possibilidades de transição são expressas numa
> frame local normalizada pela geometria aprendida.

## 7. ASM-R — Relational State Model

`ASM-R` corresponde a `J_NO_DIRECTION`. Ele remove o catálogo explícito de
direções e produz movimento por uma transição contextual:

$$
v_{\mathrm{raw}}=T(z,x),
$$

$$
v=G^{-1}T(z,x).
$$

O movimento continua direcional no sentido de ser um vetor, mas suas direções
são implícitas na função $T$. A métrica relacional permanece responsável por
condicionar a atualização.

Nome público sugerido:

> Relational State Emitter

Subtítulo técnico:

> A metric-conditioned causal state model derived from DRM research.

Pelos resultados atualmente disponíveis, `ASM-R` é o principal candidato: ele
perde para o controle seletivo em 5M, mas o ultrapassa em 30M.

## 8. ASM-D — Direct State Model

`ASM-D` corresponde ao controle direto estrutural sem geometria:

```text
token
  → estado
  → transição contextual direta
  → memória seletiva
  → mixer causal
  → emitter
```

Nome público sugerido:

> Causal State Emitter

Essa promoção seria apropriada se direção, métrica e naturalização não
justificarem seu custo, inclusive quando comparadas sem redistribuição de
parâmetros.

## 9. ASM-S — Selective State Model

`ASM-S` corresponde a `J_DIRECT_CONTROL_MATCHED`. A geometria é removida e seu
orçamento é redistribuído para a memória seletiva.

Seu núcleo é:

$$
m_t=f_t\odot m_{t-1}+w_t\odot c_t.
$$

O modelo preserva, esquece, escreve, transforma e emite. Por isso, **State** é
mais preciso que apenas **Memory**.

Nome público sugerido:

> Selective State Emitter

Esse nome será adequado se a scaling law mostrar que alocar capacidade à
memória oferece melhor resultado que modelar geometria.

## 10. ASM-M — Causal Memory State Model

`ASM-M` corresponde ao controle estreito `SSM_CONTROL`: mixer causal, residual,
memória seletiva e emitter, sem geometria nem transição contextual rica.

Nome público sugerido:

> Causal Memory Emitter

Se essa variante vencer, será necessário comparar cuidadosamente sua novidade
com SSMs seletivos, gated RNNs e outras arquiteturas modernas de memória.

## 11. Decisão pela scaling law

A avaliação contínua usa checkpoints em:

```text
1M, 2M, 5M, 10M, 20M, 30M, 50M e 100M tokens
```

Uma curva exploratória por arquitetura pode ser representada por:

$$
L(N)=L_\infty+AN^{-\alpha}.
$$

A decisão deve considerar:

1. CE nos mesmos checkpoints;
2. crossovers observados;
3. expoente de scaling estimado;
4. CE por hora de GPU;
5. throughput e memória;
6. resultados em capacidades complementares;
7. confirmação das finalistas com três seeds.

## 12. Matriz de promoção e nome público

| Resultado confirmado | Código | Nome público recomendado |
|---|---|---|
| J vence | ASM-X | DRM Language Emitter |
| Metric Subspace vence | ASM-U | DRM Subspace Emitter |
| Metric Orthonormal vence | ASM-F | Relational Frame Emitter |
| No Direction vence | ASM-R | Relational State Emitter |
| Direct Control vence | ASM-D | Causal State Emitter |
| Direct Matched vence | ASM-S | Selective State Emitter |
| SSM Control vence | ASM-M | Causal Memory Emitter |

## 13. Cenários de interpretação

### DRM explícito escala melhor

Se `ASM-X` começar pior e vencer em escala, o custo inicial de aprender a
geometria será parte de sua scaling law. O nome DRM Language Emitter poderá ser
mantido.

### A composição métrica-primeiro vence

Se `ASM-U` ou `ASM-F` vencer, o problema não era possuir direções, mas compor
direção e métrica de maneira incompatível. O resultado representa uma correção
da implementação original e um fortalecimento reformulado da teoria.

### A métrica sobrevive sem direções explícitas

Se `ASM-R` vencer, DRM permanece como origem teórica, mas sai do nome principal.
O projeto torna-se Relational State Emitter.

### Memória ou transição direta dominam

Se `ASM-D`, `ASM-S` ou `ASM-M` vencer, a arquitetura promovida deixa de ser DRM.
A teoria continua documentada e `ASM-X` permanece como variante experimental.

### Variantes vencem em regimes diferentes

Uma única arquitetura não precisa dominar todos os objetivos. A família ASM
pode promover modelos diferentes para eficiência inicial, qualidade em escala,
contexto longo ou controlabilidade.

## 14. Evolução recomendada do repositório

O repositório não deve ser renomeado antes da conclusão experimental. Depois da
confirmação, a organização interna pode evoluir para:

```text
src/aletheion_state_models/
├── core/
│   ├── state_model.py
│   ├── transition.py
│   ├── memory.py
│   ├── mixer.py
│   └── emitter.py
├── geometry/
│   ├── metric.py
│   ├── directional_basis.py
│   └── naturalization.py
└── variants/
    ├── explicit_drm.py
    ├── metric_subspace.py
    ├── relational_state.py
    ├── direct_state.py
    └── selective_state.py
```

Enquanto a scaling law não promove `ASM-F`, o modo de frame métrica permanece
na implementação geométrica compartilhada e na matriz experimental. Um módulo
`relational_frame.py` dedicado só deve ser criado se essa arquitetura for
promovida ou exigir lógica independente.

Interfaces neutras recomendadas:

- `StateModel`;
- `StateTransition`;
- `StateMemory`;
- `CausalMixer`;
- `GeometryOperator`;
- `DirectionalBasis`;
- `TokenEmitter`.

Os nomes antigos devem permanecer temporariamente como aliases depreciados
para evitar uma quebra imediata da API.

## 15. Estrutura documental futura

Depois da promoção, a documentação deve ser separada em:

- `THEORY.md`: teoria DRM, independentemente do modelo vencedor;
- `MODEL_FAMILY.md`: família ASM e suas variantes;
- `EXPERIMENTAL_EVIDENCE.md`: resultados, seeds, compute e scaling laws;
- `ARCHITECTURE.md`: somente a arquitetura promovida;
- `HISTORY.md`: origem DRM e evolução baseada nas ablações.

## 16. Estado provisório

Nenhuma mudança pública de nome está aprovada neste momento.

A nomenclatura ASM serve para impedir que o nome histórico pressione a decisão
científica. `ASM-X` preserva explicitamente o DRM dentro da família, mesmo que
outra variante seja promovida.

O modelo atualmente mais plausível, sujeito à scaling law até 100M e à
confirmação multiseed, é:

> **ASM-R — Aletheion Relational State Model**

com a arquitetura pública:

> **Relational State Emitter**

Se os resultados mudarem, a taxonomia já oferece uma identidade honesta para a
variante que realmente vencer.
