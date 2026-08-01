# Arquitetura

ASM — Aletheion State Models é uma família de modelos de estado causais
construída em torno de um estado latente `z_t`. Ela não usa atenção sobre sequências. O repositório
agora contém o DRM recorrente original, aproximações em blocos, híbridos com
memória seletiva e controles sem geometria. Esses caminhos precisam ser
diferenciados ao relatar um resultado.

A arquitetura DRM explícita recebe o nome **ASM-X**. As interfaces neutras da
família ficam em `src/aletheion_state_models/`; `src/drm_language_emitter/`
permanece como implementação compatível com checkpoints durante a migração.
Consulte [docs/MODEL_FAMILY_ptbr.md](docs/MODEL_FAMILY_ptbr.md).

## Arquitetura promovida: ASM-R

ASM-R — Aletheion Relational State Model — é a arquitetura promovida para
qualidade por token de treinamento. Ela completou três runs independentes de
100M tokens com CE congelado de validação `1,344538 ± 0,000561` (desvio-padrão
populacional). Seu caminho é:

```text
token → estado causal → transição contextual direta
      → naturalização pela métrica relacional
      → mixer causal + residual do token + memória seletiva
      → emitter
```

ASM-R preserva o condicionamento pela métrica relacional, mas remove o catálogo
explícito de direções. ASM-F permanece experimental após sua geração 1 divergir
nas duas seeds adicionais antes de 70M tokens. Consulte o
[relatório de confirmação](docs/report/037_Confirmacao_Multiseed_100M_e_Promocao_ASM_R_2026_08_01.md).

## Famílias de arquitetura atuais

| Família | Geometria | Memória seletiva | Finalidade |
|---|---|---|---|
| DRM original | campo direcional + métrica + fluxo | não | implementação de referência |
| DRM block-cumsum | deltas direcionais em blocos | opcional | aproximação causal escalável |
| J | DRM block-cumsum | forget/write | referência DRM explícita (ASM-X) |
| J_NO_* | componente removido ou contornado | forget/write | ablações causais da geometria |
| SSM_CONTROL | nenhuma | forget/write ampliada | controle pareado por parâmetros |

A variante J é híbrida. Sua memória seletiva foi adicionada depois que o DRM
original mostrou baixa eficiência amostral; ela não deve ser apresentada como
parte da concepção original independente do DRM.

## Estado latente

O estado latente recorrente é escrito como:

```math
z_t \in \mathcal{M}, \qquad z_t \approx \mathbf{z}_t \in \mathbb{R}^{d_{\text{state}}}
```

No MVP, `z_t` é representado por um vetor em `R^d_state`. Essa é uma
representação por coordenadas do manifold latente, não uma alegação de que o
objeto geométrico verdadeiro seja globalmente euclidiano.

`DRMStateInitializer` usa um estado inicial aprendido expandido para o batch.
Os tokens do prompt então movem o estado através da dinâmica DRM.

## DirectionField

`DirectionField(z)` retorna:

- `V(z) [B, n_directions, d_state]`
- `gates a(z) [B, n_directions]`
- `dimD(z) = sum_i a_i(z)`

```math
D(z) = \{V_i(z)\}_{i=1}^{n_{\text{directions}}}, \qquad
a_i(z) \in [0, 1], \qquad
\mathrm{dim}_{\text{active}}(z) = \sum_i a_i(z)
```

As direções não são ortogonalizadas. A normalização opcional mantém sua escala
controlada, mas não impõe um referencial ortonormal. Os gates definem uma
dimensão ativa local efetiva.

## RelationalMetric

A métrica é:

```math
G(z) =
\mathrm{diag}(\mathrm{softplus}(d(z)) + \epsilon)
+ U(z)U(z)^\top
```

Ela é positiva definida até o piso `eps` e mede energia/acoplamento de
velocidades e direções:

```math
E_z(v) = v^\top G(z)v
```

`pairwise_coupling(z, V)` calcula o acoplamento relacional entre direções
aprendidas sob `G(z)`.

```math
C_{ij}(z) = V_i(z)^\top G(z)V_j(z)
```

## DRMFlow

`DRMFlow` recebe `z_t`, o embedding do token atual `e_t`, as direções ativas e
os gates. Ele emite coeficientes:

```math
c_i(t) = c_i(z_t, e_t)
```

A velocidade bruta é uma combinação direcional gated:

```math
\Delta z_t^{\text{raw}}
= \sum_i a_i(z_t)c_i(z_t, e_t)V_i(z_t)
```

Portanto, a velocidade pertence explicitamente ao span das direções ativas.

Na configuração padrão, a velocidade direcional bruta é naturalizada pela
métrica aprendida:

```math
\Delta z_t = G(z_t)^{-1}\Delta z_t^{\text{raw}}
```

A implementação usa uma resolução Woodbury amortecida para a métrica diagonal
mais low-rank:

```math
\Delta z_t =
\left(G(z_t) + \lambda I\right)^{-1}\Delta z_t^{\text{raw}}
```

A intensidade da naturalização segue um schedule durante o treinamento. Isso
torna a métrica parte da lei de movimento, evitando condicionamento excessivo
imediato.

A atualização de estado é:

```math
z_{t+1} = z_t + dt\,\Delta z_t
```

## Caminho causal em blocos

Na escala de 125M, `directional_block_cumsum` divide a sequência em blocos
causais. A geometria é avaliada a partir do estado no início de cada bloco, as
velocidades locais condicionadas pelos tokens são avaliadas em paralelo e os
estados dos prefixos são aproximados por uma soma cumulativa:

```math
\tilde z_{b,t}
= z_b + \sum_{j \le t} dt\,\Delta z_{b,j}
```

Os blocos permanecem sequenciais porque o estado final do bloco $b$
inicializa o bloco $b+1$. As posições dentro de um bloco permanecem causais
em relação ao prefixo. Um mixer convolucional causal depthwise opcional corrige
os estados aproximados usando somente o contexto à esquerda.

Essa é uma aproximação de engenharia da trajetória recorrente, não uma solução
paralela exata da recorrência não linear.

## Memória seletiva na variante J

J aplica uma recorrência afim dependente do conteúdo depois do residual do
token:

```math
m_t = f_t \odot m_{t-1} + w_t \odot c_t
```

Os valores de esquecimento, escrita e candidato dependem do estado causal
anterior e do token atual. A recorrência é avaliada com um scan afim associativo
que evita divisão por produtos cumulativos que tendem a zero.

O caminho J atual é:

```text
fluxo direcional em blocos
-> naturalização métrica
-> mixer causal local
-> residual token→estado
-> memória seletiva forget/write
-> emissor de linguagem
```

J não instancia `RiskField` nos experimentos CE-only atuais.

## Ablações e controle da geometria

- `J_NO_METRIC` remove `RelationalMetric` e usa geometria identidade.
- `J_NO_NATURALIZATION` mantém os parâmetros e diagnósticos da métrica, mas não
  aplica a inversa da métrica ao fluxo.
- `J_NO_DIRECTION` substitui o campo direcional e o fluxo restrito às direções
  por uma transição neural causal direta.
- `SSM_CONTROL` remove direção, métrica, fluxo e risco, preservando o mixer, o
  residual do token, a memória seletiva e o emitter.

`J_NO_METRIC` e `J_NO_DIRECTION` são ablações estruturais e possuem menos
parâmetros que J. `SSM_CONTROL` amplia a memória seletiva para igualar o
orçamento de parâmetros de J; ele não é pareado por compute nem é uma
implementação de Mamba.

## Loss de ação

O termo de ação é a energia métrica média do rollout:

```math
\mathcal{L}_{\text{action}}
= \frac{1}{T}\sum_{t=1}^{T} \Delta z_t^\top G(z_t)\Delta z_t
```

Isso não torna o modelo um solver geodésico exato. O termo favorece trajetórias
aprendidas de menor ação sob a métrica aprendida atual.

## Emissor de linguagem

`LanguageEmitter(z)` é uma MLP pequena com RMSNorm e GELU. Ela mapeia o estado
latente atual para logits do vocabulário.

```math
\ell_t = f_{\text{emit}}(z_t), \qquad
p(x_{t+1} \mid x_{\le t}) = \mathrm{softmax}(\ell_t)
```

No language modeling supervisionado, a loss principal é a cross-entropy de
tokens:

```math
\mathcal{L}_{\text{CE}}
= -\frac{1}{T}\sum_{t=1}^{T}\log p(x_{t+1} \mid x_{\le t})
```

O objetivo de treinamento pode combinar previsão de tokens com regularização
geométrica:

```math
\mathcal{L}
= \mathcal{L}_{\text{CE}}
+ \lambda_{\text{action}}\mathcal{L}_{\text{action}}
+ \sum_k \lambda_k \mathcal{R}_k
```

Os regularizadores `R_k` incluem o target de fração ativa, a variância da
dimensão, termos de condicionamento/diversidade da métrica, proxies de
recorrência/estabilidade e penalidades opcionais de risco/piso métrico quando
habilitadas pela configuração. A variante J e suas ablações de componentes
atuais zeram os pesos auxiliares geométricos; portanto, esses runs otimizam
somente CE do próximo token.

## Geração

O helper `generation.py` atual aquece `z` com os tokens do prompt usando a
geometria recorrente original. Em seguida, ele repete:

1. emite logits a partir de `z`;
2. amostra o próximo token;
3. calcula o embedding desse token;
4. atualiza `z` por `DirectionField`, `RelationalMetric` e `DRMFlow`.

Não existe cache de atenção.

Esse helper **ainda não** reproduz as semânticas de block-cumsum, local-mixer,
memória seletiva, ablação de componentes ou SSM_CONTROL. A geração com
checkpoints da família J não deve ser apresentada como fiel até que o helper
seja unificado com o caminho de forward do treinamento.

## Por que não é um Transformer

O projeto não instancia `nn.MultiheadAttention`, não constrói projeções de
query/key/value e não executa atenção pareada entre tokens. O histórico da
sequência é comprimido no estado de trajetória `z_t`.

## Emergência geodésica

Uma geodésica no sentido DRM completo minimizaria um funcional de ação sobre
curvas admissíveis cujas velocidades permanecem em `span(D(z))`. O MVP oferece
pressão de treinamento e diagnósticos para trajetórias de baixa ação. Ele não
resolve exatamente o problema geodésico de valor de contorno.

## Topologia toroidal

O utilitário toroidal opcional representa coordenadas circulares como
`(cos theta, sin theta)`. O código não alega convergência toroidal espontânea.
Tal alegação exigiria boundedness, recorrência, estabilidade estrutural e
diagnósticos empíricos.
