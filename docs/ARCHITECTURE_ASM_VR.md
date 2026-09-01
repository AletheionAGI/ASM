# ASM-VR: proposta de um modelo relacional de rank variável

## Status de implementação

A **Fase 0 isolada** foi concluída em 2026-08-31. Ela ainda não altera o
forward, treinamento, geração ou checkpoints dos modelos ASM existentes.

Entregas validadas:

- `VariableRankState` persiste somente coordenadas efetivas compactas, mask,
  rank, frame e o slot de memória, que obrigatoriamente permanece `None` nesta
  fase;
- a API não possui campo ou propriedade `full_state`;
- projeção hard idempotente e filtro soft explicitamente não idempotente;
- transporte entre frames/ranks com forcing separado;
- diagnóstico do Jacobiano e do rank numérico de ciclos;
- probe supervisionado para informação descartada;
- experimento reproduzível em `scripts/eval_asm_vr_phase0.py`.

Resultado congelado do experimento padrão:

- erro de idempotência do projetor: `0.0`;
- recuperação do componente descartado pelo estado efetivo: `0.000112`;
- recuperação pela memória externa declarada: `1.0`;
- rank do Jacobiano do ciclo `8→3→5→8`: `3`;
- déficit de rank do ciclo: `5`.

Os artefatos estão em [`benchmarks/asm_vr_phase0/`](benchmarks/asm_vr_phase0/).

A **Fase 1 — colapso sem bypass** foi concluída em 2026-08-31. A variante
opt-in `build_variable_rank_phase1` integra rank hard por bloco ao núcleo real do
ASM-R: transição direta, métrica relacional, naturalização, updater e emitter.
O frame global é a identidade nesta fase. O controller decide uma máscara por
exemplo usando somente o primeiro token causal do bloco, sem observar o estado
que será descartado.

O cache guarda somente coordenadas efetivas padded com zeros nas posições
inativas, a máscara hard e os tokens do bloco aberto. Ele não guarda estado
ambiente, prefixo completo ou memória auxiliar, e `decode_step` não pode cair no
fallback que reexecuta o prefixo. Mixer, residual token-state, memórias,
refinamentos e solvers de bloco são rejeitados pela configuração Phase 1.

No teste integrado, estados e logits pareados ficam exatamente iguais após o
colapso, o Jacobiano no complemento descartado é `0.0`, nenhuma coordenada
inativa aparece no cache e o erro máximo streaming versus forward é
`1.0430813e-07`. Os artefatos estão em
[`benchmarks/asm_vr_phase1/`](benchmarks/asm_vr_phase1/).

A **Fase 2 — rank adaptativo sintético** foi concluída em 2026-08-31 com o
builder `build_variable_rank_phase2`, máscara hard no forward, surrogate STE no
backward, currículo de orçamento/hardening e perdas separadas de budget,
binarização e troca. O benchmark `Variable-Capacity Copy` executou cinco
controles em três seeds. O adaptativo obteve correlação rank × dificuldade
`1.0`, rank médio `9.67` e acurácia média `0.4201`, dentro do gate contra o
controle rank 8 (`0.4578`), mas sem superioridade Pareto. Os artefatos estão em
[`benchmarks/asm_vr_phase2/`](benchmarks/asm_vr_phase2/).

A **Fase 3A — linguagem em pequena escala** foi concluída em 2026-09-01 com
seis variantes, três seeds e aproximadamente 2M tokens por run. O adaptativo
atingiu rank médio `30.63` e CE de test
`3.1869`, apenas `0.0116` nat pior que o
rank fixo 32. Todos os gates operacionais passaram, incluindo orçamento hard e
streaming FP32. Porém, o rank fixo 16 obteve CE menor com menos rank; portanto
não houve vantagem Pareto. Gráficos PNG/SVG e dashboard estão em
[`benchmarks/asm_vr_phase3a/`](benchmarks/asm_vr_phase3a/).

A **Fase 3A.1** reintegrou mixer causal, residual token-state e memória seletiva
sob projeção hard entre cada componente. O fatorial `2³` selecionou mixer +
residual: validation CE `2.5676`, ganho de
`0.5853` nat frente ao scaffold
estrito, sem precisar de memória seletiva. A matriz de rank no scaffold
selecionado atingiu rank adaptativo médio `32.21`, mas test CE
`2.6768` contra
`2.6092` do fixed-32. Logo, o scaffold
projetado recupera qualidade, mas o controller ainda não é Pareto-superior.
Relatórios e gráficos estão em
[`benchmarks/asm_vr_phase3a1/`](benchmarks/asm_vr_phase3a1/).

Uma ablação AdamM posterior melhorou CE em `0.0225` nat no fixed-32 e `0.0246` nat no adaptativo, mas a interação foi somente `-0.0021` nat. Logo, AdamM melhora a otimização geral sem resgatar o controller adaptativo. Veja [`benchmarks/asm_vr_phase3a1_adamm/`](benchmarks/asm_vr_phase3a1_adamm/).

A **Fase 3A.2** comparou ASM-VR-R e ASM-VR-S em 30 runs parameter-matched,
com projeção adicional de `local_delta` antes do mixer. ASM-VR-S venceu R em
todas as 15 comparações pareadas: full `-0.0485` nat
e fixed-32 `-0.0503` nat. A confirmação AdamM em
seeds novas 71/89/107 repetiu a direção 3/3, delta médio
`-0.0451` nat. Assim, **ASM-VR-S é a base promovida
por qualidade**, embora seu throughput denso tenha sido cerca de
`5.1%` menor. O controller adaptativo continuou fora da
fronteira fixa em R e S. Veja
[`benchmarks/asm_vr_phase3a2/`](benchmarks/asm_vr_phase3a2/) e
[`benchmarks/asm_vr_phase3a2_adamm_confirm/`](benchmarks/asm_vr_phase3a2_adamm_confirm/).

A taxonomia **ASM-RS** foi formalizada como a mistura de ASM-R com ASM-S;
**ASM-VR-RS** é seu Variable-Rank. O controle `vr_rs_full` reproduziu a receita
do antigo ASM-R prático, mas foi dominado pelo S full mesmo com
`9.3%` mais
parâmetros. Veja
[`benchmarks/asm_vr_phase3a3_rs/`](benchmarks/asm_vr_phase3a3_rs/).

Antes da Fase 3B, o roadmap inclui o **ATTR — ASM–Transformer Transition Risk
Benchmark**. Essa etapa compara antecipação de hazards, previsibilidade calibrada
e restrição causal do safe set contra controles Transformer pareados, usando
heads e shield externos comuns. O protocolo está em
[`ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md`](ASM_TRANSFORMER_TRANSITION_RISK_PROTOCOL_ptbr.md).
Após publicar todos os gates ATTR, inclusive falhas, o desenvolvimento retorna à
Fase 3B.

Transition Memory continua fora do escopo atual. A Fase 3B deve começar com
ASM-VR-S congelado e controles full/fixed-rank explícitos. Uma nova alegação de
controller adaptativo exige antes superar a fronteira fixa.

## 1. Decisão

A proposta é uma direção forte para um novo ASM. Ela não copia a atenção linear ou a interpretação cerebral do BDH. Em vez disso, transforma em mecanismo computacional uma ideia própria do DRM:

> o modelo altera, durante a inferência, o subespaço de estado que continua efetivamente disponível.

O nome de trabalho recomendado é:

> **ASM-VR — Variable-Rank Relational State Model**

A linhagem inicial deve partir do **ASM-R**, que já demonstrou ser mais estável que a fatoração direcional explícita do ASM-X. O ASM-VR acrescentaria um subespaço efetivo variável e transporte entre subespaços. Uma segunda variante, **ASM-VR-TM**, acrescentaria memória de transição.

A hipótese experimental é:

\[
\boxed{
\text{rank adaptativo}
+
\text{transporte com perda controlada}
+
\text{memória de transição}
}
\]

pode produzir computação condicional, memória dependente do caminho e geometria efetiva variável sem usar self-attention sobre o prefixo.

## 2. A condição que torna a ideia realmente DRM-like

Não basta calcular

\[
\widetilde z_t=P_tz_t
\]

para produzir a saída e continuar carregando o `z_t` completo em outro caminho. Se as coordenadas desligadas permanecerem preservadas no estado ambiente, uma expansão futura apenas as revela novamente. Isso seria masking, não colapso efetivo nem histerese.

A regra estrutural do ASM-VR deve ser:

> Depois de uma transição redutora, nenhuma rota residual pode transportar os componentes descartados para o próximo estado efetivo.

Isso inclui:

- residual token-to-state;
- mixer causal;
- memória seletiva;
- fast weights;
- cache de inferência;
- diagnósticos que acidentalmente retornem ao forward como features.

O próprio controlador pode virar um canal lateral: se `P_t=P_t(z_t)`, o padrão de gates pode codificar alguns bits dos componentes que serão descartados. O MVP deve escolher uma política explícita:

1. **não interferência estrita:** calcular o próximo mask de estado já projetado, estatísticas atrasadas ou `stop_gradient`;
2. **leakage de controle contabilizado:** permitir o canal e medir quanta informação o mask transporta.

Qualquer canal que preserve o conteúdo descartado deve ser declarado **memória externa**. Nesse caso, a irreversibilidade aplica-se ao estado projetado, não ao estado conjunto ampliado.

## 3. Separar quatro objetos

A implementação deve evitar usar `G`, `P`, gates e transporte como se fossem o mesmo objeto.

### 3.1 Frame relacional

Seja

\[
Q_t\in\mathbb R^{N\times K},
\qquad
Q_t^{\mathsf T}Q_t=I,
\qquad
K\le N.
\]

As colunas de \(Q_t\) formam um catálogo máximo de direções relacionais. O MVP pode usar um frame aprendido global. Versões posteriores podem aplicar pequenas rotações dependentes do estado.

### 3.2 Intensidade de ativação

Um controlador causal produz

\[
s_t=\sigma(f_\theta(z_t,x_t,m_t))\in[0,1]^K.
\]

Durante o treinamento, `s` fornece gradientes e uma noção de rank suave:

\[
r_{\mathrm{soft}}(t)=\sum_{i=1}^K s_{t,i}.
\]

Também podem ser registrados:

\[
r_{\mathrm{stable}}
=
\frac{(\sum_i s_i)^2}{\sum_i s_i^2+\varepsilon}
\]

ou o rank entrópico. Nenhuma dessas quantidades deve ser chamada de rank algébrico exato.

### 3.3 Operador de acesso e projetor

O operador suave é

\[
A_t=Q_t\operatorname{diag}(s_t)Q_t^{\mathsf T}.
\]

Em geral,

\[
A_t^2\ne A_t.
\]

Portanto, durante o treino, ele é uma contração ou operador de acesso suave, não um projetor.

Para o caminho hard, seja \(m_t\in\{0,1\}^K\):

\[
P_t=Q_t\operatorname{diag}(m_t)Q_t^{\mathsf T},
\qquad
P_t^2=P_t,
\qquad
\operatorname{rank}P_t=\sum_i m_{t,i}.
\]

O treinamento pode usar Hard Concrete, Gumbel/straight-through ou top-k com straight-through. `rank()` não participa do gradiente; serve apenas como diagnóstico no caminho hard.

Para evitar flicker, a decisão hard pode ter histerese própria:

- uma direção inativa liga somente acima de `theta_up`;
- uma direção ativa desliga somente abaixo de `theta_down`;
- `theta_up > theta_down`;
- um dwell time mínimo é opcional.

Isso é diferente da histerese geométrica do estado, mas estabiliza os eventos que a produzem. Se o frame for dinâmico, os modos também precisam ser alinhados entre passos; ordenar apenas por intensidade falha quando dois modos se cruzam.

### 3.4 Métrica relacional

A métrica não deve ser o próprio projetor. Nas coordenadas ativas, use

\[
G_t^{\mathrm{eff}}
=
\operatorname{diag}(\gamma_t),
\qquad
\gamma_{t,i}>0.
\]

No espaço ambiente, a forma semidefinida associada é

\[
g_t=Q_t\operatorname{diag}(m_t\odot\gamma_t)Q_t^{\mathsf T}.
\]

Seu kernel contém as direções inativas. Inversões e naturalização devem ocorrer somente nas coordenadas efetivas, ou por pseudoinversa controlada. Não se deve adicionar um piso positivo em todas as direções, pois isso restauraria rank completo e repetiria a limitação atual do ASM-X/ASM-R.

O MVP acima usa projeção ortogonal euclidiana. Se o projetor precisar ser ortogonal em uma métrica ambiente `M`, a forma correta é

\[
P=U(U^{\mathsf T}MU)^{-1}U^{\mathsf T}M.
\]

Nesse caso, `P` continua idempotente, mas não precisa ser simétrico no produto euclidiano; ele satisfaz \(P^{\mathsf T}M=MP\). Essa extensão deve vir depois do caso euclidiano, com testes próprios de condicionamento.

## 4. Estado e transporte

### 4.1 Coordenadas efetivas

Se \(B_t\in\mathbb R^{N\times r_t}\) contém apenas as colunas ativas de \(Q_t\), represente o estado utilizável como

\[
a_t=B_t^{\mathsf T}z_t\in\mathbb R^{r_t},
\qquad
z_t^{\mathrm{eff}}=B_ta_t.
\]

A implementação física pode manter tensors de largura \(K\) com máscara hard durante o treino. A semântica, porém, deve ser a de coordenadas efetivas.

### 4.2 Mapa de transição

Uma forma inicial é

\[
J_t
=
C_tB_{t+1}^{\mathsf T}B_t,
\qquad
J_t:\mathbb R^{r_t}\to\mathbb R^{r_{t+1}},
\]

onde \(C_t\) é um mixer ou contração aprendida nas coordenadas efetivas.

O próximo estado é

\[
a_{t+1}=J_ta_t+w_t,
\]

\[
z_{t+1}=B_{t+1}a_{t+1}.
\]

O termo \(w_t\) é força ou escrita produzida pelo token atual. Ele deve ser distinguido do transporte. Uma dimensão recém-ativada pode receber conteúdo novo por \(w_t\), mas não deve recuperar magicamente o componente antigo descartado.

Se

\[
r_{t+1}<r_t,
\]

então

\[
\operatorname{rank}J_t\le r_{t+1}.
\]

Com expansão posterior, pós-composição determinística não recupera o rank perdido. Isso fornece o análogo computacional direto do teorema de holonomia dimensional do DRM.

### 4.3 Exemplo de ciclo

Para

\[
64\to21\to37\to64,
\]

defina

\[
H=J_{37\to64}J_{21\to37}J_{64\to21}.
\]

Então

\[
\operatorname{rank}H\le21<64,
\qquad
H\ne I.
\]

Esse resultado só é válido para o componente transportado. Escritas externas \(w_t\) tornam a dinâmica afim; para diagnosticar holonomia, mede-se o Jacobiano do estado final em relação ao estado inicial ou executa-se o loop com forcing controlado.

## 5. Memória de transição

A “Transition Memory” deve ser dividida em duas modalidades para que o experimento seja cientificamente interpretável.

### 5.1 Memória estrutural não arquivística

Armazena somente informações como:

- rank anterior e atual;
- identificadores ou resumo das direções sobreviventes;
- energia perdida estimada;
- assinatura comprimida do evento;
- ordem recente de transições.

Ela pode influenciar gates e transições futuras, mas não contém os valores exatos descartados. Essa versão mantém histerese intrínseca.

Uma recorrência possível é

\[
\mu_{t+1}
=
F_\mu(\mu_t,m_t,m_{t+1},\Delta E_t,x_t).
\]

### 5.2 Memória externa arquivística

Armazena deliberadamente uma codificação dos componentes perdidos:

\[
\ell_t=(I-P_{t+1})z_t.
\]

ASM-CM fast weights podem servir como backend desse canal. Se ele restaurar \(\ell_t\), a recuperação não contradiz DRM: o estado total foi ampliado e a informação nunca foi realmente eliminada do sistema conjunto.

Essa distinção cria uma ablação especialmente importante:

1. sem memória de transição;
2. memória estrutural sem conteúdo descartado;
3. arquivo externo dos componentes descartados.

## 6. Transições controladas por energia

Para um transporte

\[
J_t:(V_t,G_t)\to(V_{t+1},G_{t+1}),
\]

a condição dissipativa é

\[
J_t^{\mathsf T}G_{t+1}J_t\preceq G_t.
\]

Defina

\[
M_t=J_t^{\mathsf T}G_{t+1}J_t-G_t.
\]

Uma penalidade diferenciável é

\[
\mathcal L_{\mathrm{energy}}
=
\left\|\operatorname{ReLU}(\lambda(M_t))\right\|_2^2,
\]

onde somente autovalores positivos violam a condição. Para evitar uma decomposição espectral grande, o MVP pode usar vetores aleatórios \(v_j\):

\[
\mathcal L_{\mathrm{energy}}
\approx
\frac1q\sum_j
\operatorname{ReLU}
\left(
\|J_tv_j\|_{G_{t+1}}^2-
\|v_j\|_{G_t}^2
\right)^2.
\]

Outra opção é parametrizar \(J_t\) diretamente como uma contração em coordenadas embranquecidas. Isso fornece garantia estrutural, mas pode limitar a capacidade. Deve ser uma variante separada da regularização suave.

“Energia” aqui significa norma quadrática da fibra. Não significa FLOPs, energia elétrica ou custo metabólico.

## 7. Objetivo de treinamento

Um objetivo inicial é

\[
\begin{aligned}
\mathcal L={}&\mathcal L_{\mathrm{CE}}
+\lambda_{\mathrm{budget}}\mathcal L_{\mathrm{budget}}
+\lambda_{\mathrm{binary}}\mathcal L_{\mathrm{binary}}\\
&+\lambda_{\mathrm{switch}}\mathcal L_{\mathrm{switch}}
+\lambda_{\mathrm{energy}}\mathcal L_{\mathrm{energy}}
+\lambda_{\mathrm{rankvar}}\mathcal L_{\mathrm{rankvar}}.
\end{aligned}
\]

Termos possíveis:

\[
\mathcal L_{\mathrm{budget}}
=
\left(\frac1T\sum_t r_{\mathrm{soft}}(t)-r_{\mathrm{target}}\right)^2,
\]

\[
\mathcal L_{\mathrm{binary}}
=
\frac1{TK}\sum_{t,i}s_{t,i}(1-s_{t,i}),
\]

\[
\mathcal L_{\mathrm{switch}}
=
\frac1T\sum_t\|s_t-s_{t-1}\|_1.
\]

`rankvar` deve impedir a solução trivial de rank constante, mas sem forçar variação artificial em toda amostra. Uma opção é impor apenas uma variância mínima por batch depois de um warm-up.

O orçamento deve ser progressivo:

1. aprender qualidade com gates quase abertos;
2. introduzir custo de rank;
3. endurecer gates;
4. habilitar execução esparsa;
5. ajustar memória de transição.

## 8. Computação condicional real

Gates densos não reduzem FLOPs por si mesmos. Para obter custo próximo de

\[
O(r_td)
\]

é necessário evitar calcular as direções inativas.

### Treinamento

O primeiro protótipo pode ser denso. Ele valida a hipótese sem misturar ganho algorítmico e engenharia de kernel.

### Inferência

O caminho hard deve:

- compactar índices ativos;
- executar projeções e mixers apenas nessas coordenadas;
- agrupar exemplos com ranks semelhantes ou usar buckets;
- atualizar o frame em blocos, não necessariamente a cada token;
- medir o custo do controlador de gates.

Se o gate custa \(O(Nd)\), a economia desaparece. Alternativas posteriores incluem gate hierárquico, seleção em dois estágios e atualização do rank por bloco causal.

Por isso, o claim inicial deve ser “rank variável”; “aceleração” só depois de medições reais de throughput e VRAM.

## 9. Arquitetura proposta

```text
input token x_t
      |
causal state / ASM-R backbone
      |
RankController -> soft gates s_t -> hard mask m_t
      |
RelationalFrame -> active basis B_t
      |
EffectiveMetric -> G_t^eff
      |
TransitionTransport J_t
      |                    \
state forcing w_t          TransitionMemory mu_t
      |                    /
effective state a_{t+1}
      |
reconstruct z_{t+1}=B_{t+1}a_{t+1}
      |
emitter
```

A ordem conceitual é:

```text
entrada
→ estado causal
→ geometria e subespaço efetivo
→ transporte entre possibilidades
→ escrita do token
→ memória do evento de transição
→ próximo estado
```

## 10. Plano de implementação no repositório

Não se deve aumentar ainda mais `src/drm_language_emitter/model.py`. A nova responsabilidade deve entrar em módulos coesos.

Estrutura sugerida:

```text
src/aletheion_state_models/
├── geometry/
│   └── variable_rank/
│       ├── relational_frame.py
│       ├── rank_controller.py
│       ├── effective_metric.py
│       └── active_subspace.py
├── core/
│   ├── transition_transport.py
│   └── transition_memory.py
└── variants/
    └── variable_rank.py
```

Responsabilidades:

- `RelationalFrame`: mantém ou gera \(Q_t\).
- `RankController`: produz `s`, `m` e diagnósticos de rank.
- `ActiveSubspace`: projeta, compacta e reconstrói coordenadas.
- `EffectiveMetric`: define \(G_t^{\mathrm{eff}}\) sem piso ambiente de rank completo.
- `TransitionTransport`: constrói/aplica \(J_t\) e mede energia.
- `TransitionMemory`: registra eventos sem arquivar conteúdo, salvo em variante explícita.
- `variable_rank.py`: somente composição e construção da variante.

Contratos públicos como `RankObservation`, `RankEstimator` e `GeometrySnapshot` devem ficar na camada neutra `aletheion_state_models`. Adaptadores de checkpoint podem permanecer no pacote legado. Um `GeometrySnapshot` nomeado é preferível a continuar propagando tuples posicionais entre os caminhos recorrente, block-cumsum e inferência.

O forward legado deve receber o novo mecanismo por uma interface pequena, em vez de incorporar toda a lógica. O estado de inferência precisará guardar, além do estado causal:

- máscara/frame ativo atual;
- coordenadas efetivas;
- memória estrutural de transição;
- opcionalmente arquivo externo.


## 10.1 Protótipo de engenharia versus modelo semanticamente completo

Há um atalho útil no código atual: mascarar dinamicamente as colunas do fator `U` da métrica `diag + U U^T`. Isso permite validar controller, thresholds, estado de histerese, telemetria e paridade de inferência com baixo risco.

Esse protótipo deve ser chamado, por exemplo, **active factor rank**, não rank DRM. O piso diagonal mantém a métrica SPD e com rank ambiente completo. Ele também não reduz o custo do solver Woodbury enquanto os tensors continuarem padded.

A sequência recomendada é:

1. instrumentar rank numérico e rank do fator sem mudar logits;
2. validar controller hard com banda `off/on`, default desligado;
3. integrar o estado de rank ao streaming por blocos;
4. somente então substituir o estado por coordenadas canonicamente projetadas e introduzir a métrica PSD efetiva do ASM-VR verdadeiro.

Assim, o protótipo testa a engenharia, mas não recebe a conclusão geométrica antes da hora.

## 10.2 Integração segura com o motor atual

Antes de modificar o forward por blocos, convém substituir o retorno posicional grande de `_directional_cumsum_block` por uma dataclass `DirectionalBlockResult`, em uma refatoração sem mudança funcional.

Requisitos adicionais:

- todos os novos campos de configuração começam em `off` e preservam logits e checkpoints atuais;
- o estado de rank é por exemplo/batch, nunca estado global do módulo;
- no modo block-cumsum, gates e frame mudam somente na fronteira do bloco;
- o cache guarda o estado de rank **anterior** ao bloco aberto, para que recomposição incremental reproduza o forward completo;
- `collect_diagnostics=False` pode desligar métricas caras, mas nunca a dinâmica que afeta logits;
- eigendecomposição densa fica fora do hot path e roda em float32, amostrada;
- geração usa o motor `prefill/decode_step` existente e deve passar pelos mesmos testes de paridade.

## 11. Fases experimentais

### Fase 0 — operador e invariantes

Sem linguagem. Validar:

- ortogonalidade de \(Q\);
- idempotência de \(P\) no caminho hard;
- gradientes finitos no caminho soft/STE;
- rank hard igual ao número de gates;
- mapas com shapes variáveis simulados;
- desigualdade energética.

### Fase 1 — colapso sem bypass

Usar frame global fixo, gates hard e sem qualquer memória auxiliar.

Teste decisivo:

> Dois estados que diferem somente em uma direção descartada devem tornar-se indistinguíveis após o colapso e permanecer indistinguíveis sob os mesmos inputs futuros.

Esse é o teste contra histerese falsa.

### Fase 2 — rank adaptativo em tarefas sintéticas

Tarefas com capacidade variável conhecida:

- cópia com quantidade variável de itens;
- MQAR com número variável de pares ativos;
- mudança explícita de regime;
- loops de operações não comutativas;
- compressão seguida de expansão.

Medir se \(r_t\) acompanha a dificuldade e se loops distintos produzem holonomias distintas.

### Fase 3 — ASM-VR sobre ASM-R

Comparar com o mesmo orçamento:

- ASM-R;
- ASM-R com máscara de mesmo custo, mas rank fixo;
- ASM-VR sem memória;
- ASM-VR com memória estrutural;
- ASM-VR com arquivo externo;
- ASM-CM.

### Fase 4 — execução esparsa

Somente depois da qualidade e dos invariantes:

- gather/scatter de direções ativas;
- buckets de rank;
- gate por bloco;
- kernels compilados;
- medições reais de custo.

### Fase 5 — frame dinâmico e não comutatividade

Um frame fixo com masks diagonais tende a produzir transições que comutam. Para memória de ordem mais rica, introduzir:

- rotações de Householder;
- transformação de Cayley;
- mixer ortogonal de baixa dimensão;
- frame atualizado por bloco causal.

Isso deve ser posterior ao MVP, pois frame totalmente dinâmico acrescenta custo, gauge não identificável e instabilidade.

## 12. Testes obrigatórios

### Correção matemática

- `P @ P == P` no caminho hard, dentro da tolerância.
- \(P\) simétrico e PSD.
- rank hard correto.
- pseudoinversa somente no subespaço ativo.
- \(\operatorname{rank}H\le\min_t r_t\) em loops lineares controlados.
- penalidade energética zero para contrações conhecidas e positiva para expansões.

### Causalidade

- tokens futuros não mudam masks, ranks, estados ou logits anteriores.
- compact streaming reproduz o forward de referência.
- nenhuma cache retém coordenadas descartadas no modo intrínseco.

### Ausência de bypass

- remover uma direção apaga diferenças que existam somente nela.
- com `P` mantido fixo, perturbações no complemento `(I-P)` não mudam o próximo estado ou logits.
- medir o Jacobiano no complemento e treinar um probe para detectar informação descartada remanescente.
- auditar quanta informação o próprio padrão de masks carrega.
- expansão sem forcing ou arquivo externo inicializa novas direções sem recuperar os valores antigos.
- habilitar arquivo externo deve restaurar informação somente através da interface declarada.

### Treinamento

- gradientes finitos perto das mudanças de gate.
- rank não colapsa sempre para 0 ou \(K\).
- utilização não fica presa a poucas direções globais.
- troca de rank não apresenta flicker excessivo.

### Eficiência

- FLOPs estimados e medidos por rank.
- throughput, VRAM e bytes do estado.
- custo do controlador separado do custo economizado.
- distribuição de rank por token e por bloco.

## 13. Diagnósticos DRM

Para um loop controlado, registrar:

\[
\delta_{\mathrm{rank}}=r_0-\operatorname{rank}_\varepsilon(H),
\]

\[
h_F=\frac{\|H-I\|_F}{\sqrt{r_0}},
\]

\[
D=G_0-H^{\mathsf T}G_0H,
\]

além de:

- singular values de \(H\);
- energia perdida;
- subespaço sobrevivente;
- comutador \([H_A,H_B]\);
- recuperação com e sem memória externa;
- rank suave e hard ao longo da sequência.

Na dinâmica não linear, \(H\) deve ser estimado pelo Jacobiano do mapa do loop ou por perturbações locais. “A sequência terminou no mesmo token” não significa, por si só, que o loop fechou no espaço-base.

## 14. Riscos

### Colapso trivial

A CE pode preferir rank máximo. A penalidade de compute pode preferir rank mínimo. É necessário currículo e curva de Pareto qualidade/custo.

### Rank cosmético

O modelo pode desligar gates sem reduzir dependências ou FLOPs. O teste hard e a execução compactada são indispensáveis.

### Bypass de memória

ASM-R/ASM-CM contêm residual, mixer e memórias capazes de carregar informação descartada. O MVP deve desativá-los ou submetê-los à mesma projeção.

### Instabilidade de base

Um \(Q_t\) arbitrário a cada token cria ambiguidades de rotação e sinais. Começar com frame global ou rotações pequenas por bloco.

### Escrita confundida com recuperação

Após expansão, \(w_t\) pode preencher novas direções. Isso é criação de estado novo, não reconstrução do estado perdido. Os diagnósticos precisam separar ambos.

### Ganho de eficiência não garantido

Treino denso com gates não economiza compute. Sparsity irregular também pode ser pior para GPU que operações densas menores.

## 15. Critérios de avanço

O modelo só deve receber promoção além de “experimental” se demonstrar:

1. mudança real de rank hard, não apenas gates suaves;
2. ausência comprovada de bypass no modo intrínseco;
3. benefício de qualidade, capacidade ou custo contra controles de rank fixo;
4. rank correlacionado à dificuldade ou ao regime da entrada;
5. histerese/ordem mensurável em loops controlados;
6. recuperação atribuível explicitamente à memória externa quando habilitada;
7. economia medida, não inferida da contagem de gates;
8. confirmação multiseed.

## 16. Recomendação final

A combinação **Dynamic Rank Gating + Transition Memory** é mais original e mais coerente com DRM do que importar diretamente linear attention ou Hebbian updates do BDH.

O melhor primeiro modelo, porém, não deve tentar resolver tudo de uma vez. O MVP recomendado é:

> **ASM-VR-0:** ASM-R com frame aprendido fixo, gates Hard Concrete, transporte explícito, estado canonicamente projetado, sem memória auxiliar e com diagnósticos de holonomia/energia.

Depois:

> **ASM-VR-TM:** acrescentar memória estrutural de eventos.

E somente como ablação separada:

> **ASM-VR-EM:** usar fast weights do ASM-CM como memória externa dos componentes descartados.

Essa sequência permite responder três perguntas distintas:

1. O rank variável melhora o modelo?
2. A história das transições acrescenta capacidade útil?
3. Quando a informação reaparece, ela foi recriada ou restaurada por um canal externo?

Se os resultados forem positivos, o ASM-VR terá uma identidade arquitetural própria:

\[
\boxed{
\text{adaptive computation}
+
\text{path-dependent memory}
+
\text{variable effective geometry}
}
\]
