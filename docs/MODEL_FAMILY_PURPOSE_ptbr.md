# Família de Modelos ASM: Propósito Prático e Guia de Escolha

## Escopo

Este guia responde **qual modelo ASM escolher para um objetivo prático**. Ele complementa [`MODEL_FAMILY_ptbr.md`](MODEL_FAMILY_ptbr.md), que define a taxonomia, e os relatórios de benchmark, que contêm as evidências. “Promovido” sempre significa promovido para um objetivo declarado, não universalmente melhor.

## Escolha rápida

| Objetivo | Comece com | Motivo | Alerta principal |
|---|---|---|---|
| Melhor qualidade de linguagem por parâmetro na escala PMCS-64 atual | **ASM-VR-S full** | Menor CE de test pareado | O resultado depende do protocolo e da escala |
| Memória associativa durável explícita | **ASM-CM** | Único braço pareado que aprendeu MQAR curto em todas as seeds | Recall longo passou em apenas 1/3 seeds; valide cada checkpoint |
| Gargalo lógico de capacidade | **ASM-VR-S fixed-32** | Metade do rank lógico, linguagem não inferior e único a concluir streaming 32K | A implementação densa não produziu ganho físico |
| Linhagem relacional validada para qualidade/token | **ASM-R** | Evidência histórica mais forte no protocolo maior | Custos geométricos e streaming exigem verificações separadas |
| Baseline simples de estado seletivo | **ASM-S** | Concentra o orçamento em memória seletiva sem geometria relacional | Não é memória associativa explícita |
| Execução compacta dos pesos ASM-R | **ASM-C** | Cache streaming retido e limitado | Reprovou o gate histórico de MQAR curto |
| Pesquisa geométrica ou DRM | **ASM-X, ASM-U, ASM-F, ASM-RS** | Expõe hipóteses geométricas explícitas | Experimental; fidelidade teórica não é utilidade medida |
| Controles estruturais e ablações | **ASM-D, ASM-M** | Isola estado direto e memória causal mínima | Não são arquiteturas promovidas |

## Matriz de seleção prática

| Variante | Propósito prático | Escolha quando | Força principal | Limitação ou não-afirmação principal | Status |
|---|---|---|---|---|---|
| **ASM-X** | Pesquisa de modelo de estado DRM explícito | Você precisa expor direções, métrica, movimento e memória no mesmo modelo | Mapeamento mais próximo da formulação DRM explícita | O custo/benefício geométrico não justificou promoção prática | Experimental/taxonômico |
| **ASM-U** | Pesquisa de movimento em subespaço métrico | Você precisa restringir movimento a `span(V)` | Coerência geométrica forte | Candidato proposto de segunda geração sem evidência de promoção | Proposto/experimental |
| **ASM-F** | Pesquisa de frame local normalizado pela métrica | Você está testando frames explicitamente relacionais | Composição geometricamente consistente | A geração anterior divergiu em seeds adicionais | Experimental |
| **ASM-R** | Qualidade relacional por token | Você quer o backbone relacional promovido no protocolo validado | Transição contextual condicionada por métrica aprendida | Evidência histórica não é diretamente comparável à PMCS-64 | Promovido para qualidade/token relacional |
| **ASM-C** | Streaming compacto dos pesos ASM-R | Você precisa de estado retido limitado com os mesmos pesos treinados | Caminho de engenharia streaming com cache retido pequeno | MQAR histórico curto falhou; cache limitado não é recall associativo | Mecanismo streaming validado, modelo experimental |
| **ASM-C2** | Pesquisa de memória endereçável com slots fixos | Você precisa de slots K/V explícitos e limitados | Torna leitura e escrita inspecionáveis | Gates end-to-end de MQAR, ablação, paridade e regressão permanecem sem aprovação | Experimental |
| **ASM-C2-FW** | Pesquisa de memória associativa fast-weight | Você está testando memória matricial e escritas por delta rule | Probe isolado mostrou capacidade associativa forte | Probes com fases separadas não provam controle causal end-to-end | Candidato experimental |
| **ASM-CM** | Memória associativa durável com estado retido limitado | A tarefa exige retenção explícita chave/valor em atrasos longos | Aprendeu MQAR curto em 3/3 seeds PMCS-64; uma seed generalizou até 32K | Recall longo PMCS-64 foi apenas 1/3 seeds; streaming e paridade falharam | Promovido para a configuração validada anterior; configuração pareada não repromovida |
| **ASM-CM-VR** | Pesquisa de memória associativa durável rank-aware | Você precisa de memória explícita com gargalo lógico estrito de estado | Fixed-32 passou gates estruturais e da Fase 1 seed 17; full/fixed longos passaram 2/3 seeds | Seed 29 ficou não finita em 32K; adaptive também passou só 2/3; bytes densos iguais | Experimental; gate longo de promoção falhou |
| **ASM-RS** | Composição relacional + seletiva explícita | Você precisa tornar explícita a receita histórica R+S | Separa claramente componentes relacional e seletivo | Mais parâmetros e pior qualidade/custo que S no gate pareado | Validado, não promovido |
| **ASM-D** | Controle estrutural de estado direto | Você precisa de baseline neural sem geometria | Simples e útil para atribuição causal | Menor novidade e sem evidência de promoção | Controle |
| **ASM-S** | Baseline de estado seletivo e arquitetura eficiente | Você quer preservar/esquecer/escrever sem geometria relacional | Alocação simples de capacidade para memória seletiva | Não é DRM nem memória chave/valor durável explícita | Opção validada |
| **ASM-VR-S** | Controle lógico de capacidade de estado sobre ASM-S | Você precisa comparar rank full/fixo ou pesquisar rank | Full venceu linguagem PMCS-64; fixed-32 reteve qualidade em `+0,03 nat` e concluiu 32K | Caminhos atuais são densos; controller adaptativo perdeu para a fronteira fixa | Full/fixo validados; adaptativo experimental |
| **ASM-M** | Controle mínimo de memória causal | Você precisa de mixer + residual + memória seletiva estreita sem transição geométrica rica | Isola uma receita causal compacta | Deve ser comparado a RNNs gated e SSMs seletivos; não promovido | Controle/candidato |
| **ASM-VR-RS** | Variable Rank sobre a composição R+S explícita | Você precisa de controle relacional-seletivo para pesquisa de rank | Aplica a mesma interface de rank ao ASM-RS | Evidência publicada cobre somente o controle full, que não foi promovido | Definido/experimental |

`ASM-C2-FW-LM` é um identificador técnico de linhagem cujo nome público promovido é ASM-CM. `ASM-VR-R` permanece linhagem/controle de benchmark até ser formalizado separadamente como membro público.

## Para que Variable Rank é útil

Variable Rank projeta o estado sobre um prefixo de um frame compartilhado. O rank controla, portanto, **quantas direções do estado podem carregar informação** na fronteira de um bloco.

### Utilidades atuais

1. **Experimentos de gargalo de capacidade.** Comparar full e rank fixo testa se todas as dimensões do estado são necessárias.
2. **Comparação arquitetural com a mesma capacidade lógica.** A mesma política de rank pode ser aplicada a backbones distintos sem violar a regra anti-bypass.
3. **Estudos de regularização e estabilidade.** Na PMCS-64, fixed-32 ficou a `+0,0283 nat` do CE de linguagem full e foi o único braço a concluir o stream de 32K da seed 17. Isso é evidência para aquele checkpoint, não garantia universal de estabilidade.
4. **Pesquisa de controller.** Rank adaptativo poderá alocar mais direções a blocos difíceis e menos a blocos fáceis quando superar políticas fixas e a execução física se tornar compacta.
5. **Diagnóstico.** Sweeps de rank expõem a fronteira de qualidade e mostram se um controller é melhor que um orçamento constante.

### Três políticas

- **Full rank:** baseline atual mais seguro para qualidade; a projeção é logicamente a identidade.
- **Fixed rank:** ferramenta prática de pesquisa mais clara; produz gargalo explícito e reproduzível.
- **Adaptive rank:** mecanismo de pesquisa em aberto. O controller input-only atual variou e recebeu gradientes, mas ficou `+0,0631 nat` fora da fronteira fixa na Fase 3A.2 do ASM-VR-S.

### O que Variable Rank ainda não fornece

A implementação publicada ainda executa tensores densos. Rank menor **não** garante:

- menos FLOPs;
- menos VRAM;
- treino ou decoding mais rápido;
- menos bytes de estado retido;
- menor uso de energia.

A PMCS-64 confirmou esse limite. Fixed-32 e full tiveram o mesmo pico observado de memória de treino (`77,9 MiB`), e fixed-32 não foi pelo menos 5% mais rápido. Não converta “metade do rank lógico” em “metade do compute”.

Um ganho físico exige kernels compactos de gather/scatter ou baixo rank, execução agrupada para exemplos com rank semelhante, controller sensível a custo e medições no hardware-alvo.

## PMCS-64: ASM-CM versus ASM-VR-S pareados

A nova suíte usa 274.058 parâmetros no ASM-CM e 274.135 em cada braço ASM-VR-S, diferença total de `0,028%`. Os braços VR têm 4.160 parâmetros congelados no controller e, portanto, 1,49% menos parâmetros treináveis. Veja [`benchmarks/asm_cm_vs_vr_s_pmcs64/README.md`](benchmarks/asm_cm_vs_vr_s_pmcs64/README.md).

### Principais resultados medidos

- **Linguagem:** VR-S full CE `2,5168`; fixed-32 `2,5451`; CM `2,5489`.
- **Controle MQAR curto:** CM `99,95%`; VR-S full `3,96%`; fixed-32 `2,29%`.
- **MQAR 32K:** média CM `33,33%`, causada por uma seed aproximadamente perfeita e duas seeds que falharam; ambos os braços VR-S `0%`.
- **Estado streaming em 4K:** CM `131.584` bytes contra `320` bytes nos dois braços VR-S.
- **Throughput streaming em 4K:** CM `162,3` tok/s; VR-S full `1.132,3`; fixed-32 `1.135,5`.
- **Streaming 32K:** CM falhou no token 15.200; VR-S full falhou em 30.335; fixed-32 concluiu.

Esses resultados delimitam propósito em vez de nomear um vencedor geral:

- VR-S full é o modelo de linguagem pareado mais forte.
- CM contém o mecanismo associativo explícito, mas sua robustez de memória longa e streaming dependeu da seed/configuração.
- Fixed-32 foi o melhor trade-off observado de capacidade lógica/estabilidade, sem alegação de speedup físico por rank.

## Vocabulário de status

- **Promovido para um objetivo:** passou o protocolo declarado para esse objetivo.
- **Opção validada:** implementado e medido, mas não é a escolha promovida padrão.
- **Experimental:** candidato de pesquisa com gates incompletos ou reprovados.
- **Controle:** projetado principalmente para isolar um mecanismo.
- **Proposto:** definição taxonômica/de projeto sem evidência empírica suficiente.

## Limites da evidência

Sempre registre corpus, orçamento de tokens, seeds, otimizador, precisão, hardware, contagem de parâmetros, regra de abertura do test e caminho dos artefatos. Não compare scores históricos como se viessem de uma única execução pareada. Separe qualidade full-forward, streaming com estado retido, recall associativo e custo físico.
