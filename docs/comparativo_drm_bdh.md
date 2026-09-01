# DRM/ASM e BDH: comparação técnica

## Escopo e conclusão curta

Este documento compara:

1. **DRM v7**, a teoria de *Directional Relational Manifolds* descrita em [`paper/drm_v7.tex`](paper/drm_v7.tex);
2. **BDH/BDH-GPU**, a arquitetura *The Dragon Hatchling* descrita no arquivo disponível como [`paper/others/2509.26507v1.pdf`](paper/others/2509.26507v1.pdf);
3. **ASM**, a tradução computacional parcial de ideias do DRM mantida neste repositório.

A ressalva central é que **DRM e BDH não são dois modelos diretamente equivalentes**. DRM v7 é um framework geométrico para sistemas de dimensão efetiva variável. BDH-GPU é uma arquitetura concreta de linguagem, com equações de inferência, implementação em tensores e resultados experimentais. A comparação operacional mais útil é, portanto, **ASM-X/ASM-R/ASM-CM versus BDH-GPU**, mantendo DRM v7 como teoria de referência.

Em uma frase: **DRM é mais geral e mais forte como linguagem matemática para perda de dimensão, irreversibilidade e memória dependente do caminho; BDH é muito mais completo como arquitetura neural executável, escalada e empiricamente estudada**.

## 1. Os três objetos comparados

### 1.1 DRM v7: geometria estratificada

Os dados fundamentais são

\[
\mathfrak D=(M,E,g,\rho,\{S_\alpha\},\{\nabla^\alpha\},\{J_e\}).
\]

- \(M\) é o espaço de estados.
- \(E\to M\) é um fibrado de capacidade ambiente fixa \(N\).
- \(g\) é uma métrica positiva semidefinida.
- A fibra efetiva é \(\overline E_p=E_p/\ker g_p\).
- A dimensão efetiva é \(d_{\mathrm{DRM}}(p)=\operatorname{rank}g_p\).
- \(\rho:E\to TM\) determina movimentos admissíveis.
- Dentro de cada estrato de rank constante, há conexão e transporte paralelo.
- Entre estratos, mapas explícitos \(J_e\), possivelmente não invertíveis, realizam a transição.

Para um ciclo híbrido fechado, a memória geométrica é capturada pelo produto ordenado

\[
H_\gamma=P_mJ_{m-1}P_{m-1}\cdots J_1P_1.
\]

Se algum \(J_k\) tem rank menor que a dimensão inicial, então \(H_\gamma\) é deficiente em rank e não pode ser a identidade. Essa é a principal forma de **histerese geométrica irreversível** do artigo.

### 1.2 BDH/BDH-GPU: atenção linear com estado sináptico

BDH descreve \(n\) neurônios em um grafo local. Parâmetros residem na topologia e nos pesos das conexões. O estado rápido de inferência reside em pesos sinápticos dinâmicos \(\sigma(i,j)\), atualizados por uma regra do tipo Hebb.

BDH-GPU é a formulação tensorial treinável. Ela usa três matrizes principais \(E,D_x,D_y\), dimensão neuronal grande \(n\), dimensão baixa \(d\ll n\), ativações positivas após ReLU e uma matriz de estado recorrente com \(nd\) escalares por camada. Em forma compacta, a atualização de memória inclui um termo de rank 1 (a orientação \(d\times n\) ou sua transposta depende da convenção de armazenamento):

\[
\rho_{t,l}=\rho_{t-1,l}+\operatorname{LN}(E y_{t,l-1})x_{t,l}^{\mathsf T}U.
\]

seguido por leitura por atenção linear e novas ativações \(x_{t,l},y_{t,l}\). A formulação de treino pode processar tokens em paralelo; em inferência, \(\rho\) é o estado persistente.

O paper atribui a esse desenho:

- cerca de \((3+o(1))nd\) parâmetros escaláveis;
- custo de inferência \(O(ndL)\) por token na implementação simples;
- ausência de janela rígida de contexto;
- ativações positivas e empiricamente esparsas;
- localização de conceitos em neurônios e sinapses;
- interpretação equivalente como dinâmica local de grafo, com circuitos excitatório/inibitório e plasticidade hebbiana.

### 1.3 ASM no repositório: implementação parcial derivada do DRM

A taxonomia está descrita em [`MODEL_FAMILY.md`](MODEL_FAMILY.md) e [`../ARCHITECTURE.md`](../ARCHITECTURE.md):

- **ASM-X** é a variante DRM explícita: campo de direções, gates, métrica, fluxo, mixer e memória seletiva.
- **ASM-R** remove o catálogo explícito de direções e conserva uma transição contextual condicionada pela métrica.
- **ASM-CM** acrescenta memória associativa *fast-weight* limitada e é a arquitetura promovida atualmente.

No ASM-X, a métrica implementada é

\[
G(z)=\operatorname{diag}(\operatorname{softplus}(d(z))+\varepsilon)+U(z)U(z)^{\mathsf T},
\]

com atualização aproximada

\[
v_{\mathrm{raw}}=\sum_i a_i c_iV_i,\qquad
v=(G+\lambda I)^{-1}v_{\mathrm{raw}},\qquad
z_{t+1}=z_t+\Delta t\,v.
\]

Como o piso diagonal é positivo, \(G\) é definida positiva e tem **rank exato fixo**. Assim, `dimD`, gates e rank numérico não realizam o rank formal variável de DRM v7. O próprio repositório registra que ainda não implementa fibras quociente, estratificação, conexões, mapas de transição, transporte ou holonomia completos.

## 2. Comparação direta

| Eixo | DRM v7 | BDH/BDH-GPU | ASM atual |
|---|---|---|---|
| Natureza | Framework geométrico geral | Arquitetura concreta de LM/SSM | Família de LMs causais derivada do DRM |
| Estado persistente | Elemento da fibra efetiva transportado por um caminho híbrido | Matriz sináptica \(\sigma\) ou estado comprimido \(\rho\) por camada | Vetor latente, memória seletiva e, no ASM-CM, matriz *fast-weight* limitada |
| Mecanismo de memória | Holonomia, ordem do caminho, colapso e reativação de dimensão | Escrita hebbiana/atenção linear no estado sináptico | Recorrência causal, forget/write e memória associativa consolidada |
| Atenção | Não é requisito da teoria | Sim: atenção linear é mecanismo central | Sem self-attention sobre o prefixo; ASM-C2 usa endereçamento sobre slots fixos e ASM-CM usa *fast weights* |
| Dimensão | Rank efetivo varia com o estado | \(n\) e \(d\) são fixos para o modelo | Largura fixa; gates mudam atividade, não o rank formal da métrica |
| Geometria explícita | Essencial: métrica, anchor, conexão, curvatura, transporte | Não há métrica DRM, anchor ou conexão geométrica | Métrica aprendida e naturalização existem em ASM-X/ASM-R, mas não realizam o formalismo completo |
| Transições | Mapas \(J:r\to s\), inclusive não invertíveis | Atualizações suaves/discretas de estado em espaço de dimensão fixa | Atualizações neurais em espaço de largura fixa |
| Irreversibilidade | Teorema de perda de rank; sem memória externa, a informação não retorna | Pode esquecer por damping, interferência ou saturação; não há teorema análogo de rank | Forget/write e capacidade limitada geram esquecimento operacional; sem holonomia formal implementada |
| Dependência da ordem | Não comutatividade de transportes e transições é explícita | Ordem dos tokens altera cumulativamente \(\rho\) | Ordem dos tokens altera a trajetória e a memória recorrente |
| Positividade/esparsidade | A métrica é PSD; não exige ativações positivas | Ativações são positivas e cerca de 5% ficam ativas nos experimentos relatados | Estados e velocidades não são, em geral, positivos ou esparsos por construção |
| Interpretação biológica | Possível aplicação, mas sem reivindicação empírica específica | Reivindicação forte de equivalência funcional com neurônios, sinapses, Hebb e excitação/inibição | Motivação geométrica/cognitiva; não é apresentado como modelo cerebral biologicamente plausível |
| Paralelismo | Não especifica um algoritmo neural | Treino paralelo em tokens; inferência recorrente por estado linear | Recorrência original estrita e aproximações block-cumsum/scan causal |
| Estado versus contexto | Abstração independente de contexto de LM | Estado fixo em comprimento, mas grande: \(n d\) escalares por camada | ASM-CM demonstra estado retido fixo em comprimento e compacto no protocolo publicado |
| Evidência | Teoremas, exemplos e reduções; sem benchmark de LM | Scaling de 25M a 800M na tabela experimental, tradução/linguagem, sparsity e monosemanticidade | Testes, ablações e benchmarks internos; sem comparação válida direta com BDH |

## 3. Semelhanças reais

### 3.1 Estado como objeto principal

Ambos rejeitam a ideia de que inferência seja apenas uma transformação estática do token atual. O histórico modifica um estado persistente:

- no DRM, o caminho transforma a fibra efetiva por transporte ordenado;
- no BDH, tokens escrevem no estado sináptico \(\rho\) ou \(\sigma\);
- no ASM, tokens deslocam \(z_t\) e modificam memórias recorrentes.

### 3.2 Relações em vez de armazenamento literal do prefixo

BDH guarda correlações neuronais. ASM comprime o prefixo em estado e memória limitada. DRM descreve relações e possibilidades efetivas. Em todos os casos, a memória pretende ser **estrutural**, não uma simples lista imutável dos tokens anteriores.

### 3.3 Ordem importa

O produto \(P_mJ_{m-1}\cdots J_1P_1\) do DRM, a soma ordenada de escritas em \(\rho\) no BDH e as recorrências do ASM são sensíveis à sequência de eventos. Porém, só o DRM v7 transforma essa propriedade em invariantes geométricos explícitos de loops e comutadores.

### 3.4 Memória rápida e memória externa

DRM prova que uma expansão determinística não recupera informação descartada após uma redução genuína de rank; seria necessário ampliar o estado com um canal auxiliar. BDH faz exatamente da sinapse dinâmica um canal de estado rápido. ASM-CM também separa estado causal e memória associativa. Essa é a ponte conceitual mais fértil entre os trabalhos.

## 4. Diferenças decisivas

### 4.1 Rank variável contra atividade esparsa

No DRM, diminuir dimensão significa mudar \(\operatorname{rank}g\) e formar outra fibra quociente. No BDH, menos neurônios ativos não muda a dimensão matemática do estado. No ASM, fechar gates também não muda o rank exato da métrica SPD.

Portanto:

> esparsidade de ativação, número de gates ativos e rank geométrico formal são observáveis diferentes.

Chamar a sparsity do BDH ou `dimD` do ASM de “transição de rank DRM” sem uma construção adicional seria incorreto.

### 4.2 Memória por escrita versus memória por perda

BDH lembra porque **escreve correlações** no estado sináptico. DRM lembra porque um caminho pode **destruir direções e impedir a reconstrução**. O primeiro é um mecanismo de armazenamento; o segundo é um invariante de transporte irreversível. ASM-CM está mais próximo do BDH nesse eixo, pois sua memória *fast-weight* também é escrita e lida associativamente.

### 4.3 Tamanho do estado

BDH-GPU deliberadamente mantém estado da mesma ordem dos parâmetros: \(n\times d\) por camada. Isso favorece capacidade associativa e localização por neurônio, mas pode consumir muita memória. Por exemplo, a configuração tabelada de 800M usa \(n=1{.}048{.}576\), \(d=256\) e oito camadas; a contagem bruta seria cerca de 2,15 bilhões de escalares de estado antes de otimizações de precisão, sparsity ou sharding.

ASM-CM toma a direção oposta: o benchmark do repositório mede **143.360 bytes** de estado retido, invariantes entre 4K e 32K tokens, em uma RTX 4090. Esses números não permitem declarar vitória: modelos, dados, precisão, qualidade e protocolos são diferentes. Eles mostram duas prioridades arquiteturais distintas — **capacidade sináptica ampla** no BDH contra **compactação controlada** no ASM-CM.

### 4.4 Atenção

BDH-GPU é explicitamente um modelo de atenção linear em alta dimensão. ASM se apresenta como livre de self-attention/QKV sobre tokens e usa um estado causal. Logo, ambos evitam o custo de uma KV-cache que cresce com todo o prefixo, mas por mecanismos diferentes.

### 4.5 Grau de validação

DRM v7 estabelece resultados matemáticos condicionais. BDH apresenta experimentos de scaling, tradução, sparsity, sinapses monosemânticas e composição por concatenação. O ASM apresenta ablações e benchmarks próprios, mas também registra limitações importantes:

- as antigas comparações com GPT-2 foram retraídas por double shift dos targets;
- no controle pareado de 100M tokens/seed 1, o Transformer versionado registra CE 1,120721 e 69.367 tok/s, contra CE 1,344849 e 16.573 tok/s do ASM-R; esse controle não é BDH e não mede ASM-CM em 32K;
- o Transformer pareado continua com CE geral menor que ASM-CM;
- não há benchmark corrigido e controlado entre ASM e BDH;
- o DRM formal ainda não foi realizado integralmente no código.

## 5. Leitura crítica da evidência

### BDH é mais forte hoje quando a pergunta é

- “Existe uma arquitetura treinável e escalada?”
- “Como implementar atenção linear como estado sináptico?”
- “Há evidência de sparsity, localização conceitual e scaling semelhante ao GPT?”
- “Como relacionar uma LM a dinâmica local de grafo?”

O paper testa 25M–800M parâmetros na tabela de scaling, com Europarl En–PL/En–Cs, bytes UTF-8, 1,2 bilhão de tokens e GPTXL como controle. A curva descrita como igualando melhor o GPTXL é a **BDH-GPU'**, que acrescenta gating tipo xLSTM e combina predições de várias camadas; isso deve ser separado da formulação vanilla. Pela leitura visual da Figura 7, a BDH-GPU vanilla fica pior que GPTXL nas escalas mostradas, enquanto BDH-GPU' acompanha ou supera levemente o controle; como o paper não tabula essas curvas, não se devem transformar estimativas visuais em resultados numéricos oficiais. A evidência é concreta, mas concentrada em um corpus, usa um baseline GPT-2/TransformerXL e não inclui LMs modernos, benchmarks amplos de reasoning ou uma tabela numérica completa das curvas. As conclusões biológicas continuam sendo uma interpretação funcional plausível, não uma validação neurocientífica direta.

### DRM é mais forte quando a pergunta é

- “O que significa perder e reativar graus de liberdade?”
- “Quando uma volta ao mesmo estado-base deixa memória irreversível?”
- “Como separar curvatura regular de memória causada por transições?”
- “Como medir rank defect, energy defect e não comutatividade?”

Seus limites também são claros: os mapas \(J\) são dados prescritos; transições contínuas, regras canônicas para \(J\), estimação experimental e aplicações cognitivas permanecem abertas.

### ASM ocupa uma posição intermediária

O repositório transforma a intuição de trajetória relacional em modelos causais executáveis, mas os melhores resultados levaram a arquitetura promovida para longe do ASM-X mais fiel ao DRM. ASM-R remove direções explícitas; ASM-CM acrescenta memória *fast-weight*. Esse resultado aproxima o mecanismo prático do ASM-CM do tema central de estado associativo do BDH, ainda que sem a ativação positiva, a dimensão neuronal maciça e a equivalência de grafo proposta pelo BDH.

## 6. Como os modelos podem se complementar

### 6.1 Usar DRM para analisar BDH

Uma linha de trabalho plausível seria definir, para o BDH:

1. um espaço-base de estados observáveis do modelo;
2. fibras formadas por subespaços sinápticos/neurais efetivamente ativos;
3. uma métrica estimada por Fisher, covariância de ativações ou Hessiana local;
4. estratos definidos por rank efetivo robusto, e não apenas por contagem de ReLUs não nulas;
5. mapas de transição estimados quando o suporte ativo muda;
6. loops de prompts que retornam ao mesmo marcador de contexto para medir \(H_\gamma\), rank defect e comutadores.

Isso daria ao BDH diagnósticos formais de histerese que o paper ainda não possui. O estado `σ` do BDH é dirigido e, em geral, não é simétrico nem PSD; `ρ` é retangular e pode ter sinais. Nenhum dos dois pode ser identificado diretamente com a métrica `g` do DRM. Seria preciso construir e justificar um tensor PSD — por exemplo Fisher ou Gram — e cada escolha mudaria o significado do rank. ReLU também torna estratos por suporte apenas *piecewise smooth*, em tensão com as hipóteses regulares de DRM v7.

Além disso, memória recorrente não é automaticamente histerese DRM: é preciso definir um loop fechado na base e medir transporte final não identidade. Outer products aditivos isolados podem comutar; a ordem no BDH surge das ativações dependentes da história, damping/RoPE e não linearidades. Para ser DRM genuíno, seria necessário demonstrar fibras quociente e transições de rank, não apenas sparsity.

### 6.2 Usar BDH como kernel operacional inspirado pelo DRM

BDH oferece algo que DRM v7 deliberadamente não define: uma dinâmica local treinável capaz de gerar os mapas observados. Seria possível tratar a regra hebbiana e a leitura linear como candidatas para produzir empiricamente \(J_e\). Depois, os invariantes DRM avaliariam se a dinâmica resultante é dissipativa, expansiva, reversível ou dependente da ordem.

### 6.3 Possível evolução do ASM

Para aproximar ASM e BDH sem perder a identidade do projeto:

- testar ativações positivas e esparsas em uma variante isolada;
- comparar memória *fast-weight* compacta com estado \(n\times d\) amplo;
- adicionar diagnóstico de suporte/rank efetivo com limiares e intervalos de confiança;
- implementar mapas de transição explícitos e ciclos de holonomia antes de reivindicar DRM formal;
- preservar ablações para separar ganhos de métrica, memória, sparsity e capacidade.

## 7. Experimento justo recomendado

Não existe base para ordenar os modelos usando os números atuais. Um comparativo direto deve fixar:

- mesmo corpus e split;
- mesmo tokenizer;
- mesmos tokens de treino;
- faixas pareadas de parâmetros;
- mesma precisão e hardware;
- mesma regra de validação contínua;
- implementação otimizada ou, no mínimo, nível de otimização documentado.

Métricas mínimas:

1. CE/perplexidade e scaling por tokens/parâmetros;
2. tokens/s, FLOPs aproximados e tempo para atingir qualidade;
3. bytes de estado recorrente e VRAM em 512, 4K, 32K e além;
4. MQAR e recuperação associativa com distância crescente;
5. extrapolação temporal além do comprimento de treino;
6. interferência e sobrescrita de memória;
7. sparsity e estabilidade dos conceitos localizados;
8. ciclos A→B contra B→A para medir não comutatividade e histerese;
9. ablação de atenção/métrica/memória com orçamento de parâmetros e compute controlados.

## 8. Veredito

- **Como teoria matemática:** DRM v7 é mais geral e fornece resultados de irreversibilidade que BDH não tem.
- **Como arquitetura de linguagem pronta para investigação:** BDH-GPU é mais especificado, mais escalado e mais diretamente reproduzível a partir do paper.
- **Como software deste repositório:** ASM é uma família experimental real, mas não é ainda a implementação integral do DRM v7.
- **Como mecanismo de memória:** BDH e ASM-CM estão mais próximos entre si do que BDH e DRM formal; ambos usam estado associativo escrito durante a sequência.
- **Como eficiência de estado:** ASM-CM prioriza compactação; BDH prioriza um estado sináptico muito amplo. A melhor escolha depende da curva qualidade/capacidade/memória em protocolo comum.
- **Como agenda de pesquisa:** a combinação mais promissora é BDH como dinâmica neural concreta, DRM como instrumento de análise de rank, transporte e histerese, e ASM como bancada de ablação entre geometria e memória.

A conclusão responsável não é que um “vence” o outro. É que eles atacam camadas diferentes do mesmo problema: **BDH explica como escrever e consultar um estado relacional amplo; DRM explica como a estrutura de possibilidades pode mudar e deixar memória irreversível; ASM testa quais dessas ideias sobrevivem em um modelo causal compacto**.

## Fontes internas consultadas

- [`paper/drm_v7.tex`](paper/drm_v7.tex)
- [`paper/others/2509.26507v1.pdf`](paper/others/2509.26507v1.pdf)
- [`../README.md`](../README.md)
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
- [`MODEL_FAMILY.md`](MODEL_FAMILY.md)
- [`drm_philosophy.md`](drm_philosophy.md)
- [`limitations.md`](limitations.md)
- [`benchmarks/asm_cm_post_fp32/README.md`](benchmarks/asm_cm_post_fp32/README.md)
