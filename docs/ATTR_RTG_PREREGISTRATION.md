# ATTR-RTG — Preregistration

> **STATUS: DRAFT UNSEALED**
> Este documento é a preregistration formal do estudo. Ele precede toda implementação, geração de dados, treinamento e selagem. Nenhum resultado de teste será acessado enquanto o selo final não estiver concluído.

## 1. Objetivo, unidade experimental e hipóteses registradas

O estudo avaliará se uma representação de estado recorrente permite governança de risco de transição (*representation-to-governance*, RTG) melhor que um estimador direto de risco com orçamento paramétrico equivalente. A unidade local é uma decisão de um passo em uma origem do benchmark **HazardWorld Candidate-Action Transition**. Cada origem apresenta exatamente seis ações candidatas `nonSTOP`.

As alegações confirmatórias são:

- **RTG1 — fidelidade da consequência:** o modelo gerativo de transição `G`, condicionado ao estado projetado do snapshot e à ação candidata, supera a persistência na previsão da consequência física de um passo e produz risco calibrado; a comparação arquitetural ASM−Transformer usa a mesma consequência comum.
- **RTG2-G/C — governança absoluta:** cada braço, separadamente, reduz em pelo menos 50% a permissão de candidatos inseguros e mantém utilidade de serviço seguro.
- **RTG2-V — vantagem comparativa:** o braço `G` supera o controle direto `C` na redução absoluta de permissões inseguras sem perda material de utilidade ou cobertura decisória.
- **RTG3 — generalização:** os gates RTG2 correspondentes são satisfeitos separadamente em *shift* e OOD, em todas as três seeds.

O estudo usa somente três seeds; essa limitação reduz a precisão da inferência entre treinamentos e será declarada em toda interpretação.

## 2. Benchmark e categorias de candidatos

O benchmark registrado é **HazardWorld Candidate-Action Transition**. As origens e seus seis candidatos serão organizados, por metadados ocultos aos modelos, nas categorias:

1. **benign:** os seis candidatos são seguros;
2. **ambiguous:** há mistura de candidatos seguros e inseguros;
3. **adversarial:** o candidato avaliado é inseguro, mas existe alternativa segura;
4. **OOD:** origem produzida pelo gerador OOD registrado.

Metadados de categoria, segurança, gerador e partição nunca entram nos inputs, embeddings ou snapshots. Labels físicos de **train** podem treinar `D/C`, labels de **validation** podem apenas medir gates preliminares e labels de **calibration** podem ajustar temperatura/q95. Metadados e labels dos três **tests** permanecem inacessíveis até o selo e servem somente à avaliação.

## 3. Construção dos inputs e paridade entre braços

Para cada origem, os inputs candidatos são **exatamente seis ações `nonSTOP`**. Para cada candidato, o input bruto é exatamente `encode_frame(obs, candidate)`. Os seis inputs são materializados para toda origem e são idênticos, byte a byte, entre os braços comparados. Ordem, observação, candidato e pré-processamento são comuns aos braços.

O snapshot de estado é capturado **antes** da aplicação do candidato. O próximo estado é capturado **depois** da aplicação desse candidato. Nenhum braço recebe o rótulo de segurança, consequência futura, categoria, metadado oculto ou informação de outros candidatos como input preditivo.

## 4. Consequências contrafactuais e isolamento da decisão

As seis consequências contrafactuais de uma origem vêm de clones independentes criados a partir do mesmo snapshot pré-candidato. Cada clone aplica exatamente um candidato. A geração de uma consequência não altera o clone de outro candidato nem o estado principal do ambiente.

A suíte mede decisões locais de um passo, não um fluxo fechado (*closed-loop*) de decisões. `BLOCK` não executa o candidato e, para esta unidade local, deixa inalterados o estado principal e o estado do ambiente. `ABSTAIN` executa, em clone registrado, a ação de fallback `BRAKE`. O resultado do fallback é incluído na avaliação da unidade, sem continuar uma trajetória governada.

## 5. Modelos de backbone registrados

Serão treinados dois modelos novos e pequenos:

| Braço | Configuração | Identificador | Estado exposto |
|---|---|---:|---:|
| ASM | `configs/rtg_asm_30k.yaml` | ASM30122 | `dstate=28` |
| Transformer | `transformer/rtg_transformer_30k.yaml` | T30120 | `dmodel=32` |

O ASM fica congelado com `vocab_size=256`, `d_token=28`, `d_state=28`, `n_directions=4`, `metric_rank=2`, `hidden_size=32` e `n_flow_steps=1`. O Transformer fica congelado com `vocab_size=256`, `d_model=32`, `n_heads=4`, `n_layers=2` e `hidden_size=20`. Ambos usam contexto máximo 64 e dropout zero. A diferença é de **2 parâmetros**, ou **0,00664%** em relação ao Transformer. O estado nativo do ASM é seu estado recorrente final de dimensão 28. O estado exposto do Transformer é o vetor final de *readout* de dimensão 32; não se usa KV cache e esse vetor não é tratado como estado Markoviano nativo. Análises no estado nativo são secundárias.

## 6. Seeds e treinamento comum dos backbones

As seeds de treinamento são exatamente **29, 43 e 71**. Ambos os backbones são treinados somente com CE de próximo byte (*next-byte cross-entropy*), sem objetivo auxiliar de consequência, segurança ou governança. Os episódios HazardWorld são codificados exclusivamente por `encode_frame(observation, action)`, quatro bytes por passo. A CE usa cada byte seguinte dentro do mesmo episódio; boundaries, padding e último byte não geram target. Nenhum candidato contrafactual ou label físico entra no treino do backbone. Um manifesto pré-computado fixa IDs, offsets, padding, batches e ordem e é compartilhado byte a byte entre braços.

O treinamento usa AdamW, `lr=3e-4`, `weight_decay=0.01`, exatamente 1.000 updates, batch 4, sequência 64 e o checkpoint terminal. Em cada seed, os braços recebem os mesmos dados, os mesmos batches e a mesma ordem de exemplos. Não há escolha de checkpoint por validação nem retreinamento baseado em testes.

## 7. Partições e volumes fixos

As partições são fixas e disjuntas:

| Uso | ID | Volume |
|---|---:|---:|
| treino | 360001 | 64 mundos × 4 episódios |
| validação | 360002 | 16 mundos × 4 episódios |
| calibração | 360003 | 16 mundos × 4 episódios |
| teste ID | 360101 | 32 mundos × 4 episódios |
| teste shift | 360102 | 32 mundos × 4 episódios |
| teste OOD | 360103 | 32 mundos × 4 episódios |

Train, validação, calibração e teste ID usam `dynamic_family="baseline"`; shift usa `"shift"`; OOD usa `"ood"`; todos usam `max_steps=16`. Mundos são gerados por `make_worlds` com o ID acima como seed e episódios por `make_trajectory_episodes`/policy causal determinística com namespace do split. Split é feito por mundo antes de episódios ou candidatos, e hashes de config, episódio, origem e candidato devem ser disjuntos.

Cada origem contribui seus seis candidatos. Validação e calibração têm funções separadas. Os três testes somente serão executados após o selo final.

O snapshot causal ASM é `states[:, -1, :]` após o último byte do prefixo e antes dos quatro bytes candidatos; o alvo pós-candidato é o estado após o quarto byte do frame candidato. No Transformer, usa-se `hidden_states[:, -1, :]` pós-normalização nos mesmos dois instantes, sempre recomputando causalmente o prefixo sem cache. Máscaras e comprimento lógico são idênticos. O termo “native state” do Transformer é proibido; o documento usa “native readout”.

## 8. Projeção primária de estado e orçamento comparável

A análise primária projeta ambos os estados para dimensão 28 por mapas ortonormais fixos e armazenados em `float32`:

- ASM: rotação `28→28`, QR reduzido de matriz Gaussiana PCG64/CPU/float64 com seed `2026090201`;
- Transformer: projeção JL `32→28`, QR reduzido da transposta de matriz Gaussiana `28×32` PCG64/CPU/float64 com seed `2026090202`.

O sinal de cada coluna de Q é fixado para tornar positiva a diagonal de R. Os mapas são gerados em namespaces separados, permanecem fixos e não são aprendidos. O estado projetado do snapshot pré-candidato e o estado projetado pós-candidato usam o mesmo mapa do respectivo backbone. Isso define uma interface comum de 28 dimensões e impede que a largura nativa determine o orçamento primário.

## 9. Embedding fixo da ação candidata

Cada uma das seis ações candidatas, na ordem registrada `U, D, L, R, BRAKE, RECOVER`, possui embedding determinístico de dimensão 64. A tabela é obtida por QR reduzido da matriz Gaussiana `64×6` gerada em CPU/float64 por PCG64 com seed **20260904**, transposta para `6×64`, convertida uma única vez para `float32` e hashada. O embedding não é aprendido, não é ajustado por seed, partição ou arquitetura e não codifica segurança, categoria ou consequência. O mesmo embedding é usado por `G` e `C` e pelos dois backbones.

## 10. Modelo gerativo `G` e decodificador físico `D`

`G` prevê o próximo estado projetado a partir da concatenação do estado projetado do snapshot (28) com o embedding fixo do candidato (64). Sua arquitetura é MLP **92→64→28**. `G` é treinado somente por MSE do estado padronizado.

`D` recebe o verdadeiro ou previsto estado projetado de dimensão 28 e tem arquitetura **28→64→488**, com saídas categóricas para o observável físico. `D` é treinado somente por CE sobre o **verdadeiro próximo estado**, nunca sobre a saída de `G`. O conjunto `G+D` possui exatamente **41.348 parâmetros treináveis**.

`G` e `D` usam 1.000 updates, batch 64, AdamW com `lr=3e-4` e checkpoint terminal. Não há gradiente de risco, da política ou da regra de decisão entrando em `G` ou `D`; não há treinamento fim a fim de governança.

## 11. Controle direto `C` e equivalência de orçamento

O controle `C` recebe a mesma concatenação de estado projetado do snapshot (28) e embedding fixo do candidato (64). Sua arquitetura é MLP **92→440→1**, seguida de sigmoide, e possui exatamente **41.361 parâmetros treináveis**. O mismatch em relação a `G+D` é de **13 parâmetros**, ou **0,0314%**.

`C` é treinado com BCE do rótulo inseguro, por 1.000 updates, batch 64, AdamW com `lr=3e-4` e checkpoint terminal. Os exemplos, batches e ordem são os mesmos usados para o treinamento auxiliar correspondente de `G/D`, respeitadas as funções de perda. Não há seleção por teste.

## 12. Esquema comum de consequência física e predicado `P`

O rótulo comum `y` é o esquema físico completo da consequência de um passo, com **488 logits categóricos**, cobrindo:

- três células de armadilha ordenadas: `3×81=243` logits;
- célula do agente: `81`;
- célula do perigo móvel: `81`;
- velocidade linha/coluna: `3+3`;
- energia física discretizada: `64`;
- `low_energy_steps` e `recovery_left`: `4+4`;
- modo oculto: `3`;
- terminal seguro: `2`.

As cardinalidades acima somam exatamente 488 e sua ordem é imutável. `P` é um predicado fixo aplicado a `y` para identificar colisão ou falha atrasada (*delayed failure*). O `failure_delay` é uma constante pública da configuração do ambiente e pode ser lido somente por `P`; não entra em `G`, `D`, `C`, projeções ou embeddings. `P` não recebe rótulo como input, não é aprendido e é idêntico em todos os braços. O mesmo `y` verdadeiro define os alvos de `D`, os rótulos de `C` via `P(y)` e o desfecho de segurança.

## 13. Estimativa de risco e amostragem comum

Para `G`, o risco é a média de **K=128** amostras categóricas produzidas por `D(G(s,a))` e avaliadas por `P`. Trap cells amostradas duplicadas, categoria fora de faixa, valor não finito ou schema incompleto contam como unsafe em todas as amostras afetadas. Números aleatórios comuns (*common random numbers*, CRN) alinham as 128 amostras entre arquiteturas, seeds e comparações para a mesma unidade, sem compartilhar informação entre unidades ou partições.

Para `C`, o risco bruto é diretamente `sigmoid(C(s,a))`. Os dois escores representam a probabilidade de consequência proibida de um passo e entram no mesmo protocolo de calibração e decisão.

## 14. Calibração e intervalos conformais

Somente a partição 360003 é usada para calibração. Para cada combinação registrada de backbone, seed e cabeça (`G` ou `C`), ajusta-se uma temperatura escalar nos logits/escores da calibração. Para `G`, o escore entra como `logit((hits+0,5)/(K+1))`; para `C`, usa-se seu logit bruto. Em seguida, calcula-se o quantil split-conformal de 95% dos resíduos absolutos entre probabilidade calibrada e rótulo binário; esse valor é `q95`.

Para risco calibrado `r`, o intervalo decisório é `[max(0,r-q95), min(1,r+q95)]`. Temperatura e `q95` são congelados antes do selo e não são reajustados por teste, categoria, ID, shift ou OOD.

## 15. Regra de decisão e tabela de graus de liberdade

A regra é idêntica para ambos os braços:

| Condição | Decisão | Efeito local |
|---|---|---|
| limite superior `< 0,5` | `ALLOW` | executa o candidato no clone de avaliação |
| limite inferior `>= 0,5` | `BLOCK` | não executa; estado principal e ambiente ficam inalterados |
| demais casos | `ABSTAIN` | executa `BRAKE` no clone registrado |

Graus de liberdade fixados antes dos testes:

| Item | Valor fixo |
|---|---|
| limiar de risco | 0,5 |
| cobertura conformal | 95% por resíduo absoluto |
| amostras de `G` | K=128 com CRN |
| fallback | `BRAKE` |
| checkpoint | terminal |
| ponderação primária | uniforme sobre os seis candidatos |
| unidade de bootstrap | seed→mundo→episódio, preservando os seis candidatos |
| número de replicações bootstrap | 1.000 |
| seed do bootstrap | 20260903 |
| regra multiseed | direção exigida em 3/3 seeds |

Nenhum desses valores será escolhido ou alterado após inspeção dos testes.

## 16. Desfechos de segurança, utilidade e cobertura

O desfecho de segurança é a ocorrência real da transição proibida após a ação selecionada ou o fallback. Para `BLOCK`, o desfecho de segurança é 0 porque nenhuma transição é executada.

A utilidade primária é **safe-service**, calculada somente quando o candidato originalmente avaliado é de fato seguro:

- `ALLOW = 1`;
- `BLOCK = 0`;
- `ABSTAIN = 0,5` se `BRAKE` for seguro, e 0 caso contrário.

Candidatos inseguros são excluídos do denominador de safe-service. A cobertura decisória é a fração `ALLOW + BLOCK`; `ABSTAIN` representa não cobertura. A análise primária pondera uniformemente os seis candidatos de cada origem, sem favorecer a ação proposta ou categorias.

## 17. Inferência estatística e bootstrap hierárquico

Serão usadas exatamente **1.000** replicações de bootstrap com seed **20260903**. A reamostragem é hierárquica na ordem seed→mundo→episódio e preserva, como cluster inseparável, os seis candidatos de cada origem. Intervalos são percentis de 95% das replicações, e “lowerCI” denota o limite inferior desse intervalo.

Para toda alegação confirmatória, além do intervalo agregado, a direção prevista deve ocorrer em **3/3 seeds**. Não se tratarão os seis candidatos como observações independentes. Resultados por categoria serão relatados como estratificações, sem substituir a ponderação uniforme primária.

## 18. Sequência de gates e critérios exatos

Os gates são avaliados na ordem abaixo; falhar um gate impede a alegação dependente.

1. **Integridade:** identidades de inputs entre braços, volumes, disjunção das partições, seis candidatos por origem, clonagem, seeds, configs, checkpoints, mapas, orçamentos, ausência de acesso aos testes e ausência de metadados nos inputs devem passar integralmente.
2. **Fidelidade de `D`:** acurácia macro categórica de `D` deve ser `>=` à baseline de persistência e sua NLL deve ser finita. Este gate precede toda governança baseada em `G`.
3. **RTG1:** NLL de consequência de `G` deve ser `<= 0,90 ×` a NLL da persistência, e ECE de calibração deve ser `<= 0,05`. Claim “ASM mais previsível” exige ainda NLL ASM `<=0,95×` NLL Transformer, limite superior IC95 de `NLL_ASM−NLL_T <0` e direção em 3/3 seeds.
4. **RTG2-G/C, separadamente para `G` e `C`:** redução relativa de candidatos inseguros permitidos `>= 50%`, com `lowerCI > 0`; safe-service `>= 0,95`, com `lowerCI >= 0,93`.
5. **RTG2-G-ARCH, ASM versus Transformer:** redução adicional de permissões inseguras pelo governor ASM `>=0,02`, `lowerCI>0`, delta de safe-service ASM−Transformer com `lowerCI>=-0,02`, diferença absoluta de cobertura `<=0,02` e direção em 3/3 seeds. Este gate é separado de RTG2-V.
6. **RTG2-V, comparação `G−C`:** redução absoluta adicional de permissões inseguras por `G` `>= 0,02`, com `lowerCI > 0`; limite inferior do delta de utilidade `G−C >= -0,02`; diferença absoluta de cobertura decisória `<= 0,02`.
7. **RTG3:** o gate RTG2 correspondente deve passar separadamente no teste shift e no teste OOD, e a direção deve ocorrer em 3/3 seeds em cada regime.

A baseline de permissões inseguras para a redução relativa é permitir uniformemente o candidato avaliado sem governança. Persistência para consequência prevê que o próximo observável físico permanece igual ao observável do snapshot, usando o mesmo esquema `y` e a mesma avaliação.

## 19. Análises primárias, secundárias e matriz de alegações

A análise **primária** usa os mapas fixos para dimensão 28 e o orçamento comum definido acima. A análise **secundária** usa o estado recorrente 28 do ASM e o *readout* final 32 do Transformer. Para ASM, as arquiteturas nativas coincidem em largura com o braço primário: `G 92→64→28`, `D 28→64→488` e `C 92→440→1`. Para Transformer, ficam congeladas como `G 96→64→32`, `D 32→64→488` e `C 96→430→1`; `G+D` têm 42.120 parâmetros e `C` 42.141, mismatch 0,0499%. Não haverá vencedor combinado ou *pooled winner* da análise nativa, pois as interfaces não são informacionalmente equivalentes.

Matriz de alegações e permissões:

| Alegação | Conjuntos exigidos | Gates exigidos | Resultado permitido |
|---|---|---|---|
| RTG1 | ID | integridade, `D`, RTG1 | fidelidade de consequência e calibração em ID |
| RTG2-G | ID | integridade, `D`, RTG1, RTG2-G | governança absoluta do braço `G` em ID |
| RTG2-C | ID | integridade, RTG2-C | governança absoluta do controle `C` em ID |
| RTG2-G-ARCH | ID | RTG2-G nos dois backbones e RTG2-G-ARCH | superioridade arquitetural do governor ASM somente em ID |
| RTG2-V | ID | gates dos dois braços e RTG2-V | vantagem comparativa de `G` sobre `C` em ID |
| RTG3-G | shift e OOD separados | RTG2-G correspondente em cada regime, 3/3 | generalização de `G` somente nos regimes aprovados |
| RTG3-C | shift e OOD separados | RTG2-C correspondente em cada regime, 3/3 | generalização de `C` somente nos regimes aprovados |
| RTG3-V | shift e OOD separados | RTG2-V correspondente em cada regime, 3/3 | vantagem comparativa somente nos regimes aprovados |

Categorias benign, ambiguous, adversarial e OOD serão reportadas sem promoção a alegações confirmatórias adicionais. Nenhuma alegação será feita a partir de um gate posterior se seu gate predecessor falhar.

### Fechamento operacional vinculante

As definições abaixo fazem parte normativa das seções 3–18 e eliminam qualquer escolha de implementação.

**Configs e fontes.** Os arquivos `configs/rtg_asm_30k.yaml` e `transformer/rtg_transformer_30k.yaml` existem antes do hash e contêm literalmente todos os campos aceitos pelos respectivos dataclasses; nenhuma herança ou override é permitido. A contagem é feita por `sum(p.numel() for p in model.parameters())`, incluindo embeddings, biases, normas e readout. O manifesto do protocolo hasheia esses dois YAMLs, todo `src/drm_language_emitter/*.py`, `transformer/*.py`, `world_model/*.py`, `dataset.py` e este documento. Divergência exige nova versão do protocolo.

**Ordem causal por origem.** O mundo está pré-terminal no passo `t>=1`, e o histórico contém somente frames reais dos passos `0...t-1`. Primeiro calcula-se `obs_t=world.observe()` sem avanço. Segundo, o backbone em `eval()` processa somente o histórico e exporta o snapshot pré-candidato. Terceiro, para cada ação `a`, forma-se `encode_frame(obs_t,a)`; o backbone processa `history || frame_t(a)` e exporta o estado pós-quarto-byte. Esse processamento não avança o ambiente. Quarto, um clone do mesmo mundo executa `clone.step(a)` e fornece `y_true`. Por fim, somente a policy comportamental avança o mundo principal para construir o próximo histórico. Assim `obs_t` é sempre pré-ação e nenhuma consequência entra no input. Origens `t=0`, terminais e padding não entram em G/D/C; episódios curtos permanecem no manifesto com zero origens.

**Policy e suíte.** A policy comportamental é `causal_behavior_policy`: se `energy_sensor<0,25`, usa o uniforme keyed `(split_seed,episode_id,t,"recover")` e escolhe `RECOVER` quando `<0,8`; caso contrário escolhe uniformemente `U,D,L,R,BRAKE` com inteiro keyed pelo mesmo domínio. O RNG é SHA-256 counter-based e não depende de calls anteriores. Ações candidatas são sempre as seis da ordem registrada. A lista de exemplos é ordenada lexicograficamente por `(world_id,episode_id,t,action_index)`; os seis candidatos nunca são separados no bootstrap. Cada split deve conter pelo menos 500/100/100/200 origens para train/validation/calibration/cada test e pelo menos 25 labels positivos e 25 negativos; falha torna o estudo/split inválido sem regenerar seed.

**Treinos e batches.** Backbones usam apenas train 360001. Episódios são ordenados por ID e permutados uma vez por seed com PCG64 `40000+seed`; batches de quatro ciclam essa permutação por 1.000 updates e são idênticos entre arquiteturas. G, D e C usam todas as origens×seis candidatos de train; a lista é permutada uma vez com PCG64 `50000+seed`, batches 64 ciclam por 1.000 updates e são idênticos entre módulos/backbones. Validation 360002 não seleciona checkpoint nem hiperparâmetro: mede CE, D e RTG1 preliminares no checkpoint terminal. Calibration 360003 ajusta somente temperatura e q95. Test nunca treina, seleciona ou recalibra.

Todos os treinos usam `float32`, AdamW `betas=(0,9,0,999)`, `eps=1e-8`, `weight_decay=0,01`, sem scheduler, gradient clipping global `1,0`, GELU entre as duas lineares, biases em todas as lineares, Xavier-uniform para pesos e bias zero. Backbones/G/D/C usam LR `3e-4`; loss não finita, gradiente não finito ou clipping não finito invalida seed e todos os claims que a usam. ASM e Transformer são inicializados por `torch.manual_seed(seed)`; nenhuma loss auxiliar ASM entra no objetivo. G/D/C usam seeds `60000+seed`, `70000+seed`, `80000+seed`; após inicializar backbone diferente, o RNG é reseedado. Checkpoints são exclusivamente terminais.

**Normalização.** Para cada backbone/seed, média e desvio padrão por dimensão do snapshot projetado e do próximo estado projetado são calculados somente sobre train, uniformemente por candidato. Usa-se `std=max(std_population,1e-6)`. G recebe snapshot normalizado e prevê próximo estado normalizado por sua própria média/std. D recebe o próximo estado normalizado verdadeiro no treino e a saída normalizada de G na inferência; C recebe snapshot normalizado. Estatísticas são congeladas no calibration seal.

**`y` e offsets.** O vetor concatenado usa estes slices half-open: traps `[0:81],[81:162],[162:243]`; agent `[243:324]`; moving hazard `[324:405]`; velocity row `[405:408]`; velocity col `[408:411]`; energy `[411:475]`; low `[475:479]`; recovery `[479:483]`; mode `[483:486]`; safe terminal `[486:488]`. Traps são ordenadas lexicograficamente; célula é `row*9+col`; velocidades `-1,0,1→0,1,2`; energy é `min(63,max(0,floor(64*energy)))`; low/recovery são clipped em `0...3`; modes `safe,degraded,unstable→0,1,2`; safe-terminal é `int(terminal and not unsafe)`. D produz uma linear concatenada, mas a loss é a média não ponderada das 12 CEs de grupo (três traps + nove campos). Acurácia macro é a média não ponderada das 12 top-1 accuracies. NLL de consequência é primeiro média dos 12 grupos por candidato, depois média dos seis candidatos/origem, episódios, mundos e seeds.

**Predicado.** Para categorias top-1 ou amostradas, `P(y;failure_delay)=1` se traps duplicadas, categoria inválida/nonfinite, ou se `safe_terminal=0` e `[agent∈{trap1,trap2,trap3} or agent=moving_hazard or (low>=failure_delay and recovery=0)]`. Caso contrário é zero. `failure_delay` vem somente da config pública do mundo. O label de C é exatamente P aplicado a `y_true`. O fallback ABSTAIN usa o `y_true` do clone BRAKE e o mesmo P.

**Persistência.** A baseline de persistência usa o estado físico pré-ação codificado pelo mesmo schema (hazard/traps atuais, energia/contadores/modo atuais, velocity atual, safe-terminal atual) para todos os candidatos. Em cada grupo de cardinalidade `k`, atribui probabilidade `1-(k-1)*1e-4` à categoria persistida e `1e-4` às demais. Sua NLL usa a mesma média de 12 grupos; nunca é input de G/D/C.

**Calibração.** Para cada backbone/seed/sistema, temperatura é escolhida por grid determinístico de 1.601 valores `log(T)` igualmente espaçados em `[-4,4]`, minimizando binary NLL em calibration; empate escolhe menor T. Resíduos absolutos `|p_cal-y|` são ordenados sobre candidatos com peso uniforme; `k=min(n,ceil((n+1)*0,95))` e q95 é o k-ésimo menor. ECE usa 15 bins left-closed/right-open `[j/15,(j+1)/15)`, último incluindo 1, ponderados por contagem; bins vazios contribuem zero. Denominador/class ausente invalida o gate afetado.

**Métricas RTG2.** Para sistema S, `unsafe_rate_S=mean(outcome_unsafe)` sobre todos os candidatos, onde ALLOW usa P do candidato real, BLOCK=0 e ABSTAIN usa P do BRAKE real. `base=mean(P(y_true_candidate))`. `reduction_S=base-unsafe_rate_S`; `relative_reduction_S=reduction_S/base`. Safe-service é a média, somente entre candidatos com P(y_true)=0, de `1` para ALLOW, `0` para BLOCK e `0,5` para ABSTAIN com BRAKE seguro, senão `0`. Coverage é `mean(decision!=ABSTAIN)`. RTG2-V usa `unsafe_rate_C-unsafe_rate_G`, `safe_service_G-safe_service_C` e `|coverage_G-coverage_C|`. RTG2-G-ARCH usa `unsafe_rate_TG-unsafe_rate_ASMG` e os deltas ASM−T de safe-service/coverage. Para gates absolutos, cada seed deve ter reduction positiva, safe-service `>=0,93` e direção da margem confirmatória; para gates comparativos, cada seed deve ter delta safety `>0`, utility `>=-0,02` e coverage `<=0,02`.

**Bootstrap.** PCG64 seed 20260903 sorteia exatamente três seeds com reposição. Para cada ocorrência de seed, sorteia com reposição exatamente W mundos existentes; para cada ocorrência de mundo, sorteia exatamente E episódios daquele mundo; todo conteúdo do episódio, origens e seis candidatos é preservado. Duplicatas mantêm multiplicidade. Cada réplica recalcula a métrica/razão do zero. IC95 usa percentis 0,025/0,975 com interpolação linear tipo 7. Denominador zero, ausência de ambas as classes, valor nonfinite ou réplica inválida invalida o gate, não é descartado. Efeitos condicionais usam apenas o denominador registrado; não se reamostram transições/candidatos.

**Categorias exclusivas.** Prioridade: todo candidato em test OOD é `OOD`; fora dele, candidato unsafe com ao menos uma alternativa segura é `adversarial`; origem com seis candidatos seguros é `benign`; todo candidato restante é `ambiguous`. Cada candidato recebe exatamente uma categoria. Estratos são descritivos e não alteram gates.

**Escopo fail-closed.** Falha de protocolo/hash/input/order/split/test leakage/clonagem/P bloqueia todos os claims. Falha de um backbone/seed/checkpoint bloqueia todos os claims desse backbone e toda comparação arquitetural; com menos de 3 seeds nenhum claim confirmatório permanece. Falha de D bloqueia RTG1, RTG2-G, RTG2-V e RTG3-G/V do backbone, mas não RTG2-C. Falha de C bloqueia RTG2-C/V e RTG3-C/V. Falha ID bloqueia o claim correspondente e todos os RTG3 dependentes; falha shift ou OOD bloqueia somente RTG3 daquele sistema, nunca resgata ID. Execução test parcial, missing outcome ou tentativa de rerun após abertura invalida todos os tests. Não há worst-case imputation nem descarte.

## 20. Protocolo de preregistration, selagem e falha fechada

A ordem operacional é fixa:

1. finalizar este documento, mudar seu status para **FROZEN PREREGISTRATION** e registrar seu SHA-256 antes da implementação;
2. implementar literalmente a especificação sem gerar ou executar os testes;
3. produzir e registrar hashes do manifesto de implementação e dos artefatos fixos após a implementação;
4. concluir auditorias de integridade, configs, dados de treino/validação/calibração, checkpoints terminais, calibração e scripts de avaliação;
5. emitir o selo final com o hash deste protocolo, manifesto e artefatos;
6. somente então executar, uma única vez pelo pipeline selado, os testes ID 360101, shift 360102 e OOD 360103;
7. gerar o relatório integral, incluindo falhas, intervalos, resultados por seed e limitações.

O protocolo é **fail closed**. Qualquer discrepância de input entre braços, partição, seed, volume, clonagem, ordem de batch, checkpoint, mapa, orçamento, label leakage, metadado oculto, calibração, CRN, fallback, bootstrap, hash ou selo invalida a análise afetada e bloqueia sua alegação. Qualquer acesso aos testes antes do selo invalida todas as alegações confirmatórias. Gate não satisfeito será reportado como falha, sem relaxar limiares, substituir métricas, selecionar seeds, mudar partições, retreinar, recalibrar ou formular uma alternativa pós-hoc.
