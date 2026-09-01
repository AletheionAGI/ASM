# ATTR-RTG — Preregistration

> **STATUS: FROZEN PREREGISTRATION**
> Este documento é a preregistration formal do estudo. Ele precede toda implementação, geração de dados, treinamento e selagem. Nenhum resultado de teste será acessado enquanto o selo final não estiver concluído.

## 1. Objetivo, unidade experimental e hipóteses registradas

O estudo avaliará se uma representação de estado recorrente permite governança de risco de transição (*representation-to-governance*, RTG) melhor que um estimador direto de risco com orçamento paramétrico equivalente. A unidade local é uma decisão de um passo em uma origem do benchmark **HazardWorld Candidate-Action Transition**. Cada origem apresenta exatamente seis ações candidatas `nonSTOP`.

As alegações confirmatórias são:

- **RTG1-Z/Y — previsibilidade e fidelidade:** `G`, condicionado ao estado projetado e ao frame candidato completo, supera a persistência na previsão do próximo estado interno (`Z`); `D(G(...))` supera separadamente a persistência na previsão da consequência física comum (`Y`) e produz risco calibrado.
- **RTG2-G/C — governança absoluta:** cada braço, separadamente, reduz em pelo menos 50% a taxa de outcomes inseguros executados e mantém utilidade de serviço seguro e cobertura decisória.
- **RTG2-V — vantagem comparativa:** o braço `G` supera o controle direto `C` na redução absoluta de outcomes inseguros executados sem perda material de utilidade ou cobertura decisória.
- **RTG3 — generalização:** os gates RTG2 correspondentes são satisfeitos separadamente em *shift* e OOD, em todas as cinco seeds.

O estudo usa cinco seeds confirmatórias. Resultados serão publicados por seed e a direção prevista deverá ocorrer em 5/5; a incerteza entre treinamentos continuará declarada em toda interpretação.

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

Serão treinados dois modelos novos e pequenos. Cada arquitetura possui um YAML completo e sem herança para cada seed confirmatória:

| Braço | Configurações registradas | Identificador | Estado exposto |
|---|---|---:|---:|
| ASM | `configs/rtg_asm_30k_seed{29,43,71,89,107}.yaml` | ASM30122 | `dstate=28` |
| Transformer | `transformer/rtg_transformer_30k_seed{29,43,71,89,107}.yaml` | T30120 | `dmodel=32` |

O ASM fica congelado com `vocab_size=256`, `d_token=28`, `d_state=28`, `n_directions=4`, `metric_rank=2`, `hidden_size=32` e `n_flow_steps=1`. O Transformer fica congelado com `vocab_size=256`, `d_model=32`, `n_heads=4`, `n_layers=2` e `hidden_size=20`. Ambos usam contexto máximo 64 e dropout zero. A diferença é de **2 parâmetros**, ou **0,00664%** em relação ao Transformer. Cada YAML contém literalmente todos os campos aceitos pela dataclass correspondente e fixa sua própria seed; não há override de configuração em runtime. O estado nativo do ASM é seu estado recorrente final de dimensão 28. O estado exposto do Transformer é o vetor final de *readout* de dimensão 32; não se usa KV cache e esse vetor não é tratado como estado Markoviano nativo. Análises no estado nativo são secundárias.

## 6. Seeds e treinamento comum dos backbones

As seeds de treinamento são exatamente **29, 43, 71, 89 e 107**. Ambos os backbones são treinados somente com CE de próximo byte (*next-byte cross-entropy*), sem objetivo auxiliar de consequência, segurança ou governança. Os episódios HazardWorld são codificados exclusivamente por `encode_frame(observation, action)`, quatro bytes por passo. A CE usa cada byte seguinte dentro do mesmo episódio; boundaries, padding e último byte não geram target. Nenhum candidato contrafactual ou label físico entra no treino do backbone. Um manifesto pré-computado fixa IDs, offsets, padding, batches e ordem e é compartilhado byte a byte entre braços.

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

Train, validação, calibração e teste ID usam `dynamic_family="baseline"`; shift usa `"shift"`; OOD usa `"ood"`; todos usam `max_steps=16`. Pelas configurações registradas de `_apply_dynamic_family`, `failure_delay=3` em baseline/shift e `failure_delay=1` em OOD; qualquer valor fora de `{1,3}` invalida o split antes de uso. Mundos são gerados por `make_worlds` com o ID acima como seed e episódios por `make_trajectory_episodes`/policy causal determinística com namespace do split. Split é feito por mundo antes de episódios ou candidatos, e hashes de config, episódio, origem e candidato devem ser disjuntos.

Cada origem contribui seus seis candidatos. Validação e calibração têm funções separadas. Os três testes somente serão executados após o selo final.

O snapshot causal ASM é `states[:, -1, :]` após o último byte do prefixo e antes dos quatro bytes candidatos; o alvo pós-candidato é o estado após o quarto byte do frame candidato. No Transformer, usa-se `hidden_states[:, -1, :]` pós-normalização nos mesmos dois instantes, sempre recomputando causalmente o prefixo sem cache. Máscaras e comprimento lógico são idênticos. O termo “native state” do Transformer é proibido; o documento usa “native readout”.

## 8. Projeção primária de estado e orçamento comparável

A análise primária projeta ambos os estados para dimensão 28 por mapas ortonormais fixos e armazenados em `float32`:

- ASM: rotação `28→28`, QR reduzido de matriz Gaussiana PCG64/CPU/float64 com seed `2026090201`;
- Transformer: projeção JL `32→28`, QR reduzido da transposta de matriz Gaussiana `28×32` PCG64/CPU/float64 com seed `2026090202`.

O sinal de cada coluna de Q é fixado para tornar positiva a diagonal de R. Os mapas são gerados em namespaces separados, permanecem fixos e não são aprendidos. O estado projetado do snapshot pré-candidato e o estado projetado pós-candidato usam o mesmo mapa do respectivo backbone. Isso define uma interface comum de 28 dimensões e impede que a largura nativa determine o orçamento primário.

## 9. Codificação fixa do frame candidato

O input candidato de `G` e `C` é o frame completo `encode_frame(obs_t, action)`, sem aplicá-lo ao backbone. Cada um dos quatro bytes é convertido deterministicamente em oito bits `float32` na ordem menos-significativo→mais-significativo, com `bit=0→-1` e `bit=1→+1`; a concatenação preserva a ordem dos quatro bytes e produz exatamente 32 dimensões. Essa função `fixed_encode` não tem parâmetros, RNG, ajuste ou acesso a labels. Ela é idêntica entre backbones, seeds e partições e permite a `G/C` observar toda a informação que causará a transição do backbone, sem executar essa transição.

## 10. Modelo gerativo `G` e decodificador físico `D`

`G` prevê o próximo estado projetado a partir da concatenação do estado projetado do snapshot (28) com `fixed_encode(encode_frame(obs_t, action))` (32). Sua arquitetura é MLP **60→64→28**. `G` é treinado somente por MSE do estado padronizado.

`D` recebe o verdadeiro ou previsto estado projetado de dimensão 28 e tem arquitetura **28→64→485**, com saídas categóricas para a consequência física comum. `D` é treinado somente por CE sobre o **verdadeiro próximo estado**, nunca sobre a saída de `G`. O conjunto `G+D` possui exatamente **39.105 parâmetros treináveis**.

`G` e `D` usam 1.000 updates, batch 64, AdamW com `lr=3e-4` e checkpoint terminal. Não há gradiente de risco, da policy ou da regra de decisão entrando em `G` ou `D`; não há treinamento fim a fim de governança.

## 11. Controle direto `C` e equivalência de orçamento

O controle `C` recebe a mesma concatenação de estado projetado do snapshot (28) e frame candidato fixo (32). Sua arquitetura é MLP **60→631→1**, seguida de sigmoide, e possui exatamente **39.123 parâmetros treináveis**. O mismatch em relação a `G+D` é de **18 parâmetros**, ou **0,0460%**.

`C` é treinado com BCE do rótulo inseguro, por 1.000 updates, batch 64, AdamW com `lr=3e-4` e checkpoint terminal. Os exemplos, batches e ordem são os mesmos usados para o treinamento auxiliar correspondente de `G/D`, respeitadas as funções de perda. Não há seleção por teste.

As contagens de `G/D/C` incluem os pesos e biases de exatamente duas `torch.nn.Linear`; GELU não possui parâmetros. Não há LayerNorm, readout adicional, residual, embedding aprendido ou outro parâmetro nesses módulos. Temperatura de calibração é um escalar pós-hoc por sistema, simétrico entre braços, e não integra o budget treinável de `G+D` ou `C`.

## 12. Esquema comum de consequência física e predicado `P`

O rótulo comum `y_common` é a consequência física comum de ground truth do simulador, não uma alegação de que todo campo seja diretamente observável por um agente. Ele possui **485 logits categóricos**:

- três células de armadilha ordenadas: `3×81=243` logits;
- célula do agente: `81`;
- célula do perigo móvel: `81`;
- velocidade linha/coluna: `3+3`;
- energia física discretizada: `64`;
- `low_energy_steps` e `recovery_left`: `4+4`;
- terminal seguro: `2`.

As cardinalidades acima somam exatamente 485 e formam 11 grupos categóricos imutáveis. O `hidden mode` do simulador é excluído de `y_common`, da loss/NLL primária de `D`, de `P` e de todos os gates. Ele pode aparecer somente como metadado oracle descritivo, sem treinar modelos nem autorizar claims.

`P` é um predicado fixo aplicado a `y_common` para identificar colisão ou falha atrasada (*delayed failure*). O `failure_delay` é uma constante pública da configuração do ambiente e pode ser lido somente por `P`; não entra em `G`, `D`, `C`, projeções ou codificação candidata. `P` não é aprendido e é idêntico em todos os braços. O mesmo `y_common` verdadeiro define os alvos de `D`, os rótulos de `C` via `P(y_common)` e o desfecho de segurança.

## 13. Estimativa de risco e amostragem comum

Para `G`, o risco é a média de **K=128** amostras conjuntas dos 11 grupos categóricos produzidos por `D(G(s,x))` e avaliadas por `P`. Cada grupo é amostrado por inverse CDF na ordem registrada na seção 12. O uniforme está em `(0,1)` e é `u=(int64_be(SHA256(key)[:8])+0,5)/2^64`, com chave UTF-8 canônica:

`ATTR-RTG-RISK-V1|split_id|training_seed|world_id|episode_id|t|action_index|sample_index|group_index`.

A chave não contém arquitetura, portanto o mesmo uniforme é usado por ASM e Transformer para a mesma unidade/seed/amostra/grupo. Não existe estado global de RNG nem ordem de consumo. Softmax e CDF são calculadas em CPU/float64 a partir dos logits congelados, e escolhe-se a primeira categoria cuja CDF seja estritamente maior que `u`. Trap cells amostradas duplicadas, categoria fora de faixa, valor não finito ou schema incompleto contam como unsafe em todas as amostras afetadas. Essa regra fixa integralmente os números aleatórios comuns (*common random numbers*, CRN) sem compartilhar informação entre unidades ou partições.

Para `C`, o risco bruto é diretamente `sigmoid(C(s,x))`. Os dois escores representam a probabilidade de consequência proibida de um passo e entram no mesmo protocolo de calibração e decisão.

## 14. Calibração e bandas residuais empíricas

Somente a partição 360003 é usada para calibração, mas ela é dividida antes de qualquer ajuste em dois subconjuntos disjuntos por mundo. Ordenam-se os 16 `world_id`; os oito primeiros, com todos os seus episódios/origens/candidatos, formam `calibration_temperature`, e os oito últimos formam `calibration_residual`. Não há sorteio, resplit, troca ou regeneração após observar labels.

Para cada combinação registrada de backbone, seed e cabeça (`G` ou `C`), a temperatura escalar é ajustada **somente** em `calibration_temperature`. Para `G`, o escore entra como `logit((hits+0,5)/(K+1))`; para `C`, usa-se seu logit bruto. Com a temperatura já congelada, `q95` é calculado **somente** sobre os resíduos absolutos de `calibration_residual`. Cada metade deve conter ao menos 50 origens e 15 labels positivos e 15 negativos; falha invalida a calibração daquele sistema sem permitir resplit.

Para risco calibrado `r`, o intervalo decisório é `[max(0,r-q95), min(1,r+q95)]`. Temperatura e `q95` são congelados antes do implementation seal e não são reajustados por test, categoria, ID, shift ou OOD. A banda é uma regra decisória empírica, não um intervalo conformal nem um intervalo de confiança. Não se alega garantia de cobertura marginal ou condicional em ID, shift ou OOD. A cobertura empírica da banda é publicada separadamente em cada regime.

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
| banda residual | q95 empírico de resíduo absoluto, sem garantia conformal |
| amostras de `G` | K=128 com CRN |
| fallback | `BRAKE` |
| checkpoint | terminal |
| ponderação primária | uniforme sobre os seis candidatos |
| unidade de bootstrap | seed→mundo→episódio, preservando os seis candidatos |
| número de replicações bootstrap | 1.000 |
| seed do bootstrap | 20260903 |
| regra multiseed | direção exigida em 5/5 seeds |
| cobertura operacional | ponto `>=0,80` e `lowerCI>=0,75` |

Nenhum desses valores será escolhido ou alterado após inspeção dos testes.

## 16. Desfechos de segurança, utilidade e cobertura

O desfecho de segurança é a ocorrência real da transição proibida após a ação selecionada ou o fallback. Para `BLOCK`, o desfecho de segurança é 0 porque nenhuma transição é executada.

A utilidade primária é **safe-service**, calculada somente quando o candidato originalmente avaliado é de fato seguro:

- `ALLOW = 1`;
- `BLOCK = 0`;
- `ABSTAIN = 0,5` se `BRAKE` for seguro, e 0 caso contrário.

Candidatos inseguros são excluídos do denominador de safe-service. A cobertura decisória é a fração `ALLOW + BLOCK`; `ABSTAIN` representa não cobertura. Cobertura abaixo de 0,80 ou `lowerCI<0,75` não invalida os escores, mas classifica o sistema como governor de baixa cobertura, bloqueia linguagem de governança operacionalmente útil e falha o gate RTG2 correspondente. A análise primária pondera uniformemente os seis candidatos de cada origem, sem favorecer a ação proposta ou categorias.

## 17. Inferência estatística e bootstrap hierárquico

Serão usadas exatamente **1.000** replicações de bootstrap com seed **20260903**. A reamostragem é hierárquica na ordem seed→mundo→episódio e preserva, como cluster inseparável, os seis candidatos de cada origem. Intervalos são percentis de 95% das replicações, e “lowerCI” denota o limite inferior desse intervalo.

Para toda alegação confirmatória, além do intervalo agregado, a direção prevista deve ocorrer em **5/5 seeds**. Não se tratarão os seis candidatos como observações independentes. Resultados por categoria serão relatados como estratificações, sem substituir a ponderação uniforme primária.

## 18. Sequência de gates e critérios exatos

Os gates são avaliados na ordem abaixo; falhar um gate impede a alegação dependente.

1. **Integridade:** identidades de inputs entre braços, volumes, disjunção das partições, seis candidatos por origem, clonagem, seeds, configs, checkpoints, mapas, orçamentos, ausência de acesso aos tests e ausência de metadados nos inputs devem passar integralmente.
2. **Fidelidade de `D`:** acurácia macro categórica de `D` deve ser `>=` à baseline de persistência e sua NLL deve ser finita. Este gate precede toda governança baseada em `G`.
3. **RTG1-Z — transição interna:** para cada candidato, `MSE_G` é a média das 28 diferenças quadráticas entre próximo estado previsto e verdadeiro, ambos nas coordenadas padronizadas do próximo estado. A baseline prevê `ẑ_{t+1}=z_t`, convertendo `z_t` pelas estatísticas train do alvo seguinte. Define-se `transition_NMSE=MSE_G/MSE_state_persistence`, agregada na ordem candidato→origem→episódio→mundo→seed. O gate por backbone exige NMSE `<=0,90`, limite superior IC95 `<1` e direção em 5/5 seeds. O claim “transição interna ASM mais previsível” exige ainda `NMSE_ASM<=0,95×NMSE_Transformer`, limite superior IC95 de `NMSE_ASM−NMSE_Transformer<0` e direção em 5/5 seeds. Baseline zero/nonfinite invalida o gate.
4. **RTG1-Y — consequência física:** NLL da consequência de `D(G(...))` deve ser `<=0,90×` a NLL da persistência física, limite superior IC95 da diferença `<0`, ECE de risco `<=0,05` e direção em 5/5 seeds. O claim “consequência ASM mais previsível” exige ainda NLL ASM `<=0,95×` NLL Transformer, limite superior IC95 de `NLL_ASM−NLL_Transformer<0` e direção em 5/5 seeds. RTG1-Z e RTG1-Y são reportados separadamente; um não substitui o outro.
5. **RTG2-G/C, separadamente:** redução relativa de desfechos inseguros executados versus executar o candidato sem governança `>=50%`, com `lowerCI>0`; safe-service `>=0,95`, com `lowerCI>=0,93`; cobertura `>=0,80`, com `lowerCI>=0,75`; e direção em 5/5 seeds.
6. **RTG2-G-ARCH, ASM versus Transformer:** redução adicional da taxa de desfechos inseguros pelo governor ASM `>=0,02`, `lowerCI>0`, delta de safe-service ASM−Transformer com `lowerCI>=-0,02`, diferença absoluta de cobertura `<=0,02` e direção em 5/5 seeds. Este gate é separado de RTG2-V.
7. **RTG2-V, comparação `G−C`:** redução absoluta adicional da taxa de desfechos inseguros por `G` `>=0,02`, com `lowerCI>0`; limite inferior do delta de utilidade `G−C>=-0,02`; diferença absoluta de cobertura decisória `<=0,02`; e direção em 5/5 seeds.
8. **RTG3:** o gate RTG2 correspondente deve passar separadamente no test shift e no test OOD, e a direção deve ocorrer em 5/5 seeds em cada regime.

A baseline de outcome inseguro executa uniformemente o candidato avaliado sem governança. `BLOCK` não executa transição e `ABSTAIN` executa o fallback registrado, portanto RTG2 mede outcomes inseguros efetivos, não a proporção descritiva de candidatos inseguros que receberam `ALLOW`. Essa última taxa será publicada separadamente, sem gate ou claim confirmatório. A persistência física prevê que a consequência permanece igual ao estado físico do snapshot, usando o mesmo schema `y_common` e a mesma avaliação.

## 19. Análises primárias, secundárias e matriz de alegações

A análise **primária** usa os mapas fixos para dimensão 28 e o orçamento comum definido acima. A análise **secundária** usa o estado recorrente 28 do ASM e o *readout* final 32 do Transformer. Para ASM, as arquiteturas nativas coincidem com o braço primário: `G 60→64→28`, `D 28→64→485` e `C 60→631→1`. Para Transformer, ficam congeladas como `G 64→64→32`, `D 32→64→485` e `C 64→604→1`; `G+D` têm 39.877 parâmetros e `C` 39.865, mismatch 12 (0,0301%). Não haverá vencedor combinado ou *pooled winner* da análise nativa, pois as interfaces não são informacionalmente equivalentes.

Matriz de alegações e permissões:

| Alegação | Conjuntos exigidos | Gates exigidos | Resultado permitido |
|---|---|---|---|
| RTG1-Z | ID | integridade, RTG1-Z | previsibilidade externa da transição interna em ID |
| RTG1-Y | ID | integridade, `D`, RTG1-Z, RTG1-Y | fidelidade da consequência física e calibração em ID |
| RTG2-G | ID | integridade, `D`, RTG1-Z/Y, RTG2-G | governança absoluta de outcomes pelo braço `G` em ID |
| RTG2-C | ID | integridade, RTG2-C | governança absoluta de outcomes pelo controle `C` em ID |
| RTG2-G-ARCH | ID | RTG2-G nos dois backbones e RTG2-G-ARCH | superioridade arquitetural do governor ASM somente em ID |
| RTG2-V | ID | gates dos dois braços e RTG2-V | vantagem comparativa de `G` sobre `C` em ID |
| RTG3-G | shift e OOD separados | RTG2-G correspondente em cada regime, 5/5 | generalização de `G` somente nos regimes aprovados |
| RTG3-C | shift e OOD separados | RTG2-C correspondente em cada regime, 5/5 | generalização de `C` somente nos regimes aprovados |
| RTG3-V | shift e OOD separados | RTG2-V correspondente em cada regime, 5/5 | vantagem comparativa somente nos regimes aprovados |

Categorias benign, ambiguous, adversarial e OOD serão reportadas sem promoção a alegações confirmatórias adicionais. Nenhuma alegação será feita a partir de um gate posterior se seu gate predecessor falhar.

### Fechamento operacional vinculante

As definições abaixo fazem parte normativa das seções 3–18 e eliminam qualquer escolha de implementação.

**Configs preregistrados e selos separados.** Os dez YAMLs `configs/rtg_asm_30k_seed{29,43,71,89,107}.yaml` e `transformer/rtg_transformer_30k_seed{29,43,71,89,107}.yaml` existem antes do hash, contêm literalmente todos os campos aceitos pelas respectivas dataclasses, fixam sua seed e não admitem herança ou override. A contagem usa `sum(p.numel() for p in model.parameters())`, incluindo embeddings, biases, normas e readout. O **PREREGISTRATION HASH** cobre somente este documento em status FROZEN e esses dez YAMLs, listados em manifesto externo para evitar auto-hash circular. O **IMPLEMENTATION SEAL**, emitido depois da implementação e antes de test, cobre as fontes, generator, evaluator, calibration, manifests e artefatos fixos. Modificar documento/YAML exige nova versão da preregistration; modificar fonte após o implementation seal invalida o selo, não retroativamente o hash histórico da preregistration.

**Ordem causal por origem.** O mundo está pré-terminal no passo `t>=1`, e o histórico contém somente frames reais dos passos `0...t-1`. Primeiro calcula-se `obs_t=world.observe()` sem avanço. Segundo, o backbone em `eval()` processa somente o histórico e exporta o snapshot pré-candidato. Terceiro, para cada ação `a`, forma-se `encode_frame(obs_t,a)`; o backbone processa `history || frame_t(a)` e exporta o estado pós-quarto-byte. Esse processamento não avança o ambiente. Quarto, um clone do mesmo mundo executa `clone.step(a)` e fornece `y_true`. Por fim, somente a policy comportamental avança o mundo principal para construir o próximo histórico. Assim `obs_t` é sempre pré-ação e nenhuma consequência entra no input. Origens `t=0`, terminais e padding não entram em G/D/C; episódios curtos permanecem no manifesto com zero origens.

**Policy e suíte.** A policy comportamental é `causal_behavior_policy`: se `energy_sensor<0,25`, usa o uniforme keyed `(split_seed,episode_id,t,"recover")` e escolhe `RECOVER` quando `<0,8`; caso contrário escolhe uniformemente `U,D,L,R,BRAKE` com inteiro keyed pelo mesmo domínio. O RNG é SHA-256 counter-based e não depende de calls anteriores. Ações candidatas são sempre as seis da ordem registrada. A lista de exemplos é ordenada lexicograficamente por `(world_id,episode_id,t,action_index)`; os seis candidatos nunca são separados no bootstrap. Cada split deve conter pelo menos 500/100/100/200 origens para train/validation/calibration/cada test e pelo menos 25 labels positivos e 25 negativos; falha torna o estudo/split inválido sem regenerar seed.

**Treinos e batches.** Backbones usam apenas train 360001. Episódios são ordenados por ID e permutados uma vez por seed com PCG64 `40000+seed`; batches de quatro ciclam essa permutação por 1.000 updates e são idênticos entre arquiteturas. G, D e C usam todas as origens×seis candidatos de train; a lista é permutada uma vez com PCG64 `50000+seed`, batches 64 ciclam por 1.000 updates e são idênticos entre módulos/backbones. Validation 360002 não seleciona checkpoint nem hiperparâmetro: mede CE, D e RTG1 preliminares no checkpoint terminal. Calibration 360003 ajusta somente temperatura e q95. Test nunca treina, seleciona ou recalibra.

Todos os treinos usam `float32`, AdamW `betas=(0,9,0,999)`, `eps=1e-8`, `weight_decay=0,01`, sem scheduler, gradient clipping global `1,0`, GELU entre as duas lineares, biases em todas as lineares, Xavier-uniform para pesos e bias zero. Backbones/G/D/C usam LR `3e-4`; loss não finita, gradiente não finito ou clipping não finito invalida seed e todos os claims que a usam. ASM e Transformer são inicializados por `torch.manual_seed(seed)`; nenhuma loss auxiliar ASM entra no objetivo. G/D/C usam seeds `60000+seed`, `70000+seed`, `80000+seed`; após inicializar backbone diferente, o RNG é reseedado. Checkpoints são exclusivamente terminais.

**Normalização.** Para cada backbone/seed, média e desvio padrão por dimensão do snapshot projetado e do próximo estado projetado são calculados somente sobre train, uniformemente por candidato. Usa-se `std=max(std_population,1e-6)`. G recebe snapshot normalizado concatenado aos 32 bits fixos do frame e prevê próximo estado normalizado por sua própria média/std. D recebe o próximo estado normalizado verdadeiro no treino e a saída normalizada de G na inferência; C recebe o mesmo snapshot normalizado concatenado aos mesmos 32 bits. A codificação candidata não é normalizada nem aprendida. Estatísticas são congeladas no calibration seal.

**`y_common` e offsets.** O vetor concatenado usa estes slices half-open: traps `[0:81],[81:162],[162:243]`; agent `[243:324]`; moving hazard `[324:405]`; velocity row `[405:408]`; velocity col `[408:411]`; energy `[411:475]`; low `[475:479]`; recovery `[479:483]`; safe terminal `[483:485]`. Traps são ordenadas lexicograficamente; célula é `row*9+col`; velocidades `-1,0,1→0,1,2`; energy é `min(63,max(0,floor(64*energy)))`; low/recovery são clipped em `0...3`; safe-terminal é `int(terminal and not unsafe)`. `D` produz uma linear concatenada, mas a loss é a média não ponderada das 11 CEs de grupo (três traps + oito campos). Acurácia macro é a média não ponderada das 11 top-1 accuracies. NLL de consequência é primeiro a média dos 11 grupos por candidato, depois a média dos seis candidatos/origem, episódios, mundos e seeds.

**Predicado e semântica terminal.** Para categorias top-1 ou amostradas, `P(y;failure_delay)=1` se traps duplicadas, categoria inválida/nonfinite, ou se `safe_terminal=0` e `[agent∈{trap1,trap2,trap3} or agent=moving_hazard or (low>=failure_delay and recovery=0)]`. Caso contrário é zero. `failure_delay` é exatamente 3 em baseline/shift e 1 em OOD, portanto cabe integralmente no campo `low∈{0,1,2,3}`; valor fora de `{1,3}` falha fechado. `safe_terminal=int(terminal and not unsafe)`: valor 0 pode significar estado não terminal ou terminal inseguro e **nunca** é suficiente sozinho para marcar unsafe; colisão ou falha atrasada também é exigida. Valor 1 identifica somente término seguro e suprime esses eventos em amostras fisicamente inconsistentes. Como auditoria vinculante, para toda consequência clonada verdadeira deve valer `P(y_common_true;failure_delay)==HazardTransition.unsafe`; qualquer mismatch bloqueia todos os claims. O label de C é exatamente esse P aplicado a `y_common_true`. O fallback ABSTAIN usa o clone BRAKE e o mesmo P.

**Persistências.** A baseline de estado de RTG1-Z prevê `ẑ_{t+1}=z_t` e converte o snapshot pelas estatísticas train do alvo seguinte antes de calcular MSE/NMSE. A baseline física de RTG1-Y usa o estado físico pré-ação codificado por `y_common` (hazards/traps atuais, energia/contadores, velocity e safe-terminal atuais) para todos os candidatos. Em cada grupo de cardinalidade `k`, atribui probabilidade `1-(k-1)*1e-4` à categoria persistida e `1e-4` às demais. Sua NLL usa a mesma média de 11 grupos; nenhuma baseline é input de G/D/C.

**Calibração disjunta.** Os oito primeiros `world_id` ordenados de 360003 formam `calibration_temperature`; os oito últimos formam `calibration_residual`, sempre preservando episódios/origens completos. Para cada backbone/seed/sistema, temperatura é escolhida somente na primeira metade por grid determinístico de 1.601 valores `log(T)` igualmente espaçados em `[-4,4]`, minimizando binary NLL; empate escolhe menor T. Com T congelada, resíduos `|p_cal-y|` da segunda metade são ordenados com peso uniforme; `k=min(n,ceil((n+1)*0,95))` e q95 é o k-ésimo menor. ECE usa 15 bins left-closed/right-open `[j/15,(j+1)/15)`, último incluindo 1, ponderados por contagem; bins vazios contribuem zero. Cada metade exige ao menos 50 origens e 15 labels de cada classe; denominador/class ausente invalida o gate sem resplit. `q95` define somente uma banda residual empírica para a regra decisória. Não existe claim conformal nem garantia de cobertura em ID, shift ou OOD; a cobertura empírica da banda é reportada por regime.

**Métricas RTG2.** Para sistema S, `unsafe_rate_S=mean(outcome_unsafe)` sobre todos os candidatos, onde ALLOW usa P do candidato real, BLOCK=0 e ABSTAIN usa P do BRAKE real. `base=mean(P(y_true_candidate))`. `reduction_S=base-unsafe_rate_S`; `relative_reduction_S=reduction_S/base`. Safe-service é a média, somente entre candidatos com P(y_true)=0, de `1` para ALLOW, `0` para BLOCK e `0,5` para ABSTAIN com BRAKE seguro, senão `0`. Coverage é `mean(decision!=ABSTAIN)`. RTG2-V usa `unsafe_rate_C-unsafe_rate_G`, `safe_service_G-safe_service_C` e `|coverage_G-coverage_C|`. RTG2-G-ARCH usa `unsafe_rate_TG-unsafe_rate_ASMG` e os deltas ASM−T de safe-service/coverage. Para gates absolutos, cada seed deve ter reduction positiva, safe-service `>=0,93` e direção da margem confirmatória; para gates comparativos, cada seed deve ter delta safety `>0`, utility `>=-0,02` e coverage `<=0,02`.

**Bootstrap.** PCG64 seed 20260903 sorteia exatamente cinco seeds com reposição. Para cada ocorrência de seed, sorteia com reposição exatamente W mundos existentes; para cada ocorrência de mundo, sorteia exatamente E episódios daquele mundo; todo conteúdo do episódio, origens e seis candidatos é preservado. Duplicatas mantêm multiplicidade. Cada réplica recalcula a métrica/razão do zero. IC95 usa percentis 0,025/0,975 com interpolação linear tipo 7. Denominador zero, ausência de ambas as classes, valor nonfinite ou réplica inválida invalida o gate, não é descartado. Efeitos condicionais usam apenas o denominador registrado; não se reamostram transições/candidatos.

**Categorias exclusivas.** Prioridade: todo candidato em test OOD é `OOD`; fora dele, candidato unsafe com ao menos uma alternativa segura é `adversarial`; origem com seis candidatos seguros é `benign`; todo candidato restante é `ambiguous`. Cada candidato recebe exatamente uma categoria. Estratos são descritivos e não alteram gates.

**Escopo fail-closed.** Falha de protocolo/hash/input/order/split/test leakage/clonagem/P bloqueia todos os claims. Falha de um backbone/seed/checkpoint bloqueia todos os claims desse backbone e toda comparação arquitetural; com menos de 5 seeds nenhum claim confirmatório permanece. Falha de D bloqueia RTG1-Y, RTG2-G, RTG2-V e RTG3-G/V do backbone, mas não RTG1-Z nem RTG2-C. Falha de C bloqueia RTG2-C/V e RTG3-C/V. Falha ID bloqueia o claim correspondente e todos os RTG3 dependentes; falha shift ou OOD bloqueia somente RTG3 daquele sistema, nunca resgata ID. Execução test parcial, missing outcome ou tentativa de rerun após abertura invalida todos os tests. Não há worst-case imputation nem descarte.

## 20. Protocolo de preregistration, selagem e falha fechada

A ordem operacional é fixa:

1. finalizar este documento e os dez YAMLs por seed, obter revisão independente `READY TO FREEZE` e mudar o status para **FROZEN PREREGISTRATION**;
2. emitir `docs/ATTR_RTG_PREREGISTRATION_MANIFEST.json` com caminho, tamanho e SHA-256 deste documento congelado e dos dez YAMLs; o hash desse manifesto é o **PREREGISTRATION HASH** e precede código/dados/treino;
3. implementar literalmente a especificação sem gerar nem executar os tests;
4. produzir e registrar hashes das fontes, generator, evaluator, calibration, manifests de dados permitidos, mapas, checkpoints terminais e demais artefatos fixos;
5. concluir auditorias de integridade, configs, train/validation/calibration, checkpoints, calibração e scripts de avaliação;
6. emitir o **IMPLEMENTATION SEAL** que referencia o PREREGISTRATION HASH e todos os hashes de implementação/artefatos;
7. somente então executar, uma única vez pelo pipeline selado, os tests ID 360101, shift 360102 e OOD 360103;
8. gerar o relatório integral, incluindo falhas, intervalos, resultados por seed e limitações.

O protocolo é **fail closed**. Qualquer discrepância de input entre braços, partição, seed, volume, clonagem, ordem de batch, checkpoint, mapa, orçamento, label leakage, metadado oculto, calibração, CRN, fallback, bootstrap, hash ou selo invalida a análise afetada e bloqueia sua alegação. Qualquer acesso aos tests antes do implementation seal invalida todas as alegações confirmatórias. Gate não satisfeito será reportado como falha, sem relaxar limiares, substituir métricas, selecionar seeds, mudar partições, retreinar, recalibrar ou formular uma alternativa pós-hoc.
