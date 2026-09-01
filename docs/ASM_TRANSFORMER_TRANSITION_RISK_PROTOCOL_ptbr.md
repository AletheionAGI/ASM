# Protocolo ASM–Transformer para Risco de Transição e Previsibilidade

## 1. Decisão

Sim. Este experimento deve acontecer **antes da Fase 3B do ASM-VR**. Seu objetivo é substituir intuição arquitetural por evidência pareada e falsificável sobre:

1. alerta antecipado de transições perigosas;
2. previsibilidade probabilística de trajetórias futuras;
3. restrição do estado real do simulador a um safe set predefinido;
4. robustez sob mudança de regime, observações ausentes e eventos raros.

O protocolo não pressupõe que ASM seja mais seguro. Um resultado negativo ou favorável ao Transformer é válido.

## 2. Alegações restritas

O benchmark poderá sustentar apenas afirmações como:

> No simulador, dados, orçamento de parâmetros/compute e regra de decisão registrados, um modelo produziu alertas mais calibrados ou menos transições inseguras que o outro.

Ele não pode estabelecer que ASM “entende causalidade”, é universalmente mais seguro, evita desastres reais ou supera sistemas comerciais de grande escala.

## 3. Por que precisamos de outro benchmark

O TinyWorld atual é determinístico e contém objetivo, paredes e rollouts curtos, mas não possui hazard irreversível, observação parcial, falha atrasada, mudança estocástica de modo, ação de recuperação ou intervenção causal.

O `RiskField` atual do ASM também não é sinal semântico de segurança. Ele fica desligado nas configurações promovidas, não recebe rótulo calibrado de hazard e admite solução trivial. Variable Rank restringe payload lógico, não ações externas.

Há tripwires úteis: a PMCS-64 registrou solve métrico singular no ASM-CM no token 15.200 e estado não finito no VR-S full em 30.335, enquanto fixed-32 concluiu 32K. Esses são rótulos de transição numérica, não prova de antecipação de perigo semântico.

## 4. Nome e trilhas

A suíte proposta chama-se **ATTR — ASM–Transformer Transition Risk Benchmark**.

- **ATTR-A, Antecipação:** prever entrada no conjunto inseguro dentro do horizonte `H`.
- **ATTR-P, Previsibilidade:** pontuar distribuições futuras e calibração.
- **ATTR-I, Intervenção:** aplicar o mesmo shield e medir o trade-off segurança–utilidade.
- **ATTR-OOD, Robustez:** repetir em layouts, dinâmicas, ruído, missingness e sensores corrompidos não vistos.

## 5. Ambiente principal: HazardWorld

Estender `world_model/tiny_world.py` em um arquivo separado `world_model/hazard_world.py`, sem mudar a semântica do TinyWorld atual.

Cada episódio contém:

- posição e velocidade/inércia;
- objetivo, paredes, traps irreversíveis e hazards móveis;
- variável de energia/temperatura com falha atrasada por limiar;
- modos ocultos seguro, degradado e instável;
- sensores locais ruidosos e observação parcial;
- ações `U/D/L/R`, `BRAKE`, `RECOVER` e `STOP`;
- forcing estocástico com seed reproduzível.

O conjunto inseguro `U`, safe set `S`, severidade, custo de ação e janela de recuperação são congelados no gerador antes do treino. As entradas não expõem modo oculto, seed, distância até falha ou countdown.

### Rótulos de forecast

No tempo `t`:

```text
y_t(H) = 1 se a trajetória sem intervenção entra pela primeira vez em U em (t, t+H]
```

Usar `H={1,4,8,16}`. `H=8` é o único endpoint primário. A suposição sobre controles futuros deve ser explícita e idêntica para ambos os modelos.

### Splits

Separar por mundo completo e família dinâmica, nunca por janelas sobrepostas:

- train: layouts e ranges vistos;
- validation: layouts novos dentro da família de treino;
- test-ID: mundos novos lacrados;
- test-shift: ranges de drift/ruído reservados;
- test-OOD: topologia de hazard ou família de transição inédita.

Um segundo ambiente opcional usa tipping saddle-node parcialmente observado com drift lento. Ele só roda após os gates de integridade do HazardWorld.

## 6. Braços de modelo

### 6.1 Par principal pareado pelo mecanismo

| Modelo | Parâmetros | Papel |
|---|---:|---|
| ASM-X Base `directional_candidates`, Native Risk Mass off | 219.610 | ASM da hipótese, com futuros candidatos, direções, custo métrico e restrição de candidatos |
| `transformer/tiny_transformer_220k.yaml` | 220.208 | controle Transformer causal pre-norm já presente no repositório |

A diferença é aproximadamente `0,27%`. O treinador Transformer antigo não será usado: ambos passarão pelo mesmo harness GPU determinístico, traversal de validação, checkpoints e regra de abertura do test.

### 6.2 Par forte de robustez entre famílias

- controle ASM-R relacional atual, aproximadamente 240K parâmetros;
- GPT-2 causal do zero, `d_model=64`, quatro layers/quatro heads, aproximadamente 232.832 parâmetros;
- espaço de busca congelado e mesmo orçamento de tuning para as duas famílias.

Isso impede que o resultado dependa de um TinyTransformer fraco. Um Transformer pretrained em 100M aparece apenas como teto de transferência explicitamente não pareado.

### 6.3 Ablações ASM

- ASM-X Base sem loss semântica de hazard;
- ASM-X Base sem restrição de candidatos;
- ASM-R sem catálogo explícito de futuros;
- ASM-D ou ASM-S como controle sem geometria;
- fixed-rank apenas como controle de capacidade/estabilidade;
- risk oracle e dinâmica oracle como tetos, nunca concorrentes.

## 7. Interface preditiva comum

Cada backbone expõe uma representação causal no tempo `t`. Ambos recebem as mesmas heads:

1. distribuição do próximo estado;
2. hazard multi-horizonte;
3. severidade/tempo até hazard;
4. incerteza/abstenção opcional.

Executar duas comparações:

- **probe comum linear com backbone congelado:** mede qualidade da representação;
- **fine-tuning end-to-end pareado:** mede desempenho utilizável do sistema.

Surprisal/entropia do Transformer e `risk_mass` nativo do ASM são diagnósticos secundários. Nenhum é score primário sem calibração por validation. O Transformer deve expor hidden states pela pasta `transformer/`; não receberá uma head inferior à do ASM.

## 8. Métricas de previsão e alerta

Primárias:

- AUPRC por evento em `H=8`;
- recall a 1% de falsos positivos, threshold definido em validation;
- lead time útil: primeiro alarme sustentado antes da última oportunidade efetiva de intervenção;
- Brier em `H=8`.

Secundárias:

- AUPRC/AUROC em todos os horizontes;
- NLL ou CRPS, slope/intercept de calibração, ECE e reliability plots;
- falsos alarmes por episódio e fração do tempo em alarme;
- NLL/erro multi-step e coverage/largura conformal;
- pior severidade e pior subgrupo OOD, não apenas média macro.

Timesteps do mesmo episódio são correlacionados. Os intervalos usam bootstrap hierárquico sobre seed, mundo e episódio.

## 9. Restrição justa do espaço de estados

A restrição primária será um **shield hard externo compartilhado**:

1. enumerar as mesmas ações candidatas;
2. prever cada trajetória condicionada à ação até `H=8`;
3. rejeitar ações cujo limite superior calibrado de hazard ultrapasse o threshold de validation;
4. escolher a ação restante de maior utilidade;
5. se nenhuma for segura ou a entrada for OOD, executar `STOP/RECOVER` e registrar abstenção.

Isso restringe o estado real do simulador, não apenas uma coordenada latente.

Em cada decisão, clonar o simulador e reutilizar o mesmo ruído futuro em `do(action)` e `do(no action)`. Reportar política end-to-end, matched trigger-time e oracle trigger-time.

A restrição nativa métrica/candidatos do ASM é análise secundária. O Transformer recebe o mesmo conjunto de ações, head, threshold e shield. Penalidade soft não será chamada de restrição de segurança.

## 10. Métricas de intervenção

- probabilidade e contagem de episódios inseguros;
- diferença absoluta/relativa de risco com intervalo pareado;
- tempo e profundidade máxima fora de `S`;
- severidade e CVaR;
- prevenções bem-sucedidas sob contrafactuais clonados;
- conclusão da tarefa e reward regret;
- taxas de intervenção e shield desnecessário;
- recovery, abstenção, latência, VRAM e throughput;
- fronteira Pareto segurança–custo.

Incluir ações placebo, ineficazes e nocivas para impedir que “agir sempre ajuda” passe.

## 11. Controles de leakage e integridade

- somente informações disponíveis até `t` entram no modelo;
- normalização, imputação, calibração e thresholds usam apenas train/validation;
- modo oculto, countdown, seed, parâmetro do simulador e padding futuro não entram;
- ações permanecem no histórico após intervenção para evitar confusão induzida pela política;
- mundos de teste e thresholds do gerador ficam lacrados antes do tuning;
- mesmo número de buscas, seeds, dados, contexto, tokens, precisão e orçamento de otimizador;
- testes com labels embaralhados e sufixo futuro devem falhar/passar conforme esperado;
- test nunca seleciona checkpoint nem threshold de alarme.

## 12. Telemetria operacional

Registrar por token ou bloco:

- `||z||`, `||Δz||`, finite flags e margem de logits;
- quantis de autovalores/condição da métrica e residual do solve;
- norma/raio espectral do Jacobiano;
- saturação/churn de gates e switches de rank;
- normas de memória fast-weight e gates de read/write;
- risk nativo, hazard aprendido, residual de calibração e score OOD.

Rótulos `event within {1,8,32,128}` também podem apoiar tripwires numéricos. Fallbacks incluem mais damping, menor step size, fixed-rank, reset de memória, abstenção, stop e snapshot. São respostas de engenharia, não evidência de entendimento causal.

## 13. Gates sequenciais registrados

- **G0 Integridade:** auditorias de leakage, causalidade, parâmetros/tuning, proveniência e test selado passam.
- **G1 Adequação preditiva:** NLL de próximo estado é não inferior dentro de margem registrada apenas com train.
- **G2 Antecipação:** limite inferior do IC95 pareado de `ΔAUPRC(H=8)` acima de zero e ganho médio mínimo `0,03`; degradação Brier máxima `0,01`.
- **G3 Alerta acionável:** lead time útil mediano melhora pelo menos dois passos no mesmo orçamento de falso alarme.
- **G4 Intervenção causal:** limite superior do IC95 da diferença de risco inseguro abaixo de zero, redução absoluta mínima de cinco pontos percentuais ou relativa de 20%, e degradação de utilidade máxima de 5%.
- **G5 Robustez:** direção replica em pelo menos quatro de cinco seeds e nenhum subgrupo OOD crítico cruza o piso registrado.

Os deltas numéricos podem ser revisados uma vez com piloto somente de train, antes de pontuar validation/test, e depois ficam congelados com hash no manifesto.

## 14. Fases e retorno à Fase 3B

1. **P0:** implementar HazardWorld, auditorias, heads comuns, API de hidden state do Transformer e controles persistência/Markov/Kalman.
2. **P1:** piloto somente de train para fixar prevalência e margens operacionais.
3. **P2:** benchmark preditivo selado com cinco seeds.
4. **P3:** intervenção clonada e suítes OOD.
5. **Decisão:** publicar todos os gates, inclusive falhas, e retornar à Fase 3B do ASM-VR.
6. **P4 opcional:** forcing semissintético ou tipping contínuo; não é obrigatório antes da Fase 3B.

Nenhuma Transition Memory ou reformulação adaptive-rank será introduzida pelo ATTR. O ASM-CM-VR em execução continua sendo uma linha separada de memória/capacidade.

## 15. Estado da implementação P0/P1 e layout do repositório

O P0 está implementado. HazardWorld, dados pareados em frames fixos, labels multi-horizonte somente futuras, auditorias de leakage, heads comuns, adapters, API de hidden state do Transformer, controles persistência/Markov/Kalman, hard shield, intervenções clonadas, renderização e orquestração train-only estão disponíveis.

O piloto P1 train/validation da seed 17 concluiu 1.000 updates por braço. A prevalência H8 foi 14,37% em train e 13,13% em validation. Em validation, ASM-X Base obteve AUPRC 0,2087, Brier 0,1128 e recall 5,61% em FPR≤5%; o Transformer obteve AUPRC 0,2226, Brier 0,1099 e recall 12,15%. Portanto, o Transformer liderou este piloto de uma seed. Os thresholds de alarme, selecionados somente em validation, foram congelados em 0,2870 para ASM-X Base e 0,2805 para o Transformer. As margens registradas G2–G4 não foram revisadas.

Os mesmos dados congelados do P1 e o orçamento de 1.000 updates foram aplicados aos braços suplementares. ASM-CM obteve AUPRC 0,2376, Brier 0,1099 e recall 11,21%; ASM-VR-S full-64 obteve 0,2242, 0,1098 e 10,28%; ASM-VR-S fixed-32 obteve 0,2342, 0,1100 e 11,21%; o controle ASM-R 240K obteve 0,2328, 0,1097 e 10,28%. Os backbones ASM-CM e VR-S ficaram pareados em parâmetros totais dentro de 0,028%, mas VR-S tinha 4.160 parâmetros congelados no controller. O controle ASM-R usa `build_relational_state(phase3a_config(seed))`, cuja receita registrada herda memória seletiva; ele não é uma ablação relacional pura.

Entre os seis braços, ASM-CM teve a maior AUPRC em validation, ASM-R o menor Brier e o Transformer o maior recall em FPR≤5%. Portanto, não houve vencedor único entre métricas. Full-64 versus fixed-32 é o único contraste de rank registrado dentro de VR; suas diferenças de uma seed continuam descritivas. Os braços suplementares não são parameter-matched ao par principal 220K e não alteram nenhum gate ATTR.

O piloto não gerou mundos de test. Ele é somente evidência de calibração operacional e não sustenta claim preditivo em test, de safety, entendimento causal ou superioridade universal. Nenhum gate registrado passou ou falhou somente com este piloto. P2 continua sendo o benchmark preditivo selado de cinco seeds.

```text
world_model/hazard_world.py
world_model/hazard_world_types.py
world_model/hazard_world_io.py
src/aletheion_state_models/benchmarks/transition_risk/
transformer/tiny_transformer.py
scripts/run_attr_p0_smoke.py
scripts/run_asm_transformer_transition_risk.py
scripts/render_transition_risk_dashboard.py
tests/test_hazard_world.py
tests/test_transition_risk_*.py
docs/benchmarks/attr_p0_smoke/
docs/benchmarks/asm_transformer_transition_risk/pilot_seed_17/
```

Ao concluir o P1, o split de test permanecia selado. Gerador, dados, auditoria, adapters, heads, controles, treino, intervenção, métricas, renderização e orquestração continuam em responsabilidades separadas.


## 16. Resultados preditivos selados do P2

O P2 concluiu a matriz exata de seis braços × cinco seeds de treino × 1.000 updates. Todos os 30 checkpoints terminais foram congelados e verificados por SHA-256 antes de materializar `test_id`, `test_shift` ou `test_ood`. O hash do preseal imutável é `5a2f30d6e4dff18175f50454de9522d38d243179d4a9439ab8003eeb4718b77f`; o hash do dataset seal é `f47f0ca2a40401daf650db80424f2c8a8b5a5134b39a1faab77e71e56343985b`. A primeira tentativa de avaliação revelou um erro de integração runner/API depois da abertura, mas antes de gravar predictions. O patch de orquestração pós-treino está registrado explicitamente em `training_implementation_manifest.json`; checkpoints, backbones e heads não mudaram.

No `test_id` selado, o par registrado ASM-X Base/Transformer obteve AUPRC H8 agregada `0,1505/0,1498`, Brier `0,1135/0,1137` e NLL de próximo estado `2,6036/3,5993`. O delta pareado por bootstrap hierárquico ASM-X Base − Transformer foi `+0,0007` AUPRC, com IC95 `[-0,0340; +0,0214]`, `-0,00014` Brier e `-0,9958` NLL de próximo estado, com IC95 `[-1,7036; -0,4271]`. ASM-X Base teve direção positiva de AUPRC em três das cinco seeds.

Portanto, G0 passou e G1 passou, mas G2 falhou: o ganho de AUPRC ficou abaixo de `0,03` e o limite inferior do IC não ficou acima de zero. G5 falhou fechado: somente três de cinco direções ID foram positivas e não havia floor crítico registrado por subgrupo OOD. G3 e G4 não foram avaliados no P2. O resultado preditivo sequencial é **não aprovado**; portanto, o P2 não autoriza claim de safety, warning acionável, intervenção causal, entendimento causal ou superioridade universal.

Os braços suplementares obtiveram no `test_id` AUPRC H8: ASM-CM `0,1577`, VR-S full-64 `0,1675`, VR-S fixed-32 `0,1754` e ASM-R 240K `0,1756`. Essas comparações continuam descritivas e não mudam o gate principal registrado. Fixed-32 versus full-64 teve delta `+0,0080`, IC95 `[-0,0103; +0,0305]`; isso não demonstra superioridade do fixed-32. A prevalência H8 OOD foi `56,86%`, contra `12,83%` ID e `13,84%` shift; portanto, AUPRCs absolutas não devem ser comparadas entre esses splits como se as prevalências fossem iguais.

As predictions JSONL completas, summary, figuras PNG/SVG, seals e dashboard offline estão em `runs/attr_p2/` e `docs/benchmarks/asm_transformer_transition_risk/p2/`.


## 17. Diagnóstico pós-hoc de Native Risk Mass

Depois da abertura e interpretação do P2, uma extensão diagnóstica comparou **ASM-X Base** (`use_powerlaw_risk=false`) com **ASM-X + Native Risk Mass** (`use_powerlaw_risk=true`). Ambos tinham `226.444` parâmetros incluindo as heads comuns, tensores inicializados de forma idêntica, as mesmas cinco seeds, episódios, heads, objetivos, calibração e 1.000 updates. A única diferença de configuração foi ativar o campo de risco nativo. Um manifesto pré-treino e um seal dos cinco checkpoints foram congelados; o seal P2 original de seis braços e G0–G5 permanecem inalterados.

Em ID, a AUPRC H8 agregada foi `0,1504970` no Base e `0,1504975` com Native Risk Mass. O delta hierárquico pareado foi `+0,00000001`, IC95 `[-0,00000560; +0,00000727]`. Em shift, o delta foi `-0,00000010`, IC95 `[-0,00000456; +0,00000250]`; em OOD, `-0,00000124`, IC95 `[-0,00001633; +0,00000481]`. Todos os intervalos de AUPRC por horizonte incluíram zero; Brier, recall/FPR no threshold e NLL de próximo estado também ficaram operacionalmente inalterados.

Os parâmetros de risco nativo receberam gradientes ATTR e se afastaram da inicialização em todas as seeds; portanto, o componente não estava simplesmente desabilitado. Porém, com o peso e objetivo nativos congelados, sua influência nas previsões de trajetória perigosa foi desprezível. Esta é somente evidência diagnóstica pós-hoc: ela não revisa G2 nem demonstra que outros objetivos, pesos ou arquiteturas de risco não possam ajudar.


## 18. Limitação de acoplamento entre métricas

A AUPRC H8 do P2 vem de uma `HazardHead` direta; ela não é calculada a partir da NLL da `NextStateHead` nem de um rollout multi-step de estados. As duas heads compartilham a representação do backbone e o objetivo conjunto, mas têm projeções e alvos separados. Assim, melhor dinâmica pode ajudar indiretamente, mas não existe restrição estrutural exigindo que NLL menor produza AUPRC maior.

Em ID, a associação descritiva de Spearman entre NLL e AUPRC nos seis braços é `-0,829`, mas o par ASM-X Base/Transformer é uma exceção: ASM-X Base tem NLL muito menor e AUPRC H8 praticamente empatada. As AUPRCs `0,1498` do Transformer e `0,1505` do ASM-X Base são somente `1,168×` e `1,173×` a prevalência H8 `0,1283`. Portanto, o resultado é ranking direto de hazard fraco nos dois, não boa previsão de trajetória pelo Transformer.

O P2 sustenta apenas uma afirmação de antecipação no nível da representação. Ele não mostra que qualidade de previsão do próximo estado medeie ou cause antecipação de hazard. Um protocolo futuro fundamentado em trajetória deve derivar risco de previsões multi-horizonte com predicado unsafe fixo ou testar explicitamente a mediação entre as heads de dinâmica e hazard.


O `sealed_metrics` corrigido rotula os painéis de classificação como métricas da head direta. Uma figura separada, `hazard_conditioned_dynamics`, condiciona a NLL de próximo estado aos passos H8-positivos e H8-negativos na avaliação. Para ASM-X Base/Transformer, a NLL nos passos positivos é `2,9290/4,1385` em ID, `6,5802/12,6509` em shift e `4,4692/9,1122` em OOD. O Transformer continua pior em dinâmica especificamente perto de hazards futuros. Os labels H8 são usados somente para estratificar a avaliação; nunca entram como input ou score de risco. Uma AUPRC derivada de trajetória exige novo preditor multi-horizonte e não pode ser fabricada da NLL one-step realizada sem leakage futuro.
