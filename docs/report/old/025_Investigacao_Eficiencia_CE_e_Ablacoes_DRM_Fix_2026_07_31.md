# Investigação da eficiência de CE e ablações DRM Fix

Data: 2026-07-31  
Branch de desenvolvimento: `drm-fix`  
Estado: triagem A–I e rescoring determinístico F/H/I concluídos

## 1. Objetivo

Este relatório documenta a investigação iniciada após a correção do baseline
GPT-2 independente. O objetivo é descobrir por que o DRM com aproximadamente
127 milhões de parâmetros apresenta menor eficiência amostral de
next-token cross-entropy que um GPT-2 de porte semelhante, identificar os
gargalos dominantes e testar correções de maneira controlada.

O objetivo experimental não é preservar uma conclusão anterior de
superioridade. É produzir uma comparação tecnicamente válida e encontrar
mudanças que melhorem de fato o DRM.

## 2. Origem da reavaliação

Durante a preparação do rescoring determinístico, foi identificado um
desalinhamento no treino original do baseline GPT-2:

1. `MemmapTokenDataset.make_batch` já entregava `x` e `y` deslocados em um
   token;
2. o treino passava `labels=y` para `GPT2LMHeadModel`;
3. o Hugging Face aplicava internamente outro deslocamento causal.

O baseline GPT-2 original foi, portanto, treinado com deslocamento duplo. Os
resultados isolados do DRM continuaram válidos, mas a conclusão comparativa
DRM versus GPT-2 foi invalidada.

A correção passou a calcular explicitamente:

```text
CE(logits(x), y)
```

sem entregar `labels` ao mecanismo de loss interno do GPT-2. Foi adicionado um
teste de regressão para impedir a reintrodução do deslocamento duplo.

O GPT-2 corrigido alcançou, ainda no começo do treino:

```text
tokens_seen = 51.003.392
validation CE = 1,2037
```

Esse valor já era inferior aos melhores valores intermediários dos DRMs
treinados até 150M tokens, tornando necessária uma investigação arquitetural.

## 3. Hipótese inicial: faltam as fases formais do roadmap?

O projeto permanece antes da Fase 1 do roadmap formal. Ainda faltam:

- rank, kernel e estratos reais da métrica;
- anchor explícito;
- conexão e transporte;
- mapas de transição entre estratos;
- holonomia e histerese;
- diagnósticos de redução Riemanniana, sub-Riemanniana e Fisher–Rao.

Essas fases são necessárias para aproximar a implementação do formalismo do
paper. Entretanto, a maior parte delas foi planejada inicialmente como
diagnóstico. Implementar apenas diagnósticos não aumenta a capacidade causal
do language model nem reduz automaticamente CE.

A conclusão da auditoria foi:

> A ausência das fases formais pode limitar o desenvolvimento conceitual do
> DRM, mas não é a explicação imediata mais provável para a baixa eficiência
> amostral. O gargalo dominante está no caminho sequencial atualmente
> utilizado.

## 4. Alocação dos 127M parâmetros

O checkpoint DRM block64/local-mixer possui 127.266.438 parâmetros:

| Componente | Parâmetros | Percentual |
|---|---:|---:|
| Campo de direções | 48.151.616 | 37,84% |
| Métrica relacional | 53.428.736 | 41,98% |
| Flow | 11.243.072 | 8,83% |
| Emitter | 6.427.904 | 5,05% |
| Risk field | 5.515.782 | 4,33% |
| Local mixer | 2.104.576 | 1,65% |
| Token embedding | 393.216 | 0,31% |

Campo de direções e métrica concentram aproximadamente 101,6M parâmetros, ou
79,8% do modelo. O local mixer, responsável pela principal interação causal
não linear dentro de cada bloco, representa somente 1,65%.

Há ainda 5,5M parâmetros no `RiskField` mesmo com
`use_powerlaw_risk: false`. Nesse modo, o módulo retorna zeros e sua rede não
participa da previsão.

## 5. Gargalo do caminho block64

No caminho `directional_block_cumsum`, a geometria é calculada uma vez a partir
do estado inicial do bloco:

```python
base_directions, base_gates = self.direction_field(z_start)
base_metric_diag, base_metric_u = self.metric(z_start)
```

Direções, gates e métrica são reutilizados para os 64 tokens. Antes do mixer, o
estado tem a forma aproximada:

```text
z_t = z_start + soma(delta(token_i, z_start), i <= t)
```

Isso fornece um prefixo causal, mas oferece pouca composição não linear e
sensível à ordem entre tokens distantes. A ordem local é parcialmente
recuperada pelo mixer convolucional de duas camadas e kernel 8, cujo campo
receptivo é aproximadamente 15 tokens.

O GPT-2, em contraste, distribui seus parâmetros por 13 camadas sucessivas de
attention e MLP. Cada camada recompõe o contexto. O DRM atual concentra a maior
parte da capacidade em uma geometria compartilhada pelo bloco inteiro.

Hipótese principal:

> O DRM não sofre por falta de parâmetros, mas porque a maior parte deles não
> está alocada em composição causal profunda entre tokens.

## 6. Hipóteses auxiliares

### 6.1 Losses geométricas

Em uma janela de um checkpoint DRM, as contribuições observadas foram:

| Termo | Contribuição aproximada |
|---|---:|
| CE | 1,02889 |
| `metric_u_target` | 0,03148 |
| `condition` | 0,00358 |
| `dim_sparsity` | 0,00161 |
| `action` | 0,00093 |
| demais auxiliares | aproximadamente 0,00024 |

As losses auxiliares adicionavam cerca de 3,7% à loss total. Isso poderia
reduzir eficiência amostral, mas não parecia suficiente para explicar sozinho
a diferença para o GPT-2.

### 6.2 Gates

No DRM de 150M tokens auditado, a atividade soft chegou a aproximadamente
2,5%, equivalente a `dimD` próximo de 1,6 entre 64 direções.

A implementação anterior de `active_fraction_loss` era:

```python
(active_fraction - target).clamp_min(0).pow(2)
```

Ela tratava o alvo como limite máximo. Valores abaixo do alvo não eram
penalizados. Ao mesmo tempo, o modelo tinha bias negativo nos gates e
regularização de sparsity.

Foi adicionada a opção `active_fraction_loss_mode: target`, que usa erro
quadrático simétrico em torno do alvo.

## 7. Estratégia de ablação A–I

As mudanças foram separadas para identificar causalmente a fonte de qualquer
ganho. Aplicar tudo de uma vez poderia melhorar o CE, mas não mostraria qual
mudança ajudou, qual prejudicou ou quais apenas se compensaram.

As variantes A–G são parcialmente cumulativas:

```text
A = baseline atual
B = A sem losses auxiliares
C = B + gates corrigidos
D = C + residual token -> estado
E = D + bloco 16
F = D + mixer dilatado
G = reorganização profunda baseada em B/C/D
H = combinação das mudanças vencedoras, sem gates forçados
```

### A — baseline atual

Verifica a reprodutibilidade do DRM block64:

- block size 64;
- velocity cumsum;
- local mixer causal;
- hidden size 256;
- kernel 8;
- duas camadas;
- losses geométricas originais.

Parâmetros: 127.266.438.

### B — somente CE

Mantém a arquitetura de A e zera as losses auxiliares:

- action;
- sparsity e entropy;
- variance;
- metric regularization e diversity;
- recurrence e stability;
- blindspot;
- active fraction;
- metric U floor e target;
- condition;
- consistency.

Objetivo: verificar se o DRM aprende linguagem mais rapidamente quando todo o
gradiente de otimização é direcionado ao next-token CE.

Parâmetros: 127.266.438.

### C — gates corrigidos

Parte de B e adiciona:

```text
active_fraction_loss_mode = target
target_active_fraction = 0,5
lambda_active_fraction = 0,01
gate_logit_bias = 0
```

Objetivo: testar se gates quase fechados limitavam a dinâmica.

Parâmetros: 127.266.438.

### D — residual token para estado

Parte de C e adiciona uma projeção causal direta:

```text
state_t <- state_t + scale * W(token_embedding_t)
```

O residual evita que toda a informação lexical precise atravessar campo de
direções, gates, métrica e flow antes de alcançar o emitter.

Parâmetros: 129.625.734.

### E — blocos de 16

Parte de D e reduz o bloco de 64 para 16 tokens.

Objetivo: recalcular a geometria quatro vezes mais frequentemente e reduzir a
região em que direções e métrica permanecem congeladas.

Trade-off esperado: maior custo computacional e menor throughput.

Parâmetros: 129.625.734.

### F — mixer causal dilatado

Parte de D e substitui o mixer curto por:

- hidden size 512;
- quatro camadas;
- kernel 8;
- crescimento de dilatação 2;
- campo receptivo teórico de 106 tokens.

Objetivo: ampliar composição causal e sensibilidade à ordem sem pagar o custo
de recalcular toda a geometria a cada token.

Parâmetros: 132.525.446.

### G — profundidade DRM efetiva

G reorganiza o orçamento em um DRM mais estreito de dois estágios:

- `d_token = 1024`;
- `d_state = 1024`;
- `hidden_size = 3200`;
- 48 direções;
- metric rank 48;
- bases direcionais e métricas de tamanho 64;
- residual token para estado;
- um estágio adicional de refinamento DRM causal.

No refinamento, a geometria da posição `t` é condicionada ao estado causal
precedente produzido pelo primeiro estágio. Isso adiciona composição
state-dependent sem executar uma recorrência estritamente serial dentro de
cada estágio.

Parâmetros: 125.829.830.

### H — combinação parameter-matched

Após a conclusão de A–G, H foi definida como:

```text
H =
CE-only de B
+ residual token -> estado de D
+ mixer dilatado de F
- correção forçada dos gates de C
- block16 de E
- segundo estágio de G
```

H também define `instantiate_disabled_risk: false`. Quando
`use_powerlaw_risk` está desabilitado, o modelo deixa de criar os 5,5M
parâmetros sem gradiente do `RiskField`.

Parâmetros:

```text
DRM H  = 127.009.664
GPT-2  = 126.080.640
diferença = 929.024 parâmetros, aproximadamente 0,74%
```

O default de `instantiate_disabled_risk` permanece `true`, preservando
compatibilidade estrutural com checkpoints antigos.

### I — F parameter-matched e isolamento dos gates

H terminou com:

```text
validation CE = 1,8520
throughput = 15.070 tokens/s
```

H não preservou o `1,8408` de F e ficou praticamente empatada com D. Como H
removeu ao mesmo tempo os gates corrigidos e os parâmetros mortos do
`RiskField`, foi necessário um experimento isolador.

I mantém exatamente a arquitetura e o orçamento de H, alterando somente:

```text
active_fraction_loss_mode = target
lambda_active_fraction = 0,01
target_active_fraction = 0,5
gate_logit_bias = 0
```

Portanto:

```text
H versus I = efeito limpo dos gates corrigidos sob mixer dilatado
F versus I = efeito da remoção dos parâmetros mortos, com ressalva de RNG
```

I possui 127.009.664 parâmetros, igual a H.

I terminou a triagem amostrada com:

```text
validation CE = 1,8535
throughput = 15.334 tokens/s
```

## 8. Implementação

Arquivos principais:

```text
configs/drm_fix_ablation_variants.json
scripts/run_drm_fix_ablation.py
scripts/run_drm_fix_ablation.sh
src/drm_language_emitter/config.py
src/drm_language_emitter/losses.py
src/drm_language_emitter/model_components.py
src/drm_language_emitter/directional_blocks.py
src/drm_language_emitter/model.py
tests/test_forward.py
tests/test_losses.py
```

Foram adicionados:

- modo simétrico para a loss de atividade;
- residual token-to-state opcional;
- dilatação crescente no local mixer;
- `DRMRefinementLayer`;
- matriz declarativa A–G;
- execução sequencial;
- retomada por `checkpoint_latest.pt` ou `checkpoint_best.pt`;
- modo `--plan-only`;
- testes de causalidade e propagação de gradientes.

Os novos campos têm defaults retrocompatíveis. O carregamento dos checkpoints
DRM anteriores foi testado.

## 9. Validação de implementação

Antes da triagem:

- todas as variantes A–G completaram forward smoke test;
- todas produziram loss finita;
- causalidade de prefixo foi testada no residual, mixer dilatado e refinamento;
- gradientes do residual e refinamento foram verificados;
- checkpoint DRM antigo de 127.266.438 parâmetros foi carregado;
- 77 testes passaram e 1 foi ignorado;
- `git diff --check` passou.

## 10. Protocolo da triagem

Comando:

```bash
./scripts/run_drm_fix_ablation.sh \
  --variants all \
  --target-tokens 5000000 \
  --output-root runs/drm_fix_ablation_5m
```

Condições:

- seed 1;
- 5M tokens por variante;
- mesmos manifests document-disjoint;
- validação a cada 1M tokens;
- 16 batches por avaliação;
- mesma sequência determinística de validação em passos equivalentes;
- PG-19 não utilizado.

Esta triagem serve para eliminar hipóteses fracas. Não é evidência final de
generalização.

## 11. Resultados parciais A–C

| Tokens | A | B | C |
|---:|---:|---:|---:|
| 1M | 2,4047 | 2,3844 | 2,3985 |
| 2M | 2,2095 | 2,1897 | 2,1853 |
| 3M | 2,1001 | 2,0862 | 2,0859 |
| 4M | 1,9774 | 1,9572 | 1,9565 |
| 5M | 1,8789 | 1,8616 | 1,8603 |

### A versus B

B melhorou A em aproximadamente 0,0174 CE aos 5M tokens, cerca de 0,9%.
O ganho foi pequeno, mas consistente ao longo da curva.

Conclusão provisória:

> As losses auxiliares causam uma perda pequena de eficiência, mas não são o
> gargalo dominante.

### B versus C

C melhorou B em aproximadamente 0,0012 CE aos 5M tokens, um empate prático.

Os gates, porém, mudaram substancialmente:

| Diagnóstico | A | B | C |
|---|---:|---:|---:|
| Atividade soft | 9,8% | 83,5% | 54,5% |
| `dimD_mean` | 6,25 | 53,41 | 34,85 |
| Gate entropy | 0,246 | 0,359 | 0,667 |

C corrigiu efetivamente o alvo dos gates, mas isso não trouxe ganho material
de CE.

Conclusão provisória:

> Gates quase fechados não parecem ser o gargalo principal da previsão
> next-token.

### Condicionamento da métrica

Foi observado:

```text
condition_proxy A = 156
condition_proxy B = 185
condition_proxy C = 24.820
```

C produziu uma métrica muito mais mal condicionada sem melhorar CE. Portanto,
C isoladamente não deve ser promovida sem controle adicional da métrica.

## 12. Resultados D–G

### D — residual direto

```text
validation CE = 1,8526
throughput = 15.036 tokens/s
```

D melhorou C em aproximadamente 0,00775 CE, com queda de throughput próxima de
5,2%. O residual ajuda, mas não resolve isoladamente o gargalo.

### E — block16

E foi interrompida após 1M tokens:

```text
validation CE = 2,4315
throughput = 6.413 tokens/s
```

Além de ficar atrás das variantes anteriores no mesmo ponto, perdeu
aproximadamente 57% de throughput em relação a D. O custo de recalcular a
geometria quatro vezes mais frequentemente não se justificou.

### F — mixer causal dilatado

```text
validation CE = 1,8408
throughput = 14.884 tokens/s
```

F foi a vencedora da triagem:

- melhora de aproximadamente 0,03812 CE sobre A;
- melhora de aproximadamente 0,02074 CE sobre B;
- melhora de aproximadamente 0,01178 CE sobre D;
- perda de throughput de aproximadamente 6,5% contra A;
- campo receptivo causal ampliado de 15 para 106 tokens.

O resultado sustenta a hipótese de que o campo receptivo causal curto era um
gargalo real.

### G — segundo estágio DRM

```text
validation CE = 1,8675
throughput = 14.873 tokens/s
```

G ficou 0,0267 CE atrás de F com throughput praticamente idêntico. A
profundidade state-dependent testada adicionou complexidade sem melhorar a
eficiência amostral e foi descartada nessa configuração.

### Resumo final A–G

| Variante | Val CE | Throughput | Decisão |
|---|---:|---:|---|
| A | 1,8789 | 15.924 | baseline |
| B | 1,8616 | 16.036 | manter CE-only |
| C | 1,8603 | 15.865 | gates forçados descartados |
| D | 1,8526 | 15.036 | residual mantido |
| E | 2,4315 em 1M | 6.413 | interrompida |
| F | **1,8408** | 14.884 | vencedora |
| G | 1,8675 | 14.873 | descartada |

## 13. Estado atual das hipóteses

| Hipótese | Estado após A–G |
|---|---|
| Auxiliares explicam a diferença | Enfraquecida; impacto pequeno |
| Gates fechados explicam a diferença | Fortemente enfraquecida |
| Falta caminho lexical direto | Parcialmente sustentada; ganho pequeno em D |
| Geometria fica congelada por tempo demais | Solução block16 inviável pelo custo |
| Campo receptivo causal é curto | Sustentada por F |
| Falta profundidade state-dependent | Não sustentada pela configuração G |
| Fases formais ausentes são a causa imediata | Não sustentada até agora |

## 14. Resultado de H/I e rescoring determinístico

H foi executada no mesmo protocolo de 5M tokens:

```text
H validation CE = 1,8520
F validation CE = 1,8408
diferença = 0,0111 a favor de F
```

I foi executada no mesmo protocolo:

```text
I validation CE = 1,8535
H validation CE = 1,8520
```

Como o validation CE intermediário usava amostras diferentes em cada passo,
F, H e I foram reavaliadas sequencialmente nos mesmos 4.834.787 targets.

| Variante | CE determinístico | Perplexidade | Delta para F |
|---|---:|---:|---:|
| F | **1,875146** | **6,5218** | 0 |
| I | 1,884072 | 6,5802 | +0,008926 |
| H | 1,885751 | 6,5913 | +0,010606 |

Hashes:

```text
F cf19a557c25f805f86061c211778d1cee051bd34f831bb905d5c09174b98e79f
I 0a1951ac91ad3b56d8f9afa243ce868b4f99071b106ceec409b61d512dcfdccb
H 4eb80010231f169046a6f7c6d7daac70c1a7bf9a3b74f692536fe327669ad13f
```

F venceu a seed 1. I melhorou H em apenas 0,001679 CE, indicando que a
correção dos gates tem efeito pequeno quando arquitetura e sequência de
inicialização são mantidas.

F e I possuem comportamento arquitetural quase idêntico, mas a remoção física
do `RiskField` em I alterou a sequência do RNG usada para inicializar emitter e
mixer. Portanto, a diferença F–I ainda não pode ser atribuída com segurança à
configuração. Ela pode refletir sorte de inicialização.

## 15. Decisão posterior

Após o rescoring:

1. comparar I, H e F nos mesmos checkpoints de tokens;
2. registrar F como vencedora observada da seed 1;
3. não promover F apenas pelo delta de uma inicialização;
4. tornar a inicialização independente da presença do `RiskField`;
5. repetir a comparação dilated-mixer parameter-matched em múltiplas seeds;
6. promover para 30M somente após confirmar o ganho;
7. manter PG-19 congelado durante todo o desenvolvimento.

## 16. Conclusão parcial

Os primeiros resultados já mostram que simplesmente remover regularização ou
abrir gates não transforma a eficiência do DRM. Embora essas mudanças alterem
fortemente a geometria interna, o CE permanece praticamente igual.

Isso reforça o diagnóstico central:

> O problema mais provável está na composição causal entre tokens: geometria
> compartilhada por blocos longos, mixer pequeno e pouca profundidade
> state-dependent.

F demonstrou que ampliar o campo receptivo causal produz o melhor ganho entre
as mudanças testadas. O rescoring determinístico confirmou F como vencedora
observada, mas H e I mostraram que os gates explicam muito pouco da diferença.

O próximo passo não é escolher pelo menor número isolado. É controlar a
inicialização e confirmar o mixer dilatado em múltiplas seeds.

## 17. Implementação do próximo ciclo

A inicialização passou a usar sementes locais e offsets estáveis por componente.
Assim, adicionar ou remover um módulo opcional não desloca o RNG dos módulos
seguintes. Um teste automatizado confirma que ativar/desativar `RiskField` não
altera os pesos dos componentes compartilhados.

Foram acrescentadas duas variantes:

| Variante | Arquitetura | Parâmetros |
|---|---|---:|
| J | residual token→estado + memória seletiva forget/write + mixer curto | 126.080.896 |
| J_DILATED | J + mixer causal dilatado de quatro camadas | 127.191.968 |

J está a apenas 256 parâmetros do GPT-2 de referência (126.080.640). A memória
usa gates por canal para esquecimento e escrita, com recorrência causal
associativa. A implementação paralela foi comparada com uma recorrência
sequencial de referência e também possui testes de causalidade e gradientes.

Foi criada uma tarefa MQAR/associative recall que gera novas associações
chave→valor a cada lote e mede CE e acurácia somente nas posições de consulta.
Ela permite distinguir capacidade de recuperação de simples melhora no
language modeling.

## 18. Protocolo pareado e gate de promoção

O ciclo passa a ser:

1. executar F e I nas seeds 1, 2 e 3 com os mesmos dados e hiperparâmetros;
2. reavaliar cada checkpoint na mesma sequência contínua de validação;
3. executar MQAR para F, I, J e J_DILATED;
4. treinar J e J_DILATED em 5M tokens e fazer o mesmo rescoring;
5. promover uma candidata somente se ela cumprir simultaneamente:
   - três seeds pareadas;
   - redução média de CE de pelo menos 0,005;
   - vitória em pelo menos duas das três seeds;
   - desvio-padrão de CE não superior a 0,03.

O gate é mecanizado por `scripts/check_drm_fix_promotion.py`; uma decisão
reprovada encerra com código 2 e não sugere a execução de 30M. Portanto, neste
momento nenhuma variante está promovida: a implementação está pronta, mas os
novos resultados pareados ainda precisam ser produzidos.
