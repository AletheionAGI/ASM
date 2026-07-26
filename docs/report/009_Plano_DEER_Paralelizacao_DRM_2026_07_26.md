# Plano DEER Para Paralelizacao Temporal Do DRM

Data do relatorio: 2026-07-26  
Branch alvo: `drm-deer`  
Projeto auditado: `drm-language-emitter`  

## 1. Resumo executivo

O gargalo principal do `drm-language-emitter` e o loop recorrente token-a-token no `DRMEmitterModel`. Esse loop preserva a natureza do DRM, porque o estado latente `z` evolui causalmente conforme cada token e consumido. O problema e que essa forma de execucao tem baixa paralelizacao no eixo temporal: mesmo usando GPU, o treinamento executa centenas de passos dependentes por sequencia.

O metodo DEER, apresentado em "Parallelizing non-linear sequential models over the sequence length" por Lim, Zhu, Selfridge e Firmansyah, oferece uma alternativa conceitualmente adequada: manter a mesma recorrencia matematica, mas resolver a trajetoria inteira de estados como um problema de ponto fixo usando Newton paralelo e scans associativos. Em principio, isso pode preservar o DRM enquanto reduz a dependencia sequencial de `O(T)` para poucas iteracoes paralelas.

Porem, a aplicacao direta de DEER ao DRM 125M atual nao e trivial. O estado do modelo local e grande:

```text
d_state = 1536
seq_len = 512
batch_size tipico = 2 a 4
```

Um Newton denso sobre estados desse tamanho tende a exigir Jacobianos grandes, muita memoria e cuidado numerico. Trabalhos posteriores sobre DEER apontam complexidade cubica no tamanho do estado e instabilidade como riscos praticos. Portanto, a recomendacao nao e substituir imediatamente o forward do 125M, mas criar uma linha experimental incremental:

1. Prototipo DEER em recorrencia pequena e independente.
2. Prototipo DEER no DRM tiny com `d_state` reduzido.
3. Validacao de equivalencia contra o loop sequencial.
4. Variante blockwise ou quasi-Newton para escalabilidade.
5. Integracao opcional via flag, sem mudar o caminho padrao.

## 2. Problema atual

### 2.1 Loop temporal sequencial

O forward do DRM segue a forma:

```python
z = z0
for t in range(seq_len):
    e_t = token_embedding(input_ids[:, t])
    z = drm_step(z, e_t)
    logits_t = emitter(z)
```

Na implementacao atual, o loop real tambem inclui:

```text
direction_field(z)
metric(z)
risk(z)
flow(z, token, directions, gates)
metric.naturalize(...)
updater(z, dz)
losses auxiliares
diagnosticos opcionais
```

Mesmo quando `collect_diagnostics=False`, o caminho quente continua dependente de `z[t]` para produzir `z[t+1]`.

### 2.2 Por que batch nao resolve completamente

A GPU ja paraleliza o batch e as operacoes internas de matriz. Isso ajuda, mas nao elimina o gargalo de comprimento de sequencia:

```text
batch axis: paralelizavel
state/channel axis: paralelizavel
token/time axis: atualmente sequencial
```

Para `seq_len=512`, cada microbatch executa centenas de atualizacoes causais. O custo nao e apenas FLOPs; tambem ha overhead de Python, lancamentos de kernels pequenos, autograd em passos curtos e baixa ocupacao relativa comparada a blocos densos de Transformer.

### 2.3 Por que aumentar throughput pode prejudicar qualidade

Os experimentos de throughput mostraram que reduzir custo alterando muito a dinamica pode degradar qualidade. Em particular, aumentar agressivamente intervalos como:

```text
geometry_update_interval
aux_loss_interval
naturalization_interval
```

reduz custo, mas tambem muda a frequencia com que a geometria relacional corrige a trajetoria. Isso nao e apenas uma otimizacao de execucao; afeta a dinamica treinada.

O objetivo do DEER e diferente: acelerar a solucao da mesma recorrencia, nao diluir componentes do DRM.

## 3. O que DEER propoe

### 3.1 Reformulacao por ponto fixo

Considere a recorrencia:

```text
z_{t+1} = f_theta(z_t, x_t)
```

Em vez de computar `z_1, z_2, ..., z_T` sequencialmente, DEER trata toda a trajetoria como variavel:

```text
Z = [z_1, z_2, ..., z_T]
```

e define um sistema de equacoes:

```text
F(Z) = 0
```

onde cada bloco mede a violacao da recorrencia:

```text
F_t(Z) = z_{t+1} - f_theta(z_t, x_t)
```

Resolver a recorrencia vira resolver esse sistema. Newton ou variantes quasi-Newton atualizam a trajetoria inteira em paralelo.

### 3.2 Paralelizacao por linearizacao local

Em uma iteracao de Newton, lineariza-se a recorrencia ao redor de uma trajetoria chute:

```text
f_theta(z_t + delta_t, x_t)
~= f_theta(z_t, x_t) + J_t delta_t
```

Isso produz um sistema temporal com estrutura quase bidiagonal:

```text
delta_{t+1} - J_t delta_t = residual_t
```

Esse tipo de sistema pode ser resolvido com parallel scan ou algoritmos similares, reduzindo a profundidade temporal efetiva.

### 3.3 Preservacao matematica

Se o solver converge, a trajetoria final satisfaz a mesma recorrencia:

```text
z_{t+1} = f_theta(z_t, x_t)
```

Portanto, DEER pode ser visto como um solver alternativo para o DRM, nao como uma nova arquitetura. Esse e o argumento forte a favor: a identidade do modelo pode ser preservada.

## 4. Encaixe no DRM

### 4.1 Recorrencia DRM minima

Para aplicar DEER, precisamos isolar uma funcao pura:

```text
drm_transition(z_t, token_t, global_step, tick) -> z_{t+1}
```

No DRM atual, uma transicao inclui:

```text
1. calcular ou reaproveitar geometria;
2. calcular direcoes e gates;
3. calcular metrica relacional;
4. calcular fluxo dz;
5. naturalizar dz;
6. atualizar estado z;
7. opcionalmente computar perdas auxiliares.
```

O primeiro prototipo deve remover variaveis externas do solver e usar a forma mais limpa:

```text
geometry_update_interval = 1
naturalization_interval = 1
aux_loss_interval = 1
collect_diagnostics = False
```

Depois que a equivalencia funcionar, podemos reintroduzir cache e amostragem auxiliar.

### 4.1.1 Atualizacao apos prototipo `directional_candidates`

O modo `directional_candidates` melhora a semantica do passo local: dentro de cada token, ele gera candidatos por direcoes ativas, avalia custos em paralelo e combina endpoints por softmax. Isso paraleliza o eixo de direcoes, mas nao remove o loop temporal:

```text
for t in seq_len:
    z_{t+1} = directional_candidates(z_t, token_t)
```

Portanto, ele deve ser visto como a transicao `F` a ser usada por DEER/quasi-DEER:

```text
z_{t+1} = F_directional(z_t, x_t)
```

O proximo prototipo correto para atacar o `for t` deve tratar `F_directional` como funcao de ponto fixo sobre a trajetoria inteira:

```text
Z = Phi(Z, X)
```

onde:

```text
Phi_t(Z, X) = F_directional(z_{t-1}, x_t)
```

Como Newton denso exige Jacobianos grandes, a primeira implementacao nao deve formar `dF/dz`. A ordem recomendada passa a ser:

```text
1. implementar fixed-point iteration sobre toda a trajetoria;
2. adicionar Anderson acceleration sem Jacobiano;
3. medir convergencia contra o loop sequencial;
4. so depois medir wall-clock.
```

Para esta etapa, o criterio de sucesso volta a ser equivalencia numerica, porque o solver tenta resolver a mesma recorrencia, nao uma nova arquitetura.

### 4.1.2 Resultado inicial: warmstart cumulativo

O chute inicial `z0` repetido para todos os tempos e fraco. Em um probe tiny com `seq_len=16`, `batch=2`, `d_state=12`, Anderson partindo desse chute reduziu residual, mas ainda ficou longe do rollout sequencial em poucas iteracoes.

Um warmstart melhor e barato foi adicionado:

```text
1. avaliar F(z0, x_t) para todos os tokens em paralelo;
2. extrair deltas locais F(z0, x_t) - z0;
3. construir uma trajetoria inicial por soma cumulativa desses deltas.
```

Esse warmstart e exato para dinamicas aditivas independentes do estado e aproximado para o DRM. No probe tiny, ele reduziu a diferenca inicial contra o rollout sequencial de aproximadamente `0.053` para `0.0012` em max-abs. Com quatro iteracoes Anderson, a diferenca ficou abaixo de `0.001`.

Isso muda a avaliacao da linha DEER: Anderson sem warmstart forte nao parecia promissor; Anderson com warmstart cumulativo merece ser integrado como modo experimental de trajetoria.

O proximo gate e:

```text
seq_len 32/64
CUDA
comparar rollout sequencial vs Anderson com warmstart cumulativo
medir max_abs_diff, residual e wall-clock
```

### 4.2 Saidas e loss

O forward sequencial atual produz logits a partir dos estados:

```text
logits_t = emitter(z_t)
```

Um solver DEER deve produzir a mesma sequencia de estados. Entao a CE continua igual:

```text
loss = cross_entropy(emitter(Z), targets)
```

Isso permite comparar diretamente:

```text
max_abs_diff(states_sequential, states_deer)
max_abs_diff(logits_sequential, logits_deer)
abs(loss_sequential - loss_deer)
```

## 5. Riscos tecnicos

### 5.1 Custo de Jacobiano

O maior risco e o tamanho do estado:

```text
d_state = 1536
```

Um Jacobiano denso por passo teria tamanho:

```text
1536 x 1536 = 2.359.296 elementos
```

Para 512 tokens, isso fica rapidamente inviavel se armazenado de forma ingenua.

Consequencia: DEER denso direto no DRM 125M provavelmente nao e a primeira implementacao correta.

### 5.2 Instabilidade numerica

Newton pode divergir se a trajetoria inicial for ruim, se a dinamica for mal condicionada ou se os Jacobianos amplificarem erro. O DRM usa:

```text
bounded_state
metric naturalization
low-rank metric solve
gates direcionais
```

Esses componentes podem ajudar estabilidade, mas tambem tornam a linearizacao mais complexa.

### 5.3 Backpropagation

Mesmo se o forward paralelo convergir, o backward precisa ser correto. Existem duas opcoes:

```text
1. diferenciar atraves das iteracoes do solver;
2. usar diferenciacao implicita.
```

A primeira e mais simples, mas pode consumir memoria. A segunda e mais elegante, mas aumenta complexidade.

### 5.4 Interacao com caches

O DRM atual permite cache de geometria:

```text
geometry_update_interval > 1
```

Esse cache torna a funcao de transicao dependente de ticks e estados anteriores. Para DEER, isso deve ser tratado explicitamente. No prototipo inicial, o cache deve ser desativado.

### 5.5 Qualidade de treinamento

Mesmo preservando a recorrencia no limite, um solver aproximado com poucas iteracoes pode introduzir erro. Esse erro pode funcionar como ruído de treinamento. Pode ajudar ou prejudicar; precisa ser medido.

## 6. Solucoes possiveis

### 6.1 Solucao A: fast path sequencial otimizado

Antes de DEER, ainda ha uma melhoria simples:

```text
forward_train_fast
```

Esse caminho manteria o loop sequencial, mas removeria:

```text
dicts por passo
listas desnecessarias
diagnosticos
condicionais fora do modo usado
calculos auxiliares quando lambda=0
```

Vantagem:

```text
baixo risco
preserva exatamente o DRM
facil de testar
```

Limite:

```text
continua O(T) no eixo temporal
```

### 6.2 Solucao B: compile do passo DRM

O codigo ja tem suporte conceitual a:

```text
compile_drm_step
use_torch_compile
```

Compilar apenas o passo `drm_step` tende a ser mais realista que compilar o forward inteiro. O forward completo tem listas, condicionais e loops Python, enquanto o step e mais estatico.

Vantagem:

```text
baixo a medio risco
sem alteracao matematica
```

Limite:

```text
ganho depende do backend PyTorch/Windows/CUDA
nao paraleliza tempo
```

### 6.3 Solucao C: DEER tiny

Implementar DEER em escala reduzida:

```text
d_state: 16, 32, 64
seq_len: 16, 32, 64
batch: 1 a 4
```

Meta:

```text
provar equivalencia contra forward sequencial
```

Essa e a primeira etapa recomendada para a branch `drm-deer`.

### 6.4 Solucao D: DEER blockwise

Aplicar DEER dentro de blocos curtos:

```text
bloco 0: tokens 0..63
bloco 1: tokens 64..127
...
```

Os blocos continuam sequenciais entre si, mas cada bloco pode resolver estados em paralelo internamente.

Vantagem:

```text
reduz memoria
limita instabilidade
mantem compatibilidade com seq_len 512
```

Limite:

```text
nao remove toda dependencia temporal
```

### 6.5 Solucao E: quasi-Newton / ELK

Trabalhos posteriores a DEER propuseram quasi-Newton e estabilizacao inspirada em Levenberg-Marquardt/Kalman smoothing. Essa linha e mais adequada para estados grandes como o DRM 125M.

Vantagem:

```text
mais escalavel que Newton denso
potencialmente mais estavel
```

Limite:

```text
implementacao mais dificil
exige validacao numerica cuidadosa
```

### 6.6 Solucao F: DEER em subespaco

Resolver correcao da trajetoria em um subespaco:

```text
delta_z = P delta_h
dim(delta_h) << d_state
```

Isso combina bem com o DRM, porque ele ja trabalha com direcoes e bases:

```text
n_directions
direction_basis_size
metric_u_basis_size
```

Vantagem:

```text
compatibilidade conceitual com geometria direcional
menor custo de Jacobiano
```

Limite:

```text
pode deixar de resolver a recorrencia completa
passa a ser aproximacao arquitetural/solver
```

## 7. Plano de implementacao recomendado

### Fase 0 - Baseline congelado

Antes de implementar DEER, registrar resultados do forward sequencial:

```text
modelo tiny
seed fixo
input fixo
states
logits
loss
tempo
```

Critério:

```text
teste deterministico reproduzivel
```

### Fase 1 - Recorrencia generica pequena

Criar um modulo experimental:

```text
src/drm_language_emitter/deer.py
```

Com uma API independente:

```python
def sequential_solve(f, z0, inputs):
    ...

def deer_solve(f, z0, inputs, *, iterations, damping):
    ...
```

O primeiro `deer_solve` pode ser pedagogico e limitado, com foco em corretude.

Testes:

```text
tests/test_deer.py
```

Casos:

```text
recorrencia linear simples
recorrencia nao-linear pequena
comparacao contra sequential_solve
```

### Fase 2 - Adaptador DRM tiny

Criar adaptador que expose:

```text
f(z_t, token_t) = drm_transition(z_t, token_t)
```

Configuracao:

```text
d_state pequeno
d_token pequeno
n_directions pequeno
metric_rank pequeno
seq_len curto
geometry_update_interval = 1
naturalization_interval = 1
```

Critérios:

```text
max_abs_diff estados < tolerancia definida
loss_deer proxima de loss_sequential
backward sem NaN
```

### Fase 3 - Backward e treinamento smoke

Treinar por poucos steps em dataset tiny:

```text
steps: 3, 10, 50
```

Comparar:

```text
loss inicial
loss final
grad_norm
tempo por step
uso de memoria
```

### Fase 4 - Blockwise DEER

Se a fase tiny funcionar, implementar:

```text
sequence_solver = sequential | deer_blockwise
deer_block_size = 16 | 32 | 64
deer_iterations = 2 | 3 | 5
damping = ...
```

Critério:

```text
erro controlado e throughput superior em seq_len maior
```

### Fase 5 - Escala intermediaria

Testar em configuracoes entre tiny e 125M:

```text
d_state 128
d_state 256
d_state 512
seq_len 128
seq_len 256
```

So depois considerar 125M.

## 8. Metricas de sucesso

### 8.1 Corretude

```text
max_abs_diff(states) <= tolerancia
max_abs_diff(logits) <= tolerancia
abs(loss_diff) <= tolerancia
sem NaN/Inf
```

### 8.2 Treinabilidade

```text
gradientes finitos
loss diminui em smoke tests
val_ce nao degrada contra baseline sequencial
```

### 8.3 Performance

```text
tokens/sec
step_elapsed_sec
forward_backward_elapsed_sec
max_memory_mb
```

### 8.4 Qualidade final

Comparar no mesmo budget:

```text
tokens_seen
best_val_ce
final_val_ce
amostras qualitativas
```

## 9. Decisoes de engenharia

### 9.1 Nao substituir o forward padrao inicialmente

O forward sequencial deve continuar sendo o baseline autoritativo.

### 9.2 Feature flag obrigatoria

Qualquer solver novo deve ficar atras de uma flag:

```text
sequence_solver: sequential | deer
```

ou argumento de script:

```text
--sequence-solver deer
```

### 9.3 Sem mudanca silenciosa de qualidade

DEER nao deve entrar no treinamento principal ate provar:

```text
equivalencia em tiny
estabilidade em smoke train
comparacao de val_ce
```

### 9.4 Primeiro solver simples, depois solver rapido

A primeira implementacao deve priorizar clareza e teste. Otimizacoes com `torch.compile`, CUDA Graph ou kernels customizados devem vir depois.

## 10. Referencias

- Yi Heng Lim, Qi Zhu, Joshua Selfridge, Muhammad Firmansyah Kasim. "Parallelizing non-linear sequential models over the sequence length." ICLR 2024.  
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3bfbd65743e60c685a3845bd61ce15f-Abstract-Conference.html

- Xavier Gonzalez, Andrew Warrington, Jimmy T. H. Smith, Scott W. Linderman. "Towards Scalable and Stable Parallelization of Nonlinear RNNs." NeurIPS 2024.  
  https://proceedings.neurips.cc/paper_files/paper/2024/hash/0b2b199fdd52089b31d3a0120e400b2a-Abstract-Conference.html

- arXiv metadata for "Parallelizing non-linear sequential models over the sequence length."  
  https://arxiv.org/abs/2309.12252

## 11. Conclusao

DEER e uma das poucas ideias que ataca diretamente o gargalo temporal do DRM sem exigir abandonar a definicao recorrente do modelo. A tese e boa: resolver a mesma trajetoria de estados por um metodo paralelo.

Para o `drm-language-emitter`, a aplicacao direta no 125M e arriscada por causa de `d_state=1536`, custo de Jacobianos e estabilidade. O caminho correto e incremental:

```text
provar corretude pequena -> integrar DRM tiny -> validar backward -> testar blockwise -> escalar
```

Se esse caminho funcionar, a branch `drm-deer` pode virar a base de um novo modo de execucao do DRM: matematicamente recorrente, mas treinado com paralelismo temporal parcial ou aproximado.
