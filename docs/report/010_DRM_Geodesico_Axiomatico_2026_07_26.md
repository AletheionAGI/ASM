# DRM Geodesico Axiomatico: Do Loop Recorrente Ao Planejamento De Caminhos

Data do relatorio: 2026-07-26  
Branch alvo: `drm-deer-drm`  
Projeto auditado: `drm-language-emitter`  

## 1. Resumo executivo

Este relatorio registra uma mudanca conceitual importante para o `drm-language-emitter`: a implementacao atual usa conceitos DRM, mas executa a sequencia como uma recorrencia local token-a-token. Essa forma e funcional, treinavel e coerente com uma leitura dinamica do DRM, mas pode nao ser a formulacao mais fiel aos axiomas geometricos do modelo.

A hipotese central deste relatorio e:

```text
O DRM nao deve ser tratado primariamente como uma RNN que aplica deslocamentos locais.
O DRM deve ser tratado como uma geometria direcional onde o proximo estado e obtido por inferencia de caminho na metrica.
```

Em outras palavras, o modelo nao deveria apenas fazer:

```text
z_{t+1} = z_t + dz_t
```

Ele deveria resolver algo mais proximo de:

```text
z_{t+1}, caminho = argmin custo_geometrico(z_t, contexto, direcoes, metrica, risco)
```

Essa diferenca e fundamental. Na implementacao atual, o `for loop` existe porque cada estado depende do estado anterior. Na formulacao geodesica, a dependencia ainda existe matematicamente, mas o calculo deixa de ser "andar um passo pequeno" e passa a ser "encontrar o caminho mais coerente dentro da geometria direcional".

Essa abordagem parece mais alinhada com a ideia original de DRM:

```text
pontos
retas
dimensoes direcionais
metrica relacional
caminhos minimais
orientacao interna
```

O objetivo nao e abandonar o DRM. E tornar a implementacao mais DRM.

## 1.1 Correcao importante apos revisao tecnica

Esta proposta nao deve ser tratada como descoberta conceitual nova em relacao ao material teorico do DRM. Ela deve ser tratada como refinamento de implementacao.

O PDF do DRM ja define a nocao de geodesica relacional, em particular a ideia de estacionariedade da energia relacional sob variacoes com extremos fixos. A implementacao atual ja contem uma aproximacao fraca dessa direcao por meio do `Action Loss`:

```text
L_action = mean(Delta z^T G(z) Delta z)
```

Esse termo penaliza energia de movimento local, mas nao resolve o problema de contorno de uma geodesica. Portanto, a distincao correta e:

```text
Action Loss atual:
  penaliza passos energeticamente caros durante uma trajetoria gerada localmente.

DRM geodesico proposto:
  tenta escolher o destino ou caminho resolvendo diretamente uma energia relacional.
```

Ou seja, o relatorio nao propoe trocar o objetivo teorico do DRM. Ele propoe tornar a implementacao mais proxima da definicao geodesica ja existente, saindo de uma penalizacao local para uma resolucao explicita, ainda que aproximada, de caminho.

Tambem ha um limite importante: a energia geodesica usada para escolher caminho nao deve depender do `target` supervisionado em tempo de inferencia. Usar `CE(emitter(z_next), target)` dentro do solver seria vazamento do alvo durante o forward de treino e nao corresponderia a uma geodesica pura. A CE pode continuar sendo a loss externa de treinamento, mas nao deve ser o criterio interno do solver geodesico inicial.

## 2. Diagnostico da implementacao atual

### 2.1 O DRM atual e uma RNN geometrica

Hoje o `DRMEmitterModel` opera conceitualmente assim:

```text
1. inicia um estado z0;
2. le o token t;
3. calcula geometria local em z_t;
4. calcula um deslocamento dz_t;
5. atualiza z_t para z_{t+1};
6. repete para o proximo token.
```

A estrutura e:

```text
z_{t+1} = update(z_t, flow(z_t, token_t, geometry(z_t)))
```

Essa forma preserva causalidade e memoria, mas e essencialmente uma RNN nao-linear com componentes geometricos.

### 2.2 O problema do deslocamento local

O deslocamento local `dz` induz uma interpretacao operacional:

```text
o modelo esta caminhando pouco a pouco no espaco latente
```

Isso pode ser util, mas tambem reduz a geometria DRM a uma integracao local. Se os axiomas dizem que um ponto possui varias retas/dimensoes direcionais e que a metrica define caminhos, entao o passo seguinte nao deveria ser apenas um vetor local. Ele deveria emergir de uma escolha de caminho.

Em termos geometricos, o problema e:

```text
dz local nao equivale necessariamente a caminho geodesico
```

Um vetor local pode apontar para uma direcao plausivel, mas nao garante que o modelo esteja escolhendo a rota mais curta, mais coerente ou menos arriscada na metrica relacional.

### 2.3 O loop como sintoma

O `for loop` nao e o problema conceitual por si so. Ele e sintoma de uma decisao de implementacao:

```text
resolver a sequencia por integracao causal local
```

Se a formulacao for trocada para resolucao de caminho, o loop pode continuar existindo em algum nivel, mas deixa de ser o eixo principal da inteligencia do modelo.

O modelo passaria de:

```text
estado atual -> pequeno deslocamento -> proximo estado
```

para:

```text
estado atual + geometria + contexto -> rota/destino -> proximo estado
```

## 3. Reinterpretacao axiomatica do DRM

### 3.1 Axioma 1: ponto

Um estado `z` representa um ponto em uma variedade relacional.

```text
z in M
```

Esse ponto nao e apenas uma memoria vetorial. Ele e uma posicao em uma geometria interna.

### 3.2 Axioma 2: retas/dimensoes direcionais

Em cada ponto, existem direcoes possiveis:

```text
D(z) = {d_1(z), d_2(z), ..., d_k(z)}
```

Essas direcoes nao devem ser entendidas apenas como features. Elas representam possibilidades de movimento, interpretacao, relacao e transformacao.

### 3.3 Axioma 3: metrica relacional

A metrica define custo, distancia, compatibilidade e curvatura local:

```text
g_z(u, v)
```

Ela determina se uma direcao e barata, cara, estavel, instavel, coerente ou arriscada.

### 3.4 Axioma 4: contexto como campo de restricoes

O token ou contexto nao deve ser apenas uma entrada para gerar `dz`. Ele deve deformar ou condicionar o problema de caminho:

```text
contexto -> restricoes sobre direcoes validas
contexto -> atratores de destino
contexto -> penalidades metricas
```

### 3.5 Axioma 5: proximo estado como solucao de caminho

O proximo estado deveria ser definido por uma solucao:

```text
z_{t+1} = endpoint(gamma*)
```

onde:

```text
gamma* = argmin_gamma E(gamma; z_t, contexto_t, D, g, R)
```

e `E` e uma energia/custo de caminho.

## 4. Formulacao proposta

### 4.1 Energia de caminho

Uma formulacao inicial:

```text
E(gamma) =
  comprimento_metrico(gamma)
  + alinhamento_contextual(gamma, token/contexto)
  + penalidade_de_risco(gamma)
  + penalidade_de_curvatura(gamma)
  + penalidade_de_incoerencia_direcional(gamma)
```

Em forma discreta:

```text
gamma = [z_0, z_1, ..., z_K]
```

e:

```text
E(gamma) =
  sum_i metric_energy(z_i, z_{i+1} - z_i)
  + context_cost(z_K, token/contexto)
  + risk_cost(z_i)
  + curvature_cost(z_{i-1}, z_i, z_{i+1})
```

### 4.2 Proximo estado como destino, nao deslocamento

Em vez de aprender diretamente:

```text
dz = flow(z, token)
z_next = z + dz
```

o DRM geodesico pode aprender:

```text
direcoes candidatas
custos metricos
atrator contextual
solver de rota
z_next = endpoint do caminho escolhido
```

Isso muda a semantica:

```text
flow local -> planner geometrico
```

### 4.3 Consulta geometrica interna

A ideia operacional deve ser descrita sem linguagem de agencia. O mecanismo proposto e uma otimizacao ou selecao diferenciavel condicionada pela geometria do estado:

```text
1. dado z, gerar mapa local de direcoes;
2. avaliar quais direcoes reduzem custo;
3. simular ou inferir trajetorias curtas;
4. escolher destino coerente;
5. emitir token a partir do destino.
```

Tecnicamente, isso nao implica consciencia, introspeccao ou agencia. E apenas uma mudanca de mecanismo: a transicao passa a depender de uma avaliacao explicita de custos geometricos, em vez de um unico deslocamento local produzido diretamente pelo fluxo.

## 5. Relacao com DEER

DEER continua relevante, mas muda de papel.

Na leitura anterior:

```text
DEER = acelerar a recorrencia existente
```

Na leitura geodesica:

```text
DEER = uma possivel tecnica para resolver trajetorias/caminhos em paralelo
```

Ou seja, DEER nao e a solucao conceitual inteira. Ele e um solver candidato.

O nucleo conceitual passa a ser:

```text
DRM como problema de caminho em geometria direcional
```

e DEER, quasi-Newton, shooting methods, relaxation, parallel scan ou otimizacao diferenciavel sao formas possiveis de resolver esse problema.

## 6. Arquiteturas candidatas

### 6.1 DRM Local-Step atual

Forma:

```text
z_{t+1} = update(z_t, dz_t)
```

Vantagens:

```text
simples
causal
ja implementado
treinavel
```

Limites:

```text
sequencial
interpreta geometria como passo local
pode nao realizar caminho minimo
```

### 6.2 DRM Geodesic-Step

Forma:

```text
z_{t+1} = geodesic_step(z_t, token_t, D(z_t), g(z_t))
```

O passo ainda e causal, mas o update e obtido por um mini-solver de caminho.

Vantagens:

```text
mais fiel a metrica
mantem interface semelhante ao modelo atual
permite implementacao incremental
```

Limites:

```text
mais caro por passo
exige solver estavel
```

### 6.3 DRM Block-Geodesic

Forma:

```text
Z_{t:t+B} = solve_path(z_t, tokens_{t:t+B})
```

O modelo resolve um bloco de estados de uma vez.

Vantagens:

```text
reduz dependencia token-a-token
abre paralelismo temporal parcial
mais alinhado a DEER
```

Limites:

```text
precisa controlar erro e memoria
mais complexo que o step local
```

### 6.4 DRM Global Trajectory Solver

Forma:

```text
Z_{1:T} = solve_trajectory(z_0, tokens_{1:T})
```

Essa e a versao mais ambiciosa.

Vantagens:

```text
maxima coerencia com planejamento de caminho
maior potencial de paralelismo temporal
```

Limites:

```text
alto risco
memoria alta
backward complexo
```

## 7. Plano de pesquisa e implementacao

### Fase 1 - Formalizar a energia geodesica sem vazamento de alvo

Definir uma energia minima usando componentes ja existentes:

```text
metric_energy
risk_mass
gates direcionais
direction basis
```

Primeira energia proposta:

```text
E(z_next | z, token) =
  metric_energy(z, z_next - z)
  + risk_penalty(z_next)
  + direction_alignment_penalty(z_next - z, D(z))
  + context_compatibility(z_next, token)
```

Essa formulacao ainda usa `z_next - z`, mas nao aprende somente um deslocamento. Ela escolhe `z_next` minimizando energia geometrica e compatibilidade contextual.

O termo supervisionado de CE deve ficar fora do solver:

```text
z_next = geodesic_step(z, token)
loss = CE(emitter(z_next), target)
```

Essa separacao evita circularidade. Durante inferencia, o solver continua disponivel porque depende apenas de `z`, `token/contexto`, direcoes, metrica e risco, nao do proximo token verdadeiro.

### Fase 2 - Solver local diferenciavel

Implementar um solver pequeno:

```text
z_candidate = z + initial_delta
for k in solver_steps:
    z_candidate = z_candidate - alpha * grad_z E(z_candidate)
```

Isso ainda tem loop interno, mas muda a semantica: o loop deixa de ser a recorrencia principal e vira otimizacao de caminho.

Critério:

```text
melhorar ou igualar CE em modelo tiny
sem NaN
tempo aceitavel
```

Antes de avancar para fases mais caras, medir obrigatoriamente:

```text
wall_clock por forward
wall_clock por forward_backward
max_memory_mb
tokens/sec
custo relativo vs local_step
```

Se o `geodesic_step` tiny custar mais que 2x o baseline sem ganho claro de CE/estabilidade, a linha deve ser pausada ou reformulada antes de `multi-candidate`, caminhos curtos ou blocos temporais.

### Fase 3 - Multi-candidato direcional

Gerar varios candidatos a partir das direcoes:

```text
z_i = z + a_i d_i
```

Avaliar custo de cada candidato:

```text
score_i = -E(z_i)
```

Combinar:

```text
z_next = sum_i softmax(score)_i z_i
```

Essa fase e altamente paralelizavel sobre direcoes.

### Fase 4 - Caminhos curtos

Em vez de escolher apenas `z_next`, resolver caminhos discretos pequenos:

```text
gamma = [z, h1, h2, z_next]
```

Isso permite curvatura e evita reduzir tudo a reta local.

### Fase 5 - Blocos temporais

Depois que o step geodesico funcionar, aplicar em blocos:

```text
solve_path(z_start, tokens[t:t+B])
```

Aqui DEER ou quasi-Newton volta como ferramenta para resolver a consistencia entre estados do bloco.

## 8. Como testar se a ideia e melhor

### 8.1 Teste de equivalencia nao e suficiente

Para DEER puro, equivalencia ao loop sequencial e criterio central. Para DRM geodesico, nao queremos apenas reproduzir o loop antigo; queremos uma dinamica mais fiel.

Portanto, os testes devem medir:

```text
CE
estabilidade
qualidade de amostras
geometria dos caminhos
energia media
risco medio
uso das direcoes
```

### 8.2 Testes tiny

Primeiros testes:

```text
configs/tiny.yaml
configs/tiny_drm_stronger.yaml
seq_len 32/64
steps 100/400/1000
```

Comparar:

```text
DRM local-step
DRM geodesic-step
DRM multi-candidate
```

### 8.3 Testes qualitativos

Mesmo em base model, o texto deve evitar:

```text
caracteres corrompidos
pseudo-palavras excessivas
colapso repetitivo
```

### 8.4 Testes geometricos

Medir:

```text
comprimento medio do caminho
energia metrica media
alinhamento com direcoes ativas
diversidade de direcoes
condicao da metrica
risco acumulado
```

## 9. Implicacoes para throughput

Essa abordagem nao garante throughput imediato. Na verdade, um solver geodesico local pode inicialmente ser mais lento que o loop atual.

Esse risco nao e secundario. Um solver iterativo dentro de cada token pode multiplicar o custo por `solver_steps`, alem de exigir autograd adicional para gradientes internos. Como o DRM sequencial ja e muito mais lento que modelos Transformer comparaveis, a primeira validacao experimental deve ser custo real em parede, nao apenas elegancia matematica.

O ganho potencial vem depois:

```text
1. candidatos direcionais avaliados em paralelo;
2. caminhos curtos resolvidos em paralelo;
3. blocos temporais resolvidos por DEER/quasi-Newton;
4. menos steps temporais locais para obter a mesma transformacao semantica.
```

O objetivo inicial deve ser qualidade e coerencia axiomatica, nao apenas tokens/s.

## 10. Decisoes propostas

### 10.1 Renomear a linha de pesquisa

O nome `drm-deer` e util, mas estreito. A solucao conceitual nao e apenas DEER. O nome mais preciso e:

```text
drm-deer-drm
```

Interpretação:

```text
DEER como ferramenta
DRM como formulacao principal
```

### 10.2 Nao chamar isso de otimizacao de throughput

Esta linha deve ser tratada como:

```text
reformulacao axiomatica do mecanismo de transicao DRM
```

Throughput e consequencia possivel, nao criterio inicial.

Ainda assim, ha um limite pragmatico: se a reformulacao geodesica nao mostrar ganho mensuravel de qualidade em tiny, nao ha justificativa para pagar o custo de fases mais complexas.

### 10.3 Preservar baseline

O modelo atual deve continuar existindo como:

```text
sequence_mode = local_step
```

A nova abordagem deve entrar como:

```text
sequence_mode = geodesic_step
sequence_mode = geodesic_block
```

### 10.4 Implementar em camadas

Ordem recomendada:

```text
1. documentar axiomas
2. criar energia geodesica simples sem CE interna
3. testar solver local tiny
4. medir custo wall-clock antes de escalar
5. testar multi-candidatos direcionais
6. testar caminhos curtos
7. testar blocos com DEER/quasi-Newton
```

## 11. Perguntas abertas

1. O token deve definir um atrator de destino, uma restricao de caminho ou ambos?
2. A metrica deve ser calculada apenas no ponto inicial ou ao longo do caminho?
3. As direcoes devem ser bases locais, retas globais ou campos vetoriais?
4. O emissor deve ler apenas o endpoint ou tambem estatisticas do caminho?
5. A loss deve supervisionar apenas o token seguinte ou tambem a energia/caminho?
6. O caminho minimo deve ser unico ou uma mistura de caminhos plausiveis?

## 12. Conclusao

A critica ao `for loop` revelou uma questao mais profunda: a implementacao atual usa a linguagem do DRM, mas sua execucao ainda e a de uma RNN geometrica local. Isso nao invalida o modelo. Tambem nao significa que a proposta geodesica seja automaticamente melhor. A formulacao correta e mais estreita: o DRM teorico ja contem uma nocao de geodesica relacional, enquanto o codigo atual implementa uma aproximacao local regularizada por energia.

A abordagem mais condizente com DRM e formular a transicao como inferencia de caminho:

```text
ponto + direcoes + metrica + contexto -> caminho/destino
```

Essa formulacao aproxima a implementacao do criterio geodesico relacional: escolher estados e caminhos por custo geometrico, em vez de apenas penalizar deslocamentos locais depois que eles ja foram gerados.

O proximo passo recomendado e implementar um `geodesic_step` experimental em escala tiny, sem CE interna no solver, sem substituir o forward atual, e medir simultaneamente qualidade e custo real. Se a geometria pura nao melhorar nada nessa escala, as fases mais caras nao devem avancar.
