# DRM 125M - Estrategias Para Acelerar b8/Anderson Causal

Data do relatorio: 2026-07-29  
Escopo: levantar, com base no codigo atual e na literatura, as possibilidades tecnicas para transformar o caminho DRM `b8 + Anderson causal` em um caminho competitivo de throughput sem perder a qualidade observada em 37M.  
Status: relatorio de arquitetura e pesquisa aplicada. Nao e conclusao experimental fechada.

## 1. Resumo executivo

Os probes recentes mostraram tres fatos importantes:

1. O caminho rapido `block64 velocity iter0` chega perto da meta de throughput local: ~13.2k tok/s em 125M/1M tokens.
2. Esse caminho rapido perde qualidade de forma grande contra GPT-2 125M: `val_ce=3.2831` contra `3.0195` em 1M tokens.
3. O sampled b8 teacher quase nao transferiu qualidade: melhor caso `val_ce=3.2739`, ainda muito longe do GPT-2 e pagando throughput.

Portanto, a qualidade nao parece estar em uma regularizacao facil. Ela parece vir do forward efetivo do mecanismo local: b8, iteracoes de ponto fixo/Anderson, e a mistura causal de curto alcance dentro do bloco.

A conclusao tecnica e direta:

```text
Nao basta ensinar o caminho rapido a imitar b8.
Precisamos tornar o proprio mecanismo b8/Anderson executavel como primitivo paralelo/fundido.
```

As duas linhas mais promissoras sao:

1. **Kernel/scan especializado para b8 Anderson causal**, preservando a semantica atual o maximo possivel.
2. **Reformulacao associativa/afim do bloco b8**, transformando parte da dinamica em scan paralelizavel, inspirada em Mamba/selective scan e RNNs lineares paralelizaveis.

Minha recomendacao: implementar primeiro um **MVP Triton/CUDA de b8 local fused**, sem tentar resolver tudo de uma vez. O alvo inicial deve ser fundir `flow + naturalize + updater + residual` para blocos b8 e reduzir launch/autograd overhead. Em paralelo, estudar a reformulacao afim como caminho de segunda geracao.

## 2. Contexto empirico local

Resultados relevantes informados nos probes 125M:

| Configuracao | Tokens/s | Val CE 1M | Leitura |
|---|---:|---:|---|
| GPT-2 125M real | ~43.0k | 3.0195 | baseline forte |
| DRM b8 iter2 candidate | ~0.7k-0.9k inicio | nao finalizado no mesmo probe | qualidade potencial, custo proibitivo |
| DRM b8 iter2 velocity | ~0.9k | nao finalizado | remover candidate ajuda pouco |
| DRM b8 iter2 velocity stride4 | ~1.5k | nao finalizado | stride ajuda, insuficiente |
| DRM super64 local8 iter2 velocity | ~4.1k | probe interrompido | melhora throughput, ainda longe |
| DRM b64 velocity iter0 | ~13.2k | 3.2831 | rapido, qualidade ruim |
| DRM b64 velocity iter0 + sampled teacher | ~10.8k-11.2k | melhor 3.2739 | ganho pequeno demais |

Esse padrao sugere que:

- a meta de throughput bruta e possivel no modelo DRM;
- o problema e preservar o mecanismo local de qualidade;
- aumentar bloco ou remover Anderson recupera throughput mas perde CE;
- regularizacao amostrada nao compensa a perda de semantica.

## 3. Onde o codigo esta gastando tempo

No codigo atual:

- `src/drm_language_emitter/model.py::_forward_directional_cumsum` percorre blocos em Python.
- `src/drm_language_emitter/model.py::_directional_cumsum_block_base` calcula o warmstart local, aplica endpoint/fixed-point/Anderson e retorna estados para todos os tokens.
- `src/drm_language_emitter/model.py::_apply_block_anderson` chama `causal_anderson_solve`.
- `src/drm_language_emitter/deer.py::causal_anderson_solve` ja usa prefix sums para os Gram matrices e `torch.linalg.solve` batched.

Com `seq_len=512` e `block_size=8`:

```text
512 / 8 = 64 blocos por microbatch
iter2 ~= warmstart + 2 imagens de ponto fixo
64 * 3 = 192 avaliacoes pequenas por microbatch
grad_accum=8 -> 1536 avaliacoes pequenas por optimizer step
```

O gargalo nao e so o `solve`. E a granularidade: muitas chamadas pequenas, pouco trabalho por kernel, baixa ocupacao e grafo autograd fragmentado.

## 4. Principios vindos da literatura

### 4.1 Anderson acceleration

Anderson acceleration e uma tecnica para acelerar iteracoes de ponto fixo. Walker e Ni descrevem a relacao com GMRES/quasi-Newton e discutem detalhes praticos de implementacao. Fonte: SIAM, Walker & Ni 2011, "Anderson Acceleration for Fixed-Point Iterations": https://epubs.siam.org/doi/10.1137/10078356X

Aplicacao ao DRM:

- Faz sentido como acelerador local de uma trajetoria de estados.
- A versao causal atual ja corrigiu a dependencia de futuro usando prefix Gram.
- O problema restante e de execucao GPU, nao de validade matematica basica.

### 4.2 Prefix scan / parallel scan

Blelloch formalizou prefix sums como bloco fundamental para paralelizar computacoes que parecem sequenciais. Fonte: "Prefix Sums and Their Applications", CMU-CS-90-190: https://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html

A literatura CUDA mostra que scan eficiente precisa ser work-efficient, usar shared memory, evitar bank conflicts e reduzir overhead. Fonte: NVIDIA GPU Gems 3, "Parallel Prefix Sum (Scan) with CUDA": https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda

Aplicacao ao DRM:

- `torch.cumsum` resolve so o caso aditivo simples.
- Anderson causal precisa de scans segmentados sobre pequenos blocos e historicos.
- A unidade correta provavelmente e "muitos blocos b8 em paralelo", nao um loop Python por bloco.

### 4.3 IO-aware kernels

FlashAttention mostrou que wall-clock pode melhorar muito quando a computacao e reorganizada para reduzir trafego HBM/SRAM e fundir kernels, sem mudar a matematica da atencao. Fonte: Dao et al. 2022, arXiv:2205.14135: https://arxiv.org/abs/2205.14135

Aplicacao ao DRM:

- O DRM tem um problema parecido em espirito: nao e so FLOPs, e schedule de memoria/launch.
- A solucao deve manter estados/residuos de b8 em registradores/shared memory e escrever para HBM so o necessario.
- Backward pode recomputar partes em vez de salvar tudo, se isso reduzir memoria e overhead.

### 4.4 Selective scan / Mamba

Mamba usa parametros dependentes do input e um algoritmo hardware-aware para executar recorrencia em modo paralelo. Fonte: Gu & Dao 2023/2024, arXiv:2312.00752: https://arxiv.org/abs/2312.00752

Aplicacao ao DRM:

- A ideia relevante nao e copiar Mamba, mas observar o padrao: transformar uma recorrencia em uma primitiva scan-friendly.
- Se parte do update DRM puder ser expressa como composicao de mapas afins locais, a sequencia deixa de exigir loop temporal estrito.

### 4.5 RNNs lineares paralelizaveis

Martin & Cundy mostram que recorrencias com dependencias sequenciais lineares podem ser paralelizadas no eixo temporal por scan e reportam speedups com kernel CUDA. Fonte: arXiv:1709.04057: https://arxiv.org/abs/1709.04057

Aplicacao ao DRM:

- O DRM atual e nao linear em `z`, entao nao entra automaticamente nesse caso.
- Mas uma aproximacao local afim de cada subpasso pode permitir scan associativo dentro do bloco.

### 4.6 Compiladores e lowering para scan

ScanWeaver, ainda recente, representa uma linha de pesquisa diretamente relevante: transformar recorrencias afins em scans associativos e baixar para GPU/MLIR. Fonte: arXiv:2606.00601: https://arxiv.org/abs/2606.00601

Aplicacao ao DRM:

- Isso sugere uma rota de longo prazo: elevar o bloco DRM para uma recorrencia afim/semiafim e gerar scan.
- Ainda e pesquisa, mas a direcao combina com o problema.

### 4.7 Triton e custom kernels no ecossistema PyTorch

Triton foi proposto como linguagem/compilador para workloads tensoriais tiled quando bibliotecas padrao nao servem bem. Fonte: Tillet, Kung & Cox, MAPL/PLDI 2019: https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations

PyTorch documenta integracao de kernels Triton definidos pelo usuario com `torch.compile`, inclusive via `torch.library.triton_op`. Fonte: https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html

Aplicacao ao DRM:

- Triton e provavelmente a ferramenta mais pragmatica para um MVP.
- CUDA C++ pode ser melhor no limite, mas aumenta custo de desenvolvimento no Windows.

### 4.8 Deep Equilibrium Models

DEQ usa root-finding/fixed-point e implicit differentiation para reduzir memoria efetiva. Fonte: Bai, Kolter & Koltun, NeurIPS 2019 / arXiv:1909.01377: https://arxiv.org/abs/1909.01377

Aplicacao ao DRM:

- O DRM ja usa uma ideia local de ponto fixo.
- Implicit differentiation pode ser uma rota para reduzir memoria/backward, mas nao resolve automaticamente o throughput do forward b8.

## 5. Espaco completo de alternativas

### A. Kernel fused exato para b8/Anderson causal

Objetivo: preservar o maximo possivel da semantica atual `b8 iter2`, removendo overhead de Python, autograd fragmentado e kernels pequenos.

Como seria:

```text
entrada: z_start [B, d], tokens [B, n_blocks, 8, d_token]
kernel: para cada bloco b8:
  calcula warmstart local
  executa iteracoes Anderson
  calcula residuos e Gram prefix dentro do bloco
  resolve sistema pequeno h<=4
  escreve states [B, n_blocks, 8, d_state]
```

O primeiro MVP nao precisa fundir todas as MLPs. Pode comecar fundindo:

- construcao de residuals;
- Gram prefix;
- ridge;
- solve pequeno;
- combinacao Anderson.

O passo seguinte funde `flow/naturalize/updater`.

Estimativa:

| Nivel | Throughput esperado | Risco de qualidade | Complexidade |
|---|---:|---|---|
| Fused Anderson only | 1.5k-3k | baixo | media |
| Fused transition + Anderson | 4k-8k | baixo | alta |
| CUDA completo com backward custom | 8k-15k | baixo | muito alta |

Pragmatismo: melhor opcao para preservar qualidade, mas precisa engenharia de kernel.

### B. Segmented batched b8: "muitos b8 como batch", sem kernel custom completo

Objetivo: remover o loop externo por bloco transformando `[B, T]` em `[B * n_blocks, 8]`.

Hoje parte do superblock ja faz algo parecido, mas a semantica nao e exatamente b8 encadeado. A alternativa e executar todos os blocos b8 de uma camada/sequence em uma chamada batched, usando starts aproximados ou prefix de endpoints.

Variantes:

1. blocos independentes a partir de starts estimados;
2. duas passagens: coarse endpoints -> refine batched;
3. wavefront: computa endpoints de blocos em grupos.

Estimativa:

| Variante | Throughput esperado | Risco |
|---|---:|---|
| starts aproximados | 4k-8k | alto CE |
| coarse + refine | 3k-6k | medio |
| wavefront por grupos | 2k-5k | medio-baixo |

Risco: se o estado inicial de cada b8 precisa ser exato, essa aproximacao perde causalidade semantica ou qualidade. O superblock atual ja indicou ganho de throughput, mas ainda nao provou recuperar CE.

### C. Anderson "matrix-free" com sistema fechado para h pequeno

Objetivo: evitar `torch.linalg.solve` generico e overhead de montar tensores para `h=4`.

Como:

- Para `history_size=4`, resolver os sistemas `4x4` por Cholesky/LDL manual em kernel.
- Usar fp32 internamente.
- Aproveitar que o tamanho e estatico.
- Evitar materializar grandes tensores intermediarios.

Estimativa:

| Escopo | Throughput esperado |
|---|---:|
| substituir solve generico apenas | +10%-30% |
| solve + Gram prefix fused | +30%-80% |
| solve + residual + combination fused | 1.5x-3x |

Leitura: necessario dentro da solucao A, mas sozinho provavelmente nao chega em 15k.

### D. Associative scan por aproximacao afim local

Objetivo: transformar a transicao DRM dentro do bloco em composicao de mapas locais:

```text
z_t ~= A_t z_{t-1} + b_t
```

A composicao de mapas afins e associativa:

```text
(A2, b2) o (A1, b1) = (A2 A1, A2 b1 + b2)
```

Assim, todos os prefixos podem ser calculados por parallel scan.

Como obter `A_t` e `b_t`:

1. diagonal/low-rank Jacobian local;
2. gate-conditioned affine transition;
3. linearizacao em torno de `z_start`;
4. aprender diretamente `A_t`/`b_t` como caminho rapido.

Estimativa:

| Variante | Throughput esperado | Risco de qualidade |
|---|---:|---|
| diagonal affine | 15k-30k | alto |
| low-rank affine | 8k-20k | medio |
| learned affine + b8 correction | 10k-20k | medio |
| affine + occasional exact b8 | 8k-15k | medio-baixo |

Essa e a rota mais elegante de pesquisa, mas tambem a que mais muda a arquitetura. Ela pode virar um DRM "scan-native".

### E. Reformular Anderson como atencao causal local

O proprio relatorio anterior ja levantou a hipotese: Anderson causal b8 pode estar funcionando como uma mini-atencao causal local sobre residuos/iterates.

Se isso for verdade, uma alternativa e substituir Anderson por uma camada explicita de mistura local:

```text
features por token: residual, delta, gate, metric, risk
janela causal: 8 tokens
saida: mistura/correcao do estado
```

Implementacoes possiveis:

- depthwise causal conv sobre estados/residuos;
- local causal attention b8;
- kernel FlashAttention-style para janela fixa 8;
- MLP de mistura triangular fixa.

Estimativa:

| Variante | Throughput esperado | Risco |
|---|---:|---|
| causal conv b8 | 20k-40k | alto-medio |
| local attention b8 fused | 10k-25k | medio |
| MLP triangular b8 | 15k-30k | medio |

Ponto forte: pode recuperar a "mistura causal local" sem solver.  
Ponto fraco: muda a narrativa e precisa provar que substitui Anderson.

### F. CUDA Graphs / launch overhead reduction

Objetivo: capturar a sequencia fixa de kernels para reduzir overhead de launch.

Como:

- shapes fixos (`batch=2`, `seq=512`, `block=8`);
- warmup;
- captura do optimizer step;
- replay com buffers estaticos.

Estimativa:

| Caso | Ganho |
|---|---:|
| overhead de launch dominante | 1.2x-2x |
| computacao/autograd dominante | <1.2x |

Nao resolve sozinho, mas e barato de testar depois que shapes ficarem estaveis.

### G. `torch.compile` agressivo e `torch.scan`

PyTorch tem `torch.compile` para otimizar regioes e `torch.scan` como operador de controle estruturado ainda prototipo. Fontes: `torch.compile` docs https://docs.pytorch.org/docs/stable/generated/torch.compile.html e `torch.scan` docs https://docs.pytorch.org/docs/stable/higher_order_ops/scan.html

Possibilidades:

- compilar `_directional_cumsum_block_base`;
- isolar `_apply_block_anderson` em funcao pura;
- evitar graph breaks;
- usar `fullgraph=True` em microfuncoes;
- experimentar `torch.scan` para inner recurrence.

Estimativa:

| Variante | Throughput esperado |
|---|---:|
| compile parcial atual | 1.1x-1.5x |
| compile sem graph breaks em bloco | 1.5x-2.5x |
| scan/compile bem sucedido | 2x-4x |

Risco: em Windows/CUDA/PyTorch, isso pode ser instavel. Deve ser exploratorio, nao aposta unica.

### H. Backward custom com recomputacao

Objetivo: reduzir memoria e grafo autograd salvando menos intermediarios.

Inspiracao:

- FlashAttention troca armazenamento de intermediarios por recomputacao controlada.
- DEQ usa implicit differentiation para backprop atraves de ponto fixo.

Aplicacao:

- forward salva `z_start`, tokens, parametros ou seeds;
- backward recomputa trajetoria b8 local;
- gradiente passa por um operador custom.

Estimativa:

| Variante | Throughput | Memoria |
|---|---:|---:|
| checkpointing manual | neutro a +20% | melhora |
| backward fused/recompute | +20%-80% | melhora forte |
| implicit local backward | incerto | melhora forte |

Risco: alto para corretude de gradiente. Deve vir depois de estabilizar forward.

### I. Alternancia de modos por step

Objetivo: pagar b8/Anderson em uma fracao dos optimizer steps.

Ja sabemos que sampled teacher nao funcionou bem. Mas alternancia direta de forward pode ser diferente:

```text
step % 4 == 0: b8 iter2
outros: b64 velocity
```

Estimativa de throughput medio:

```text
b8 iter2: ~0.8k
b64 iter0: ~13k
1/4 b8 -> media harmonica aproximada ~2.4k-3.0k
1/8 b8 -> ~4.0k-5.0k
```

Risco: se os steps rapidos treinarem um modelo diferente, o b8 ocasional pode nao corrigir. E mais uma sonda cientifica que uma solucao final.

### J. Distillation offline de trajetorias b8

Objetivo: treinar um operador rapido para prever a correcao b8/Anderson, mas com dataset de trajetorias e supervisionamento mais rico que MSE esparsa online.

Como:

1. coletar pares `(z_start, tokens_b8, states_fast, states_anderson)`;
2. treinar um `correction_head` local;
3. congelar/testar no LM;
4. depois treinar end-to-end.

Estimativa:

| Fase | Throughput final | Risco |
|---|---:|---|
| correction MLP pequena | 10k-20k | medio |
| local attention correction | 8k-15k | medio-baixo |

Por que pode funcionar melhor que sampled teacher: o professor nao compete diretamente com CE no mesmo treino; o aluno ve muitos exemplos densos e aprende a correcao.

### K. Aumentar batch efetivo / paralelismo de dados

O batch atual `BatchSize=2, SeqLen=512, GradAccum=8` usa pouco paralelismo por chamada. Aumentar microbatch poderia melhorar ocupacao, se couber em VRAM.

Possibilidades:

- `BatchSize=4, GradAccum=4`;
- gradient checkpointing para caber batch maior;
- DDP multi-GPU;
- sequence packing com mais blocos independentes.

Estimativa:

| Mudanca | Throughput esperado |
|---|---:|
| batch 2 -> 4 se couber | +20%-80% |
| DDP 2 GPUs | ~1.6x-1.9x global |
| packing mais agressivo | +10%-40% |

Nao resolve sozinho o b8, mas aumenta o teto.

### L. Precisao numerica e quantizacao interna

Possibilidades:

- manter Gram/solve em fp32;
- usar bf16 para states/deltas;
- TF32 para matmuls;
- fp8 nos MLPs se hardware suportar;
- quantizar apenas teacher/correcao.

Estimativa:

| Mudanca | Ganho |
|---|---:|
| garantir bf16 matmul | +10%-30% |
| fp8 MLPs | +20%-80% se suportado |
| solve fp32 pequeno | necessario para estabilidade |

Risco: Anderson e sensivel a condicionamento. Nao baixar precisao do solve sem testes.

### M. Remover componentes por perfil real

Antes de kernelizar tudo, medir:

- tempo em `direction_field`;
- tempo em `metric`;
- tempo em `flow`;
- tempo em `risk`;
- tempo em `candidate_step`;
- tempo em `causal_anderson_solve`;
- tempo no backward de cada componente.

Ferramentas:

- `torch.profiler`;
- Nsight Systems;
- Nsight Compute;
- PyTorch `torch.compile` graph break logs;
- contagem de kernels por optimizer step.

Estimativa: perfil nao melhora throughput diretamente, mas evita otimizar o trecho errado.

## 6. Matriz de decisoes

| Opcao | Preserva qualidade b8 | Chance de chegar 15k tok/s | Tempo de implementacao | Risco tecnico | Recomendacao |
|---|---|---:|---:|---|---|
| A. Kernel fused b8/Anderson | alta | media-alta | alto | alto | principal |
| B. Segmented batched b8 | media | media | medio | medio | testar antes de CUDA completo |
| C. Solve/Gram especializado | alta | baixa-media sozinho | medio | medio | subcomponente de A |
| D. Associative affine scan | media | alta | alto | alto | pesquisa de segunda geracao |
| E. Local attention/conv substituta | media | alta | medio | medio-alto | sonda forte |
| F. CUDA Graphs | alta | baixa | baixo-medio | baixo | complemento |
| G. torch.compile/scan | alta se funcionar | baixa-media | baixo-medio | medio | exploratorio |
| H. Backward custom/recompute | alta | media | alto | alto | depois do forward |
| I. Alternancia por step | media | baixa-media | baixo | medio | diagnostico |
| J. Distillation offline | media | media-alta | medio | medio | sonda paralela |
| K. Batch/DDP | alta | media global | baixo-medio | baixo | operacional |
| L. Precisao/FP8 | media-alta | baixa-media | medio | medio | depois de perfil |
| M. Profiling serio | alta | indireta | baixo | baixo | fazer agora |

## 7. Recomendacao de sequencia

### Fase 0: perfil antes de escrever kernel

Objetivo: confirmar onde o tempo realmente esta em b8 iter2.

Entregaveis:

- script `profile_drm_b8_anderson_torch_profiler.py`;
- tabela por op;
- numero de kernels por step;
- tempo forward/backward separado;
- memoria alocada e pico.

Decisao esperada:

```text
Se Anderson solve <20% do tempo, nao otimizar solve primeiro.
Se MLP transition/backward domina, fundir/compilar transition primeiro.
Se launch overhead domina, CUDA Graphs/Triton batched primeiro.
```

### Fase 1: b8 batched sem mudar matematica

Objetivo: reduzir loop Python e chamar o mesmo bloco b8 como batch grande.

Implementar um caminho experimental:

```text
directional_batched_block8_anderson
```

Entrada logica:

```text
z_starts: [B, n_blocks, d_state]
tokens:   [B, n_blocks, 8, d_token]
```

Inicialmente, `z_starts` pode vir de uma passada coarse. Medir se qualidade cai.

Meta:

- 1M tokens;
- throughput >3k;
- CE melhor que b64 velocity;
- se sim, 10M.

### Fase 2: Triton MVP para Anderson local

Objetivo: especializar `history_size=4`, `block_size=8`, fp32 solve, bf16 states.

Primeiro kernel:

- recebe residuals/images;
- calcula Gram prefix;
- resolve h pequeno;
- mistura images.

Segundo kernel:

- inclui residual computation.

Terceiro kernel:

- inclui parte do transition se o perfil justificar.

Meta:

- recuperar exatamente ou quase exatamente o output PyTorch;
- tolerancia numerica definida;
- benchmark forward/backward.

### Fase 3: substituto local aprendivel

Se o kernel exato ficar caro demais, testar uma substituicao explicita do papel de Anderson:

```text
local causal mixer b8:
  inputs = states_fast, residuals, deltas, gates
  output = corrected_states
```

Variantes:

- triangular MLP;
- causal depthwise conv;
- local attention b8 fused;
- correction low-rank.

Meta:

- CE perto do b8 iter2 em 10M;
- throughput >10k.

### Fase 4: DRM scan-native

Reformular o bloco como composicao de mapas afins/low-rank:

```text
z_t = A_t z_{t-1} + b_t
```

ou:

```text
z_t = z_{t-1} + U_t phi_t(V_t z_{t-1} + c_t)
```

com uma decomposicao que permita scan ou scan aproximado.

Meta:

- arquitetura nova, nao apenas flag;
- causalidade formal;
- throughput na faixa 15k-30k;
- qualidade comparavel ao b8 Anderson.

## 8. Possivel arquitetura do kernel exato

### Layout sugerido

```text
B = microbatch
S = seq_len
L = block_size = 8
N = S / L
D = d_state
H = history_size <= 4
I = anderson_iterations <= 2

tokens: [B, N, L, d_token]
states: [B, N, L, D]
```

Um CTA pode processar:

- um bloco b8 de um batch item;
- ou dois/quatro blocos b8 se `D` for pequeno o bastante;
- ou um tile de `D`.

### Dados em shared/registers

Manter local:

- `z_start`;
- `states[L, D_tile]`;
- `images[H, L, D_tile]`;
- `residuals[H, L, D_tile]`;
- acumuladores Gram `[L, H, H]` ou prefix compactado;
- coeficientes `[L, H]`.

### Cuidados

- `D` provavelmente e grande demais para um CTA carregar inteiro confortavelmente; tile por dimensao.
- Gram exige reducao sobre `D`, entao precisa reducao entre tiles.
- Uma primeira versao pode calcular Gram em PyTorch e kernelizar so solve/combine, mas o ganho sera limitado.
- Backward e mais dificil que forward; considerar recomputation.

## 9. Por que o sampled teacher falhou

O resultado foi:

```text
baseline b64 velocity iter0: val_ce=3.2831, ~13.2k tok/s
sampled i16 w0.05:          val_ce=3.2739, ~11.2k tok/s
sampled i8 w0.05:           val_ce=3.2739, ~11.1k tok/s
GPT-2 125M:                 val_ce=3.0195, ~43.0k tok/s
```

Isso indica:

- o professor b8 local nao esta sendo incorporado com forca suficiente;
- aumentar frequencia de 1/16 para 1/8 nao ajudou;
- o erro principal nao e uma pequena deriva regularizavel;
- o caminho rapido provavelmente esta em outra dinamica.

Portanto, manter sampled teacher como linha principal seria desperdicio. Pode ficar como ferramenta auxiliar de ablation, nao como solucao.

## 10. Plano de provas antes de 150M

Cada candidato novo deve passar por:

1. teste causal de prefixo;
2. equivalencia numerica contra b8 PyTorch em forward pequeno;
3. backward finito;
4. probe 1M com GPT-2 reference;
5. probe 10M seed 1;
6. so entao multi-seed.

Criterio minimo para continuar:

```text
1M: CE melhor que b64 baseline por >=0.03
10M: distancia contra GPT-2 menor que b64 baseline
throughput: >=8k tok/s em 125M
```

Criterio para escalar:

```text
10M: CE competitivo ou tendencia clara
throughput: >=10k tok/s
sem quebra causal
sem instabilidade de loss
```

## 11. Minha recomendacao objetiva

Ordem que eu seguiria:

1. **Profiling serio do b8 iter2** com `torch.profiler` e, se possivel, Nsight.
2. **MVP Triton do subproblema Anderson**: Gram/solve/combine para `L=8,H=4`.
3. **Batchificar todos os b8 como segmentos** para reduzir Python loop, mesmo que primeiro use starts aproximados so para medir teto.
4. **Testar local causal mixer** como substituto aprendivel de Anderson, porque pode recuperar o papel de mini-atencao local com muito mais throughput.
5. **So depois CUDA completo/backward custom**, se os passos 2-4 mostrarem que a qualidade b8 e preservavel.

Nao recomendo agora:

- insistir em sampled teacher online;
- declarar b64 velocity como competitivo so por throughput;
- escalar 150M com caminho que ja perdeu em 1M contra GPT-2;
- escrever CUDA completo sem perfil, porque ha alto risco de otimizar o trecho errado.

## 12. Fontes consultadas

- Walker, H. F.; Ni, P. "Anderson Acceleration for Fixed-Point Iterations", SIAM Journal on Numerical Analysis, 2011. https://epubs.siam.org/doi/10.1137/10078356X
- Blelloch, G. E. "Prefix Sums and Their Applications", CMU-CS-90-190, 1990. https://www.cs.cmu.edu/afs/cs.cmu.edu/project/scandal/public/papers/CMU-CS-90-190.html
- Harris, M.; Sengupta, S.; Owens, J. D. "Parallel Prefix Sum (Scan) with CUDA", GPU Gems 3. https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
- Sengupta, S.; Harris, M.; Garland, M. "Efficient Parallel Scan Algorithms for GPUs", NVIDIA Technical Report NVR-2008-003. https://research.nvidia.com/publication/2008-12_efficient-parallel-scan-algorithms-gpus
- Dao, T.; Fu, D. Y.; Ermon, S.; Rudra, A.; Re, C. "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", arXiv:2205.14135. https://arxiv.org/abs/2205.14135
- Gu, A.; Dao, T. "Mamba: Linear-Time Sequence Modeling with Selective State Spaces", arXiv:2312.00752. https://arxiv.org/abs/2312.00752
- Martin, E.; Cundy, C. "Parallelizing Linear Recurrent Neural Nets Over Sequence Length", arXiv:1709.04057. https://arxiv.org/abs/1709.04057
- Wu, Q.; Zolnikov, P. "ScanWeaver: Compiler-Driven Parallelization of Affine Recurrences via Associative Scan Lowering", arXiv:2606.00601. https://arxiv.org/abs/2606.00601
- Tillet, P.; Kung, H. T.; Cox, D. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations", MAPL/PLDI 2019. https://research.ibm.com/publications/triton-an-intermediate-language-and-compiler-for-tiled-neural-network-computations
- PyTorch docs, "Using User-Defined Triton Kernels with torch.compile". https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html
- PyTorch docs, "`torch.compile`". https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- PyTorch docs, "Control Flow - Scan". https://docs.pytorch.org/docs/stable/higher_order_ops/scan.html
- Bai, S.; Kolter, J. Z.; Koltun, V. "Deep Equilibrium Models", NeurIPS 2019 / arXiv:1909.01377. https://arxiv.org/abs/1909.01377
