# Plano de melhoria linguística do ASM-CM e ASM-CM-E

Data: 3 de agosto de 2026

## Objetivo

Este documento define um programa experimental para reduzir a cross-entropy
linguística do **ASM-CM** e do **ASM-CM-E** sem perder as propriedades que
justificam essas arquiteturas:

- memória associativa recuperável e durável;
- MQAR de contexto longo;
- streaming com estado limitado;
- paridade entre forward completo e decode incremental;
- baixo crescimento de cache por stream;
- compatibilidade com linguagem natural.

O objetivo não é melhorar CE a qualquer custo. Uma variante que reduz CE, mas
perde recuperação em 32K ou passa a preservar o prefixo completo, não representa
um avanço do ASM-CM.

## Ponto de partida congelado

### ASM-CM

- identificador técnico: ASM-C2-FW-LM;
- parâmetros: 84.011.396;
- backbone originado do ASM-R treinado com 100M tokens;
- CE congelado da seed 1: 1,32940050;
- memória fast-weight durável em FP32;
- currículo misto de linguagem e MQAR;
- arquitetura publicamente promovida por memória compacta, não por superar o
  Transformer em CE.

### ASM-CM-E

- parâmetros: 85.208.200;
- acréscimo epistêmico: 1.196.804 parâmetros, ou aproximadamente 1,42%;
- CE congelado preliminar da seed 1: 1,32953025;
- diferença diante do ASM-CM: +0,00012975 CE, aproximadamente +0,0098%;
- venceu o ASM-CM em CE MQAR em cinco das sete distâncias do currículo da seed
  1, incluindo redução aproximada de 7,9% em 4K;
- confirmação de 32K e seeds 2 e 3 ainda em andamento na data deste documento.

O resultado linguístico atual deve ser interpretado como empate. A diferença é
muito inferior à resolução prática necessária para declarar superioridade.

## Diagnóstico

Os dois modelos não receberam um pré-treinamento linguístico independente e
extenso após a introdução da memória. Eles herdaram o backbone ASM-R de 100M
tokens e passaram por uma especialização conservadora que favorece preservação:

```text
checkpoint ASM-R
        ↓
80% batches de linguagem + 20% batches MQAR
        ↓
learning rate baixo no backbone
        ↓
learning rate maior na memória
        ↓
distilação fixa do ASM-R
        ↓
ASM-CM ou ASM-CM-E
```

Essa estratégia foi correta para demonstrar compatibilidade entre linguagem e
memória, mas possui quatro limites para melhoria linguística:

1. o professor ASM-R impõe um teto quando a distilação permanece forte;
2. a sequência linguística de especialização tem 128 tokens, enquanto o
   rescoring congelado usa 512;
3. 20% dos batches são destinados ao benchmark associativo;
4. o orçamento linguístico adicional é pequeno diante do pré-treinamento.

Portanto, mais tokens devem ajudar, mas o primeiro experimento deve verificar se
é possível aproveitar melhor os tokens antes de multiplicar o orçamento.

## Princípio experimental

Cada alteração será testada contra seu próprio checkpoint congelado:

```text
ASM-CM  → ASM-CM-L
ASM-CM-E → ASM-CM-E-L
```

O sufixo `-L` identifica apenas uma continuação linguística experimental. Ele
não constitui um novo nome público ou promoção.

Todas as comparações devem usar:

- mesmo corpus e ordenação de amostras;
- mesmos budgets de tokens;
- mesmas seeds;
- mesmo tokenizer;
- mesmo hardware e precisão;
- rescoring sobre a sequência completa de validação congelada;
- MQAR congelado, sem selecionar o melhor checkpoint por amostras diferentes.

## Fase 0 — congelar a linha de base

Antes de qualquer continuação, registrar para cada seed:

- SHA-256 do checkpoint;
- SHA-256 dos manifests;
- CE e perplexidade no corpus linguístico completo;
- CE e acurácia MQAR em 40, 80, 160, 320, 512, 1K, 4K e 32K;
- throughput de prefill e decode;
- pico de VRAM;
- bytes de cache por stream;
- paridade BF16;
- médias dos gates de leitura, escrita, consolidação e, no ASM-CM-E, confiança
  epistêmica.

Essa fase impede que a continuação seja comparada com números derivados de
protocolos anteriores ou checkpoints diferentes.

## Fase 1 — recuperação linguística de baixo custo

### Hipótese

Uma fase curta com maior proporção linguística, sequências maiores e distilação
decrescente pode reduzir CE sem apagar MQAR.

### Currículo recomendado

#### Estágio A — estabilização

- 90% linguagem e 10% MQAR;
- sequência linguística de 256 tokens;
- distilação 0,5;
- backbone LR de 1e-5;
- memória e gates LR de 5e-5;
- 5M tokens linguísticos.

#### Estágio B — recuperação

- 95% linguagem e 5% MQAR;
- sequência linguística de 512 tokens;
- distilação reduzida linearmente de 0,25 para 0,05;
- backbone LR de 5e-6;
- memória e gates LR de 2e-5;
- 15M tokens linguísticos.

#### Estágio C — consolidação

- 95% linguagem e 5% MQAR;
- sequência de 512 tokens;
- distilação 0,05;
- cosine decay até 10% do LR inicial;
- 10M tokens linguísticos.

O orçamento total inicial será de 30M tokens linguísticos por modelo e seed.
Checkpoints devem ser salvos em 5M, 10M, 20M e 30M.

### Por que manter MQAR

MQAR deixa de ocupar 20% do treinamento, mas não é removido. O replay de 5% é
uma proteção contra esquecimento catastrófico da memória associativa.

## Fase 2 — ablação do cronograma

Antes de executar três seeds completas, uma seed deve comparar:

| Variante | Linguagem/MQAR | Seq. | Distilação | Pergunta |
|---|---:|---:|---:|---|
| Controle | 80/20 | 128 | 0,5 fixa | reproduz o protocolo atual? |
| L90 | 90/10 | 256 | 0,5 fixa | mais linguagem já reduz CE? |
| L95 | 95/5 | 512 | 0,5 fixa | sequência longa ajuda? |
| L95-D | 95/5 | 512 | 0,5 → 0,05 | o professor estava limitando o aluno? |
| L95-ND | 95/5 | 512 | zero | retirar distilação causa deriva? |

Cada braço deve consumir exatamente o mesmo número de tokens linguísticos. O
número de passos não deve ser usado como orçamento porque muda com batch e
comprimento de sequência.

Somente as duas melhores variantes seguem para três seeds.

## Fase 3 — objetivo composto

A loss recomendada é:

$$
\mathcal{L}=
\mathcal{L}_{\mathrm{CE}}
+\lambda_{\mathrm{KD}}(t)\mathcal{L}_{\mathrm{KD}}
+\lambda_{\mathrm{MQAR}}\mathcal{L}_{\mathrm{MQAR}}
+\lambda_{\mathrm{cal}}\mathcal{L}_{\mathrm{cal}}.
$$

Para ASM-CM, inicialmente:

$$
\lambda_{\mathrm{cal}}=0.
$$

Para ASM-CM-E, a calibração somente deve ser ativada após existir um conjunto
com targets explícitos de confiabilidade. Usar apenas entropia ou acerto do
próximo token como sinônimo de incerteza epistêmica seria insuficiente.

O conjunto de calibração futuro deve conter:

- consulta com associação disponível;
- consulta sem memória correspondente;
- memórias conflitantes;
- memória corrompida;
- chave conhecida com valor novo;
- exemplos fora da distribuição;
- consulta que deve recorrer ao armazenamento externo.

O ASM-CM-E deve aprender não apenas a recuperar, mas a reconhecer quando não
possui evidência para recuperar.

## Fase 4 — continuação em escala

Se a fase de 30M tokens produzir ganho real, a melhor configuração deve seguir
uma curva contínua:

```text
100M herdados
  + 30M continuação
  + 100M continuação
  + 300M continuação
  + 500M continuação
```

Os valores representam tokens adicionais, não o total histórico. O treinamento
deve salvar milestones e manter estado do otimizador para produzir uma curva
genuína, evitando reinícios em cada ponto.

Para cada milestone, ajustar:

$$
L(N)=L_{\infty}+AN^{-\alpha}.
$$

Comparar:

- CE por token;
- CE por hora;
- inclinação $\alpha$;
- piso estimado $L_{\infty}$;
- custo de VRAM;
- MQAR por distância;
- confiança epistêmica e erro de calibração.

Não se recomenda iniciar diretamente com 1B de tokens. Primeiro é necessário
demonstrar que a curva de 30M a 300M ainda melhora sem colapso de memória.

## Fase 5 — corpus e tokenizer

Somente depois de validar o regime de treinamento devem ser alterados corpus ou
tokenizer, pois isso cria uma nova variável experimental.

### Corpus

Além da Wikipédia, incluir progressivamente:

- documentos longos;
- diálogos e narrativas;
- código, se fizer parte do produto pretendido;
- dados multilíngues, especialmente português e inglês;
- PG-19 ou outro corpus longo com procedência documentada.

Misturas devem ser versionadas, congeladas e acompanhadas por hashes.

### Tokenizer

Um tokenizer novo exige novo embedding e emitter, quebrando compatibilidade
direta com os checkpoints atuais. Deve ser tratado como experimento separado,
não como ajuste da continuação linguística.

## Gates de promoção

Uma continuação será aprovada somente quando satisfizer simultaneamente:

1. melhoria média de pelo menos 0,005 CE linguístico contra o checkpoint de
   origem;
2. vitória em pelo menos duas das três seeds;
3. desvio-padrão de CE não superior a 0,01;
4. MQAR 32K de pelo menos 95% em todas as seeds;
5. nenhuma distância curta abaixo de 95%;
6. cache constante entre 512 e 32K;
7. paridade BF16 aprovada;
8. regressão de throughput inferior a 10%;
9. ausência de NaN, divergência de gates ou crescimento não limitado do estado.

Para o ASM-CM-E, acrescentar futuramente:

10. calibração superior ao gate sem supervisão;
11. redução de falsas recuperações em consultas sem resposta;
12. capacidade de abstention sem queda excessiva de cobertura.

## Ordem de implementação

1. concluir a confirmação ASM-CM-E em andamento;
2. congelar os seis checkpoints, três ASM-CM e três ASM-CM-E;
3. criar um runner de continuação por tokens, com resume real;
4. implementar distilação decrescente;
5. implementar replay MQAR configurável por probabilidade;
6. executar a ablação de uma seed e 10M tokens por braço;
7. selecionar no máximo duas configurações;
8. executar 30M tokens e três seeds;
9. fazer rescoring completo e MQAR 32K;
10. promover a melhor configuração somente se todos os gates passarem;
11. então iniciar a curva de 100M a 500M tokens adicionais.

## Comando proposto para a futura suíte

O runner ainda deverá ser implementado. A interface planejada é:

```bash
./scripts/run_asm_cm_language_improvement.sh \
  --models ASM_CM,ASM_CM_E \
  --seeds 1,2,3 \
  --language-tokens 30000000 \
  --language-probability-final 0.95 \
  --language-seq-len-final 512 \
  --distillation-start 0.5 \
  --distillation-end 0.05 \
  --mqar-replay-probability 0.05 \
  --output-root runs/asm_cm_language_improvement_30m
```

Essa interface não deve ser simulada usando apenas número fixo de passos. O
runner deverá contabilizar separadamente tokens linguísticos, exemplos MQAR e
tokens totais processados.

## Decisão recomendada

Mais tokens são provavelmente necessários para uma melhoria linguística
material, mas a primeira ação não deve ser um treinamento longo e caro. A ordem
mais informativa é:

```text
confirmar ASM-CM-E
        ↓
10M tokens e uma seed para ablação de currículo
        ↓
30M tokens e três seeds para confirmação
        ↓
100M–500M tokens adicionais para scaling law
```

Isso separa três perguntas que não devem ser confundidas:

1. o currículo atual limita o CE?
2. mais dados continuam reduzindo o CE?
3. a memória permanece funcional durante a melhoria linguística?

O melhor resultado não será simplesmente o menor CE. Será a menor CE alcançada
sem abandonar memória associativa durável e streaming compacto.
