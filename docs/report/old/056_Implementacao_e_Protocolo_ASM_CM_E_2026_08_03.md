# ASM-CM-E — implementação e protocolo experimental

Data: 3 de agosto de 2026

## Estado da proposta

O **ASM-CM-E — Aletheion Compact Memory Model with Epistemic Gating** é uma
variante experimental do ASM-CM. Ele não substitui a arquitetura promovida nem
altera retroativamente seus benchmarks. Sua finalidade é testar se confiança
aprendida pode reduzir leituras e escritas inadequadas sem prejudicar a memória
associativa durável, a linguagem ou o estado limitado.

O código foi derivado do repositório `gnai-creator/Epistemic_Softmax`, criado
por Felipe Maya Muniz em 2025, e incorporado em
`src/drm_language_emitter/utils/epistemic_softmax.py`.

## Por que não houve uma substituição global de softmax

O ASM-CM promovido usa uma matriz fast-weight. A recuperação é feita por
produto entre query e matriz, não por softmax sobre slots:

$$
r_t = q_t^{\mathsf T}M_t.
$$

Substituir a cross-entropy linguística por uma distribuição suavizada também
mudaria simultaneamente a função de treinamento, a calibração e a eficiência
amostral. Já substituir o softmax da memória por uma mistura uniforme faria
uma consulta incerta combinar memórias possivelmente incompatíveis.

O ASM-CM-E adota uma interpretação mais conservadora: quando não há evidência
suficiente, a memória reduz a operação em vez de fabricar uma média.

## Operador epistêmico local

Duas redes produzem evidência local e consenso contextual:

$$
q_{\mathrm{local}} = \sigma(Q_{\mathrm{local}}(h_t)),
\qquad
q_{\mathrm{consenso}} = \sigma(Q_{\mathrm{consenso}}(h_t)).
$$

Ambas representam confiabilidade positiva. Isso elimina a ambiguidade da
implementação de origem, cujos comentários alternavam entre interpretar os
gates como incerteza e como confiança. A confiança e a incerteza são:

$$
c_t=q_{\mathrm{local}}q_{\mathrm{consenso}},
\qquad
u_t=1-c_t.
$$

O módulo `EpistemicSoftmax` original adaptado continua disponível para futuras
ablações do output head, mas não participa da primeira variante ASM-CM-E.

## Integração na memória

O mecanismo mantém os gates funcionais já aprendidos pelo ASM-CM e acrescenta
confiança epistêmica independente para leitura e escrita:

$$
g^{\mathrm{read}}_t =
\sigma(R(h_t))c^{\mathrm{read}}_t,
$$

$$
g^{\mathrm{write}}_t =
\operatorname{selective}(\sigma(W(h_t)))c^{\mathrm{write}}_t.
$$

A atualização passa a ser, esquematicamente:

$$
z_{t+1}=z_t+s_r g^{\mathrm{read}}_t r_t,
$$

$$
M_{t+1}=f_tM_t+g^{\mathrm{write}}_t\Delta M_t.
$$

A mesma confiança de escrita modula a consolidação lenta. O tamanho persistente
da memória permanece limitado porque os gates não acrescentam estado por token.

## Compatibilidade e inicialização

- `epistemic_memory_gating=false` preserva exatamente o caminho ASM-CM.
- A opção somente é válida com memória `fast_weight`.
- Os novos módulos ficam sob `addressable_memory.*`, mantendo o grupo de
  learning rate específico da memória.
- A confiança inicial padrão é 0,9. Assim, a variante começa próxima do
  comportamento do ASM-CM, sem o enfraquecimento aproximado de 75% que seria
  produzido por dois gates sigmoid inicialmente em 0,5.
- Checkpoints ASM-CM continuam carregáveis quando a opção está desabilitada.
  ASM-CM-E precisa treinar seus novos gates; não é uma troca de inferência em
  checkpoint congelado.

## Diagnósticos implementados

Para leitura e escrita são expostos:

- confiança;
- incerteza;
- evidência local;
- consenso contextual.

Também permanecem disponíveis os gates funcionais, normas de leitura,
retenção, consolidação e magnitude da memória.

## Hipóteses

### H1 — abstention útil

O ASM-CM-E deve aprender a reduzir operações quando o contexto não sustenta
uma recuperação ou consolidação confiável.

### H2 — preservação da memória durável

A acurácia MQAR em 32K deve permanecer em pelo menos 95% nas três seeds.

### H3 — compatibilidade linguística

A regressão média de CE diante do ASM-CM pareado não pode exceder 0,02.

### H4 — estado limitado

O cache deve permanecer constante entre 512, 4K e 32K.

## Protocolo pareado

Cada seed ASM-CM-E parte do mesmo checkpoint ASM-R usado para criar a linhagem
ASM-CM correspondente. O treinamento mantém:

- currículo MQAR de 40, 80, 160, 320, 512, 1K e 4K;
- mistura inicial de 80% linguagem e 20% MQAR;
- distilação do ASM-R;
- learning rates separados para backbone e memória;
- três seeds independentes;
- rescoring linguístico congelado;
- avaliação MQAR até 32K;
- medição de cache no decode compacto.

Os gates preregistrados são:

1. todos os currículos curtos aprovados;
2. MQAR 32K de pelo menos 95% em todas as seeds;
3. cache constante em todos os comprimentos;
4. regressão média de CE de no máximo 0,02.

O resultado deve ser interpretado como uma ablação de segurança e seletividade,
não como promoção automática. Mesmo que passe, ainda será necessário medir
calibração, falsos positivos de recuperação, OOD e decisão de escalonamento.

## Comando completo

```bash
./scripts/run_asm_cm_e_suite.sh
```

É possível reduzir o custo apenas para smoke testing, sem valor conclusivo:

```bash
CURRICULUM='40:20,80:10' \
LANGUAGE_EVAL_TOKENS=100000 \
LONG_EVAL_BATCHES=4 \
./scripts/run_asm_cm_e_suite.sh
```

Os artefatos finais serão gravados em:

```text
runs/asm_cm_e_suite/
├── seed_1/
├── seed_2/
├── seed_3/
├── decision.json
└── report.md
```

## Próxima fase caso seja aprovado

O experimento seguinte deve fornecer supervisão explícita de confiabilidade:
consultas respondíveis, consultas sem memória correspondente, memórias
conflitantes, corrupção controlada e dados fora da distribuição. O objetivo
será medir se `u_t` prevê falha real, e não apenas se o novo gate preserva CE e
MQAR. Só então ASM-CM-E poderá atuar como roteador entre memória local,
armazenamento episódico, LLM local, LLM remota e abstention.
