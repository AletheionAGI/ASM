# Contribuição geométrica em J e próximas ablações

Data: 2026-07-31  
Branch experimental: `drm-fix`  
Escala: aproximadamente 126M parâmetros, 5M tokens, seeds 1/2/3

## 1. Pergunta

A variante J combinou a dinâmica geométrica DRM com memória seletiva
forget/write e obteve uma grande melhora de CE. Isso criou uma questão causal:

> O ganho vem da geometria DRM ou apenas da memória seletiva semelhante a um
> SSM gated?

Para responder, foi criado `SSM_CONTROL`, que mantém embedding, mixer causal
curto, residual token→estado, memória seletiva e emitter, mas não instancia:

- `DirectionField`;
- `RelationalMetric`;
- `DRMFlow`;
- `RiskField`.

Os parâmetros removidos foram transferidos para a largura da memória seletiva.
Assim, J possui 126.080.896 parâmetros e `SSM_CONTROL` possui 126.076.000,
diferença de apenas 4.896 parâmetros (0,0039%).

## 2. Protocolo

- três seeds pareadas: 1, 2 e 3;
- exatamente 5.005.312 tokens observados por execução;
- mesmos manifests de treino e validação;
- rescoring sequencial sobre os mesmos 4.834.787 targets;
- seleção independente do mínimo de amostras variáveis;
- comparação conjunta de CE, dispersão e throughput.

## 3. Resultado por seed

| Seed | J CE | SSM_CONTROL CE | Ganho de J |
|---:|---:|---:|---:|
| 1 | 1,761031 | 1,806133 | 0,045102 |
| 2 | 1,764080 | 1,814285 | 0,050204 |
| 3 | 1,756632 | 1,799135 | 0,042504 |
| **Média** | **1,760581** | **1,806518** | **0,045937** |

Agregados:

| Variante | CE médio | Desvio-padrão | CE mínimo | CE máximo | PPL aproximada |
|---|---:|---:|---:|---:|---:|
| J | **1,760581** | **0,003057** | 1,756632 | 1,764080 | 5,82 |
| SSM_CONTROL | 1,806518 | 0,006191 | 1,799135 | 1,814285 | 6,09 |

A pior seed de J ainda foi melhor que a melhor seed do controle. Portanto, não
existe sobreposição entre as faixas observadas.

## 4. Custo computacional

| Seed | J tokens/s | SSM_CONTROL tokens/s | Controle/J |
|---:|---:|---:|---:|
| 1 | 13.754 | 33.424 | 2,43x |
| 2 | 13.756 | 34.758 | 2,53x |
| 3 | 13.811 | 34.386 | 2,49x |

O controle é pareado por parâmetros, não por compute. J oferece CE
substancialmente melhor, mas custa aproximadamente 2,5 vezes mais tempo de
treino.

## 5. Conclusão permitida

Neste controle, a memória seletiva sozinha não explica todo o ganho:

```text
memória seletiva + geometria DRM
                  >
memória seletiva sem geometria DRM
```

J passou pelo gate de três seeds, ganho mínimo de 0,005 CE, maioria de seeds e
desvio-padrão máximo de 0,03. Isso é evidência causal de contribuição do
conjunto geométrico implementado.

O resultado ainda não prova:

- que todos os componentes geométricos são necessários;
- que a métrica é o componente responsável;
- que J supera um Mamba real;
- que o ganho persiste em 30M ou 150M tokens;
- que a vantagem de CE compensa o custo de compute.

## 6. Próxima matriz causal

### 6.1 J completa

Referência positiva:

```text
campo direcional + fluxo + métrica/naturalização
+ residual + memória seletiva + mixer + emitter
```

Resultado congelado em 5M: CE médio 1,760581.

### 6.2 J sem métrica — `J_NO_METRIC`

Remover a instanciação de `RelationalMetric` e usar identidade no caminho da
velocidade:

\[
\Delta z_t = \sum_k \alpha_k(z_t,x_t)v_k(z_t)
\]

Objetivo: testar o valor do módulo métrico completo e sua parametrização. Como
a remoção libera muitos parâmetros, devem ser reportadas duas leituras:

1. ablação estrutural com as larguras compartilhadas de J;
2. controle posterior com os parâmetros redistribuídos.

A primeira identifica necessidade causal; a segunda mede competitividade sob
orçamento.

### 6.3 J sem campo direcional — `J_NO_DIRECTION`

Remover direções e gates dependentes do estado. Para não apagar também a
capacidade de transição, substituí-los por uma projeção direta causal
\(T(z_t,x_t)\rightarrow\Delta z_t\), inicializada em stream independente.

Objetivo: testar se decompor o movimento em direções ativas oferece algo além
de uma transição neural direta. O substituto e sua contagem precisam ser
declarados; este teste não deve ser descrito como remoção de todo o fluxo.

### 6.4 J sem naturalização — `J_NO_NATURALIZATION`

Manter `RelationalMetric` calculada e diagnosticada, mas forçar:

\[
g(z_t)^{-1}\Delta z_t \longrightarrow \Delta z_t
\]

Na prática, `metric_naturalization_strength = 0`. Todos os demais módulos,
parâmetros e streams de inicialização permanecem iguais.

Objetivo: isolar se aplicar a métrica à trajetória melhora CE. Esse é o teste
causal mais limpo para a contribuição da naturalização.

### 6.5 J sem RiskField — já realizado

J já usa:

```text
instantiate_disabled_risk = false
lambda_blindspot = 0
```

Portanto, não existe `RiskField` na J atual e não há novo treino a executar.
F/I já demonstraram que retirar o scaffold desabilitado, mantendo
inicialização estável, não muda o CE. J é simultaneamente a variante completa
atual e a variante sem RiskField.

## 7. Protocolo dos próximos testes

Executar somente as três variantes novas e usar os checkpoints J já congelados:

```text
J_NO_METRIC
J_NO_DIRECTION
J_NO_NATURALIZATION
```

Para cada uma:

- seeds 1, 2 e 3;
- 5M tokens;
- mesma sequência de treino;
- mesmo scheduler e hiperparâmetros;
- rescoring contínuo dos 4.834.787 targets;
- MQAR nas mesmas seeds;
- parâmetros, throughput, VRAM e tempo total registrados;
- comparação pareada contra J.

Critério por componente:

- J melhora pelo menos 0,005 CE em média;
- J vence pelo menos duas das três seeds;
- desvio-padrão da candidata não excede 0,03;
- faixas e deltas por seed são publicados;
- MQAR é usado como evidência complementar, não substituto do CE.

Interpretação:

| Resultado | Conclusão |
|---|---|
| J > J_NO_METRIC | O módulo métrico completo contribui |
| J ≈ J_NO_METRIC | A métrica parametrizada é redundante nessa escala |
| J > J_NO_DIRECTION | O campo direcional contribui além da transição direta |
| J ≈ J_NO_DIRECTION | A decomposição direcional é redundante |
| J > J_NO_NATURALIZATION | Precondicionar o fluxo pela métrica contribui |
| J ≈ J_NO_NATURALIZATION | A métrica não influencia CE pelo caminho proposto |
| Variante reduzida > J | O componente removido prejudica otimização |

## 8. Ordem de execução

1. implementar e testar os três caminhos de ablação;
2. confirmar contagens e independência de inicialização;
3. fazer smoke até além do passo 50 para detectar NaN;
4. executar MQAR curto;
5. executar 5M × três seeds;
6. fazer rescoring contínuo;
7. decidir quais componentes permanecem;
8. promover somente a arquitetura mínima vencedora para 30M;
9. adicionar um Mamba real de orçamento comparável.

## 9. Decisão atual

- Memória seletiva: contribuição confirmada contra I.
- Geometria DRM como conjunto: contribuição confirmada contra SSM_CONTROL.
- Mixer dilatado: ganho insuficiente contra J.
- RiskField desabilitado: não necessário.
- Métrica isolada: pendente.
- Campo direcional isolado: pendente.
- Naturalização isolada: pendente.
- Promoção para 30M: aguardando a decomposição geométrica e MQAR.

## 10. Implementação da matriz

As três variantes foram implementadas:

| Variante | Intervenção | Parâmetros |
|---|---|---:|
| J | referência completa atual, sem RiskField | 126.080.896 |
| J_NO_METRIC | remove RelationalMetric e usa identidade | 72.652.160 |
| J_NO_DIRECTION | troca campo+fluxo restrito por transição causal direta | 83.206.400 |
| J_NO_NATURALIZATION | mantém métrica, mas não precondiciona o fluxo | 126.080.896 |

`J_NO_METRIC` e `J_NO_DIRECTION` são ablações estruturais, não controles
pareados por parâmetros. Essa diferença é intencional nesta primeira etapa:
mede-se se remover o mecanismo destrói capacidade. Se uma variante reduzida
empatar com J, uma segunda etapa redistribuirá os parâmetros liberados.

Como todas as perdas auxiliares geométricas estão zeradas, a métrica afeta CE
somente através da naturalização. Portanto, `J_NO_METRIC` e
`J_NO_NATURALIZATION` devem produzir a mesma trajetória e o mesmo CE quando os
streams compartilhados de inicialização são idênticos. Executar ambas confirma
essa equivalência e mede separadamente a economia de parâmetros; não são duas
hipóteses funcionais independentes no protocolo CE-only.

O launcher `scripts/run_drm_geometry_component_suite.sh` executa, em ordem:

1. suíte completa de testes;
2. forward smoke das quatro variantes;
3. MQAR nas seeds 1, 2 e 3;
4. treino de 5M tokens nas quatro variantes e três seeds;
5. rescoring contínuo;
6. decisões automáticas de J contra cada ablação.

Comando:

```bash
./scripts/run_drm_geometry_component_suite.sh
```

Diretório padrão:

```text
runs/drm_geometry_component_ablation_5m
```

É possível reduzir somente o MQAR para um ensaio operacional:

```bash
MQAR_STEPS=10 \
OUTPUT_ROOT=runs/drm_geometry_component_operational_smoke \
./scripts/run_drm_geometry_component_suite.sh
```

Esse segundo comando ainda executa os treinos de 5M; ele não é um dry-run da
suíte inteira.
