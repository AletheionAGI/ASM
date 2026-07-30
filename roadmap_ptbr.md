# Roadmap formal de implementação do DRM

Data original: 2026-07-11
Alinhado ao repositório e ao `docs/paper/drm_v6.tex`: 2026-07-30

Versão detalhada histórica em inglês: [roadmap.md](roadmap.md).

## Estado atual

O projeto está no **Gate 0 — validação empírica independente e fidelidade de
runtime**, antes da Fase 1 da implementação formal.

O marco ativo é o benchmark independente 125M:

- manifests de treino e validação da Wikipédia separados por documento;
- PG-19 externo congelado;
- auditorias de contaminação aprovadas com zero sobreposições;
- smoke test CUDA aprovado em uma RTX 4090;
- três runs DRM e três GPT-2, com 150M tokens cada;
- seleção por validação antes de uma única consulta ao teste externo.

O Gate 0 termina somente quando:

1. os seis runs forem concluídos;
2. a seleção de checkpoint usar os mesmos tokens determinísticos de validação;
3. cada checkpoint selecionado for avaliado uma vez no PG-19;
4. média, dispersão e diferença entre modelos forem publicadas;
5. a geração reproduzir o sequence mode/local mixer treinado;
6. flags públicas inertes forem implementadas ou rejeitadas.

## Base já implementada

- `DRMEmitterModel`: estado latente causal sobre sequências.
- `DirectionField`: direções e gates dependentes do estado.
- `DRMFlow`: velocidade no span das direções.
- `RelationalMetric`: métrica SPD `diag + U Uᵀ`.
- naturalização métrica via identidade de Woodbury.
- atualização de estado limitada opcionalmente.
- caminhos recorrente, directional cumsum, block cumsum e superblock.
- Anderson causal e fixed point experimentais.
- mixer convolucional causal local.
- diagnósticos de gates, ação, condição, recorrência e estabilidade.
- datasets memmap com verificação de tamanho e SHA-256.
- checkpoints `weights_only=True`.
- preparação documental e auditoria de corpus externo.
- avaliação congelada e determinística do test set.

## Limitações formais atuais

Ainda não existem:

- métrica positiva semidefinida com kernel verdadeiro;
- fibra efetiva `E/Ker(g)`;
- estratos de rank constante;
- anchor formal `rho:E→TM`;
- conexão relacional;
- transporte paralelo intrastatum;
- mapas explícitos de transição de rank `J`;
- transporte híbrido ordenado;
- holonomia de transição e invariantes de histerese;
- redução Fisher–Rao implementada;
- fechamento toroidal.

`use_toroidal_state` e `tie_embeddings` são aceitos pela configuração, mas
estão inertes.

## Distinção conceitual essencial

O paper define:

\[
d_{\mathrm{DRM}}(p)=\operatorname{rank}(g_p),
\qquad
\overline E_p=E_p/\operatorname{Ker}(g_p).
\]

A métrica neural atual é:

\[
G(z)=\operatorname{diag}(\operatorname{softplus}(d(z))+\varepsilon)
     +U(z)U(z)^\top.
\]

Como a diagonal é estritamente positiva, `G` é positiva definida e seu rank
exato é sempre `d_state`. Portanto:

- `dimD = sum(gates)` não é `rank(g)`;
- um rank numérico por tolerância pode ser um diagnóstico operacional;
- esse rank deve ser rotulado como aproximação espectral;
- para implementar literalmente o paper será necessária uma métrica PSD que
  permita kernel verdadeiro.

## Dependência formal do paper v6

```text
métrica PSD
→ kernel e fibra quociente
→ estratos de rank constante
→ anchor compatível com o kernel
→ conexão e transporte intrastatum
→ mapas explícitos de transição
→ transporte híbrido
→ holonomia e histerese
```

O tuple formal completo é:

```text
(M, E, g, rho, {S_alpha}, {nabla_alpha}, {J_e})
```

O código atual implementa apenas uma analogia neural de `M`, dinâmica
direcional, métrica SPD e energia de trajetória.

## Resumo das fases

| Fase | Entrega | Estado | Prioridade |
|---:|---|---|---|
| 0 | Validação independente e fidelidade de runtime | Em andamento | Crítica |
| 1 | Rank métrico, kernel e estratos | Não iniciada | Crítica |
| 2 | Anchor e movimento admissível | Não iniciada | Alta |
| 3 | Conexão e transporte intrastatum | Não iniciada | Alta |
| 4 | Mapas de transição e critérios energéticos | Não iniciada | Alta |
| 5 | Holonomia híbrida e histerese | Não iniciada | Alta |
| 6 | Reduções clássicas e Fisher–Rao | Não iniciada | Média |

Fechamento toroidal é uma trilha opcional posterior. O paper afirma
explicitamente que boundedness e recorrência não implicam um toro.

---

# Fase 1 — rank métrico, kernel e estratos

## Objetivo

Separar três medidas hoje confundidas:

1. atividade direcional dos gates;
2. rank numérico/espectral;
3. rank matemático exato.

## Primeira integração

Somente diagnóstica, sem mudar o treinamento congelado.

Novo módulo sugerido:

```text
src/drm_language_emitter/effective_geometry.py
```

Diagnósticos:

- autovalores da métrica;
- rank numérico absoluto e relativo;
- participation rank;
- stable rank;
- massa espectral próxima do kernel;
- gap espectral;
- histórico de labels de estrato;
- comparação explícita com `gate_dimD`.

Uma tolerância relativa pode ser definida como:

\[
\tau=\max(\tau_{\mathrm{abs}},
          \tau_{\mathrm{rel}}\lambda_{\max}).
\]

Então:

\[
r_\tau(G)=\#\{\lambda_i(G)>\tau\}.
\]

Esse valor deve ser exportado como `metric_numerical_rank`, não
`metric_exact_rank`.

## Testes

- matrizes PSD sintéticas com ranks 0, 1, intermediário e completo;
- invariância aproximada sob mudança uniforme de escala;
- estabilidade em `float32` e `float64`;
- resultados finitos em CPU e CUDA;
- distinção entre gates e rank métrico;
- métrica neural atual reporta rank exato completo.

## Critério de conclusão

- diagnósticos sem efeito no forward de treino;
- tolerâncias documentadas e congeláveis;
- nenhum uso de `dimD` como sinônimo de `rank(g)`;
- estimador validado em exemplos do paper.

---

# Fase 2 — anchor e movimento admissível

## Objetivo

Introduzir explicitamente:

\[
\rho:E\to TM
\]

e verificar:

\[
\operatorname{Ker}(g_p)\subseteq\operatorname{Ker}(\rho_p).
\]

## Implementação incremental

- anchor identidade como padrão;
- módulo opcional `anchor.py`;
- diagnóstico de compatibilidade com o kernel;
- rank da mobilidade observável;
- curvas admissíveis separadas de simples updates neurais.

## Critério de conclusão

- comportamento padrão permanece inalterado;
- anchor não identidade possui testes de forma e gradiente;
- incompatibilidade com kernel é detectada;
- “anchor” dos solvers geodésicos não é confundido com `rho`.

---

# Fase 3 — conexão e transporte intrastatum

## Objetivo

Definir transporte apenas dentro de estratos de rank constante, sobre a fibra
quociente efetiva.

## Primeira implementação

- aproximação diagnóstica por alinhamento de frames;
- verificação de compatibilidade métrica;
- invertibilidade dentro do estrato;
- drift do frame e erro de transporte dos gates;
- rejeição de transporte regular através de mudança de rank.

Arquivos sugeridos:

```text
src/drm_language_emitter/transport.py
tests/test_transport.py
```

## Critério de conclusão

- frames idênticos produzem drift próximo de zero;
- permutações são alinhadas;
- transporte regular é invertível;
- energia métrica é preservada dentro da tolerância;
- mudanças de estrato exigem mapas `J`, não transporte regular.

---

# Fase 4 — mapas de transição e energia

## Objetivo

Identificar mapas:

\[
J_{r\to s}:V_-\to V_+
\]

em eventos de mudança de rank.

## Diagnósticos

- rank de `J`;
- norma de operador métrica;
- perda de energia;
- defeito de rank;
- classificação dissipativa, isométrica ou expansiva;
- violação do critério:

\[
J^\top G_+J\preceq G_-.
\]

## Testes

- colapso `2→1`;
- reativação `1→2`;
- projeção ortogonal dissipativa;
- transição expansiva;
- composição de transições dissipativas.

## Critério de conclusão

- eventos de rank têm mapas explícitos;
- rank perdido não reaparece deterministicamente sem canal auxiliar;
- critérios matriciais reproduzem os exemplos do paper.

---

# Fase 5 — holonomia híbrida e histerese

## Objetivo

Compor:

\[
H_\gamma=P_mJ_{m-1}P_{m-1}\cdots J_1P_1.
\]

## Invariantes

- defeito de rank;
- `||H-I||`;
- defeito energético;
- persistência espectral;
- comutadores de transição;
- dependência da ordem;
- subespaços sobrevivente e perdido.

## Testes

- loop sem transição retorna identidade;
- loop com colapso de rank não pode retornar identidade;
- histerese existe mesmo com transportes intrastatum identidade;
- loops dissipativos formam composição não expansiva;
- ordem de transições não comutativas altera o resultado.

## Critério de conclusão

- exemplos computacionais reproduzem os teoremas centrais do paper;
- holonomia neural é distinguida de proxies de recorrência;
- relatórios registram o caminho híbrido completo.

---

# Fase 6 — reduções e Fisher–Rao

## Objetivo

Verificar computacionalmente as hipóteses das reduções:

- Riemanniana;
- sub-Riemanniana;
- Fisher–Rao.

Fisher–Rao não é simplesmente “usar a Hessiana da loss”. É necessário
identificar a fibra efetiva com o espaço tangente dos parâmetros e comparar a
métrica com:

\[
\mathbb E_\theta[
  \partial_i\log p(X;\theta)
  \partial_j\log p(X;\theta)
].
\]

## Critério de conclusão

- hipóteses verificadas explicitamente;
- métricas comparadas em exemplos pequenos;
- aproximação empírica não é apresentada como equivalência formal;
- custo computacional e tolerâncias documentados.

---

# Trilha opcional — fechamento toroidal

Não faz parte da sequência central. Só deve ser ativada com evidência de:

- variedade invariante compacta e conexa;
- campos completos;
- independência pontual;
- comutatividade;
- ação transitiva de `R^n`;
- estabilizador como lattice de rank completo.

Aplicar `sin/cos`, limitar a norma ou observar recorrência não prova essas
condições.

---

# Ordem prática recomendada

Enquanto o benchmark congelado estiver em execução:

- implementar apenas diagnósticos novos e isolados;
- não alterar `RelationalMetric`, `DRMEmitterModel.forward` ou scripts ativos;
- adicionar testes PSD sintéticos;
- preparar a Fase 1 em branch ou mudanças locais separadas.

Após o benchmark:

1. corrigir paridade entre geração e forward local-mixer;
2. corrigir semântica default de `geometry_report`;
3. implementar ou rejeitar flags inertes;
4. concluir a Fase 1 diagnóstica;
5. decidir, com ablação própria, se a métrica de treino deve permitir kernel
   exato;
6. avançar pelas Fases 2–6.

## Afirmação científica adequada hoje

```text
drm-language-emitter é um protótipo neural de modelagem de linguagem
inspirado em Directional Relational Manifolds.
```

Somente após implementar e validar kernel, estratos, anchor, conexão,
transições e holonomia será adequado afirmar que o código realiza
computacionalmente a estrutura formal completa do paper.
