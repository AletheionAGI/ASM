# Plano de implementação ASM-C2 — memória recuperável e endereçável

> **Estado em 2026-08-01:** implementação concluída para o núcleo, integração
> causal/streaming, variantes 16/32/64, controles NOREAD/NOWRITE/memória
> embaralhada, testes e runner com gates automáticos. A validação experimental
> na RTX 4090 reprovou o controle curto por endereçamento quase uniforme. A
> correção esparsa e seu microteste estão descritos no
> [report 048](048_Diagnostico_ASM_C2_e_Plano_Memoria_Esparsa_2026_08_01.md).

## 1. Motivação experimental

O ASM-C resolveu o problema de crescimento do estado de inferência:

- cache retido constante de 6.144 bytes até 32K;
- pico medido de VRAM constante em 387,53 MiB;
- retenção de 99,6% do throughput entre 4K e 32K;
- throughput de 503,4 tokens/s em 32K, 2,97 vezes o caminho ASM-R anterior.

Contudo, a suíte MQAR curta mostrou que execução compacta não implica memória
recuperável. Após 20.000 passos pareados:

| Variante | Acurácia | CE |
|---|---:|---:|
| Transformer pareado | 100,00% | 0,000379 |
| ASM-C | 36,21% | 1,549430 |
| ASM-C memória 2x | 35,96% | 1,459153 |
| ASM-S | 35,42% | 1,769726 |

O Transformer já alcançava 99,78% em 5.000 passos. Nenhum ASM atingiu 50%, e
dobrar a largura interna da memória reduziu CE sem melhorar a acurácia final.

Esses resultados enfraquecem duas explicações simples:

1. faltavam apenas mais passos de adaptação;
2. faltava apenas uma projeção forget/write mais larga.

A hipótese de trabalho passa a ser estrutural: a memória atual sabe preservar,
esquecer e sobrescrever um vetor comprimido, mas não possui uma operação
explícita de busca por conteúdo.

## 2. Objetivo do ASM-C2

ASM-C2 será a segunda geração do **Aletheion Compact State Model**, combinando:

1. estado recorrente compacto para continuidade global;
2. bloco local limitado para execução streaming;
3. banco de memória de tamanho fixo;
4. leitura endereçável por conteúdo;
5. escrita e substituição seletivas;
6. custo de memória independente do comprimento total da sequência.

O objetivo não é recriar self-attention sobre todo o prefixo. O modelo poderá
consultar um número fixo de slots, mas não armazenará uma chave e um valor para
cada token passado.

Para uma sequência com comprimento acumulado $T$, número de slots $S$ e
dimensão de memória $d_m$, o alvo de complexidade é:

$$
\text{memória persistente}=O(Sd_m),
\qquad
\text{custo de acesso por token}=O(Sd_m),
$$

com $S$ fixo e independente de $T$.

## 3. Arquitetura proposta

### 3.1 Estado de inferência

O estado compacto será ampliado de:

```text
estado concluído + bloco aberto + contador
```

para:

```text
estado concluído
+ bloco aberto limitado
+ contador de posição
+ chaves de memória [batch, slots, d_memory]
+ valores de memória [batch, slots, d_memory]
+ uso/idade dos slots [batch, slots]
```

Nenhum `input_ids` completo, histórico do emitter ou memória proporcional a
$T$ poderá reaparecer no cache.

### 3.2 Leitura por conteúdo

A cada token, uma consulta será formada pelo estado e pela entrada:

$$
q_t=W_q[z_t;x_t].
$$

Para chaves $K_t$ e valores $M_t$ nos $S$ slots:

$$
a_t=\mathrm{softmax}\left(\frac{q_tK_t^\top}{\sqrt{d_m}}\right),
\qquad
r_t=a_tM_t.
$$

O vetor recuperado $r_t$ será integrado à transição ASM-R por uma projeção
residual com gate inicial pequeno:

$$
\widetilde z_t=z_t+\gamma_t\odot W_r r_t.
$$

Esse acesso é semelhante a attention sobre **slots fixos de memória**, mas não
é self-attention entre todos os tokens. A documentação deverá dizer isso
explicitamente; ASM-C2 não deve ser anunciado genericamente como “sem
attention” sem qualificar o mecanismo.

### 3.3 Escrita seletiva

O controlador produzirá:

- intensidade de escrita $w_t$;
- chave candidata $k_t$;
- valor candidato $m_t$;
- vetor de apagamento $e_t$;
- preferência por slots semelhantes, livres ou antigos.

Uma atualização inicial simples será:

$$
K_{t+1}=K_t\odot(1-w_t e_t)+w_t k_t,
$$

$$
M_{t+1}=M_t\odot(1-w_t e_t)+w_t m_t,
$$

onde $w_t$ é distribuído sobre os slots. A primeira implementação deve usar
endereçamento suave e diferenciável. Escrita discreta, top-k ou straight-
through só deve ser considerada após a versão suave estar correta.

### 3.4 Política de substituição

O ASM-C2 precisa evitar que todos os slots convirjam para o mesmo conteúdo. A
prioridade de escrita combinará:

1. similaridade com uma chave existente, para atualizar uma associação;
2. baixa utilização, para ocupar um slot livre;
3. idade, para substituir memória antiga;
4. gate de novidade, para não escrever todo token.

Diagnósticos obrigatórios:

- entropia das leituras;
- entropia das escritas;
- fração de slots usados;
- colisões de chave;
- idade média dos slots;
- frequência de escrita;
- norma do vetor recuperado;
- contribuição residual da memória ao estado.

## 4. Variantes experimentais

### ASM-C2-16

Banco com 16 slots. Controle de menor custo, útil para verificar se o mecanismo
funciona antes de aumentar capacidade.

### ASM-C2-32

Variante principal inicial. O protocolo MQAR usa oito pares; 32 slots oferecem
folga sem tornar a busca grande.

### ASM-C2-64

Controle de capacidade. Deve mostrar se o ganho cresce com slots ou se o
problema está na política de escrita/leitura.

### ASM-C2-NOREAD

Mantém parâmetros e escrita, mas bloqueia a leitura no estado. Controla se o
ganho vem realmente de recuperação.

### ASM-C2-NOWRITE

Mantém o caminho de leitura sobre memória inicial fixa, mas não atualiza slots.
Controla ganhos provenientes apenas de parâmetros ou residual adicional.

### Controles externos

- ASM-C atual;
- ASM-S;
- Transformer pareado;
- opcionalmente ASM-C2 com memória embaralhada durante avaliação.

O número de slots altera principalmente estado persistente e computação, não
necessariamente parâmetros. Comparações devem relatar parâmetros, bytes de
cache, FLOPs aproximados e tokens por segundo separadamente.

## 5. Organização de código

Arquivos planejados:

```text
src/aletheion_state_models/
  core/
    addressable_memory.py
  variants/
    compact_addressable.py

src/drm_language_emitter/
  inference.py                 # extensão compatível do InferenceState

scripts/
  run_asm_c2_mqar_suite.sh
  compare_asm_c2_controls.py
  plot_asm_c2_results.py

tests/
  test_addressable_memory.py
  test_asm_c2_streaming.py
```

`AddressableMemory` deverá implementar uma interface neutra e não depender de
classes DRM. A variante `build_compact_addressable` combinará esse componente
com a transição ASM-R/ASM-C existente.

A implementação reutiliza `benchmark_asm_r_long_streaming.py`, cujo protocolo
é compatível com checkpoints ASM-C2, evitando duplicar lógica de streaming e
MQAR longo.

## 6. Configuração proposta

Novos campos, desabilitados por padrão:

```text
addressable_memory: false
addressable_memory_slots: 32
addressable_memory_dim: 128
addressable_memory_heads: 1
addressable_memory_read_scale: 0.1
addressable_memory_write_bias: -2.0
addressable_memory_temperature: 1.0
addressable_memory_usage_decay: 0.99
addressable_memory_age_bias: 1.0
```

O ASM-R e ASM-C atuais devem manter exatamente o mesmo comportamento quando
`addressable_memory=false`.

## 7. Compatibilidade de checkpoints

ASM-C2 deverá carregar todos os pesos compatíveis do checkpoint ASM-R de 100M.
Somente os módulos novos serão inicializados aleatoriamente.

O carregamento deve:

1. listar todas as chaves novas esperadas;
2. rejeitar qualquer incompatibilidade fora de `addressable_memory.*`;
3. registrar hashes do checkpoint de origem;
4. registrar seed e inicialização da memória;
5. nunca apresentar ASM-C2 como checkpoint integralmente pré-treinado.

Uma futura avaliação de linguagem exigirá pré-treinamento próprio do ASM-C2,
pois adaptação MQAR de módulos novos não mede integração linguística completa.

## 8. Fases de implementação

### Fase 1 — núcleo matemático

- implementar estado de chaves, valores, uso e idade;
- implementar leitura suave;
- implementar escrita, apagamento e substituição;
- garantir tensores finitos em FP32 e BF16;
- expor diagnósticos sem alterar logits quando desabilitado.

Critério: testes unitários de escrita e recuperação determinística passam.

### Fase 2 — integração causal

- integrar a leitura antes da transição do estado;
- atualizar a memória somente com informação disponível até $t$;
- conectar memória ao `InferenceState` compacto;
- preservar estado entre blocos;
- impedir fallback para prefixo completo.

Critério: alterar tokens futuros não modifica memória, estados ou logits do
prefixo.

### Fase 3 — paridade treino/decode

- comparar forward completo com prefill + decode token a token;
- testar fronteiras de bloco;
- testar serialização e retomada do estado;
- medir erro FP32 e BF16;
- testar batch maior que um.

Critério: paridade FP32 dentro da tolerância definida e divergência BF16
explicitamente medida.

### Fase 4 — MQAR curto

- executar ASM-C, ASM-C2-16, ASM-C2-32, ASM-C2-64 e Transformer;
- milestones 500, 1K, 2K e 5K;
- usar batches pareados e 4.096 targets de validação;
- exigir pelo menos 80% antes de qualquer teste longo.

Critério principal: ASM-C2 ultrapassa 80% até 5K. Meta competitiva: aproximar-
se do Transformer sem perder streaming limitado.

### Fase 5 — ablações da memória

- NOREAD;
- NOWRITE;
- memória embaralhada;
- slots 16/32/64;
- política apenas por similaridade versus similaridade + uso/idade.

Critério: o ganho desaparece quando leitura ou escrita útil é removida.

### Fase 6 — retenção longa

Somente variantes aprovadas no controle curto avançam para 512–32K. Medir:

- acurácia e CE MQAR;
- intervalos de confiança;
- cache e pico de VRAM;
- throughput incremental;
- ocupação e colisão dos slots;
- degradação por distância e número de associações.

Critério: cache e VRAM permanecem limitados e a acurácia fica materialmente
acima do acaso em distâncias longas.

### Fase 7 — regressão de linguagem

- avaliar CE congelado do checkpoint com módulos novos inicialmente neutros;
- adaptar ou pré-treinar ASM-C2 em 5M tokens como triagem;
- comparar CE, tempo, memória e throughput com ASM-C;
- somente depois considerar 30M/100M.

Critério: nenhum ganho MQAR será promovido se causar regressão material não
justificada em linguagem ou destruir a vantagem streaming.

## 9. Testes obrigatórios

### Unidade

- shapes de leitura e escrita;
- distribuição de atenção sobre slots soma um;
- escrita desabilitada não altera memória;
- leitura desabilitada não altera estado;
- uso e idade permanecem limitados;
- ausência de NaN/Inf;
- gradientes alcançam chaves, valores e controladores.

### Causalidade

- prefixos idênticos produzem memória idêntica;
- perturbações futuras não alteram slots anteriores;
- estado salvo e retomado reproduz a continuação;
- nenhuma informação do target entra na escrita.

### Limite de memória

- bytes do cache constantes entre 4K e 32K;
- nenhuma dimensão do cache depende de `tokens_seen`;
- profiler confirma ausência de prefixo completo;
- pico de VRAM cresce no máximo 10% entre 4K e 32K.

### Qualidade

- controle curto MQAR com 4.096 targets;
- intervalos de confiança de 95%;
- três seeds para a variante candidata;
- comparação pareada com ASM-C e Transformer.

## 10. Gates de promoção

ASM-C2 só poderá avançar se cumprir simultaneamente:

1. acurácia MQAR curta de pelo menos 80%;
2. ganho reproduzido em pelo menos duas de três seeds;
3. NOREAD/NOWRITE confirmam que a memória causa o ganho;
4. cache limitado até 32K;
5. crescimento de pico de VRAM no máximo 10% entre 4K e 32K;
6. retenção de throughput em 32K de pelo menos 80% do valor em 4K;
7. paridade causal e retomada de estado aprovadas;
8. regressão de CE de linguagem documentada e dentro do limite definido antes
   do experimento.

Passar em MQAR não basta para promover ASM-C2 como modelo de linguagem. Passar
em streaming não basta para promovê-lo como memória. As duas propriedades
precisam coexistir.

## 11. Comando final planejado

Após a implementação, a interface pública desejada será:

```bash
./scripts/run_asm_c2_mqar_suite.sh
```

Ela deverá:

1. rodar testes unitários e de causalidade;
2. executar o controle curto;
3. bloquear automaticamente a fase longa se nenhuma variante atingir 80%;
4. executar ablações da vencedora;
5. executar streaming 4K–32K;
6. produzir JSON, CSV, gráficos e relatório final.

Essa interface está implementada. A suíte também confirma o candidato em três
seeds, mede paridade BF16 e executa regressão congelada de CE antes da decisão
final.

## 12. Decisão de desenvolvimento

O desenvolvimento de CE permanece pausado como objetivo primário. O CE será
usado como teste de regressão durante a construção do ASM-C2.

A pergunta científica prioritária é:

> Um estado recorrente de tamanho limitado, auxiliado por um banco igualmente
> limitado de memória endereçável, consegue recuperar associações com eficiência
> sem reintroduzir armazenamento proporcional ao prefixo?

Se a resposta for positiva, ASM-C2 terá uma vantagem estrutural concreta para
investigar. Se for negativa mesmo após controles corretos, a evidência indicará
que a família ASM atual não possui um mecanismo competitivo de memória
associativa e deverá ser reformulada antes de novos investimentos de escala.
