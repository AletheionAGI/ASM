# Controles oracle, dense-slot e fast-weight do ASM-C2

## Motivação

O probe isolado anterior terminou com 6,09% para a memória densa, 2,51% para
Top-1 e 3,43% para Top-2, contra 1,65% do controle sem memória. O acaso é
1,5625%. A memória transportou algum sinal, mas o roteamento esparso tornou a
associação ainda mais difícil e nenhuma variante se aproximou do gate de 95%.

Isso mostra que aumentar o ASM-C2 ou executar novamente a suíte completa não é
justificado. A falha deve ser decomposta entre protocolo, roteamento e
armazenamento.

## Novos controles

### ORACLE_SLOT

Usa o identificador da chave como endereço determinístico e recupera o valor
gravado no mesmo endereço. Não aprende e não é uma arquitetura candidata. Sua
única função é validar geração MQAR, posições de target, máscara e leitura.

### DENSE_SLOT_LONG

Mantém o módulo denso anterior e estende sua curva de 2 mil para 10 mil passos.
Isso verifica se os 6,09% eram apenas uma fase inicial lenta ou um platô.

### FAST_WEIGHT

Substitui a escolha aprendida de slots por uma memória associativa contínua de
capacidade fixa. Para cada par, uma representação aprendida do valor é escrita
atomicamente na linha endereçada pela chave. A leitura recupera essa linha e um
emitter aprendido decodifica o valor:

$$
M_t = M_{t-1} \odot (1 - e_{k_t}) + e_{k_t} v_t^{\mathsf T}
$$

$$
r_t = e_{q_t}^{\mathsf T} M_t
$$

O controle temporal do MQAR é explícito neste microteste. Isso permite testar
armazenamento e decodificação sem confundi-los com um roteador discreto ainda
não aprendido. Uma versão geral futura precisará aprender quando escrever e
ler, sem depender das faixas de tokens do benchmark.

## Gate

- `ORACLE_SLOT` deve atingir pelo menos 99,9%;
- `FAST_WEIGHT` deve atingir pelo menos 95%;
- o oráculo sozinho nunca autoriza promoção;
- se o fast-weight falhar, a integração ASM-C2 continua bloqueada;
- `DENSE_SLOT_LONG` é diagnóstico e não participa do gate.

## Comando

```bash
./scripts/run_asm_c2_memory_learnability.sh
```

Resultados:

```text
runs/asm_c2_memory_learnability/results.json
runs/asm_c2_memory_learnability/report.md
```

`run_asm_c2_sparse_probe.sh` foi mantido como alias de compatibilidade para não
quebrar comandos já documentados.
