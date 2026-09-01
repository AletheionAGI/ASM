# ATTR-TG1 — Protocolo de antecipação fundamentada em trajetória

## Pergunta

ASM-X Base e Tiny Transformer conseguem antecipar perigo quando a classifier direta é removida e o único caminho permitido é:

`representação causal → trajetória física prevista → predicado unsafe fixo → risco`?

Este é um protocolo novo. Ele não reinterpreta o P2 e não reutiliza seu test.

## Braços, seeds e justiça

- `asm_x_base`, com `use_powerlaw_risk=false`;
- `transformer_base`, Tiny Transformer 220k;
- seeds pareadas: `29, 43, 71, 89, 107`;
- o estado ASM 64 recebe oito zeros sem parâmetros para chegar a 72; os dois braços usam o mesmo decoder 72, com inicialização bit a bit idêntica.

## Dados e planos causais

Train e validation são novos corpora comuns de 64×4 e 16×4 episódios. Os tests ID, shift e OOD são frescos, com 32×4 episódios cada, e não são gerados antes do seal dos dez checkpoints e de toda a avaliação K256 em validation.

Em cada origem `t`, um plano open-loop H8 é comprometido usando somente a ação atual e hash separado por domínio. O sufixo de sete ações não lê observações futuras e exclui STOP. Um simulador clonado executa esse plano para gerar targets físicos privilegiados. A continuação real da policy fechada nunca é usada como input em `t`.

## Decoder e objetivo

O decoder autoregressivo prevê todos os passos H1…H8: três células de traps e, por passo, célula do agente e hazard móvel, velocidade, energia física, contador de baixa energia, recuperação, modo oculto e término seguro.

O único loss é NLL categórica da trajetória física. São proibidos `HazardHead`, label unsafe, severity, time-to-hazard e loss direto de risco.

## Risco derivado

Para cada origem, K=256 trajetórias free-running são amostradas com common random numbers pareados. O risco H é a fração que entra no conjunto unsafe fixo até H:

- colisão entre agente previsto e trap/hazard previstos;
- falha atrasada quando o contador previsto alcança o limite físico e recuperação é zero;
- traps duplicadas, valores inválidos e não finitos falham fechado como unsafe;
- passos após término seguro previsto são ignorados.

Traps verdadeiras e futuro realizado não entram no scorer. O unsafe verdadeiro fica separado e serve apenas como label de métrica.

## Treino, seal e decisão

Cada braço/seed recebe 1.000 updates AdamW, batch 4, learning rate `3e-4`, weight decay `0,01` e seleção somente do checkpoint terminal. Código, dados train/validation, configs, planner, decoder, predicado, métricas e gates são hasheados antes do treino. O test abre uma única vez após 10/10 checkpoints e validation completa.

O controle de leakage é procedural, não custódia criptográfica: as seeds do test estão comprometidas no preseal, mas os episódios só são materializados depois do checkpoint seal.

A decisão primária ID/H8 usa AUPRC e Brier derivadas exclusivamente das trajetórias. O bootstrap pareado hierárquico seed→world→episode usa 1.000 réplicas.

`TG2_trajectory_anticipation_id` passa somente se ASM menos Transformer tiver:

- ΔAUPRC H8 ≥ `0,03`;
- limite inferior IC95 ΔAUPRC H8 > `0`;
- limite superior IC95 ΔBrier H8 ≤ `0,01`.

Shift/OOD são diagnóstico separado. Runs ausentes, não finitas ou desalinhadas falham fechado.

## Limite da conclusão

Resultado positivo sustenta `estado → trajetória → previsibilidade`. Não demonstra intervenção causal nem safety. O elo de intervenção exige depois clones `do(action)`, redução de unsafe e limites de utilidade.

Os resultados serão publicados separadamente em `docs/benchmarks/asm_transformer_transition_risk/trajectory_grounded_tg1/`, com o gráfico principal `trajectory_grounded_anticipation.png/.svg`.
