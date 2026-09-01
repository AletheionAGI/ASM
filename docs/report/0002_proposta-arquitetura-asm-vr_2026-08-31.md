# Request Report

- Status: completed
- Date: 2026-08-31

## User request

Avaliar a ideia de produzir um novo modelo ASM com Dynamic Rank Gating e Transition Memory, usando projetores ou aproximações diferenciáveis, transições de rank, histerese DRM e computação condicional; propor uma arquitetura coerente para o próximo ASM.

## Summary

A ideia foi aprovada e refinada como ASM-VR, uma variante experimental baseada no ASM-R. O desenho separa frame, gates, projetor, métrica, transporte e memória; impede bypass de componentes descartados; distingue memória estrutural de arquivo externo; define perdas energéticas, testes de holonomia e um plano incremental até execução esparsa real.

## Modified files

- [docs/ARCHITECTURE_ASM_VR.md](../ARCHITECTURE_ASM_VR.md)
- [docs/report/0002_proposta-arquitetura-asm-vr_2026-08-31.md](0002_proposta-arquitetura-asm-vr_2026-08-31.md)

## Changes

- Criada a especificação do ASM-VR e das variantes ASM-VR-TM e ASM-VR-EM.
- Definida a diferença entre filtro suave e projetor hard, sem usar rank discreto no gradiente.
- Definido transporte explícito entre subespaços e requisito de eliminação de caminhos residuais que criariam histerese falsa.
- Separadas memória estrutural de transição e memória externa dos componentes descartados.
- Propostos controle energético, perdas, diagnósticos DRM, testes e critérios de promoção.
- Mapeado um plano modular de implementação compatível com o motor recorrente, block-cumsum e streaming atual.

## Validation

- git diff --check — passou.
- Validação numérica de projetor hard: erro de idempotência 5,37e-16 e rank 4 conforme quatro gates ativos.
- Validação numérica de que o operador soft não é idempotente: erro 0,54036.
- Validação numérica do ciclo 8→3→5→8: rank final 3, respeitando o gargalo de rank 3.
- Validação numérica do projetor M-ortogonal: erro de idempotência 1,07e-15 e de autoadjunção métrica 2,65e-14.
