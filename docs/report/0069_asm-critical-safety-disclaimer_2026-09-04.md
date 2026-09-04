# Request Report

- Status: completed
- Date: 2026-09-04

## User request

Ler os artefatos ATTR-RTG/ATTR-RTG-RCMZ e inserir um disclaimer bem grande logo no começo do README.md e do README_ptbr.md do ASM, recomendando não escalar os modelos por dificuldade de governança e risco de segurança, e deixando claro que download e uso são responsabilidade de quem os realizar.

## Summary

Foi adicionado, como primeiro conteúdo dos dois READMEs, um aviso crítico e destacado para não escalar nem implantar modelos ASM.

O aviso se apoia nos resultados registrados do ATTR-RTG: somente `Transformer.RTG1-Z` passou; todos os gates RTG2 de governança e RTG3 de generalização shift/OOD falharam; o governor `G` reduziu outcomes inseguros em 0% no benchmark. O texto também registra as limitações do ATTR-RTG-RCMZ: evidência local, de administrador único e sem atestação independente, com todas as 40 células shift/OOD e os seis contrastes entre modelos inválidos. Essas evidências não provam que toda variante ASM seja insegura, mas não sustentam alegações de governança robusta, segurança para scaling ou implantação autônoma.

O disclaimer explica os riscos do estado recorrente persistente e da memória gravável, incluindo dificuldade de inspeção e contenção, prompt injection, envenenamento de estado ou memória, retenção indevida, influência entre sessões, saídas inseguras e falhas sob mudança de distribuição. Ele proíbe o uso recomendado em produção, agentes autônomos, operações privilegiadas e decisões críticas ou de alto impacto, e lista controles mínimos para pesquisa isolada sem apresentá-los como garantia de segurança.

Também ficou explícito, em inglês e português, que baixar, copiar, treinar, fazer fine-tuning, escalar, modificar, distribuir, integrar ou usar código, pesos, checkpoints, derivados ou saídas ocorre por decisão e risco exclusivos de quem o fizer. Essa pessoa ou entidade assume as responsabilidades por segurança, privacidade, conformidade, implantação, monitoramento, incidentes e danos. No limite permitido pela lei, autor, titular, mantenedores e contribuidores não oferecem garantia nem assumem responsabilidade. O aviso complementa os termos da licença e não promete isenção jurídica absoluta fora do que a lei permitir.

## Modified files

- [README.md](../../README.md)
- [README_ptbr.md](../../README_ptbr.md)
- [docs/report/0069_asm-critical-safety-disclaimer_2026-09-04.md](0069_asm-critical-safety-disclaimer_2026-09-04.md)

## Changes

- Adicionado aviso crítico de segurança como primeiro conteúdo do README.md.
- Adicionada versão equivalente em português como primeiro conteúdo do README_ptbr.md.
- Vinculados os avisos aos resultados registrados ATTR-RTG e às limitações oficiais do ATTR-RTG-RCMZ.
- Declarada a assunção de risco e responsabilidade por quem baixar, treinar, escalar, modificar, distribuir, integrar ou usar os modelos, no limite permitido pela lei e em conjunto com a licença.

## Validation

- Verificação automatizada de posição, conteúdo obrigatório e links dos dois READMEs — passou.
- Verificação dos destinos docs/benchmarks/attr_rtg/README.md e LICENSE — passou.
- git diff --check — passou.
