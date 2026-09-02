# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Informar o próximo passo após o reseal e lock do ATTR-RTG-RCMZ-V1.

## Summary

O próximo passo é o usuário iniciar a execução oficial em outro terminal usando o supervisor e o novo lock V5. O agente não deve iniciar o treino.

```bash
cd /home/gnai-creator/dev/ai/ASM
PYTHONPATH=src:. .venv/bin/python -m attr_rtg_rcmz.official_supervisor \
  --output-dir artifacts/attr_rtg_rcmz/official \
  --lock-file locks/attr_rtg_rcmz_v1/LOCAL_PROTOCOL_LOCK.json \
  --lock-sha256 6ed6bd2b39460695a891c27c51cc417453e59eb10ad519660219f84a8cb8950e \
  --receipt artifacts/attr_rtg_rcmz/official/supervisor.json
```

Não fechar esse terminal. O supervisor transmite o output e impõe o deadline de 20 horas. O runner publica heartbeat, ETA, fase, seed, braço, update e VRAM. Em um segundo terminal, acompanhar:

```bash
cd /home/gnai-creator/dev/ai/ASM
tail -f artifacts/attr_rtg_rcmz/official/run.log
```

E consultar o estado estruturado com:

```bash
cat artifacts/attr_rtg_rcmz/official/status.json
```

Antes de interpretar resultados, esperar `status.json` e `supervisor.json` indicarem `COMPLETED`. Se ocorrer `CRASH` ou `TIMEOUT`, não reiniciar, reseedar ou selecionar checkpoint silenciosamente; preservar `TOMBSTONE.json`, `run.log`, `status.json` e o receipt para diagnóstico. O tempo de conclusão antes das 22h não está comprovado por uma run oficial anterior; o ETA do próprio runner será a evidência operacional após o início.

## Modified files

- [docs/report/0056_proximo-passo-execucao-oficial-rcmz_2026-09-02.md](0056_proximo-passo-execucao-oficial-rcmz_2026-09-02.md)

## Changes

- Documentado o comando oficial atualizado e o procedimento de acompanhamento sem iniciar o treino.

## Validation

- None recorded.
