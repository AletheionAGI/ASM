# Request Report

- Status: completed
- Date: 2026-09-02

## User request

Explicar qual é o próximo passo após aprovar o desenho V10 e a readiness local, sem autorizar o freeze global.

## Summary

## Próximo passo imediato

Abrir a **cerimônia de readiness externa**, sem modificar V10. O primeiro gate é nomear os responsáveis e fixar o ambiente final e os trust roots reais.

O usuário deve fornecer ou designar, por canal seguro, somente identificadores públicos e acessos controlados — nunca private keys ou shares no chat:

1. **Ambiente final/clean-room:** operador independente, host/GPU final, pipeline da imagem content-addressed e responsável pelo reproducible build/supply-chain closure.
2. **Roster de custódia:** N principals ordenados, threshold M, organização/identidade, Ed25519 public keys e trust roots independentes.
3. **HSM:** modelo/provedor, attestation root, authorization policy e operador da cerimônia private all-N/one-attempt.
4. **Beacon e anterioridade:** provider, round futuro, deadline, ledger append-only e TSA independente.
5. **Watchdog/relay:** administrador independente, endpoint fixo, public key, trusted timer e janela real de 24 horas.
6. **WORM:** serviço/tenant, retention/object-lock policy, inclusion proof interface, trusted clock e key-custody owner.
7. **Continuous attestation:** verifier, measured-boot policy e enforcement T0–T5 de read-set/syscall/device/ioctl/no-egress.

## Ordem de execução

1. Preencher e assinar o roster/ownership dos sete grupos.
2. Fazer o build content-addressed da imagem final e registrar hashes/trust roots.
3. Executar o external preflight fail-closed. Qualquer campo ausente retorna `BLOCK`.
4. Fazer os testes de HSM/beacon/ledger/watchdog/WORM/attestation e a captura real de 24 horas.
5. Arquivar receipts em um novo evidence manifest append-only, sem alterar V10/Evidence V7.
6. Submeter o snapshot e ambiente exatos aos quatro reviewers.
7. Se os quatro retornarem `GLOBAL READY TO FREEZE: YES`, solicitar uma nova autorização explícita do usuário para criar o freeze manifest/hash.

A ação prática imediata é o usuário nomear o **coordenador da cerimônia**, o **operador do ambiente final** e os **N principals/M threshold**, além de indicar os serviços externos selecionados. Credenciais privadas devem permanecer nos HSMs/canais dos respectivos operadores.

## Modified files

- [docs/report/0047_proximo-passo-readiness-externa-rcm_2026-09-02.md](0047_proximo-passo-readiness-externa-rcm_2026-09-02.md)

## Changes

- Documentada a ordem operacional para fechar os sete grupos externos e chegar a uma futura decisão de freeze.

## Validation

- Decisão V10 aplicada: preservar snapshot/evidência e não criar freeze antes da readiness global
