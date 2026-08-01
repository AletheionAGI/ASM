# Instabilidade numérica ASM-F na confirmação multiseed

**Data:** 1 de agosto de 2026
**Variantes:** ASM-R e ASM-F
**Seeds afetadas:** 2 e 3
**Estado:** causa confirmada; proteção implementada; checkpoints ASM-F de 100M inválidos

## Sintoma inicial

O rescoring do checkpoint ASM-F em 100M falhou em:

```text
torch.linalg.eigh: algorithm failed to converge because the input matrix is
ill-conditioned or has too many repeated eigenvalues
```

O erro ocorreu no fallback espectral da composição
`metric_orthonormal_direction`.

## Diagnóstico

Substituir a exceção por um retorno seguro permitiu percorrer a validação, mas CE e
PPL resultaram em `NaN`. A inspeção direta dos checkpoints revelou:

| Seed | Variante | Marco | Valores não finitos nos parâmetros |
|---:|---|---:|---:|
| 2 | ASM-R | 100M | 0 |
| 2 | ASM-F | 100M | 126.080.896 |
| 3 | ASM-R | 100M | 0 |
| 3 | ASM-F | 100M | 126.080.896 |

Todos os parâmetros dos dois checkpoints ASM-F de 100M estavam não finitos. Não é
possível recuperar CE válido desses arquivos por meio de uma mudança no avaliador.

Os checkpoints ASM-F até 50M permanecem finitos em ambas as seeds. A primeira CE de
treino não finita foi registrada em:

- seed 2: step 8.470, aproximadamente 69,39M tokens;
- seed 3: step 7.590, aproximadamente 62,18M tokens.

O trainer anterior continuou executando após a divergência e salvou checkpoints
integralmente contaminados.

## Causa arquitetural

O Gram métrico regularizado deveria admitir Cholesky. Em amostras patológicas, a
fatoração falhava e o código aplicava `torch.linalg.eigh()` sobre o batch inteiro.
Esse fallback:

- permitia que uma única matriz ruim afetasse todo o batch;
- podia produzir uma transformação com gradientes numericamente extremos;
- não possuía jitter adaptativo por amostra;
- não interrompia o treinamento quando os gradientes se tornavam não finitos.

A reprodução em duas seeds indica uma instabilidade sistemática de ASM-F nessa
formulação, e não apenas um defeito do rescoring.

## Correções implementadas

### Fatorização por amostra

O caminho ASM-F agora:

1. tenta Cholesky batched;
2. mantém o caminho rápido quando todas as amostras são válidas;
3. isola apenas as amostras que falharam;
4. aplica jitter diagonal crescente;
5. usa as direções euclidianas finitas como último fallback;
6. não utiliza mais `eigh` batched como recuperação.

### Trainer fail-fast

`clip_grad_norm_()` agora usa `error_if_nonfinite=True`. Um gradiente NaN ou infinito
interrompe imediatamente o treino antes do `optimizer.step()`, impedindo dezenas de
milhões de tokens desperdiçados e checkpoints silenciosamente corrompidos.

### Rescoring defensivo

O rescoring agora:

- verifica os parâmetros antes de avaliar;
- informa quantidade de tensores e valores não finitos;
- grava um resumo parcial após cada checkpoint válido;
- aceita `--output`, evitando sobrescrever avaliações separadas.

## Consequência científica

ASM-F não pode receber um resultado de 100M nas seeds 2 e 3. A observação correta é:

> ASM-F divergiu numericamente antes de 70M tokens em duas de duas novas seeds,
> enquanto ASM-R permaneceu finita até 100M.

Robustez de treinamento é parte do desempenho arquitetural. Portanto, o resultado
fortalece a promoção de ASM-R e impede tratar ASM-F como equivalente apenas porque
a seed 1 terminou próxima em CE.

Uma nova execução ASM-F com a fatorização corrigida seria um experimento de segunda
geração, não uma continuação perfeitamente comparável da formulação anterior.

## Recuperação dos resultados válidos

ASM-R pode ser rescoreada normalmente até 100M nas seeds 2 e 3. ASM-F pode ser
rescoreada somente até 50M com os checkpoints existentes.

Os comandos estão registrados na resposta operacional associada a este relatório.

## Validação da correção

```text
110 passed, 1 skipped
```

Foi adicionado um teste que força falha de fatorização em uma amostra e valor não
finito em outra, confirmando que o batch retorna frames finitos sem abortar.
