# Diagnóstico ASM-C2 e plano de memória esparsa

## Resultado do ASM-C2 suave

Nenhuma variante passou o controle MQAR curto de 80%:

| Variante | Acurácia em 5K | CE | Tempo |
|---|---:|---:|---:|
| Transformer | 99,78% | 0,021904 | 91 s |
| ASM-C | 33,64% | 2,004768 | 89 s |
| ASM-C2-16 | 32,89% | 2,002722 | 426 s |
| ASM-C2-32 | 32,89% | 2,012975 | 426 s |
| ASM-C2-64 | 33,20% | 2,008809 | 427 s |

O custo do ASM-C2 foi aproximadamente 4,8 vezes o ASM-C sem ganho mensurável.
O número de slots também não alterou materialmente o resultado.

## Diagnóstico dos slots

Os gates estavam ativos, portanto a falha não foi causada por memória fechada:

| Slots | Read gate | Write gate | Entropia de leitura | Máximo teórico |
|---:|---:|---:|---:|---:|
| 16 | 0,942 | 0,462 | 2,666 | 2,773 |
| 32 | 0,956 | 0,586 | 3,357 | 3,466 |
| 64 | 0,963 | 0,831 | 4,056 | 4,159 |

A entropia de leitura ficou entre 96% e 98% do máximo. Escrita e leitura foram
quase uniformes: as associações foram misturadas por todos os slots, em vez de
serem separadas em endereços recuperáveis.

## Correção implementada

A segunda parametrização do ASM-C2 acrescenta:

1. leitura top-k;
2. escrita top-1;
3. straight-through para preservar gradientes de seleção;
4. temperatura reduzida de 1,0 para 0,25;
5. chave de escrita formada com o token anterior;
6. valor formado com o token atual;
7. regularização opcional das entropias de leitura e escrita;
8. diagnósticos de gate, entropia, ocupação e norma recuperada.

O vínculo explícito token anterior→token atual corresponde à estrutura serial
dos pares chave→valor do MQAR. Ele não injeta targets futuros: no instante $t$,
somente o embedding causal de $t-1$ e o token atual estão disponíveis.

## Microteste isolado

Antes de repetir o modelo de 85M, a memória será testada num probe sem
recorrência auxiliar. O estado do token depende apenas do token atual; somente
o banco de slots pode transportar informação entre posições.

Variantes:

- `NO_MEMORY`;
- `DENSE`;
- `SPARSE_TOP1`;
- `SPARSE_TOP2`.

Gates:

- alguma memória esparsa deve alcançar pelo menos 95%;
- deve superar o controle sem memória em pelo menos 20 pontos percentuais.

Comando:

```bash
./scripts/run_asm_c2_sparse_probe.sh
```

Saída:

```text
runs/asm_c2_sparse_probe/results.json
```

Se o microteste falhar, o rerun completo ficará bloqueado. Nesse caso, a
política de binding/alocação deve ser corrigida novamente antes de envolver o
ASM-C. Se passar, o próximo runner comparará ASM-C2 Sparse 16/32/64 com ASM-C e
Transformer usando o protocolo curto pareado.
