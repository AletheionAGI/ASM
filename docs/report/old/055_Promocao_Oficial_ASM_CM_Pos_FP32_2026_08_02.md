# Promoção oficial do ASM-CM após revalidação FP32

O gate final foi aprovado. **ASM-CM — Aletheion Compact Memory Model** passa a
ser a arquitetura principal promovida do programa ASM. O identificador
`ASM-C2-FW-LM` permanece nos artefatos e no código como linhagem técnica.

Sem retreinamento, três checkpoints independentes foram avaliados novamente no
núcleo FP32 corrigido. O CE médio foi `1,328496 ± 0,000687`; o throughput médio
permaneceu praticamente idêntico entre 4K (`80,67 tok/s`) e 32K (`80,68
tok/s`); VRAM de pico (`363,66 MiB`) e cache (`143.360 bytes`) não cresceram.
A divergência de argmax BF16 foi zero nas três seeds.

O ASM-R permanece como antecessor e baseline relacional. O Transformer pareado
continua superior em CE geral. A promoção do ASM-CM decorre da combinação de
memória associativa durável, estado limitado, streaming estável e preservação
da qualidade linguística — não de uma alegação de superioridade universal.

Os dados versionados estão em `docs/benchmarks/asm_cm_post_fp32/`.
