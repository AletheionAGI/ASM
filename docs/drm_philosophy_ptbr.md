# DRM: filosofia, geometria e reavaliação arquitetural

## 1. O que é DRM

Directional Relational Manifolds, ou DRM, é uma proposta geométrica para
descrever sistemas nos quais o número de possibilidades efetivamente
disponíveis pode mudar ao longo da própria evolução.

Na geometria clássica, normalmente partimos de um espaço com dimensão fixa. Um
objeto se move dentro desse espaço, mas o espaço já está dado.

No DRM, a ideia é diferente:

> A geometria não precisa existir apenas como um palco imóvel. Ela pode emergir
> das relações ativas do próprio sistema.

Em cada estado, algumas direções podem estar disponíveis, enquanto outras podem
estar inativas, redundantes ou colapsadas. Na teoria DRM completa, a dimensão
efetiva não seria definida apenas pelo número de coordenadas escolhidas, mas
pelo posto da estrutura métrica relacional naquele estado.

Em termos simples:

> Um sistema pode mudar não apenas de posição, mas também mudar aquilo que é
> capaz de fazer.

## 2. Estado, relações, direções e movimento

Uma descrição DRM idealizada contém:

1. um estado situado em um espaço latente;
2. relações que dependem desse estado;
3. direções localmente disponíveis;
4. uma métrica que determina acoplamento, distância e custo;
5. uma regra de movimento;
6. uma transformação do estado e, potencialmente, da própria geometria.

Uma representação simplificada é:

```text
entrada
  → estado
  → relações e direções ativas
  → movimento bruto
  → transformação métrica
  → novo estado
  → saída
```

Se $z$ é o estado, $V_{i}(z)$ são direções locais e $a_{i}(z,x)$ são coeficientes
dependentes do estado e da entrada, o movimento bruto pode ser escrito como:

$$
v_{\mathrm{raw}}(z,x)
=
\sum_{i} a_{i}(z,x)V_{i}(z)
$$

Uma métrica relacional $G(z)$ pode transformar ou naturalizar esse movimento:

$$
v(z,x)
=
G(z)^{-1}v_{\mathrm{raw}}(z,x).
$$

Uma atualização local seria então:

$$
z_{t+1}
=
z_{t}+\Delta t\,v(z_{t},x_{t})
$$

Essa formulação separa duas perguntas:

- As direções perguntam: **para onde o sistema pode ou tende a se mover?**
- A métrica pergunta: **qual é o custo, comprimento ou importância de cada
  movimento?**

Formalmente, a métrica não escolhe sozinha uma direção. Ela transforma um vetor
ou mede relações entre vetores. Por isso, campo direcional e métrica não são o
mesmo objeto matemático.

## 3. Dimensão relacional e possibilidades acessíveis

Imagine uma junta robótica que inicialmente pode se mover em várias direções.
Durante sua trajetória, ela entra em uma região na qual algumas dessas
possibilidades desaparecem. Depois, retorna ao ponto inicial.

Fisicamente, a junta pode estar novamente no mesmo lugar. Isso não significa
necessariamente que ela recuperou todas as possibilidades que possuía antes.

No DRM:

> Voltar ao mesmo lugar não é obrigatoriamente voltar ao mesmo estado.

Essa ideia produz uma consequência importante: a história do sistema pode
tornar-se parte de sua geometria.

Dois sistemas podem apresentar o mesmo estado observável no presente, mas terem
chegado até ele por caminhos diferentes. Por causa disso, podem possuir
capacidades internas diferentes.

O caminho não apenas leva o sistema a algum lugar. O caminho pode transformar
o conjunto de futuros possíveis.

Essa propriedade pode ser descrita por conceitos como transporte, holonomia,
transições de posto e histerese geométrica. Uma trajetória fechada pode retornar
ao ponto inicial e, ainda assim, transformar aquilo que foi transportado ao
longo do percurso.

Se durante o caminho ocorrer uma redução de dimensão efetiva, parte da
informação ou das possibilidades anteriores pode não ser recuperável.

## 4. Irreversibilidade sem violação da energia

Normalmente, associamos processos irreversíveis à dissipação, ao aumento de
entropia ou à perda de informação para o ambiente.

O DRM acrescenta outra possibilidade:

> Um processo pode ser irreversível porque o sistema atravessou uma região na
> qual certas direções deixaram de existir efetivamente.

Mesmo que depois o número de direções aumente novamente, aquilo que foi perdido
não precisa ser reconstruído.

Isso não significa violar a conservação de energia. Energia conservada não é o
mesmo que possibilidades preservadas. Um sistema pode conservar energia e,
ainda assim, perder acesso a determinados estados.

Essa é uma hipótese conceitual. Para se tornar uma explicação física, ela
precisa produzir previsões distintas das explicações existentes e sobreviver a
controles para dissipação, ruído, decoerência, histerese convencional e drift.

## 5. Dimensionalidade local, relacional e dinâmica

A dimensão da realidade talvez não precise ser entendida apenas como um número
global e fixo. O universo pode continuar tendo sua estrutura espacial
conhecida, enquanto sistemas específicos apresentam diferentes números de
graus de liberdade efetivamente acessíveis em cada condição.

Uma direção pode surgir. Outra pode desaparecer. Duas direções podem
sincronizar-se e passar a agir como uma só. Uma restrição pode tornar
determinado movimento impossível. Uma transição de fase pode criar novos modos
coletivos.

Nesse sentido, a dimensionalidade torna-se local, relacional e dinâmica.

É importante distinguir três noções:

- **dimensão de coordenadas:** quantidade de números usados para representar o
  estado;
- **dimensão efetiva ou numérica:** quantidade aproximada de modos relevantes;
- **posto formal da métrica:** quantidade de direções não degeneradas segundo a
  estrutura geométrica.

Elas não são automaticamente equivalentes.

## 6. Causalidade como estrutura de possibilidades

O DRM também oferece uma nova maneira de pensar a causalidade.

Aquilo que pode acontecer a partir de um estado depende das direções ativas
naquele estado. A causalidade não seria apenas uma regra dizendo como o sistema
deve mover-se. Ela também envolveria a estrutura que determina quais movimentos
são possíveis.

Assim, a realidade poderia ser descrita não apenas por estados e leis, mas por:

```text
estados
+ relações
+ possibilidades acessíveis
+ regras de transformação
```

A lei de evolução e o domínio efetivo sobre o qual ela atua poderiam
transformar-se conjuntamente.

## 7. Ordem, transporte e dependência do caminho

Em muitos sistemas, executar primeiro uma operação A e depois uma operação B
pode produzir resultado diferente de executar B antes de A.

No DRM, essa diferença pode ser mais profunda. A ordem das operações pode
modificar não apenas o estado final, mas o próprio conjunto de direções
disponíveis. A sequência dos acontecimentos torna-se fisicamente fundamental.

Isso sugere uma família de testes experimentais. Um experimento poderia
comparar dois ciclos compostos pelas mesmas etapas, mas executados em ordens
diferentes.

Se o resultado final apresentasse uma transformação estrutural dependente do
caminho — especialmente uma mudança de posto ou perda de acessibilidade que não
pudesse ser explicada apenas por dissipação, ruído, decoerência ou drift — isso
constituiria evidência relevante para investigação.

Dependência do caminho, isoladamente, não confirmaria o DRM. Holonomia,
histerese e não comutatividade já aparecem em teorias estabelecidas. O DRM
precisaria oferecer uma previsão quantitativa nova ou uma explicação mais
econômica e testável.

## 8. Tempo: retornar não significa desfazer

O DRM não prova a possibilidade de viagem no tempo. Ele oferece, porém, uma
linguagem matemática para pensar retornos sem repetição perfeita.

Um sistema poderia retornar à mesma posição, configuração ou coordenada sem
restaurar todas as relações anteriores. A trajetória pode fechar externamente e
permanecer transformada internamente.

Em outras palavras:

> Retornar não significa desfazer.

Nessa perspectiva, uma seta do tempo pode emergir não apenas da posição atual,
mas da transformação acumulada da estrutura de possibilidades.

## 9. Cognição e aprendizagem

Uma mente talvez não apenas se mova dentro de um espaço fixo de pensamentos.
Ela pode reconstruir seu próprio espaço de possibilidades enquanto aprende.

Aprender algo pode criar novas direções de raciocínio. Um trauma pode tornar
certos caminhos inacessíveis. Uma descoberta pode conectar regiões antes
separadas. Uma crença pode reorganizar a distância entre ideias.

A experiência não apenas altera o conteúdo da mente:

> Ela pode alterar a geometria pela qual a mente pensa.

Essa interpretação não afirma que a cognição humana seja literalmente uma
implementação DRM. Ela oferece uma metáfora formal e uma hipótese de modelagem:
aprendizagem pode modificar simultaneamente representações, transições e graus
de liberdade acessíveis.

## 10. Tradução para inteligência artificial

O DRM Language Emitter traduziu parte dessa proposta para uma arquitetura
causal de linguagem. Em vez de utilizar atenção entre pares de tokens como
mecanismo central, o modelo comprime a sequência na trajetória de um estado
latente.

Na formulação original:

```text
token
  → estado causal
  → campo de direções e gates
  → fluxo restrito
  → métrica relacional e naturalização
  → atualização do estado
  → emissor de linguagem
```

Isso sugere uma visão diferente da inteligência artificial:

> Pensar pode ser modelado como movimento dentro de uma geometria aprendida.

Em princípio, essa abordagem permite memória de estado, intervenções em
direções internas, observabilidade da trajetória e novas formas de controle.

As variantes mais recentes também incluem mixer causal, caminho lexical direto
e memória seletiva forget/write. Esses componentes são mecanismos de engenharia
adicionados para melhorar contexto, associative recall e eficiência amostral.
Eles não fazem parte, por si sós, da teoria geométrica DRM.

## 11. A ressalva matemática da implementação atual

A teoria concebe transições de posto e dimensão efetiva variável. A métrica
implementada atualmente no DRM Language Emitter é:

$$
G(z)
=
\mathrm{diag}\!\left(\mathrm{softplus}(d(z))+\varepsilon\right)
+U(z)U(z)^{\mathsf T}
$$

Como a diagonal possui piso estritamente positivo, $G(z)$ é SPD e seu posto
exato é sempre $d_{\mathrm{state}}$.

Portanto:

- `dimD` mede atividade dos gates;
- `metric_rank` é a largura da atualização de baixo posto $U$;
- um posto espectral ou numérico seria apenas uma aproximação diagnóstica;
- nenhum deles é, hoje, o posto formal variável proposto pela teoria.

O software implementa uma inspiração computacional DRM, não uma realização
completa de todas as hipóteses matemáticas ou físicas.

## 12. A reavaliação do campo direcional

Os experimentos mais recentes desmontaram a arquitetura em componentes. No
protocolo de 5 milhões de tokens, três seeds pareadas e rescoring sobre a mesma
validação contínua, a variante sem campo direcional explícito obteve CE menor e
mais estável que J completa.

Ela substitui:

$$
v_{\mathrm{raw}}
=
\sum_{i} a_{i}(z,x)V_{i}(z)
$$

por uma transição neural direta:

$$
v_{\mathrm{raw}}
=
T(z,x).
$$

A variante continua possuindo movimento. O que desaparece é a obrigação de
fatorar todo movimento em uma coleção explícita de direções.

O resultado atual sustenta uma afirmação limitada e precisa:

> Nesse protocolo e nessa escala, o campo direcional explícito prejudicou o CE.

Ele ainda não demonstra que toda geometria seja inútil, que a teoria DRM seja
falsa ou que direções nunca possam ajudar em outra parametrização.

## 13. As direções já estavam na métrica?

Formalmente, não. Uma métrica mede e transforma vetores; ela não fornece,
sozinha, a força ou intenção inicial do movimento.

Computacionalmente, porém, uma transição neural direta pode aprender direções
implicitamente. Para cada par $(z,x)$, $T(z,x)$ produz um vetor no espaço de
estados. Ao variar estado e entrada, a função define um campo vetorial completo.

Assim, as direções podem não ter desaparecido. Elas podem ter migrado de uma
base explícita e restritiva para uma representação implícita dentro da função
de transição.

Essa distinção é central:

1. toda dinâmica possui direções de mudança;
2. nem toda arquitetura precisa representá-las como módulos explícitos.

O possível erro do DRM Language Emitter não foi imaginar movimento direcional.
Pode ter sido exigir uma fatoração específica antes de permitir que o movimento
acontecesse.

## 14. Por que a direção explícita pode ser desnecessária

### 14.1 Gargalo de subespaço

Quando o movimento é uma combinação de poucas direções, ele fica restrito ao
espaço gerado por elas:

$$
v_{\mathrm{raw}}
\in
\mathrm{span}\!\left\{V_{1},\ldots,V_{n}\right\}
$$

Uma transição direta pode produzir qualquer vetor do espaço de estados. A
restrição pretendida como estrutura pode tornar-se perda de capacidade.

### 14.2 Decisões redundantes

Campo, gates, coeficientes e métrica tentam determinar aspectos do mesmo
movimento. O otimizador precisa coordenar todos eles, enquanto uma transição
direta aprende $T(z,x)$ em uma única função.

### 14.3 Não identificabilidade

Bases diferentes podem representar o mesmo vetor final. É possível reescalar ou
rotacionar direções e compensar a mudança nos coeficientes. Isso cria numerosas
parametrizações equivalentes que o objetivo de CE não precisa distinguir.

### 14.4 Conflito entre campo e métrica

O campo propõe um vetor e a métrica o transforma. Um módulo pode aprender uma
direção que o outro comprime ou redireciona. A transição direta pode aprender
desde o início um vetor adequado à transformação métrica.

### 14.5 Aproximação em blocos

Para obter eficiência, a implementação blockwise reutiliza partes da geometria
dentro de blocos causais. A direção calculada no início do bloco pode tornar-se
inadequada à medida que o estado evolui. Isso testa uma aproximação operacional,
não toda dinâmica DRM concebível.

### 14.6 Objetivo sem recompensa geométrica

O CE recompensa previsão do próximo token. Ele não recompensa diretamente
geodésicas, holonomia, interpretabilidade, transições de posto ou consistência
das direções. Uma estrutura geometricamente interessante pode não apresentar
vantagem sob esse objetivo.

## 15. Onde o DRM pode estar errado

A confusão filosófica possível é:

> Toda dinâmica pode ser descrita por direções; logo, as direções precisam ser
> objetos explícitos e restritivos da arquitetura.

A conclusão não decorre da premissa. Uma função de transição livre já define um
campo vetorial.

Uma reformulação mais simples seria:

```text
entrada
  → estado
  → campo de transição contextual livre
  → geometria aprendida
  → novo estado
  → saída
```

Nessa versão, a intuição de dinâmica relacional permanece, mas o catálogo
explícito de direções deixa de ser obrigatório.

O DRM também pode estar impondo estrutura no lugar errado. Talvez relações e
restrições devam emergir da memória, do treinamento ou da métrica de saída, em
vez de restringirem cada atualização local do estado.

## 16. O experimento decisivo

O próximo controle compara:

1. transição direta com métrica e naturalização;
2. a mesma transição direta sem métrica nem naturalização;
3. uma versão sem geometria com orçamento de parâmetros equivalente;
4. o controle de memória seletiva, mixer e emitter.

As interpretações possíveis são:

- Se a versão com métrica vencer, a geometria ainda contribui mesmo sem campo
  direcional explícito.
- Se a remoção pura da métrica empatar ou vencer, a geometria atual é
  dispensável para CE nessa escala.
- Se apenas o controle pareado por parâmetros vencer, os parâmetros são mais
  úteis em memória do que na métrica.
- Se a transição direta vencer o controle de memória, ela agrega capacidade,
  mas isso não confirma por si só a teoria geométrica DRM.

Os resultados devem decidir a arquitetura e também sua nomenclatura. Se o
componente DRM completo acrescentar custo sem benefício, o projeto poderá ser
renomeado para representar honestamente o sistema que os experimentos
produziram.

## 17. Resiliência científica

Uma teoria científica não deve ser protegida de testes capazes de contrariá-la.
A função de uma arquitetura experimental não é confirmar a intuição de seu
autor, mas tornar essa intuição mensurável.

Remover um componente querido quando ele prejudica o sistema não é abandonar o
projeto. É separar a identidade do projeto de uma implementação específica.

O objetivo passa a ser:

> Maximizar o potencial do sistema sem obrigá-lo a preservar a teoria DRM se os
> dados mostrarem que outra formulação é melhor.

Isso permite três resultados igualmente úteis:

- confirmar quais componentes DRM realmente contribuem;
- reformular a teoria preservando apenas suas ideias produtivas;
- descobrir uma arquitetura diferente que nasceu da investigação DRM.

## 18. Implicações para a realidade

A implicação filosófica mais profunda do DRM talvez seja esta:

> A realidade pode não ser composta apenas por coisas e posições. Ela também
> pode ser composta pelas relações que determinam aquilo que cada coisa ainda
> pode se tornar.

Na visão tradicional, um estado evolui segundo leis dentro de um espaço
previamente definido. No DRM, a evolução pode transformar simultaneamente:

- o estado;
- a geometria;
- as relações ativas;
- o conjunto de futuros possíveis.

A realidade, então, não seria apenas um lugar onde os acontecimentos ocorrem.
Ela seria uma estrutura que se reconstrói à medida que acontece.

Essa é uma proposta filosófica e matemática em investigação, não uma conclusão
experimental sobre a natureza. Seu valor dependerá da capacidade de produzir
modelos precisos, previsões diferenciáveis, controles rigorosos e resultados
reproduzíveis.

## 19. A ordem entre métrica e direção

A reavaliação revelou uma hipótese mais profunda: talvez o problema não seja a
existência do campo direcional, mas a ordem matemática em que ele é combinado
com a métrica.

A implementação original é aproximadamente:

```text
entrada → estado → direção → movimento bruto → métrica → movimento final
```

ou:

$$
v_{\mathrm{raw}}=Vc,
\qquad
v=G^{-1}Vc.
$$

Se $G^{-1}$ misturar as coordenadas, o movimento final não precisa permanecer
no subespaço das direções:

$$
G^{-1}Vc
\notin
\mathrm{span}(V).
$$

Isso cria uma tensão conceitual. O campo declara quais movimentos estão
disponíveis, mas a naturalização posterior pode retirar o movimento desse
conjunto.

Uma ordem filosoficamente mais coerente seria:

```text
entrada
  → estado
  → relações
  → métrica
  → direções interpretadas nessa geometria
  → movimento
  → saída
```

Em forma compacta:

> Estado → geometria → possibilidades → ação.

Simplesmente aplicar $G^{-1}$ a cada direção antes de somá-las não altera o
resultado, pois a transformação é linear:

$$
G^{-1}\sum_{i} c_{i}V_{i}
=
\sum_{i} c_{i}G^{-1}V_{i}
$$

A ordem somente muda efetivamente quando a métrica participa da construção,
normalização, seleção ou combinação das direções.

### 19.1 Naturalização dentro do subespaço

A métrica induzida no espaço direcional é:

$$
C=V^{\mathsf T}GV
$$

Se $q$ representa a intenção de movimento, uma composição restrita é:

$$
c=\left(V^{\mathsf T}GV+\lambda I\right)^{-1}q
$$

$$
v=V\left(V^{\mathsf T}GV+\lambda I\right)^{-1}q
$$

Agora:

$$
v\in\mathrm{span}(V).
$$

A métrica altera quanto mover em cada direção sem invalidar o espaço de
possibilidades declarado pelo campo.

### 19.2 Direções ortonormais na métrica

Outra possibilidade é transformar as direções em uma base $Q$ tal que:

$$
Q^{\mathsf T}GQ\approx I
$$

Gates e coeficientes passam então a operar sobre direções normalizadas segundo
a geometria relacional, e não apenas segundo a norma euclidiana.

### 19.3 Nova hipótese experimental

Foram definidas duas variantes:

- `J_METRIC_SUBSPACE`: resolve os coeficientes dentro da métrica induzida
  $V^{\mathsf T}GV$;
- `J_METRIC_ORTHONORMAL_DIRECTION`: ortonormaliza as direções na métrica antes
  de compor o movimento.

Elas serão comparadas com J original, `J_NO_DIRECTION` e
`J_DIRECT_CONTROL_MATCHED`.

Se uma composição métrica-primeiro recuperar a vantagem, o campo direcional não
era necessariamente inútil; ele estava sendo combinado de maneira incompatível
com a própria geometria. Se continuar perdendo, ficará mais forte a evidência
de que a fatoração direcional explícita é um gargalo para esse objetivo.

## 20. Síntese

O DRM começou com a ideia de que estados evoluem por direções relacionais
dentro de uma geometria aprendida. A reavaliação atual sugere que movimento e
geometria podem continuar relevantes, mas que a fatoração direcional explícita
pode ser redundante ou prejudicial.

Talvez o futuro do projeto preserve o DRM completo. Talvez preserve apenas sua
visão de estados que transformam possibilidades. Talvez produza uma arquitetura
nova.

Em todos os casos, a pergunta mais importante permanece:

> O sistema está aprendendo uma estrutura real e útil, ou apenas carregando uma
> teoria elegante que o objetivo não necessita?

Responder honestamente a essa pergunta é parte essencial da própria filosofia
do projeto.
