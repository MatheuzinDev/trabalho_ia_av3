# Plano de Implementacao - Problema 1

## Objetivo

Resolver o primeiro problema do trabalho AV3 de IA usando algoritmos de busca/otimizacao meta-heuristica em dominio continuo.

Funcao objetivo:

```text
f(x1, x2) = x1^2 + x2^2
```

Dominio:

```text
x1, x2 in [-100, 100]
```

Tipo de problema:

```text
Minimizacao
```

Otimo conhecido:

```text
x* = (0, 0)
f(x*) = 0
```

## Algoritmos a implementar

Implementar manualmente os tres algoritmos exigidos no enunciado:

- Hill Climbing
- Local Random Search - LRS
- Global Random Search - GRS

Nao usar bibliotecas com implementacoes prontas desses metodos.

## Regras gerais do experimento

- Executar `100` rodadas por algoritmo.
- Cada rodada deve ter no maximo `1000` iteracoes.
- Armazenar a melhor solucao final de cada rodada.
- Aplicar restricao de caixa para todos os candidatos gerados.
- Rejeitar ou corrigir candidatos fora do dominio `[-100, 100]`.
- Implementar parada antecipada quando nao houver melhora por `t` iteracoes.
- Considerar uma tolerancia numerica para identificar solucao aceitavel proxima do otimo.
- Gerar uma tabela final com a moda das solucoes encontradas por algoritmo.

## Representacao da solucao

Cada candidato sera representado por um vetor com duas componentes:

```text
x = [x1, x2]
```

O valor de avaliacao sera calculado por:

```text
f(x) = x1^2 + x2^2
```

Como o problema e de minimizacao, uma solucao e melhor quando seu valor de `f(x)` e menor.

## Hill Climbing

Regras especificas:

- Comecar no limite inferior do dominio:

```text
xbest = [-100, -100]
```

- Gerar candidatos na vizinhanca de `xbest` respeitando:

```text
|xbest - y| <= epsilon
```

- Usar inicialmente `epsilon = 0.1`, conforme indicado no enunciado.
- Testar valores menores de `epsilon` para identificar o menor valor que encontra uma solucao aceitavel.
- Aceitar o candidato apenas se ele melhorar a funcao objetivo.
- Parar por maximo de iteracoes, por ausencia de melhora ou ao atingir solucao aceitavel.

Pontos de atencao:

- Com `1000` iteracoes e partida em `[-100, -100]`, `epsilon = 0.1` pode nao ser suficiente para chegar perto de `(0, 0)` se cada iteracao andar pouco.
- Sera necessario analisar empiricamente uma abertura adequada ou uma estrategia de amostragem da vizinhanca que permita progresso consistente.

## Local Random Search - LRS

Regras especificas:

- Gerar `xbest` inicial com distribuicao uniforme no dominio.
- Gerar candidatos ao redor de `xbest` usando perturbacao aleatoria local.
- Usar desvio-padrao `0 < sigma < 1`.
- Testar valores de `sigma` para identificar o menor valor que encontra uma solucao aceitavel.
- Aceitar o candidato apenas se ele melhorar a funcao objetivo.
- Parar por maximo de iteracoes, por ausencia de melhora ou ao atingir solucao aceitavel.

Exemplo de geracao de candidato:

```text
y = xbest + Normal(0, sigma)
```

## Global Random Search - GRS

Regras especificas:

- Gerar cada novo candidato com distribuicao uniforme em todo o dominio.
- Comparar o candidato com a melhor solucao atual.
- Atualizar `xbest` se o candidato melhorar a funcao objetivo.
- Parar por maximo de iteracoes, por ausencia de melhora ou ao atingir solucao aceitavel.

Observacao:

- O enunciado menciona `sigma` tambem para GRS, mas GRS global normalmente nao usa desvio-padrao, pois gera candidatos uniformes no espaco inteiro.

## Parametros iniciais sugeridos

```text
R = 100
max_iter = 1000
patience = 100
tolerancia_otimo = 1e-3
epsilon_inicial = 0.1
sigma_inicial = 0.5
```

Valores a testar:

```text
epsilon_values = [0.01, 0.05, 0.1, 0.5, 1.0]
sigma_values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75]
```

Esses valores podem ser ajustados apos a primeira execucao experimental.

## Arquivos a criar

Estrutura proposta para o problema 1:

```text
.
├── main.py
├── requirements.txt
├── PLANO.md
├── src/
│   ├── __init__.py
│   ├── problema1.py
│   ├── busca_continua.py
│   ├── estatisticas.py
│   └── graficos.py
└── resultados/
    └── problema1/
```

## Responsabilidade dos arquivos

`main.py`:

- Ponto de entrada do experimento.
- Configura parametros.
- Executa os tres algoritmos.
- Salva resultados.
- Exibe resumo no terminal.

`src/problema1.py`:

- Define a funcao objetivo.
- Define o dominio.
- Define se o problema e de minimizacao.
- Centraliza informacoes conhecidas do problema.

`src/busca_continua.py`:

- Implementa Hill Climbing.
- Implementa LRS.
- Implementa GRS.
- Implementa validacao de dominio.
- Implementa parada antecipada.

`src/estatisticas.py`:

- Calcula moda das solucoes.
- Calcula media, desvio, minimo e maximo dos valores finais.
- Calcula iteracoes medias.
- Escreve tabelas CSV.

`src/graficos.py`:

- Gera grafico de convergencia.
- Gera grafico das solucoes finais.

`resultados/problema1/`:

- Recebe tabelas CSV e imagens geradas.
- Nao deve ser enviado para o GitHub.

## Saidas esperadas

Arquivos gerados ao executar o experimento:

```text
resultados/problema1/rodadas.csv
resultados/problema1/resumo.csv
resultados/problema1/convergencia.png
resultados/problema1/solucoes_finais.png
```

O terminal deve exibir:

- Melhor solucao encontrada por algoritmo.
- Melhor valor de `f(x)` por algoritmo.
- Moda das solucoes finais por algoritmo.
- Frequencia da moda.
- Media e desvio dos valores finais.
- Numero medio de iteracoes.
- Menor `epsilon` encontrado para Hill Climbing.
- Menor `sigma` encontrado para LRS.

## Validacao

O resultado sera considerado correto se:

- Os tres algoritmos executarem as `100` rodadas sem erro.
- Todas as solucoes finais respeitarem o dominio.
- As tabelas forem geradas corretamente.
- Os graficos forem salvos corretamente.
- Ao menos um algoritmo encontrar solucao proxima de `(0, 0)`.
- A moda das solucoes for apresentada com arredondamento adequado.

## Ordem de execucao

1. Criar os arquivos base do projeto.
2. Implementar a funcao objetivo e o dominio.
3. Implementar Hill Climbing.
4. Implementar LRS.
5. Implementar GRS.
6. Implementar execucao de 100 rodadas por algoritmo.
7. Implementar resumo estatistico.
8. Implementar gravacao dos CSVs.
9. Implementar graficos.
10. Rodar o experimento completo.
11. Ajustar hiperparametros se necessario.
12. Registrar os resultados finais para uso no relatorio.
