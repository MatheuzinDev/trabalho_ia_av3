# Trabalho AV3 - Busca e Otimização Meta-Heurística

## Requisitos

- Python 3
- `numpy`
- `matplotlib`

## Como rodar a Parte 1

A Parte 1 executa os algoritmos Hill Climbing, Local Random Search e Global Random Search nos problemas contínuos.

Primeiro, crie e ative o ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependências:

```bash
python3 -m pip install -r requirements.txt
```

Execute o projeto:

```bash
python3 main.py
```

O problema executado é escolhido no arquivo `main.py`, na variável:

```python
SELECTED_PROBLEM_ID = "problema5"
```

Para rodar outro problema, altere esse valor para uma das opções:

```text
problema1
problema2
problema3
problema4
problema5
problema6
```

Exemplo: para rodar o problema 1, altere para:

```python
SELECTED_PROBLEM_ID = "problema1"
```

Depois execute novamente:

```bash
python3 main.py
```

Os resultados serão salvos na pasta:

```text
resultados/problemaX/
```

Dentro dessa pasta são gerados arquivos como:

```text
hiperparametros.csv
rodadas.csv
resumo.csv
convergencia.png
solucoes_finais.png
solucoes_finais_zoom.png
```
