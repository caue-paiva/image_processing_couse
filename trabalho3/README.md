# Trabalho 3 - Processamento de Imagens

## Arquivos esperados

Antes de executar, coloque os arquivos da base no mesmo diretorio dos scripts:

- `pets.csv`
- `pets256.zip`
- `pets_original.zip`

O codigo extrai os zips automaticamente para `data/` na primeira execucao.

## Ambiente

Crie e ative um ambiente virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Instale as dependencias:

```bash
python -m pip install -r requirements.txt
```

## Execucao

Execute o pipeline completo:

```bash
python main.py all
```

Para validar os arquivos gerados:

```bash
python main.py validate
```

Os resultados sao salvos em `outputs/`, incluindo metricas, matrizes de confusao, visualizacoes BoVW e exemplos de busca.

## Etapas individuais

Tambem e possivel executar partes do pipeline:

```bash
python main.py prepare
python main.py extract
python main.py classify
python main.py retrieve
python main.py bovw
python main.py visualize
python main.py report-assets
```

Use `--force` para recalcular descritores ou caches existentes.
