# Plano do Relatorio

## Objetivo

Escrever o relatorio final do trabalho em LaTeX, usando os resultados gerados em `outputs/` e cobrindo fielmente os requisitos do enunciado. O arquivo `Trabalho_3.pdf` deve servir apenas como referencia de organizacao e nivel de detalhe, sem copia ou reescrita do texto do exemplo.

## Estrutura

1. Introducao
   - Problema, tarefas de classificacao e busca, e perguntas experimentais.
2. Base de Dados e Protocolo Experimental
   - CSV, bases `pets256` e `pets_original`, 367 imagens, 43 classes, diferenca em relacao ao enunciado, filtro da classificacao e metricas.
3. Descritores: Semantica, Implementacao e Hipoteses
   - GCH, LBP, GLCM/Haralick, HOG, Correlograma de Cores e Gabor.
   - Para cada descritor: justificativa, semantica capturada, implementacao, hipotese para classificacao, hipotese para busca e limitacoes esperadas.
4. Resultados Quantitativos
   - Classificacao, busca e BoVW, com tabelas e figuras geradas pelo pipeline.
5. Analise por Descritor
   - Discussao individual confrontando hipoteses com resultados de classificacao e busca.
6. Analise Comparativa e Estudos de Caso
   - Comparacao entre tarefas, combinacoes compactas versus todos os descritores e exemplos visuais de busca.
7. Hipoteses sobre a Distribuicao BoVW
   - Interpretacao das projecoes t-SNE e UMAP.
8. Conclusao
   - Resumo dos achados, melhores metodos, limitacoes e trabalhos futuros.

## Ferramentas LaTeX

Usar LaTeX diretamente, sem Pandoc.

- `latexmk` versao 4.88.
- `pdflatex` do TeX Live 2026.
- `xelatex` e `lualatex` estao disponiveis, mas nao sao necessarios.
- `pandoc` esta disponivel, mas nao sera usado.
- `tectonic` nao esta instalado.

Comando de compilacao:

```bash
mkdir -p .cache/texmf-var
TEXMFVAR="$(pwd)/.cache/texmf-var" latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=relatorio relatorio/relatorio.tex
```

## Verificacao

- `relatorio/relatorio.tex` existe.
- `relatorio/relatorio.pdf` e gerado com sucesso.
- As figuras referenciadas existem em `outputs/figures/` e `outputs/retrieval_examples/`.
- O PDF contem as secoes descritas neste plano.
- O relatorio apresenta:
  - justificativa e hipotese para cada descritor;
  - acuracias e matrizes de confusao;
  - resultados de busca e exemplos visuais;
  - BoVW com metricas e visualizacoes;
  - hipoteses sobre a distribuicao dos descritores BoVW.

