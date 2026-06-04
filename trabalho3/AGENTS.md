# Trabalho 3 - Processamento de Imagens

Este diretorio contem o Trabalho 3 da disciplina, cujo objetivo e criar um sistema de busca e classificacao de animais de estimacao usando descritores de imagens.

## Fonte de Verdade

- O arquivo `trabalho-descricao.md` e a fonte de verdade para requisitos do trabalho.
- O arquivo `Trabalho_3.pdf` pode ser usado apenas como referencia de organizacao e comparacao, nunca como requisito principal nem como texto-base.

## Requisitos do Trabalho

- Selecionar 5 ou mais descritores de imagens.
- Incluir pelo menos 1 descritor que nao foi visto em aula.
- Extrair descritores para todas as imagens da base.
- Avaliar cada descritor individualmente e tambem diferentes combinacoes de descritores.

## Classificacao

- Dividir a base valida em treino, validacao e teste na proporcao 80/10/10.
- Pets com menos de 3 fotos nao entram na tarefa de classificacao.
- Pode usar `scikit-learn` ou outra implementacao externa para o classificador, desde que nao demande GPU.
- O custo computacional principal deve estar na extracao de caracteristicas.
- Apresentar acuracias e matrizes de confusao.

## Busca

- Usar a mesma colecao de descritores da classificacao.
- Implementar busca por similaridade: dada uma imagem de entrada, retornar um ranking de imagens da base.
- Usar distancia euclidiana simples par a par para diferentes combinacoes de descritores.
- Implementar Bag of Visual Words (BoVW).
- Comparar histogramas de visual words usando distancia euclidiana.
- Criar visualizacoes do descritor BoVW usando metodos como UMAP ou t-SNE.

## Relatorio

Para cada descritor, o relatorio deve:

- Justificar a escolha do descritor e a semantica visual capturada.
- Criar hipoteses sobre o funcionamento do descritor nas tarefas de classificacao e busca.
- Apresentar resultados de classificacao, incluindo acuracias e matrizes de confusao.
- Apresentar resultados de busca e exemplos que destaquem caracteristicas interessantes.
- Criar hipoteses sobre a distribuicao dos descritores nas visualizacoes BoVW.

## Bibliotecas

Para este trabalho, sao permitidas bibliotecas coerentes com `trabalho-descricao.md`, incluindo:

- `numpy`
- `scipy`
- `imageio`
- `scikit-learn`
- `matplotlib`
- `umap-learn`

Nao usar Pillow/PIL ou OpenCV para leitura, escrita ou processamento principal de imagens, a menos que o enunciado seja alterado explicitamente.

## Ambiente

- Ambiente virtual local do trabalho: `.venv`.
- Executar comandos Python preferencialmente com `.venv/bin/python`.
- O relatorio e escrito em LaTeX e compilado com `latexmk`.

## Entrega

- Codigo desenvolvido.
- Relatorio conforme os requisitos do enunciado.
