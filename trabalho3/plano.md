# Plano de Implementacao - Trabalho 3

Este plano segue `trabalho-descricao.md` como fonte da verdade. O arquivo `Trabalho_3.pdf` foi usado apenas como referencia de organizacao do relatorio, nivel de detalhe esperado nas secoes e exemplos de analise.

## 1. Objetivo

Implementar um sistema de busca e classificacao de animais de estimacao baseado em descritores classicos de imagem. O sistema deve:

1. Extrair pelo menos 5 descritores de imagens para toda a base, incluindo pelo menos 1 descritor nao visto em aula.
2. Avaliar os descritores, individualmente e em combinacoes, na tarefa de classificacao.
3. Avaliar os mesmos descritores, individualmente e em combinacoes, na tarefa de busca por similaridade.
4. Implementar Bag of Visual Words (BoVW), comparar histogramas de visual words por distancia euclidiana e visualizar sua distribuicao com t-SNE ou UMAP.
5. Produzir relatorio com justificativas, hipoteses, metricas, matrizes de confusao, exemplos de busca e analise das visualizacoes.

## 2. Restricoes de Biblioteca

Para leitura e processamento de imagens:

- Usar `imageio.v2.imread`.
- Usar `numpy` para manipulacao de pixels, histogramas, normalizacao e distancias.
- Usar `scipy` para filtros, convolucoes, interpolacao e operacoes cientificas.

Nao usar:

- Pillow/PIL.
- OpenCV.
- Bibliotecas de imagem equivalentes.

Para aprendizado de maquina e avaliacao, conforme permitido pela descricao:

- Usar `scikit-learn` para classificadores, `StandardScaler`, `KMeans`, metricas, matrizes de confusao e t-SNE.
- Usar `umap-learn` somente se estiver disponivel no ambiente; caso contrario, usar t-SNE como metodo principal de projecao.

## 3. Estrutura Sugerida do Projeto

Criar a seguinte estrutura:

```text
trabalho3/
  trabalho-descricao.md
  Trabalho_3.pdf
  pets.csv
  pets256.zip
  pets_original.zip
  plano.md
  data/
    pets256/
    pets_original/
  src/
    config.py
    io_utils.py
    color.py
    texture.py
    shape.py
    bovw.py
    features.py
    split_data.py
    classification.py
    retrieval.py
    visualization.py
    report_assets.py
    main.py
  outputs/
    features/
    splits/
    metrics/
    figures/
    retrieval_examples/
  relatorio/
    relatorio.md ou relatorio.tex
```

`pets256.zip` e `pets_original.zip` devem ser extraidos para `data/` antes da execucao do pipeline. A base operacional principal sera `data/pets256/`, pois ja contem imagens `256x256` redimensionadas e renomeadas. A pasta `data/pets_original/` sera mantida como referencia dos arquivos originais, mas nao deve ser usada para extracao principal de descritores para evitar variacao de escala, extensao e resolucao.

## 4. Preparacao da Base

### 4.1 Entrada esperada

A base disponivel neste diretorio contem:

- `pets.csv`: arquivo de metadados com 367 imagens e 43 classes.
- `pets256.zip`: base redimensionada e renomeada, com 367 imagens em `pets256/<classe>/<arquivo>.jpg`.
- `pets_original.zip`: base com tamanhos, nomes e extensoes originais, tambem com 367 imagens em `pets_original/<classe>/<arquivo_original>`.

O CSV tem as colunas:

```text
class_id, class_name, filename
```

Observacao de leitura: o arquivo possui espacos apos as virgulas no cabecalho e nas linhas. Ao ler com Python, usar `csv.DictReader(..., skipinitialspace=True)` ou fazer `strip()` nos nomes das colunas e valores.

Exemplo de linha:

```text
0, simba, simba/00000.jpg
```

Interpretacao:

- `class_id`: identificador numerico da classe.
- `class_name`: nome padronizado do pet/classe.
- `filename`: caminho relativo dentro de `pets256/`.

### 4.2 Resumo da base

Validacao feita sobre `pets.csv`, `pets256.zip` e `pets_original.zip`:

- Total de imagens no CSV: 367.
- Total de classes: 43.
- Total de imagens em `pets256.zip`: 367.
- Total de imagens em `pets_original.zip`: 367.
- As 43 pastas/classes dos dois zips batem com as 43 classes do CSV.
- Nao ha divergencia de contagem por classe entre CSV e zips.
- Menor quantidade de imagens por classe: 1.
- Maior quantidade de imagens por classe: 20.
- Classes com menos de 3 fotos: 5 classes, totalizando 5 imagens.
- Classes validas para classificacao: 38 classes, totalizando 362 imagens.

Nota: `trabalho-descricao.md` menciona 42 animais de estimacao, mas os arquivos atualmente presentes (`pets.csv`, `pets256.zip` e `pets_original.zip`) contem 43 classes. O relatorio deve mencionar os numeros reais obtidos a partir do CSV entregue e, se necessario, observar essa diferenca entre a descricao textual e a base recebida.

Distribuicao por classe:

```text
hamtaro: 20
kefka: 20
billy_franzen: 17
lulu: 17
luna: 17
daisy: 16
de_victor: 15
kyara: 15
spike: 15
emilia: 14
milka: 13
zed: 13
nina: 12
nyx: 11
de_enzo: 10
pets_antes_do_banho: 10
kookie: 9
pipita: 9
francisco: 8
bob: 7
mel: 7
ada_pipoca: 6
alan: 6
billy: 6
mel_nunes: 6
bella: 5
bruce: 5
chico: 5
dakota: 5
haru: 5
lily: 5
meeko: 5
pudim: 5
simba: 5
spike_amaral: 5
tofu: 5
tsuki: 5
zoe: 3
de_bruno_1: 1
de_bruno_2: 1
de_bruno_3: 1
de_bruno_4: 1
de_bruno_5: 1
```

Para classificacao, excluir:

```text
de_bruno_1, de_bruno_2, de_bruno_3, de_bruno_4, de_bruno_5
```

Para busca, usar as 367 imagens, pois a restricao de menos de 3 fotos foi definida apenas para classificacao.

### 4.3 Preparacao local dos arquivos

Extrair os zips para `data/`:

```bash
mkdir -p data
unzip pets256.zip -d data
unzip pets_original.zip -d data
```

Apos extrair, ignorar arquivos de metadados macOS:

- Diretorios `__MACOSX/`.
- Arquivos `.DS_Store`.
- Arquivos iniciados por `._`.

O pipeline deve montar os caminhos das imagens com:

```text
data/pets256/<filename_do_csv>
```

Exemplo:

```text
data/pets256/simba/00000.jpg
```

### 4.4 Leitura

Implementar em `src/io_utils.py`:

- Leitura de `pets.csv`.
- Normalizacao de campos do CSV com `strip()`.
- Uso de `class_id`, `class_name` e `filename` como metadados principais.
- Leitura das imagens com `imageio.v2.imread`.
- Conversao segura para `float32` em intervalo `[0, 1]`.
- Tratamento de imagens RGB/RGBA, descartando canal alfa quando existir.
- Validacao de dimensoes esperadas em `pets256`: `256x256`.
- Validacao de existencia de todos os arquivos referenciados pelo CSV.

### 4.5 Filtro para classificacao

Para a tarefa de classificacao:

- Remover classes com menos de 3 fotos.
- Neste dataset, isso remove exatamente 5 classes e 5 imagens: `de_bruno_1`, `de_bruno_2`, `de_bruno_3`, `de_bruno_4`, `de_bruno_5`.
- A classificacao deve usar 362 imagens de 38 classes.
- Manter todas as imagens na tarefa de busca, salvo se a analise decidir reportar tambem uma versao filtrada. A descricao so restringe pets com menos de 3 fotos na classificacao.

## 5. Descritores Globais

Selecionar 6 descritores para ter margem alem do minimo exigido. Pelo menos um deles deve ser explicitamente marcado como nao visto em aula.

### 5.1 Histograma Global de Cores em HSV

Semantica:

- Captura distribuicao global de cores da imagem.
- Deve ajudar a separar pets por cor de pelagem e fundo dominante.

Implementacao:

- Converter RGB para HSV manualmente com `numpy`.
- Quantizar `H`, `S` e `V`.
- Sugestao: `16 x 4 x 4 = 256` bins.
- Normalizar por soma L1.

Hipotese:

- Deve funcionar bem para pets visualmente distintos por cor, mas falhar quando diferentes pets tiverem pelagens e fundos parecidos.

### 5.2 LBP - Local Binary Patterns

Semantica:

- Captura textura local.
- Deve representar padroes de pelo, rugosidade e microtexturas.

Implementacao:

- Converter imagem para escala de cinza.
- Implementar LBP circular com interpolacao simples ou LBP 8-vizinhos como versao inicial.
- Usar histograma normalizado dos codigos.
- Se houver tempo, implementar LBP uniforme para reduzir dimensionalidade.

Hipotese:

- Isoladamente pode ser limitado, mas deve complementar cor em combinacoes.

### 5.3 Haralick/GLCM

Semantica:

- Captura textura estatistica por co-ocorrencia de niveis de cinza.

Implementacao:

- Quantizar cinza para 32 ou 64 niveis.
- Construir matrizes GLCM para distancias `1, 2, 3` e angulos `0, 45, 90, 135`.
- Extrair contraste, homogeneidade, energia, correlacao e entropia.
- Agregar media e desvio por propriedade.

Hipotese:

- Deve ser mais estavel que LBP para textura global, mas menos sensivel a padroes locais especificos.

### 5.4 HOG - Histogram of Oriented Gradients

Semantica:

- Captura forma, contornos e orientacao de bordas.

Implementacao:

- Converter para cinza.
- Redimensionar para `128x128` se necessario usando `scipy.ndimage.zoom`.
- Calcular gradientes com filtros de Sobel ou diferencas finitas.
- Dividir em celulas `16x16`.
- Usar 9 bins de orientacao.
- Normalizar blocos `2x2` com L2.

Hipotese:

- Pode ajudar em diferencas de pose e silhueta, mas deve sofrer com variacao de enquadramento e alta dimensionalidade.

### 5.5 Correlograma de Cores

Semantica:

- Captura relacao espacial entre cores, complementando o histograma global.

Implementacao:

- Converter para HSV.
- Usar canal `H` ou uma quantizacao compacta de HSV.
- Para cada distancia `1, 3, 5`, medir frequencia de pixels com mesma cor quantizada em vizinhancas deslocadas.
- Normalizar o vetor final.

Hipotese:

- Deve ajudar em pets com manchas e padroes espaciais de cor, mas pode ser sensivel a pose e fundo.

### 5.6 Gabor - Descritor Nao Visto em Aula

Semantica:

- Captura textura em multiplas frequencias e orientacoes.
- Sera o descritor explicitamente escolhido como "nao visto em aula".

Implementacao:

- Converter para cinza.
- Criar banco de filtros de Gabor com `scipy`/`numpy`.
- Sugestao: 4 ou 5 frequencias e 6 ou 8 orientacoes.
- Para cada resposta, extrair media, desvio padrao e energia.
- Normalizar o vetor final.

Hipotese:

- Deve ser forte para pelagens com listras, manchas, pelos longos e texturas orientadas.

## 6. Extracao e Armazenamento de Features

Implementar em `src/features.py`:

- Funcao unica para extrair todos os descritores de uma imagem.
- Cache em `outputs/features/*.npz` para evitar recomputacao.
- Arquivos separados por descritor:

```text
outputs/features/gch.npz
outputs/features/lbp.npz
outputs/features/glcm.npz
outputs/features/hog.npz
outputs/features/correlogram.npz
outputs/features/gabor.npz
outputs/features/bovw.npz
```

Cada `.npz` deve conter:

- `X`: matriz `n_imagens x n_features`.
- `y`: classes.
- `files`: nomes dos arquivos.
- `descriptor_name`.

## 7. Combinacoes de Descritores

Avaliar descritores individuais e combinacoes representativas:

1. Cada descritor isolado.
2. `GCH + LBP`.
3. `GCH + GLCM`.
4. `GCH + LBP + GLCM`.
5. `GCH + LBP + GLCM + Gabor`.
6. `Todos`.

Antes de concatenar:

- Normalizar cada bloco de descritor separadamente com scaler ajustado no treino.
- Concatenar os blocos ja padronizados para evitar que descritores de maior escala dominem a distancia ou o classificador.

## 8. Tarefa de Classificacao

### 8.1 Split

Implementar em `src/split_data.py`:

- Filtrar classes com menos de 3 imagens.
- Split estratificado:
  - Treino: 80%.
  - Validacao: 10%.
  - Teste: 10%.
- Usar `random_state` fixo para reprodutibilidade.
- Salvar indices em `outputs/splits/split.json`.

Observacao importante:

- Como algumas classes podem ter exatamente 3 imagens, o split estratificado 80/10/10 pode falhar dependendo do `scikit-learn`. Se isso acontecer, implementar split manual garantindo pelo menos 1 imagem por classe no teste quando possivel e priorizando a proporcao global solicitada. Documentar a regra no relatorio.

### 8.2 Classificador

Usar um classificador simples e justificavel:

- Principal: SVM (`SVC`) com kernels `linear` e `rbf`.
- Alternativa se SVM ficar instavel: k-NN ou Random Forest como baseline adicional.

Procedimento:

1. Ajustar `StandardScaler` somente no treino.
2. Testar hiperparametros no conjunto de validacao:
   - SVM linear: `C in {0.1, 1, 10}`.
   - SVM RBF: `C in {0.1, 1, 10}`, `gamma in {"scale", 0.01, 0.001}`.
3. Escolher melhor configuracao por acuracia de validacao.
4. Reportar resultado final no teste.

### 8.3 Metricas

Para cada descritor/combinacao:

- Acuracia de validacao.
- Acuracia de teste.
- F1-score macro e ponderado.
- Matriz de confusao no teste.

Gerar figuras:

- Matriz de confusao para o melhor descritor individual.
- Matriz de confusao para a melhor combinacao.
- Opcional: matriz de confusao para um descritor simples de referencia, como GCH.

## 9. Tarefa de Busca por Similaridade

### 9.1 Busca par a par

Implementar em `src/retrieval.py`:

- Para cada imagem de consulta, comparar contra todas as outras imagens.
- Excluir a propria imagem do ranking.
- Usar distancia euclidiana simples.
- Usar os mesmos descritores e combinacoes da classificacao.
- Normalizar features antes da distancia; para combinacoes, normalizar cada bloco antes de concatenar.

### 9.2 Metricas de busca

Reportar:

- Precision@1.
- Precision@5.
- Precision@10.
- Average Precision por query.
- mAP geral.

Uma imagem retornada e considerada correta se tiver a mesma classe/pet da query.

### 9.3 Exemplos visuais

Gerar exemplos de busca em `outputs/retrieval_examples/`:

- Pelo menos 3 queries para o melhor metodo.
- Pelo menos 1 exemplo de sucesso claro.
- Pelo menos 1 exemplo de falha interessante.
- Mostrar query e top-k resultados.
- Marcar resultados corretos/incorretos com borda verde/vermelha.

## 10. Bag of Visual Words

### 10.1 Descritor local sem OpenCV/PIL

Como OpenCV nao pode ser usado, implementar descritores locais densos manualmente:

1. Converter imagem para cinza.
2. Dividir a imagem em patches densos, por exemplo `16x16` com passo `8` ou `16`.
3. Para cada patch, extrair um descritor local simples:
   - Histograma de orientacoes de gradiente local, similar a HOG compacto.
   - Opcionalmente concatenar media/desvio do patch para capturar intensidade.
4. Normalizar cada descritor local.

Essa escolha atende ao requisito de BoVW sem depender de ORB/SIFT/OpenCV.

### 10.2 Dicionario visual

Implementar em `src/bovw.py`:

- Coletar descritores locais de todas as imagens de treino ou de toda a base para a tarefa de busca.
- Se houver muitos patches, amostrar no maximo um numero fixo, por exemplo `50_000`.
- Rodar `KMeans`.
- Testar `K = 100` e `K = 200`, escolhendo um padrao para o relatorio.

### 10.3 Histograma BoVW por imagem

Para cada imagem:

- Atribuir cada descritor local ao centroide mais proximo.
- Construir histograma de visual words.
- Normalizar L1.

### 10.4 Busca com BoVW

- Comparar histogramas BoVW com distancia euclidiana.
- Reportar as mesmas metricas de busca: mAP, P@1, P@5, P@10.
- Comparar diretamente com o melhor descritor global/combinacao.

### 10.5 Visualizacao BoVW

Gerar visualizacao 2D dos histogramas BoVW:

- Metodo principal: t-SNE via `scikit-learn`.
- Metodo opcional: UMAP se `umap-learn` estiver disponivel.

Figuras esperadas:

- Scatter plot colorido por classe/pet.
- Versao com legenda simplificada ou sem legenda completa caso existam muitas classes.
- Opcional: destacar apenas classes com maior numero de imagens para melhorar legibilidade.

## 11. Relatorio

Organizar o relatorio seguindo uma estrutura semelhante ao PDF de exemplo, mas com resultados proprios:

### 11.1 Introducao

Incluir:

- Contexto do problema.
- Objetivo de classificar e buscar pets.
- Resumo da abordagem por descritores.

### 11.2 Base de Dados

Incluir:

- Fonte dos dados:
  - `pets.csv` como arquivo de metadados.
  - `pets256.zip` como base principal usada nos experimentos.
  - `pets_original.zip` como referencia dos arquivos originais.
- Numero total de imagens: 367.
- Numero total de classes/pets: 43.
- Distribuicao de imagens por classe.
- Quantas classes/imagens foram usadas na classificacao apos remover pets com menos de 3 fotos: 38 classes e 362 imagens.
- Quais classes foram removidas da classificacao: `de_bruno_1`, `de_bruno_2`, `de_bruno_3`, `de_bruno_4`, `de_bruno_5`.
- Informar que a busca usa as 367 imagens, pois o filtro de menos de 3 fotos se aplica apenas a classificacao.

### 11.3 Metodologia

Para cada descritor:

- Nome.
- Semantica capturada.
- Justificativa da escolha.
- Detalhes de implementacao.
- Hipotese inicial para classificacao e busca.

Tambem incluir:

- Split treino/validacao/teste.
- Classificador e selecao de hiperparametros.
- Procedimento de busca por distancia euclidiana.
- Procedimento BoVW.
- Metodo de visualizacao.

### 11.4 Resultados de Classificacao

Incluir:

- Tabela por descritor/combinacao com dimensao, acuracia de validacao, acuracia de teste, F1 macro e F1 ponderado.
- Matrizes de confusao relevantes.
- Analise textual dos melhores e piores descritores.

### 11.5 Resultados de Busca

Incluir:

- Tabela por descritor/combinacao com mAP, P@1, P@5 e P@10.
- Exemplos visuais de rankings.
- Discussao sobre casos de sucesso e falha.

### 11.6 Resultados BoVW

Incluir:

- Configuracao do descritor local.
- Tamanho do vocabulario visual.
- Metricas de busca BoVW.
- Comparacao com descritores globais.
- Visualizacao t-SNE/UMAP.

### 11.7 Analise e Hipoteses

Para cada descritor:

- Comparar hipotese inicial com resultado observado.
- Explicar possiveis causas de desempenho bom/ruim.
- Relacionar resultado de classificacao com resultado de busca.

Para BoVW:

- Criar hipoteses sobre clusters, sobreposicoes e outliers na visualizacao.
- Discutir se as visual words parecem separar pets, especies, fundos ou texturas.

### 11.8 Conclusao

Incluir:

- Principais achados.
- Melhor descritor individual.
- Melhor combinacao.
- Comparacao entre busca por descritores globais e BoVW.
- Limitacoes.
- Possiveis trabalhos futuros.

## 12. Scripts de Execucao

Criar `src/main.py` com etapas acionaveis por argumento:

```bash
/Users/caue.lira/Desktop/facul/proc_imagens/venv/bin/python3 src/main.py extract
/Users/caue.lira/Desktop/facul/proc_imagens/venv/bin/python3 src/main.py classify
/Users/caue.lira/Desktop/facul/proc_imagens/venv/bin/python3 src/main.py retrieve
/Users/caue.lira/Desktop/facul/proc_imagens/venv/bin/python3 src/main.py bovw
/Users/caue.lira/Desktop/facul/proc_imagens/venv/bin/python3 src/main.py visualize
/Users/caue.lira/Desktop/facul/proc_imagens/venv/bin/python3 src/main.py all
```

Tambem criar um modo de configuracao central em `src/config.py`:

- Caminho da base principal: `data/pets256`.
- Caminho da base original de referencia: `data/pets_original`.
- Caminho do CSV: `pets.csv`.
- Semente aleatoria.
- Lista de descritores.
- Lista de combinacoes.
- Parametros do BoVW.
- Parametros de figuras.

## 13. Ordem de Implementacao

1. Criar estrutura de pastas e `config.py`.
2. Implementar leitura de CSV/imagens e validacao da base.
3. Implementar descritores globais na ordem:
   - GCH.
   - LBP.
   - GLCM.
   - HOG.
   - Correlograma.
   - Gabor.
4. Implementar cache de features.
5. Implementar split estratificado e filtro de classes com menos de 3 fotos.
6. Implementar classificacao com SVM e metricas.
7. Implementar busca euclidiana e metricas de retrieval.
8. Implementar geracao de exemplos visuais de busca.
9. Implementar descritores locais densos para BoVW.
10. Implementar KMeans, histogramas BoVW e busca BoVW.
11. Implementar t-SNE/UMAP para visualizacao BoVW.
12. Gerar tabelas e figuras.
13. Escrever relatorio com analise baseada nos resultados reais.
14. Rodar pipeline completo do zero para validar reprodutibilidade.

## 14. Validacao Final

Antes da entrega, verificar:

- `pets256.zip` e `pets_original.zip` foram extraidos para `data/`.
- `pets.csv` foi lido com `skipinitialspace=True` ou campos normalizados com `strip()`.
- O pipeline encontrou 367 imagens e 43 classes antes do filtro de classificacao.
- As imagens referenciadas por `filename` existem em `data/pets256/`.
- Todas as imagens sao lidas com `imageio.v2.imread`.
- Nenhum uso de PIL/Pillow/OpenCV foi introduzido.
- Existem pelo menos 5 descritores e pelo menos 1 nao visto em aula.
- Todas as imagens possuem features extraidas.
- Classes com menos de 3 fotos foram excluidas da classificacao, resultando em 362 imagens e 38 classes.
- A busca foi avaliada com as 367 imagens.
- Split 80/10/10 esta documentado e salvo.
- A classificacao reporta acuracia e matriz de confusao.
- A busca usa distancia euclidiana par a par.
- BoVW usa histograma de visual words e distancia euclidiana.
- Ha visualizacao BoVW com t-SNE ou UMAP.
- O relatorio contem justificativa, hipotese e analise para cada descritor.
- O relatorio contem exemplos de busca com discussao.
- O codigo roda com:

```bash
/Users/caue.lira/Desktop/facul/proc_imagens/venv/bin/python3 src/main.py all
```

## 15. Riscos e Decisoes a Documentar

### Split com poucas imagens por classe

Risco:

- Classes com apenas 3 imagens podem dificultar um split estratificado exato em 80/10/10.

Decisao:

- Tentar split estratificado padrao primeiro.
- Se falhar, usar split manual por classe e documentar a regra.

### Alta dimensionalidade

Risco:

- HOG e combinacoes grandes podem causar overfitting na classificacao e piorar busca por distancia euclidiana.

Decisao:

- Sempre reportar dimensao dos vetores.
- Comparar descritores individuais com combinacoes compactas.

### BoVW sem ORB/SIFT/OpenCV

Risco:

- Descritores locais densos implementados manualmente podem ser menos informativos que ORB/SIFT.

Decisao:

- Explicar no relatorio que a implementacao preserva as restricoes de biblioteca.
- Avaliar BoVW como metodo exigido, comparando honestamente com descritores globais.

### Visualizacao com muitas classes

Risco:

- Scatter plot com 42 classes pode ficar visualmente poluido.

Decisao:

- Gerar uma figura completa e uma figura complementar destacando classes com mais imagens.
