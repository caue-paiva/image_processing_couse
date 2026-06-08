# Artefatos para o relatorio

## Classificacao

| method | descriptors | dim | best_model | val_acc | test_acc | f1_macro | f1_weighted |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gch | gch | 256 | C=1.0;gamma=scale;kernel=rbf | 0.3333 | 0.1111 | 0.0431 | 0.0671 |
| lbp | lbp | 256 | C=10.0;gamma=scale;kernel=rbf | 0.5556 | 0.1667 | 0.0940 | 0.1213 |
| glcm | glcm | 10 | C=1.0;gamma=scale;kernel=rbf | 0.2500 | 0.1111 | 0.0766 | 0.1111 |
| hog | hog | 1764 | C=0.1;kernel=linear | 0.4167 | 0.2778 | 0.1963 | 0.2389 |
| correlogram | correlogram | 48 | C=1.0;gamma=scale;kernel=rbf | 0.2500 | 0.1944 | 0.1280 | 0.1843 |
| gabor | gabor | 120 | C=10.0;gamma=0.01;kernel=rbf | 0.3056 | 0.2222 | 0.1599 | 0.2378 |
| gch+lbp | gch+lbp | 512 | C=0.1;kernel=linear | 0.5000 | 0.2778 | 0.1782 | 0.2176 |
| gch+glcm | gch+glcm | 266 | C=10.0;gamma=0.001;kernel=rbf | 0.3889 | 0.2222 | 0.1453 | 0.1981 |
| gch+lbp+glcm | gch+lbp+glcm | 522 | C=0.1;kernel=linear | 0.5000 | 0.2778 | 0.1678 | 0.1963 |
| gch+lbp+glcm+gabor | gch+lbp+glcm+gabor | 642 | C=0.1;kernel=linear | 0.4722 | 0.3056 | 0.2261 | 0.2447 |
| all | gch+lbp+glcm+hog+correlogram+gabor | 2454 | C=0.1;kernel=linear | 0.5000 | 0.3889 | 0.2610 | 0.3479 |


## Resumo das matrizes de confusao

| method | test_acc | f1_macro | correct | total | top_confusions |
| --- | --- | --- | --- | --- | --- |
| gch | 0.1111 | 0.0431 | 4 | 36 | hamtaro->de_victor (2); ada_pipoca->kefka (1); alan->billy_franzen (1) |
| lbp | 0.1667 | 0.0940 | 6 | 36 | ada_pipoca->hamtaro (1); alan->spike (1); bella->emilia (1) |
| glcm | 0.1111 | 0.0766 | 4 | 36 | milka->hamtaro (2); ada_pipoca->luna (1); alan->daisy (1) |
| hog | 0.2778 | 0.1963 | 10 | 36 | hamtaro->emilia (3); ada_pipoca->lulu (1); alan->de_enzo (1) |
| correlogram | 0.1944 | 0.1280 | 7 | 36 | bob->de_victor (2); lulu->hamtaro (2); ada_pipoca->daisy (1) |
| gabor | 0.2222 | 0.1599 | 8 | 36 | ada_pipoca->hamtaro (1); alan->kefka (1); bella->kefka (1) |


## Busca

| method | descriptors | dim | mAP | P@1 | P@5 | P@10 |
| --- | --- | --- | --- | --- | --- | --- |
| gch | gch | 256 | 0.0997 | 0.2262 | 0.1248 | 0.0940 |
| lbp | lbp | 256 | 0.1183 | 0.2643 | 0.1439 | 0.1125 |
| glcm | glcm | 10 | 0.0877 | 0.1635 | 0.1003 | 0.0801 |
| hog | hog | 1764 | 0.0837 | 0.1662 | 0.0926 | 0.0706 |
| correlogram | correlogram | 48 | 0.0952 | 0.1989 | 0.1117 | 0.0888 |
| gabor | gabor | 120 | 0.0973 | 0.1907 | 0.1188 | 0.0926 |
| gch+lbp | gch+lbp | 512 | 0.1327 | 0.2807 | 0.1619 | 0.1294 |
| gch+glcm | gch+glcm | 266 | 0.1082 | 0.2534 | 0.1390 | 0.1025 |
| gch+lbp+glcm | gch+lbp+glcm | 522 | 0.1346 | 0.2970 | 0.1640 | 0.1278 |
| gch+lbp+glcm+gabor | gch+lbp+glcm+gabor | 642 | 0.1329 | 0.2888 | 0.1668 | 0.1283 |
| all | gch+lbp+glcm+hog+correlogram+gabor | 2454 | 0.1173 | 0.2561 | 0.1422 | 0.1093 |


## BoVW

| method | k | dim | mAP | P@1 | P@5 | P@10 |
| --- | --- | --- | --- | --- | --- | --- |
| bovw | 200 | 200 | 0.1070 | 0.2125 | 0.1335 | 0.1044 |


## Figuras principais

- `outputs/figures/class_distribution.png`: ok
- `outputs/figures/confusion_best_individual.png`: ok
- `outputs/figures/confusion_best_combination.png`: ok
- `outputs/figures/confusion_gch_baseline.png`: ok
- `outputs/figures/confusion_gch.png`: ok
- `outputs/figures/confusion_lbp.png`: ok
- `outputs/figures/confusion_glcm.png`: ok
- `outputs/figures/confusion_hog.png`: ok
- `outputs/figures/confusion_correlogram.png`: ok
- `outputs/figures/confusion_gabor.png`: ok
- `outputs/figures/confusion_gch_lbp.png`: ok
- `outputs/figures/confusion_gch_glcm.png`: ok
- `outputs/figures/confusion_gch_lbp_glcm.png`: ok
- `outputs/figures/confusion_gch_lbp_glcm_gabor.png`: ok
- `outputs/figures/confusion_all.png`: ok
- `outputs/figures/bovw_tsne_all.png`: ok
- `outputs/figures/bovw_umap_all.png`: ok
- `outputs/figures/bovw_tsne_top_classes.png`: ok

## Exemplos de busca

- `outputs/retrieval_examples/best_failure_1.png`
- `outputs/retrieval_examples/best_success_1.png`
- `outputs/retrieval_examples/best_success_2.png`
