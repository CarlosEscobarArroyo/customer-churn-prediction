# Ablación de time slices — XGBoost tuneado, test out-of-period fijo

> Generado por `05_modelling/07_ablacion_slices.py` el 2026-09-06 17:29.
> Test = últimos 4 meses etiquetados (885 filas, prevalencia 0.278), gap 6 meses. Train = últimos K meses de observación. Downsized = mismo n de filas muestreadas de todos los meses (1 fila por vendedora mientras n ≤ nº de vendedoras), promedio de 5 semillas.

## Resultados

|   K_meses |   n_train |   AUC multi-slicing |   Δ vs K=1 |   lift10 |   AUC downsized |   ± seeds |   Δ downsized vs K=1 |
|----------:|----------:|--------------------:|-----------:|---------:|----------------:|----------:|---------------------:|
|         1 |       214 |              0.7379 |     0      |     2.04 |          0.7186 |    0.0221 |              -0.0193 |
|         2 |       424 |              0.7456 |     0.0077 |     2    |          0.7353 |    0.0116 |              -0.0026 |
|         3 |       679 |              0.74   |     0.0021 |     1.96 |          0.7306 |    0.0146 |              -0.0073 |
|         6 |      1463 |              0.7551 |     0.0172 |     2.21 |          0.7355 |    0.0046 |              -0.0024 |
|        12 |      3098 |              0.7569 |     0.019  |     2.17 |          0.7493 |    0.0032 |               0.0114 |
|        24 |      6243 |              0.7534 |     0.0156 |     2.21 |          0.7606 |    0.0018 |               0.0227 |
|        36 |      9401 |              0.7591 |     0.0212 |     2.08 |          0.7617 |    0.0046 |               0.0238 |
|        60 |     14185 |              0.7643 |     0.0265 |     2.37 |          0.7626 |    0.0035 |               0.0247 |
|       100 |     28735 |              0.7638 |     0.0259 |     2.17 |        nan      |  nan      |             nan      |

## Lectura

- De K=1 a todo el histórico el AUC cambia +0.0259 (lift decil 2.04 → 2.17).
- Mejor K: 60 meses (AUC 0.7643).
- Si la columna *downsized* sube con K a igual n, la ganancia es diversidad temporal, no volumen (Gattermann-Itschert & Thonemann 2021, Fig. 9).

## Tabla completa

|   K_meses | variante                                               |   n_train |   n_vend |    AUC |   AUC_std |   lift10 |
|----------:|:-------------------------------------------------------|----------:|---------:|-------:|----------:|---------:|
|         1 | multi-slicing (últimos K meses)                        |       214 |      214 | 0.7379 |    0      |   2.0441 |
|         1 | downsized (mismo n, todos los meses, 1 fila/vendedora) |       214 |      214 | 0.7186 |    0.0221 |   1.9541 |
|         2 | multi-slicing (últimos K meses)                        |       424 |      357 | 0.7456 |    0      |   2.0032 |
|         2 | downsized (mismo n, todos los meses, 1 fila/vendedora) |       424 |      424 | 0.7353 |    0.0116 |   2.0114 |
|         3 | multi-slicing (últimos K meses)                        |       679 |      497 | 0.74   |    0      |   1.9623 |
|         3 | downsized (mismo n, todos los meses, 1 fila/vendedora) |       679 |      679 | 0.7306 |    0.0146 |   1.9541 |
|         6 | multi-slicing (últimos K meses)                        |      1463 |      804 | 0.7551 |    0      |   2.2076 |
|         6 | downsized (mismo n, todos los meses, 1 fila/vendedora) |      1463 |     1463 | 0.7355 |    0.0046 |   2.0032 |
|        12 | multi-slicing (últimos K meses)                        |      3098 |     1213 | 0.7569 |    0      |   2.1667 |
|        12 | downsized (mismo n, todos los meses, 1 fila/vendedora) |      3098 |     3098 | 0.7493 |    0.0032 |   2.1667 |
|        24 | multi-slicing (últimos K meses)                        |      6243 |     1823 | 0.7534 |    0      |   2.2076 |
|        24 | downsized (mismo n, todos los meses)                   |      6243 |     3122 | 0.7606 |    0.0018 |   2.3057 |
|        36 | multi-slicing (últimos K meses)                        |      9401 |     2326 | 0.7591 |    0      |   2.085  |
|        36 | downsized (mismo n, todos los meses)                   |      9401 |     3861 | 0.7617 |    0.0046 |   2.3384 |
|        60 | multi-slicing (últimos K meses)                        |     14185 |     3007 | 0.7643 |    0      |   2.3711 |
|        60 | downsized (mismo n, todos los meses)                   |     14185 |     4718 | 0.7626 |    0.0035 |   2.3466 |
|       100 | multi-slicing (últimos K meses)                        |     28735 |     6187 | 0.7638 |    0      |   2.1667 |
