# Sensibilidad al filtro de población — XGBoost tuneado

> Generado por `05_modelling/08_poblacion.py` el 2026-09-06 17:52. Regla vigente = `compras_hist >= 1` (primera fila). `vend_ultimo_mes` = vendedoras en alcance en el último mes observado del dataset.

## Tamaño de la población por regla

| regla                   |   n_rows | % filas   |   n_vend |   vend_ultimo_mes |   prevalencia |
|:------------------------|---------:|:----------|---------:|------------------:|--------------:|
| compras_hist >= 1       |    30821 | 100%      |     6356 |               192 |         0.309 |
| compras_hist >= 2       |    24465 | 79%       |     4272 |               181 |         0.274 |
| compras_hist >= 3       |    20193 | 66%       |     3096 |               166 |         0.25  |
| compras_hist >= 6       |    12811 | 42%       |     1521 |               137 |         0.199 |
| compras_hist >= 12      |     6638 | 22%       |      605 |                92 |         0.14  |
| meses_activos_u12 >= 2  |    28342 | 92%       |     5975 |               163 |         0.285 |
| meses_activos_u12 >= 3  |    20433 | 66%       |     3720 |               128 |         0.227 |
| meses_activos_u12 >= 6  |     8447 | 27%       |     1210 |                61 |         0.12  |
| meses_activos_u12 >= 9  |     3085 | 10%       |      374 |                29 |         0.051 |
| meses_activos_u12 >= 12 |      532 | 2%        |       68 |                 4 |         0.011 |

## AUC: modelo actual evaluado en el subconjunto vs. modelo reentrenado en él

| regla                   |   gkf_AUC_modelo_actual |   gkf_AUC_reentrenado |   oot_AUC_modelo_actual |   oot_AUC_reentrenado |   n_test |
|:------------------------|------------------------:|----------------------:|------------------------:|----------------------:|---------:|
| compras_hist >= 1       |                  0.7523 |                0.7523 |                  0.7638 |                0.7638 |      885 |
| compras_hist >= 2       |                  0.7581 |                0.7586 |                  0.7637 |                0.7616 |      831 |
| compras_hist >= 3       |                  0.7627 |                0.7638 |                  0.7637 |                0.7629 |      775 |
| compras_hist >= 6       |                  0.7731 |                0.7727 |                  0.7602 |                0.7582 |      619 |
| compras_hist >= 12      |                  0.7874 |                0.7826 |                  0.7321 |                0.7273 |      412 |
| meses_activos_u12 >= 2  |                  0.7432 |                0.7433 |                  0.7337 |                0.7337 |      763 |
| meses_activos_u12 >= 3  |                  0.7405 |                0.7406 |                  0.7184 |                0.7137 |      598 |
| meses_activos_u12 >= 6  |                  0.7522 |                0.7502 |                  0.6185 |                0.6147 |      279 |
| meses_activos_u12 >= 9  |                  0.7395 |                0.6888 |                  0.693  |                0.3982 |      119 |
| meses_activos_u12 >= 12 |                  0.6068 |                0.3951 |                nan      |              nan      |       12 |

## Lift (reentrenado)

| regla                   |   prevalencia |   gkf_liftPR_reentrenado |   oot_lift10_reentrenado |
|:------------------------|--------------:|-------------------------:|-------------------------:|
| compras_hist >= 1       |         0.309 |                    1.774 |                    2.167 |
| compras_hist >= 2       |         0.274 |                    1.9   |                    2.332 |
| compras_hist >= 3       |         0.25  |                    2.011 |                    2.452 |
| compras_hist >= 6       |         0.199 |                    2.231 |                    2.576 |
| compras_hist >= 12      |         0.14  |                    2.645 |                    2.233 |
| meses_activos_u12 >= 2  |         0.285 |                    1.751 |                    2.321 |
| meses_activos_u12 >= 3  |         0.227 |                    1.871 |                    2.008 |
| meses_activos_u12 >= 6  |         0.12  |                    2.426 |                    1.216 |
| meses_activos_u12 >= 9  |         0.051 |                    2.092 |                    0     |
| meses_activos_u12 >= 12 |         0.011 |                    0.922 |                  nan     |

## Notas

- Las AUC entre reglas NO son comparables como calidad del modelo: cada regla cambia la población y la prevalencia. La comparación válida es por fila: modelo actual vs reentrenado.
- Filtrar excluye vendedoras del alcance del modelo; en producción esas vendedoras no reciben score.
