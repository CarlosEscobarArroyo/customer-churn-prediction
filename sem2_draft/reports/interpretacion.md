# Interpretación (sem2)

Generado por `06_interpretacion/interpretacion.ipynb`. Modelo: **lightgbm**, explicado sobre el OOT
(2025-12 → 2026-03, 879 filas).

AUC OOT del modelo explicado: 0.7931 (registrado en `05`: 0.7931).

## 1 · Método
TreeSHAP en log-odds (1 modelo). Valor base: 0.287 en probabilidad. Diferencia máxima entre la
probabilidad explicada por SHAP y la predicha: 1.1e-15.

## 2 · Importancia global
|                |   shap_abs_medio |     pct |   rho_valor_shap |   rank_shap |   rank_permutacion |
|:---------------|-----------------:|--------:|-----------------:|------------:|-------------------:|
| n_ped_u12      |           0.2883 | 21.8552 |          -0.9899 |           1 |                  4 |
| monto_u12      |           0.267  | 20.2394 |          -0.977  |           2 |                  5 |
| compras_hist   |           0.2665 | 20.2076 |          -0.9131 |           3 |                  1 |
| n_prod_u12     |           0.2109 | 15.9908 |          -0.9821 |           4 |                  2 |
| monto_cv_u12   |           0.1818 | 13.7828 |           0.7357 |           5 |                  3 |
| ticket_prom_u3 |           0.1045 |  7.9241 |          -0.688  |           6 |                  6 |

`rho_valor_shap` > 0: valores altos de la variable elevan el riesgo; < 0: lo reducen.
Spearman entre el ranking SHAP y el de permutación de `03` (6 variables): 0.14.

## 3 · Casos locales (valor de la variable y SHAP en log-odds)
|                | bajo            | medio           | alto          |
|:---------------|:----------------|:----------------|:--------------|
| probabilidad   | 0.05            | 0.32            | 0.67          |
| n_ped_u12      | 38.00 (-0.65)   | 7.00 (-0.14)    | 1.00 (+0.39)  |
| monto_u12      | 9070.12 (-0.68) | 1545.69 (+0.14) | 32.98 (+0.41) |
| compras_hist   | 26.00 (-0.44)   | 8.00 (-0.00)    | 1.00 (+0.11)  |
| n_prod_u12     | 126.00 (-0.33)  | 28.00 (-0.09)   | 2.00 (+0.44)  |
| monto_cv_u12   | 0.93 (-0.09)    | 1.42 (+0.16)    | 3.46 (+0.23)  |
| ticket_prom_u3 | 179.67 (+0.16)  | 191.98 (+0.10)  | 32.98 (+0.05) |

## Figuras
`img/shap_resumen.png`, `img/shap_dependencia.png`, `img/shap_local_bajo.png`, `img/shap_local_medio.png`,
`img/shap_local_alto.png`.
