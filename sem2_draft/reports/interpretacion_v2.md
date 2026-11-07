# Interpretación v2

Modelo: xgboost__seleccionadas (experimento b2ad959cb4c424b4), explicado sobre el OOT
(2025-12 → 2026-03, 879 filas).
AUC OOT del modelo explicado: 0.7844 (registrado en 05: 0.7844).

## Método

TreeSHAP en log-odds. Valor base: 0.289 en probabilidad. Error máximo de exactitud: 1.7e-06.

## Importancia global

|                |   shap_abs_medio |   porcentaje |   rho_valor_shap |   rank_shap |   rank_permutacion |
|:---------------|-----------------:|-------------:|-----------------:|------------:|-------------------:|
| n_ped_u12      |           0.2932 |      23.5558 |          -0.9816 |           1 |                  4 |
| n_prod_u12     |           0.2774 |      22.2918 |          -0.9746 |           2 |                  2 |
| monto_u12      |           0.2715 |      21.8183 |          -0.9601 |           3 |                  5 |
| compras_hist   |           0.1655 |      13.3016 |          -0.9182 |           4 |                  1 |
| monto_cv_u12   |           0.1304 |      10.4795 |           0.6874 |           5 |                  3 |
| ticket_prom_u3 |           0.1064 |       8.553  |          -0.5638 |           6 |                  6 |

`rho_valor_shap` > 0: valores altos de la variable elevan el riesgo; < 0: lo reducen.
Spearman entre el ranking SHAP y el de permutación de la fase 03: 0.26.

## Casos locales (valor de la variable y SHAP en log-odds)

|                | bajo            | medio           | alto           |
|:---------------|:----------------|:----------------|:---------------|
| probabilidad   | 0.05            | 0.31            | 0.65           |
| n_ped_u12      | 14.00 (-0.51)   | 6.00 (-0.16)    | 2.00 (+0.27)   |
| n_prod_u12     | 102.00 (-0.56)  | 21.00 (+0.16)   | 5.00 (+0.39)   |
| monto_u12      | 7202.82 (-0.51) | 1351.79 (+0.09) | 110.94 (+0.43) |
| compras_hist   | 15.00 (-0.19)   | 16.00 (-0.21)   | 4.00 (+0.13)   |
| monto_cv_u12   | 0.92 (-0.17)    | 2.16 (+0.20)    | 2.44 (+0.15)   |
| ticket_prom_u3 | 521.06 (-0.05)  | 199.97 (+0.02)  | 55.47 (+0.15)  |

## Figuras

`img/shap_v2_resumen.png`, `img/shap_v2_dependencia.png`, `img/shap_v2_local_bajo.png`,
`img/shap_v2_local_medio.png`, `img/shap_v2_local_alto.png`.
