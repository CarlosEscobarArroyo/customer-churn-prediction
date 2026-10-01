# Actividad comercial histórica del departamento

## Resultado

**Ninguna variante de actividad superó el AUC mensual de la base (0.790383).** La mejor variante de actividad por esa métrica fue **Base + comparación a 6 y 12 meses** (0.789922, delta -0.000461). Las diferencias en detección al top 10 % se muestran por separado y no justifican por sí solas sustituir el modelo.

La variante de **seis meses** detectó **301** casos frente a **295** entre las mismas **431** observaciones priorizadas al top 10 %. La precisión pasó de **68,45 % a 69,84 %** (+1,39 puntos porcentuales). Mejoró el conteo en 7 de los 16 meses, empeoró en 2 y empató en 7. En el top 20 %, en cambio, detectó 541 frente a 546. Por tanto, el beneficio observado se limita al grupo de mayor riesgo y requiere confirmación independiente; no implica una mejora general de discriminación. Combinar seis y doce meses también detectó 301 casos en el top 10 %, añadiendo más columnas.

La hipótesis es que la caída de ventas, pedidos o vendedoras activas de un departamento aporte contexto adicional al historial individual. Se probaron cuatro variantes sobre las 42 variables transaccionales y se incluyó el nombre del departamento codificado como control.

| variant                           |   auc_mean |   auc_monthly_mean |   delta_auc_monthly_mean |   tp_top10 |   tp_top20 |   ap_mean |
|:----------------------------------|-----------:|-------------------:|-------------------------:|-----------:|-----------:|----------:|
| Base transaccional                |   0.789880 |           0.790383 |                 0.000000 |        295 |        546 |  0.596209 |
| Base + nombre del departamento    |   0.788920 |           0.789691 |                -0.000692 |        292 |        537 |  0.587713 |
| Base + actividad reciente         |   0.789204 |           0.789670 |                -0.000713 |        297 |        542 |  0.592562 |
| Base + comparación a 6 meses      |   0.789461 |           0.789904 |                -0.000479 |        301 |        541 |  0.594935 |
| Base + comparación a 12 meses     |   0.789287 |           0.789716 |                -0.000667 |        294 |        544 |  0.592687 |
| Base + comparación a 6 y 12 meses |   0.789446 |           0.789922 |                -0.000461 |        301 |        542 |  0.594796 |

`auc_mean` promedia AUC de cuatro bloques; `auc_monthly_mean` promedia AUC dentro de los 16 meses y mide la discriminación mensual. `tp_top10` y `tp_top20` cuentan abandonos detectados entre 431 y 870 observaciones priorizadas, respectivamente. `ap_mean` es la media de Average Precision de los bloques.

## Indicadores construidos

Para cada vendedora se calcula la media mensual de tres magnitudes del departamento, descontando su propia contribución:

1. Monto de ventas.
2. Número de pedidos.
3. Número de vendedoras activas (con al menos un pedido en ese mes).

Se usa la ventana reciente de tres meses completos, `[t−3, t−1]`, y las ventanas equivalentes desplazadas seis meses, `[t−9, t−7]`, y doce meses, `[t−15, t−13]`.

| Variante | Columnas adicionales |
|---|---|
| Actividad reciente | 3 niveles recientes en `log(1+x)` + 1 indicador de ausencia: **4** |
| Comparación a 6 meses | 3 niveles recientes + 3 históricos + 3 cambios + 2 indicadores de ausencia: **11** |
| Comparación a 12 meses | 3 niveles recientes + 3 históricos + 3 cambios + 2 indicadores de ausencia: **11** |
| Comparación a 6 y 12 meses | 9 niveles + 6 cambios + 3 indicadores de ausencia: **18** |
| Nombre del departamento | Control categórico codificado mediante one-hot del experimento anterior; no añade las magnitudes de actividad |

Los nombres exactos para cada variante son:

- `actividad_reciente`: `regional_log_ventas_h0`, `regional_log_pedidos_h0`, `regional_log_vendedoras_activas_h0`, `regional_sin_historia_h0`.
- `actividad_6m`: `regional_log_ventas_h0`, `regional_log_pedidos_h0`, `regional_log_vendedoras_activas_h0`, `regional_log_ventas_h6`, `regional_log_pedidos_h6`, `regional_log_vendedoras_activas_h6`, `regional_cambio_ventas_6m`, `regional_cambio_pedidos_6m`, `regional_cambio_vendedoras_activas_6m`, `regional_sin_historia_h0`, `regional_sin_historia_h6`.
- `actividad_12m`: `regional_log_ventas_h0`, `regional_log_pedidos_h0`, `regional_log_vendedoras_activas_h0`, `regional_log_ventas_h12`, `regional_log_pedidos_h12`, `regional_log_vendedoras_activas_h12`, `regional_cambio_ventas_12m`, `regional_cambio_pedidos_12m`, `regional_cambio_vendedoras_activas_12m`, `regional_sin_historia_h0`, `regional_sin_historia_h12`.
- `actividad_6m_12m`: `regional_log_ventas_h0`, `regional_log_pedidos_h0`, `regional_log_vendedoras_activas_h0`, `regional_log_ventas_h6`, `regional_log_pedidos_h6`, `regional_log_vendedoras_activas_h6`, `regional_log_ventas_h12`, `regional_log_pedidos_h12`, `regional_log_vendedoras_activas_h12`, `regional_cambio_ventas_6m`, `regional_cambio_pedidos_6m`, `regional_cambio_vendedoras_activas_6m`, `regional_cambio_ventas_12m`, `regional_cambio_pedidos_12m`, `regional_cambio_vendedoras_activas_12m`, `regional_sin_historia_h0`, `regional_sin_historia_h6`, `regional_sin_historia_h12`.

## Consistencia por período

AUC por bloque, en orden cronológico:

| variant            |        0 |        1 |        2 |        3 |
|:-------------------|---------:|---------:|---------:|---------:|
| transaccional      | 0.785349 | 0.794934 | 0.781867 | 0.797368 |
| mas_departamento   | 0.785685 | 0.792243 | 0.782556 | 0.795196 |
| actividad_reciente | 0.785170 | 0.793766 | 0.780637 | 0.797242 |
| actividad_6m       | 0.784894 | 0.794978 | 0.779927 | 0.798045 |
| actividad_12m      | 0.785089 | 0.794549 | 0.780246 | 0.797263 |
| actividad_6m_12m   | 0.784915 | 0.795396 | 0.779742 | 0.797730 |

Meses con AUC superior a la base, de 16:

| variant            |   meses_mejora |
|:-------------------|---------------:|
| actividad_12m      |              6 |
| actividad_6m       |              6 |
| actividad_6m_12m   |              8 |
| actividad_reciente |              3 |
| mas_departamento   |              6 |

## Componentes individuales

| variant            | model   |   auc_mean |   auc_monthly_mean |   tp_top10 |
|:-------------------|:--------|-----------:|-------------------:|-----------:|
| transaccional      | logreg  |   0.787625 |           0.787839 |        293 |
| transaccional      | xgboost |   0.788335 |           0.789059 |        298 |
| mas_departamento   | logreg  |   0.785820 |           0.786509 |        291 |
| mas_departamento   | xgboost |   0.788146 |           0.788888 |        296 |
| actividad_reciente | logreg  |   0.787017 |           0.787462 |        295 |
| actividad_reciente | xgboost |   0.788151 |           0.789044 |        296 |
| actividad_6m       | logreg  |   0.787554 |           0.787997 |        292 |
| actividad_6m       | xgboost |   0.787918 |           0.788729 |        297 |
| actividad_12m      | logreg  |   0.786815 |           0.787336 |        291 |
| actividad_12m      | xgboost |   0.788018 |           0.788691 |        297 |
| actividad_6m_12m   | logreg  |   0.787406 |           0.788212 |        292 |
| actividad_6m_12m   | xgboost |   0.787815 |           0.788512 |        293 |

## Cobertura y calidad

| partition   |   rows |   missing_department |   missing_window_h0 |   missing_window_h6 |   missing_window_h12 |
|:------------|-------:|---------------------:|--------------------:|--------------------:|---------------------:|
| development |  28311 |                   70 |                1085 |                3935 |                 6650 |
| validation  |   4379 |                    3 |                   3 |                   3 |                    3 |

El panel utilizado llega hasta 2024-12-01 y comienza en 2016-11-01; contiene 738,657 filas de 11,358 vendedoras hasta ese corte. Hay 140 observaciones activas sin departamento y 50,799.81 de monto sin ubicación, de un total de 18,766,934.03 (**0.271 %**). Estas ventas no se asignan arbitrariamente a ninguna región. En validación, tres filas carecen de contexto regional; las demás tienen las tres ventanas disponibles.

## Ejemplo del contexto para enero de 2025

Promedios de octubre–diciembre de 2024 y cambio contra octubre–diciembre de 2023, para los ocho departamentos con más ventas recientes. Esta tabla descriptiva incluye a todas las vendedoras; las variables del modelo, en cambio, descuentan a la vendedora evaluada. Aquí los cambios sí se expresan como porcentajes convencionales.

| departamento   |   monto_mensual_medio |   pedidos_mensuales_medios |   activas_mensuales_medias |   cambio_ventas_12m_pct |   cambio_activas_12m_pct |
|:---------------|----------------------:|---------------------------:|---------------------------:|------------------------:|-------------------------:|
| lima           |              36874.36 |                      99.67 |                      74.67 |                  -42.65 |                   -26.07 |
| piura          |              12563.87 |                      31.00 |                      26.33 |                  -10.04 |                   -13.19 |
| ancash         |              12408.70 |                      28.67 |                      24.00 |                  -44.82 |                   -34.55 |
| ica            |               9664.96 |                      23.33 |                      17.00 |                   -4.26 |                   -10.53 |
| la libertad    |               8683.78 |                      23.67 |                      22.33 |                  -24.94 |                   -17.28 |
| loreto         |               7359.64 |                      16.33 |                      13.67 |                  -32.06 |                   -26.79 |
| arequipa       |               7115.32 |                      20.67 |                      16.67 |                  -37.58 |                   -19.35 |
| junin          |               6971.43 |                      19.33 |                      14.33 |                  -23.17 |                   -28.33 |

## Alcance temporal y limitaciones

- Se conserva el protocolo temporal de desarrollo: cuatro bloques de cuatro meses (octubre 2023–enero 2025), gap de seis meses, 4.379 observaciones vendedora–mes y 16 meses de validación. Es validación reutilizada para seleccionar variables, **no el OOT final independiente**.
- La logística usa los últimos 36 meses permitidos y XGBoost la historia permitida completa; el ensemble mantiene 50/50 y los hiperparámetros exportados. No se ejecuta Optuna ni se buscan ventanas o umbrales a partir de la validación.
- Para la observación de mes `t`, la fecha máxima de información regional es `t−1`. Por ejemplo, para enero de 2025, la actividad reciente promedia octubre–diciembre de 2024; seis meses antes, abril–junio de 2024; doce meses antes, octubre–diciembre de 2023.
- La fuente es el panel mensual completo, incluyendo vendedoras nuevas y primeras compras. No se suman ventanas móviles del dataset de churn, que duplicarían ventas y excluirían primeras compras. El indicador de churn no interviene en los agregados.
- En cada ventana se resta la actividad histórica de la propia vendedora antes de transformar o comparar: los indicadores reflejan a las demás vendedoras del departamento. Dos vendedoras del mismo departamento pueden tener una pequeña diferencia por esta exclusión individual.
- Es válido usar compras ya observadas anteriores a cada predicción, incluidas las de meses posteriores al corte del entrenamiento: se asume acceso mensual al panel de ventas. Ninguna etiqueta del bloque de validación interviene en esta actualización.
- Departamento procede del maestro actual; no hay historia de mudanzas o cambios de ubicación. La correspondencia del maestro con el dataset de desarrollo se verificó, pero no garantiza residencia histórica. Los registros sin ubicación no se asignan a un departamento ficticio.
- Los ceros representan meses observados sin actividad; se exige un panel denso desde la primera compra hasta la última fecha requerida. Las ventanas anteriores a la cobertura del panel quedan ausentes. La mediana de imputación se aprende solo en cada entrenamiento; los indicadores de ausencia entran al modelo.
- Las ventas conservan el monto nominal original: los cambios pueden mezclar cantidad, precio y composición de compra. Por eso se incluyen también pedidos y vendedoras activas. La media de activas es un promedio de conteos mensuales, no el total de personas únicas de los tres meses.
- El cambio normalizado `(reciente − pasado)/(reciente + pasado)` está entre −1 y 1: negativo indica caída, positivo crecimiento y cero estabilidad. No es una tasa porcentual convencional; cuando ambos valores son cero se define como cero y cuando falta historia queda ausente.
- Los casos detectados se cuentan como vendedora–mes, no como personas distintas. Las métricas top 10 % y top 20 % seleccionan el mismo número de observaciones mensuales que la base. No se exportó ni sustituyó un modelo de producción.


## Reproducción y verificación

- Fuentes: `data/processed/panel_denso_mensual.csv` y `data/processed/dim_vendedor.csv`.
- Ejecutar `.venv/bin/python -m scripts.regional_activity_ablation`. Se conservan los parámetros en `reports/master_features_v1/reconstructed_tuning/` y la base en `reports/master_features_comparison_v1/`.
- Resultados, predicciones por bloque y componente, hashes, agregados departamento–mes, variables y cobertura: `reports/regional_activity_v1/`.
- [Notebook ejecutado](../05_modelling/12_actividad_departamental.ipynb).

Las pruebas verifican fechas exactas, exclusión de la propia vendedora, separación Lima/Callao, ausencia de influencia de compras futuras o etiquetas, ceros frente a historia ausente y rechazo de duplicados o huecos en el panel. Se contrastaron 108 valores de contexto directamente con las transacciones mensuales de otras vendedoras. Se verifica la misma población, pesos 50/50 y métricas desde las predicciones. Los archivos fuente se conservan intactos.
