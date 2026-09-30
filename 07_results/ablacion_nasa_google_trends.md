# Evaluación de NASA POWER y Google Trends

## Resultado

**Ninguna variante superó el AUC mensual de la base.** Las diferencias en casos detectados deben leerse junto con el AUC y la consistencia por período; no justifican reemplazar el ensemble actual.

Se evaluaron cinco ampliaciones del ensemble transaccional, sin Optuna y con los parámetros ya exportados. La base conserva 42 variables. Todos los ensambles promedian 50 % logística (últimos 36 meses de entrenamiento) y 50 % XGBoost (historia permitida completa).

Cuatro bloques de validación: octubre 2023–enero 2024, febrero–mayo 2024, junio–septiembre 2024 y octubre 2024–enero 2025; gap de seis meses. Son 4.379 observaciones en 16 meses. Al priorizar el 10 % por mes se seleccionan 431 observaciones; al 20 %, 870. Se conserva exactamente la misma población y criterio de desempate que la base.

| variant                              |   n_added_columns |   auc_mean |   auc_monthly_mean |   delta_auc_monthly_mean |   tp_top10 |   tp_top20 |   ap_mean |
|:-------------------------------------|------------------:|-----------:|-------------------:|-------------------------:|-----------:|-----------:|----------:|
| Base transaccional                   |                 0 |   0.789880 |           0.790383 |                 0.000000 |        295 |        546 |  0.596209 |
| Base + NASA POWER                    |                13 |   0.788447 |           0.789106 |                -0.001277 |        290 |        546 |  0.592996 |
| Base + Trends catálogo (z)           |                 5 |   0.788900 |           0.789900 |                -0.000483 |        296 |        545 |  0.591106 |
| Base + Trends 20 términos (z)        |                20 |   0.790110 |           0.789632 |                -0.000751 |        296 |        549 |  0.596205 |
| Base + Trends 20 índices originales  |                20 |   0.781773 |           0.789183 |                -0.001200 |        296 |        547 |  0.576426 |
| Base + NASA + Trends 20 términos (z) |                33 |   0.789175 |           0.788699 |                -0.001684 |        292 |        545 |  0.592194 |

`n_added_columns` incluye el indicador de ausencia de NASA. `tp_top10` y `tp_top20` son casos de churn identificados en la selección mensual, sumados sobre los 16 meses. `ap_mean` es la precisión promedio (AP) media de los cuatro bloques.

## Consistencia temporal

AUC por bloque (0 a 3, en orden cronológico):

| variant             |        0 |        1 |        2 |        3 |
|:--------------------|---------:|---------:|---------:|---------:|
| transaccional       | 0.785349 | 0.794934 | 0.781867 | 0.797368 |
| mas_nasa            | 0.785275 | 0.792948 | 0.780129 | 0.795436 |
| mas_trends_catalogo | 0.784227 | 0.794798 | 0.780867 | 0.795709 |
| mas_trends          | 0.789752 | 0.794783 | 0.777875 | 0.798032 |
| mas_trends_indices  | 0.788944 | 0.792934 | 0.749996 | 0.795217 |
| mas_nasa_trends     | 0.789671 | 0.793362 | 0.776536 | 0.797129 |

Meses con AUC superior a la base, de 16:

| variant             |   meses_mejora |
|:--------------------|---------------:|
| mas_nasa            |              5 |
| mas_nasa_trends     |              4 |
| mas_trends          |              6 |
| mas_trends_catalogo |              6 |
| mas_trends_indices  |              6 |
| transaccional       |              0 |

## Componentes del ensemble

| variant             | model   |   auc_mean |   auc_monthly_mean |   tp_top10 |
|:--------------------|:--------|-----------:|-------------------:|-----------:|
| transaccional       | logreg  |   0.787625 |           0.787839 |        293 |
| transaccional       | xgboost |   0.788335 |           0.789059 |        298 |
| mas_nasa            | logreg  |   0.785467 |           0.786085 |        281 |
| mas_nasa            | xgboost |   0.787707 |           0.788612 |        299 |
| mas_trends_catalogo | logreg  |   0.787266 |           0.788239 |        293 |
| mas_trends_catalogo | xgboost |   0.787225 |           0.788475 |        295 |
| mas_trends          | logreg  |   0.788062 |           0.787944 |        292 |
| mas_trends          | xgboost |   0.787612 |           0.788177 |        296 |
| mas_trends_indices  | logreg  |   0.761784 |           0.788016 |        290 |
| mas_trends_indices  | xgboost |   0.788601 |           0.788792 |        294 |
| mas_nasa_trends     | logreg  |   0.785877 |           0.785863 |        286 |
| mas_nasa_trends     | xgboost |   0.787508 |           0.788322 |        294 |

## Variables incorporadas

NASA POWER añade 13 columnas: diez agregados climáticos, dos cambios interanuales y un indicador de ausencia.

| Variable | Definición |
|---|---|
| `nasa_temperatura_media_c` | Media mensual de temperatura diaria media, °C |
| `nasa_temperatura_maxima_media_c` | Media mensual de las máximas diarias, °C |
| `nasa_temperatura_minima_media_c` | Media mensual de las mínimas diarias, °C |
| `nasa_temperatura_maxima_c` | Máxima de las máximas diarias del mes, °C |
| `nasa_temperatura_minima_c` | Mínima de las mínimas diarias del mes, °C |
| `nasa_humedad_relativa_pct` | Humedad relativa media mensual, % |
| `nasa_precipitacion_total_mm` | Suma de precipitación diaria del mes, mm |
| `nasa_precipitacion_maxima_diaria_mm` | Mayor precipitación diaria del mes, mm |
| `nasa_dias_lluvia_ge1mm` | Número de días con precipitación de al menos 1 mm |
| `nasa_viento_m_s` | Velocidad media mensual del viento, m/s |
| `nasa_temperatura_delta12_c` | Temperatura media menos la del mismo mes del año anterior, °C |
| `nasa_precipitacion_delta12_mm` | Precipitación mensual menos la del mismo mes del año anterior, mm |
| `nasa_sin_dato` | 1 si falta alguno de los diez agregados climáticos del mes |

Los agregados proceden de días en tiempo solar local (LST). Las diferencias anuales se unen por el mes exacto del año anterior. La fuente local cubre 148 provincias, 107 celdas y 120 meses entre enero de 2016 y diciembre de 2025.

Google Trends contiene 20 términos mensuales, enero de 2016–febrero de 2026:

- Catálogo (variante de cinco términos): venta por catalogo, catalogo, catalogo de ropa, vender por catalogo, glamour.
- Los otros 15: yanbal, natura, unique, esika, belcorp, shein, temu, gamarra, ropa por mayor, trabajo desde casa, trabajo, prestamo, ofertas, ropa, cts.

Las variables `trends_<término>_zexp` contienen el z-score expansivo. La variante `mas_trends_indices` contiene `trends_<término>_indice` (0–100). No se mezclan índices y z-scores en la misma variante. `mas_nasa_trends` combina las 13 columnas NASA con los 20 z-scores.

## Cobertura

| partition   |   rows |   nasa_missing_location_or_month |   nasa_missing_yearly_change |   trends_missing_index |   trends_missing_zexp |
|:------------|-------:|---------------------------------:|-----------------------------:|-----------------------:|----------------------:|
| development |  28311 |                              454 |                         2418 |                      0 |                 23426 |
| validation  |   4379 |                               42 |                           42 |                      0 |                     0 |

La ausencia de cambios climáticos anuales al inicio se debe a la falta del año previo. Los z-scores faltan en meses iniciales y en series sin dispersión. En validación no falta ningún índice ni z-score de Trends; faltan datos climáticos en 42 observaciones (0,96 %).

## Interpretación y límites

- La comparación principal para priorizar vendedoras es el AUC medio dentro del mes, junto con los casos detectados al mismo presupuesto mensual. El AUC de bloques de cuatro meses también compara observaciones de meses diferentes.
- NASA cambia por provincia y mes, aunque provincias que comparten celda meteorológica comparten valores. La capital provincial aproxima la ubicación; no es una medición individual ni el promedio de toda la provincia.
- Trends nacional es igual para todas las vendedoras del mes. Por sí solo no distingue a dos vendedoras de ese mes; puede afectar el nivel de probabilidades y las interacciones en árboles. Reentrenar también cambia los coeficientes de las variables transaccionales.
- **No se usó `google_trends_pe_regiones.csv` como predictor.** Tiene 25 departamentos y dos términos, pero ninguna fecha mensual. El código de extracción existente solicita el resumen de 2016-01 a 2026-02; asignarlo a años anteriores introduciría información posterior al corte. No hay series departamento × mes en este archivo.
- Las búsquedas son índices de interés relativo, no volúmenes absolutos. Cada término se descargó por separado: sus niveles no deben compararse como cantidades de búsquedas entre términos. Los índices originales están normalizados en toda la ventana de descarga.
- El z-score expansivo usa únicamente valores hasta el mes de referencia: `(valor − media histórica) / desviación estándar histórica`, con un mínimo de 12 meses. Cancela un factor multiplicativo común, pero no corrige el redondeo, muestreo ni revisiones históricas de la fuente. Si la serie es constante, queda nulo.
- Temu tiene una historia inicial de ceros: su z-score queda ausente en 23.426 de las 28.311 filas de desarrollo y recién está disponible para observaciones de agosto de 2023 (referencia mayo de 2023). La ausencia de dispersión no se presenta como una señal negativa de interés. Las columnas sin datos en un entrenamiento quedan en cero mediante el imputador; ese modelo no puede aprender su efecto.
- Para ambas fuentes se asumió un rezago de **tres meses**: por ejemplo, enero de 2025 usa octubre de 2024. No se buscó el mejor rezago. Son descargas históricas actuales, sin versiones ni fechas de publicación de cada valor: el resultado sigue siendo exploratorio y no acredita disponibilidad histórica. La provincia proviene del maestro actual.
- Imputación por mediana ajustada exclusivamente en cada entrenamiento; no se descartan observaciones con datos faltantes. No se incorporan códigos geográficos, sexo, edad, antigüedad ni los indicadores económicos del Excel a estas variantes.
- Se reutilizan validaciones ya usadas para desarrollar el modelo. No es un holdout final independiente ni una demostración de significancia estadística. Los conteos son observaciones vendedora–mes, no necesariamente personas distintas.


## Archivos y reproducción

- Fuentes locales: `data/external/nasa_power/clima_mensual_provincias.parquet`, `protocol.json`, `diccionario.json`; `data/external/google_trends/google_trends_pe.csv`.
- Modelo base: `reports/master_features_comparison_v1/`. Parámetros: `reports/master_features_v1/reconstructed_tuning/`.
- Resultados, predicciones por fold y componente, hashes, cobertura, fechas de entrenamiento y lista exacta de variables: `reports/climate_trends_v1/`.
- Ejecutar `.venv/bin/python -m scripts.climate_trends_ablation`. No descarga fuentes ni busca hiperparámetros; los resultados terminados se reutilizan únicamente si coincide el protocolo.
- [Notebook ejecutado](../05_modelling/11_ablacion_nasa_google_trends.ipynb).

Se verificaron las uniones temporales, invariancia ante datos futuros, normalización expansiva, parámetros, pesos del ensemble y población de validación. El clima reconstruido coincide con el cache original en las 28.311 filas de desarrollo y las 13 variables. Las fuentes y el modelo anterior se conservan.
