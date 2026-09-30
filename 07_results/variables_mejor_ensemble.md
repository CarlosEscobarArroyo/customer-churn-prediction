# Variables y resultados del mejor ensemble temporal

Documento elaborado el 20 de septiembre de 2026 a partir de las salidas guardadas del experimento del 19 de septiembre de 2026.

**Comparación posterior:** la [ablación de datos maestros](ablacion_datos_maestros.md) reentrenó la base y probó sexo, edad, antigüedad y ubicación con parámetros fijos. Este documento conserva las variables y métricas históricas de la corrida del 19 de septiembre; el informe de ablación contiene los resultados de la nueva ejecución.

## Modelo y criterio de elección

El candidato `mean_logreg_xgboost` obtuvo el mayor **ROC AUC temporal medio** entre los candidatos del [notebook 07](../05_modelling/07_ventanas_recientes_ensemble.ipynb):

| Componente | Ventana de filas de entrenamiento | Peso |
|---|---|---:|
| Regresión logística | Últimos 36 meses permitidos en cada entrenamiento | 50 % |
| XGBoost | Toda la historia permitida en cada entrenamiento | 50 % |

El score combinado es `0,5 × score_logreg + 0,5 × score_xgboost`. Ambos componentes utilizan las mismas **42 variables numéricas: 36 provenientes del SQL y 6 derivadas en Python**. La logística aplica `StandardScaler`; XGBoost recibe las variables sin ese escalado. Los scores no se han validado como probabilidades calibradas.

La ventana de 36 meses recorta las filas utilizadas para ajustar logística; no recorta la historia con la que se construyen las variables de cada fila. El ajuste exportado se realizó con datos hasta enero de 2025.

## Resultados registrados

| Métrica | Resultado |
|---|---:|
| ROC AUC medio de los cuatro bloques temporales | **0,78961** |
| Desviación estándar del AUC entre bloques (`ddof=0`) | 0,00709 |
| Average precision media de los cuatro bloques | 0,59533 |
| Lift medio mensual del top 10 % | 2,27715× |
| Mejor modelo individual: Random Forest con toda la historia, AUC medio | 0,78833 |
| Mejora del ensemble frente a ese modelo individual | +0,00129 de AUC |

La evaluación comprende **4.379 observaciones vendedora-mes**, en cuatro bloques de cuatro meses entre octubre de 2023 y enero de 2025, con seis meses completos de separación respecto del entrenamiento de cada bloque. Una vendedora puede aparecer en varios meses.

Los siguientes resultados de priorización se registraron en el [resumen de la sesión](resumen_sesion_2026-09-19.md). Se selecciona el mayor score dentro de cada mes, redondeando hacia abajo:

| Proporción priorizada mensual | Contactos mensuales promedio, redondeados | Precisión agregada | Recall agregado | Lift mensual medio |
|---|---:|---:|---:|---:|
| Top 10 % | 27 | 69,1 % | 22,1 % | 2,28× |
| Top 20 % | 54 | 63,2 % | 40,8 % | 2,08× |
| Top 30 % | 82 | 57,6 % | 55,8 % | 1,89× |

Precisión y recall acumulan observaciones; el lift promedia los valores mensuales. El criterio de elección fue AUC: el promedio de los cuatro algoritmos tiene un lift top 10 % mensual mayor (2,31948×), pero un AUC ligeramente menor (0,78952).

**Alcance:** son resultados de selección sobre validaciones reutilizadas para ajustar hiperparámetros y elegir ventanas y ensembles. No constituyen una evaluación final independiente ni demuestran superioridad estadística. El OOT histórico no se evaluó en este experimento.

## Definiciones comunes

- Cada fila representa una vendedora que compró en el mes observado `t` y tiene al menos un mes anterior con compra.
- El objetivo `churn = 1` significa ausencia de compras en los seis meses siguientes, de `t+1` a `t+6`.
- `u3`, `u6` y `u12` corresponden a ventanas de 3, 6 y 12 meses calendario que incluyen `t`. Se calculan sobre un panel mensual denso desde la primera compra, con ceros en los meses sin actividad.
- `prev3` corresponde a los meses `t-5` a `t-3`; `acum` incluye toda la historia disponible hasta `t`.
- Los montos conservan la unidad monetaria del dataset. Los conteos de productos acumulados suman los productos distintos de cada mes: un mismo producto puede contarse otra vez en meses diferentes.
- La numeración siguiente reproduce el orden reconstruido de entrada al modelo; no es un ranking de importancia.

## Las 42 variables

### Frecuencia de compra

| Nº | Variable | Definición |
|---:|---|---|
| 1 | `meses_activos_u3` | Meses con al menos una compra en los últimos 3 meses. |
| 2 | `meses_activos_u6` | Meses con al menos una compra en los últimos 6 meses. |
| 3 | `meses_activos_u12` | Meses con al menos una compra en los últimos 12 meses. |
| 4 | `n_ped_u3` | Pedidos de los últimos 3 meses. |
| 5 | `n_ped_u6` | Pedidos de los últimos 6 meses. |
| 6 | `n_ped_u12` | Pedidos de los últimos 12 meses. |

### Monto y variabilidad

| Nº | Variable | Definición |
|---:|---|---|
| 7 | `monto_u3` | Suma de montos de los últimos 3 meses. |
| 8 | `monto_u6` | Suma de montos de los últimos 6 meses. |
| 9 | `monto_u12` | Suma de montos de los últimos 12 meses. |
| 10 | `monto_mean_u12` | Promedio del monto mensual en la ventana de 12 meses disponible; incluye meses sin compra. |
| 11 | `monto_std_u12` | Desviación estándar muestral del monto mensual en esa ventana (`STDDEV` en SQL). |
| 12 | `monto_cv_u12` | `monto_std_u12 / monto_mean_u12`. |
| 13 | `monto_ult_vs_media` | Monto del mes observado dividido por `monto_mean_u12`. |

### Recencia, historia y diversidad

| Nº | Variable | Definición |
|---:|---|---|
| 14 | `meses_desde_compra_previa` | Meses entre `t` y el último mes con compra anterior a `t`. |
| 15 | `compras_hist` | Número de meses con compra anteriores a `t`; excluye el mes actual. |
| 16 | `n_prod_u12` | Suma del número de productos distintos comprados en cada mes de la ventana de 12 meses. |
| 17 | `n_cat_max_u12` | Máximo número de categorías distintas compradas en un mes de la ventana de 12 meses. |

### Tendencias

| Nº | Variable | Definición |
|---:|---|---|
| 18 | `tend_monto_u3_vs_prev3` | `(monto_u3 − monto_prev3) / (monto_u3 + monto_prev3)`. |
| 19 | `tend_nped_u3_vs_prev3` | `(n_ped_u3 − nped_prev3) / (n_ped_u3 + nped_prev3)`. |

### Acumulados históricos

| Nº | Variable | Definición |
|---:|---|---|
| 20 | `monto_acum` | Monto acumulado desde el inicio de la historia disponible hasta `t`. |
| 21 | `n_ped_acum` | Pedidos acumulados hasta `t`. |
| 22 | `n_prod_acum` | Suma histórica de productos distintos por mes hasta `t`. |
| 23 | `n_cat_max_acum` | Máximo histórico de categorías distintas compradas en un mismo mes, hasta `t`. |
| 24 | `ticket_acum` | `monto_acum / n_ped_acum`. |
| 25 | `monto_por_prod_acum` | `monto_acum / n_prod_acum`. |
| 26 | `monto_mensual_acum` | Monto acumulado dividido por los meses con compra hasta `t`, incluido el mes actual. |

### Diferencias respecto de meses anteriores

Se compara el monto o número de pedidos de `t` con el de un mes específico anterior, no con una suma móvil.

| Nº | Variable | Definición |
|---:|---|---|
| 27 | `d_monto_m1` | Monto de `t` menos monto de `t−1`. |
| 28 | `d_monto_m3` | Monto de `t` menos monto de `t−3`. |
| 29 | `d_monto_m6` | Monto de `t` menos monto de `t−6`. |
| 30 | `d_monto_m9` | Monto de `t` menos monto de `t−9`. |
| 31 | `d_monto_m12` | Monto de `t` menos monto de `t−12`. |
| 32 | `d_nped_m1` | Pedidos de `t` menos pedidos de `t−1`. |
| 33 | `d_nped_m3` | Pedidos de `t` menos pedidos de `t−3`. |
| 34 | `d_nped_m6` | Pedidos de `t` menos pedidos de `t−6`. |
| 35 | `d_nped_m9` | Pedidos de `t` menos pedidos de `t−9`. |
| 36 | `d_nped_m12` | Pedidos de `t` menos pedidos de `t−12`. |

### Variables derivadas en Python

Son las seis columnas añadidas por `engineer()` en [temporal_optuna.py](../scripts/temporal_optuna.py).

| Nº | Variable | Fórmula |
|---:|---|---|
| 37 | `ticket_prom_u12` | `monto_u12 / n_ped_u12`. |
| 38 | `ticket_prom_u3` | `monto_u3 / n_ped_u3`. |
| 39 | `intensidad_u3` | `n_ped_u3 / meses_activos_u3`: pedidos por mes activo reciente. |
| 40 | `basket_size_u12` | `n_prod_u12 / n_ped_u12`: productos distintos por mes acumulados, divididos por pedidos. |
| 41 | `recencia_norm` | `meses_desde_compra_previa × meses_activos_u12 / 12`. |
| 42 | `tasa_act_reciente_vs_hist` | `(meses_activos_u3 / 3) / (meses_activos_u12 / 12)`. |

Las divisiones de `engineer()` devuelven cero cuando el denominador es cero. `TransactionFeatures` imputa a cero los nulos estructurales de las dos tendencias, `monto_cv_u12`, `monto_ult_vs_media`, `monto_por_prod_acum` y las diez diferencias mensuales. Verifica que las entradas resultantes sean numéricas y finitas.

## Exclusiones y diferencia con los datos actuales

No entran como predictores `id_vendedor`, `mes_obs`, `mes_rank`, `churn`, `sexo`, `edad`, `antiguedad_meses`, `tipo_vendedor`, `departamento` ni `provincia`.

El CSV local actual `data/processed/churn_dataset.csv` tiene **53 columnas**. Contiene siete variables adicionales que no pertenecen al esquema de 42 variables de la corrida documentada:

- `camp_saltadas`
- `camp_part_u12`
- `tasa_camp_u3`
- `tasa_camp_u6`
- `tasa_camp_u12`
- `pct_directo_u12`
- `es_nueva_u12`

El transformador actual conserva todas las columnas transaccionales que recibe. Aplicarlo al CSV actual produciría **49 predictores**, no 42. Por tanto, los resultados de AUC anteriores no deben atribuirse a una corrida con esas siete variables adicionales.

## Trazabilidad y verificación

- Las métricas principales y la composición ganadora se contrastaron con las salidas guardadas del [notebook 07](../05_modelling/07_ventanas_recientes_ensemble.ipynb), celdas de comparación y resumen. El [notebook 06](../05_modelling/06_optuna_validacion_temporal.ipynb) registra 42 variables en los mejores trials.
- El orden y la lista de variables se reconstruyeron del `SELECT` final de `00_dataset_construction/qry_churn.sql` en el commit **`e180c2b`**, que contiene el notebook 07 ejecutado, y de `TransactionFeatures` / `engineer()`. Ese SQL tiene 46 columnas: 4 de identificación/objetivo, 6 maestras y 36 transaccionales; Python añade las otras 6.
- Las definiciones se contrastaron con el SQL y las fórmulas de Python. El [SQL actual](../00_dataset_construction/qry_churn.sql) conserva estas definiciones y añade las variables de campañas indicadas arriba.
- El notebook registra la exportación y recarga de `reports/recent_windows_v1/best_candidate.joblib`. Ese binario y los directorios `reports/recent_windows_v1/` y `reports/optuna_temporal_v1/` **no están presentes en esta copia local**. Esta revisión no recargó el ensemble, no repitió el entrenamiento y no recalculó sus métricas desde predicciones; utilizó los resultados persistidos en los notebooks y el resumen de sesión.
