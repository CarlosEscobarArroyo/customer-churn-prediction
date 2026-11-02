# Variables del ensemble

El ensemble combina regresión logística (36 meses de entrenamiento) y XGBoost (toda la historia permitida). **Ambos utilizan las mismas 42 variables transaccionales.**

Convenciones: `u3/u6/u12` = últimos 3, 6 o 12 meses, incluyendo el mes observado; `acum` = historia disponible hasta ese mes; `m1/m3/m6/m9/m12` = comparación con 1, 3, 6, 9 o 12 meses atrás.

| Grupo | Variables | Qué representan |
|---|---|---|
| Frecuencia (6) | `meses_activos_u3`, `meses_activos_u6`, `meses_activos_u12`, `n_ped_u3`, `n_ped_u6`, `n_ped_u12` | Meses con compra y cantidad de pedidos por ventana. |
| Montos y variación (7) | `monto_u3`, `monto_u6`, `monto_u12`, `monto_mean_u12`, `monto_std_u12`, `monto_cv_u12`, `monto_ult_vs_media` | Montos comprados, promedio mensual, dispersión y monto del mes observado respecto a su media. |
| Recencia e historia (2) | `meses_desde_compra_previa`, `compras_hist` | Meses desde la compra anterior y número de meses con compra previos al observado. |
| Productos y categorías (2) | `n_prod_u12`, `n_cat_max_u12` | Suma de productos distintos por mes y máximo mensual de categorías en 12 meses. |
| Tendencias (2) | `tend_monto_u3_vs_prev3`, `tend_nped_u3_vs_prev3` | Cambio normalizado de montos y pedidos entre los últimos tres meses y los tres anteriores. |
| Historia acumulada (7) | `monto_acum`, `n_ped_acum`, `n_prod_acum`, `n_cat_max_acum`, `ticket_acum`, `monto_por_prod_acum`, `monto_mensual_acum` | Montos, pedidos, suma de productos distintos mensuales, máximo mensual de categorías y razones históricas por pedido, producto y mes activo. |
| Cambios de monto (5) | `d_monto_m1`, `d_monto_m3`, `d_monto_m6`, `d_monto_m9`, `d_monto_m12` | Monto del mes observado menos el monto del mes de referencia. |
| Cambios de pedidos (5) | `d_nped_m1`, `d_nped_m3`, `d_nped_m6`, `d_nped_m9`, `d_nped_m12` | Pedidos del mes observado menos los pedidos del mes de referencia. |
| Razones derivadas (6) | `ticket_prom_u12`, `ticket_prom_u3`, `intensidad_u3`, `basket_size_u12`, `recencia_norm`, `tasa_act_reciente_vs_hist` | Ticket por pedido; pedidos por mes activo; productos por pedido; recencia ajustada por actividad; actividad reciente relativa. |

Las seis razones derivadas se calculan así:

- `ticket_prom_u12` = `monto_u12 / n_ped_u12`.
- `ticket_prom_u3` = `monto_u3 / n_ped_u3`.
- `intensidad_u3` = `n_ped_u3 / meses_activos_u3`.
- `basket_size_u12` = `n_prod_u12 / n_ped_u12`.
- `recencia_norm` = `meses_desde_compra_previa × meses_activos_u12 / 12`.
- `tasa_act_reciente_vs_hist` = `(meses_activos_u3 / 3) / (meses_activos_u12 / 12)`.

**No son entradas del modelo:** identificadores, fechas, `churn`, sexo, edad, antigüedad, tipo de vendedora, departamento ni provincia. `churn` es la etiqueta a predecir; identificadores y fechas organizan las particiones y las predicciones.

Fuentes: [SQL de construcción](../00_dataset_construction/qry_churn.sql) y [notebook del ensemble](../05_modelling/07_ventanas_recientes_ensemble.ipynb).
