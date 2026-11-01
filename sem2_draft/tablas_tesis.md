# Tablas para la tesis

Números copiados de `reports/variables.md`, `reports/modelado.md`, `reports/evaluacion.md` y
`reports/interpretacion.md`. Si se re-ejecuta un notebook, revisar contra esos reportes.

## Tabla 1. Variables utilizadas en el estudio (42)

| Tipo | Categoría | Variaciones de la variable | N.º | Fuente |
|---|---|---|:--:|---|
| Longitud, recencia, frecuencia y valor monetario (LRFM) | Longitud | Meses con compra en todo el historial previo a t (***compras_hist***). | 1 | Datos de pedidos |
| | Recencia | Meses desde la compra previa (*meses_desde_compra_previa*); recencia ponderada por la actividad reciente (*recencia_norm*). | 2 | Datos de pedidos |
| | Frecuencia | Meses activos y n.º de pedidos por ventana u3/u6/u12 (*meses_activos_u{3,6,12}*, *n_ped_u{3,6,12}*, con ***n_ped_u12*** seleccionada); intensidad = pedidos por mes activo (*intensidad_u3*); actividad reciente vs. histórica (*tasa_act_reciente_vs_hist*). | 8 | Datos de pedidos |
| | Valor monetario | Monto por ventana u3/u6/u12 (*monto_u{3,6,12}*, con ***monto_u12*** seleccionada); media, desviación estándar y coeficiente de variación a 12 m (*monto_mean_u12*, *monto_std_u12*, ***monto_cv_u12***); último mes vs. media (*monto_ult_vs_media*); ticket promedio u3/u12 (***ticket_prom_u3***, *ticket_prom_u12*). | 9 | Datos de pedidos |
| Dinámica temporal (tendencias y deltas) | Tendencia normalizada | Cambio relativo de los últimos 3 meses vs. los 3 previos, en [−1, 1], para monto y n.º de pedidos (*tend_monto_u3_vs_prev3*, *tend_nped_u3_vs_prev3*). | 2 | Datos de pedidos |
| | Delta features | Diferencia del mes t respecto al mes t−n, con n ∈ {1, 3, 6, 9, 12}, para monto (*d_monto_m{1,3,6,9,12}*) y n.º de pedidos (*d_nped_m{1,3,6,9,12}*). | 10 | Datos de pedidos |
| Features acumulativas (relación de largo plazo) | Volumen histórico | Monto, n.º de pedidos y n.º de productos acumulados sobre toda la historia hasta t (*monto_acum*, *n_ped_acum*, *n_prod_acum*). | 3 | Datos de pedidos |
| | Ratios históricos | Ticket histórico (*ticket_acum*), monto por producto (*monto_por_prod_acum*) y monto por mes activo (*monto_mensual_acum*). | 3 | Datos de pedidos |
| Diversidad de producto | Amplitud de catálogo | Productos distintos a 12 m (***n_prod_u12***); máx. de categorías por mes a 12 m y en toda la historia (*n_cat_max_u12*, *n_cat_max_acum*); productos por pedido (*basket_size_u12*). | 4 | Datos de pedidos + maestro de productos |

uN = últimos N meses incluyendo el mes de observación t. En **negrita** las 6 variables seleccionadas.
Se excluyen los datos maestros de la vendedora (sexo, edad, antigüedad, tipo y ubicación) porque solo
se guarda su valor vigente, sin historial (§4.2).

## Tabla 2. Variables seleccionadas (6)

Selección con XGBoost de referencia (parámetros fijos): importancia por permutación y luego ablación
forward; se conserva la variable si el AUC-ROC promedio en validación temporal sube más de 0.001.

| Orden | Variable | Caída de AUC-ROC al permutarla (promedio 4 bloques) | AUC-ROC promedio en validación temporal (4 bloques) al agregarla | % del \|SHAP\| medio en test out-of-time | Efecto sobre el riesgo de churn |
|--:|---|:--:|:--:|:--:|---|
| 1 | compras_hist | 0.0089 | 0.6939 | 20.2 | Más historia → menor riesgo |
| 2 | n_prod_u12 | 0.0071 | 0.7760 | 16.0 | Más productos → menor riesgo |
| 3 | monto_cv_u12 | 0.0044 | 0.7804 | 13.8 | Más irregularidad → mayor riesgo |
| 4 | n_ped_u12 | 0.0040 | 0.7825 | 21.9 | Más pedidos → menor riesgo |
| 5 | monto_u12 | 0.0035 | 0.7844 | 20.2 | Más monto → menor riesgo |
| 6 | ticket_prom_u3 | 0.0013 | 0.7879 | 7.9 | Mayor ticket → menor riesgo |

Con el XGBoost de referencia: 42 variables → AUC-ROC 0.7828; 6 variables → 0.7879.

## Tabla 3. Comparación de modelos tras la optimización

| Modelo | Variables | Ventana | AUC-ROC promedio en validación temporal (4 bloques) | Desv. est. del AUC-ROC (4 bloques) | Lift top-10 % promedio mensual (validación) | AUC-ROC en test out-of-time |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **LightGBM** | 6 | Toda la historia | **0.7911** | 0.0084 | **2.31** | **0.7931** |
| XGBoost | 6 | Toda la historia | 0.7910 | 0.0094 | 2.28 | — |
| CatBoost | 6 | Toda la historia | 0.7904 | 0.0090 | 2.28 | — |
| Random Forest | 6 | Toda la historia | 0.7898 | 0.0086 | 2.29 | — |
| Regresión logística | 42 | 36 meses | 0.7871 | 0.0076 | 2.24 | — |

Validación temporal: 4 bloques de 4 meses entre feb-2024 y may-2025; cada bloque se entrena solo con
la historia anterior, con 6 meses de brecha. Hiperparámetros con Optuna (100 trials por modelo);
variables y ventana elegidas después, con los hiperparámetros fijos. Test out-of-time: dic-2025 →
mar-2026 (879 filas), evaluado una sola vez con el modelo final; por protocolo, los demás modelos no se
evalúan ahí.
