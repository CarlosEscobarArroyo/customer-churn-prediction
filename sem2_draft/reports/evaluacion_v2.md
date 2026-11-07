# Evaluación v2

Fecha UTC: 2026-11-06. Experimento de modelado: b2ad959cb4c424b4. Modelo: xgboost__seleccionadas (6 variables: compras_hist, n_prod_u12, monto_cv_u12, n_ped_u12, monto_u12, ticket_prom_u3), estrategia sin_tratamiento, ventana de 48 meses.

Entrenamiento: 2021-06 → 2025-05 (12,240 filas). Configuración fija desde la fase 04; reentrenada con xgboost 3.2.0 (el modelado usó 3.4.1).

## Verificación con GroupKFold

| validacion | auc_promedio | desviacion_auc | lift_promedio |
| --- | --- | --- | --- |
| temporal_4_bloques (fase 04) | 0.7930 | 0.0098 | 2.3025 |
| group_kfold | 0.7785 | 0.0083 | 2.2361 |
| stratified_kfold | 0.7808 | 0.0113 | 2.2283 |

GroupKFold − StratifiedKFold: -0.0023 de AUC. Mismo pool que el modelo final (ventana de 48 meses). Solo se interpreta la diferencia: los folds aleatorios mezclan épocas y entrenan con meses futuros.

## Test OOT

2025-12 → 2026-03: 879 filas, 524 vendedoras, prevalencia 0.2765. AUC 0.7844 (IC 95 %: 0.7480–0.8165), por bootstrap de vendedoras con 2.000 repeticiones.

| metrica | valor | ic95_inferior | ic95_superior |
| --- | --- | --- | --- |
| AUC-ROC | 0.7844 | 0.7480 | 0.8165 |
| PR-AUC | 0.5249 | 0.4644 | 0.5966 |
| Lift 10 % mensual | 2.1219 |  |  |

### Por mes

| mes | filas | tasa_churn | auc | lift_10 |
| --- | --- | --- | --- | --- |
| 2025-12 | 273 | 0.3114 | 0.7750 | 2.2601 |
| 2026-01 | 192 | 0.2708 | 0.7746 | 1.5547 |
| 2026-02 | 197 | 0.2640 | 0.7887 | 1.9939 |
| 2026-03 | 217 | 0.2488 | 0.8126 | 2.6790 |

### Según porcentaje contactado

| porcentaje_contactado | contactadas | precision | recall | lift |
| --- | --- | --- | --- | --- |
| 5 | 44 | 0.5227 | 0.0947 | 1.8909 |
| 10 | 88 | 0.5909 | 0.2140 | 2.1375 |
| 15 | 132 | 0.5985 | 0.3251 | 2.1649 |
| 20 | 176 | 0.5739 | 0.4156 | 2.0758 |
| 25 | 220 | 0.5545 | 0.5021 | 2.0059 |
| 30 | 264 | 0.5455 | 0.5926 | 1.9731 |
| 40 | 352 | 0.4830 | 0.6996 | 1.7470 |
| 50 | 440 | 0.4455 | 0.8066 | 1.6113 |

### Matriz de confusión

| corte | VP | FP | FN | VN | precision | recall | porcentaje_contactado |
| --- | --- | --- | --- | --- | --- | --- | --- |
| p >= 0.5 | 111 | 88 | 132 | 548 | 0.5578 | 0.4568 | 22.6394 |
| decil superior (p >= 0.617) | 52 | 36 | 191 | 600 | 0.5909 | 0.2140 | 10.0114 |

Curvas: `05_evaluacion/evaluacion_v2_oot.png`.
