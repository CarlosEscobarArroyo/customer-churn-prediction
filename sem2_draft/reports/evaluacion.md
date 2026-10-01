# Evaluación (sem2)

Generado por `05_evaluacion/evaluacion.ipynb`. Modelo final: **lightgbm+xgboost+catboost** (6 variables:
`compras_hist`, `n_prod_u12`, `monto_cv_u12`, `n_ped_u12`, `monto_u12`, `ticket_prom_u3`), según `04_modelado/modelo_final.json`.

## 1 · Verificación complementaria (pool de desarrollo, 29,183 filas)
|                                    |    auc |    std |   lift10 | folds                                    |
|:-----------------------------------|-------:|-------:|---------:|:-----------------------------------------|
| temporal 4 bloques (selección, 04) | 0.7912 | 0.0089 |   2.344  | [0.8001, 0.7846, 0.7999, 0.7803]         |
| GroupKFold(5) por vendedora        | 0.7342 | 0.0041 |   2.0788 | [0.7278, 0.7332, 0.7407, 0.7344, 0.7351] |
| StratifiedKFold(5) sin grupos      | 0.734  | 0.0072 |   2.0804 | [0.7468, 0.7283, 0.7279, 0.7368, 0.7302] |

GroupKFold − StratifiedKFold: +0.0002 de AUC.

## 2 · Test out-of-time (una sola evaluación)
OOT: 2025-12 → 2026-03, 879 filas, prevalencia 0.276.
Entrenamiento: todo el pool de desarrollo (hasta 2025-05, brecha de 6 meses).

| métrica | valor |
|---|---:|
| AUC-ROC | 0.7926 |
| PR-AUC | 0.5681 |
| Lift decil superior (media mensual) | 2.27 |

Por miembro:
|          |    auc |   pr_auc |
|:---------|-------:|---------:|
| lightgbm | 0.793  |   0.5733 |
| xgboost  | 0.7939 |   0.5575 |
| catboost | 0.7897 |   0.5438 |

Por mes:
| mes_obs   |   n |   churn |    auc |   lift10 |
|:----------|----:|--------:|-------:|---------:|
| 2025-12   | 273 |  0.3114 | 0.7901 |   2.2601 |
| 2026-01   | 192 |  0.2708 | 0.7817 |   2.332  |
| 2026-02   | 197 |  0.264  | 0.7908 |   2.1933 |
| 2026-03   | 217 |  0.2488 | 0.8135 |   2.2963 |

Precisión / recall / lift por % de base contactada:
|   pct_contactado |   n |   precision |   recall |   lift |
|-----------------:|----:|------------:|---------:|-------:|
|                5 |  44 |       0.659 |    0.119 |  2.384 |
|               10 |  88 |       0.625 |    0.226 |  2.261 |
|               15 | 132 |       0.644 |    0.35  |  2.329 |
|               20 | 176 |       0.591 |    0.428 |  2.137 |
|               25 | 220 |       0.573 |    0.519 |  2.072 |
|               30 | 264 |       0.545 |    0.593 |  1.973 |
|               40 | 352 |       0.486 |    0.704 |  1.757 |
|               50 | 440 |       0.464 |    0.84  |  1.677 |

Matrices de confusión:
| umbral                      |   TP |   FP |   FN |   TN |   precision |   recall |   contactados_% |
|:----------------------------|-----:|-----:|-----:|-----:|------------:|---------:|----------------:|
| p >= 0.5                    |  130 |   98 |  113 |  538 |       0.57  |    0.535 |          25.939 |
| decil superior (p >= 0.635) |   55 |   33 |  188 |  603 |       0.625 |    0.226 |          10.011 |

Lectura: el AUC de selección (0.7912, 4 bloques de 2024-02 a 2025-05) y el AUC OOT (0.7926) difieren en
+0.0014; la diferencia combina el sesgo de selección y la dificultad propia del período.
Precisión y recall a p ≥ 0.5 dependen del umbral; para comparar modelos, la métrica es el AUC y, para campañas, el lift.
