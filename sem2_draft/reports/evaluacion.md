# Evaluación (sem2)

Generado por `05_evaluacion/evaluacion.ipynb`. Modelo final: **lightgbm** (6 variables:
`compras_hist`, `n_prod_u12`, `monto_cv_u12`, `n_ped_u12`, `monto_u12`, `ticket_prom_u3`), según `04_modelado/modelo_final.json`.

## 1 · Verificación complementaria (pool de desarrollo, 29,183 filas)
|                                    |    auc |    std |   lift10 | folds                                    |
|:-----------------------------------|-------:|-------:|---------:|:-----------------------------------------|
| temporal 4 bloques (selección, 04) | 0.7911 | 0.0084 |   2.312  | [0.7994, 0.7847, 0.7993, 0.7808]         |
| GroupKFold(5) por vendedora        | 0.7338 | 0.005  |   2.0817 | [0.7357, 0.7259, 0.7406, 0.7311, 0.7357] |
| StratifiedKFold(5) sin grupos      | 0.7334 | 0.0074 |   2.0843 | [0.7466, 0.7273, 0.7275, 0.7364, 0.7292] |

GroupKFold − StratifiedKFold: +0.0004 de AUC.

## 2 · Test out-of-time (una sola evaluación)
OOT: 2025-12 → 2026-03, 879 filas, prevalencia 0.276.
Entrenamiento: todo el pool de desarrollo (hasta 2025-05, brecha de 6 meses).

| métrica | valor |
|---|---:|
| AUC-ROC | 0.7931 |
| PR-AUC | 0.5740 |
| Lift decil superior (media mensual) | 2.19 |

Por miembro:
|          |    auc |   pr_auc |
|:---------|-------:|---------:|
| lightgbm | 0.7931 |    0.574 |

Por mes:
| mes_obs   |   n |   churn |    auc |   lift10 |
|:----------|----:|--------:|-------:|---------:|
| 2025-12   | 273 |  0.3114 | 0.7917 |   2.1412 |
| 2026-01   | 192 |  0.2708 | 0.7798 |   2.332  |
| 2026-02   | 197 |  0.264  | 0.789  |   1.9939 |
| 2026-03   | 217 |  0.2488 | 0.8148 |   2.2963 |

Precisión / recall / lift por % de base contactada:
|   pct_contactado |   n |   precision |   recall |   lift |
|-----------------:|----:|------------:|---------:|-------:|
|                5 |  44 |       0.705 |    0.128 |  2.549 |
|               10 |  88 |       0.625 |    0.226 |  2.261 |
|               15 | 132 |       0.636 |    0.346 |  2.302 |
|               20 | 176 |       0.602 |    0.436 |  2.179 |
|               25 | 220 |       0.564 |    0.51  |  2.039 |
|               30 | 264 |       0.545 |    0.593 |  1.973 |
|               40 | 352 |       0.486 |    0.704 |  1.757 |
|               50 | 440 |       0.464 |    0.84  |  1.677 |

Matrices de confusión:
| umbral                      |   TP |   FP |   FN |   TN |   precision |   recall |   contactados_% |
|:----------------------------|-----:|-----:|-----:|-----:|------------:|---------:|----------------:|
| p >= 0.5                    |  123 |   91 |  120 |  545 |       0.575 |    0.506 |          24.346 |
| decil superior (p >= 0.631) |   55 |   33 |  188 |  603 |       0.625 |    0.226 |          10.011 |

Lectura: el AUC de selección (0.7911, 4 bloques de 2024-02 a 2025-05) y el AUC OOT (0.7931) difieren en
+0.0021; la diferencia combina el sesgo de selección y la dificultad propia del período.
Precisión y recall a p ≥ 0.5 dependen del umbral; para comparar modelos, la métrica es el AUC y, para campañas, el lift.
