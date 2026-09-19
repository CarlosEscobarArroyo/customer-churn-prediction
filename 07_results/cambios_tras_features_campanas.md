# Cambios de cifras tras agregar las features de campañas (2026-09)

Se re-ejecutó el pipeline completo (02 → 06) sobre `qry_churn.sql` con la familia
**CAMPAÑAS** (7 columnas: `camp_saltadas`, `camp_part_u12`, `tasa_camp_u3/u6/u12`,
`pct_directo_u12`, `es_nueva_u12`). Las familias PAGO, RED y MIX se evaluaron y
descartaron (`05_modelling/experimentos/reports/features_nuevas_ablation.md`).
Los notebooks de tuning con Optuna **no** se re-ejecutaron: los modelos tuneados se
reentrenaron con los hiperparámetros de `05_modelling/*_best_params.json` sobre el
nuevo set de features (`05_modelling/retrain_tuned.py`, fit en el train del split OOT,
igual que la sección final de los notebooks de tuning).

Columna *antes* = lo que citan `experimentacion_capitulo.md` y
`resumen_ejecutivo_evaluacion.md`; columna *ahora* = outputs de los notebooks
re-ejecutados. Los MD del capítulo no se editaron.

> **Ojo con la comparación**: el capítulo se escribió sobre una extracción anterior del
> dataset cuyo bloque test OOT tenía **803 filas y prevalencia 26.2 %**; la extracción
> vigente (misma query, más meses en la fuente) da **885 filas y prevalencia 27.8 %**
> (30,821 filas totales, churn 30.9 %). Parte de las diferencias en OOT viene del cambio
> de ventana, no de las features. La comparación limpia features-vs-features es la
> ablación (`features_nuevas_ablation.md`, mismo dataset): XGBoost base 0.7444 GKF /
> 0.7625 OOT → +campañas 0.7476 / 0.7650.

## 1. Tamaño del set de features

| Cifra | Antes (capítulo) | Ahora |
|---|---:|---:|
| Columnas crudas del SQL | 46 | 53 |
| Variables tras preprocessing (03) | 78 | 85 |
| Variables tras ingeniería (04, +6 derivadas) | 84 | 91 |
| Variables retenidas por permutación (04) | 68 de 84 (descartadas 16) | 71 de 91 (descartadas 20) |
| AUC OOT (RF de selección) todas → seleccionadas | 0.7625 → 0.7628 | 0.7674 → 0.7659 |
| Variable más importante en la selección (RF, permutación en train) | `ticket_acum` | `tasa_camp_u6` (`ticket_acum` 2.ª) |

Las 7 de campañas quedaron entre las seleccionadas; en el top 20 de la selección
aparecen `tasa_camp_u6` (1.ª), `tasa_camp_u3` (7.ª), `tasa_camp_u12` (8.ª),
`camp_saltadas` (15.ª) y `pct_directo_u12` (20.ª).

## 2. Comparación de algoritmos — baseline sin tuning (Tabla 1 del capítulo)

`05_modelling/01_modelos_baseline.ipynb`. Test OOT antes: 803 filas; ahora: 885.

| Modelo | AUC GroupKFold antes → ahora | AUC OOT antes → ahora | Desv. AUC/mes antes → ahora | Lift decil OOT antes → ahora |
|---|---:|---:|---:|---:|
| Regresión Logística | 0.7417 → 0.7431 | 0.7676 → 0.7637 | 0.012 → 0.005 | 2.25 → 2.21 |
| Random Forest | 0.7418 → 0.7450 | 0.7627 → 0.7657 | 0.023 → 0.028 | 2.44 → 2.33 |
| XGBoost | 0.7418 → 0.7454 | 0.7540 → 0.7569 | 0.016 → 0.025 | 2.29 → 2.33 |

Lectura: los tres siguen empatados dentro de la std mensual; el mejor OOT ahora es
Random Forest (0.7657) y el mejor GroupKFold XGBoost (0.7454).

## 3. Modelos tuneados (Tabla 2 del capítulo)

Hiperparámetros sin cambios; fit sobre el train OOT con las 71 features. GroupKFold(5)
por vendedora sobre todo el dataset (protocolo de `pipeline.evaluate`, el mismo que usó la
ablación). Nota: el capítulo cita "AUC interna (CV)" del bucle interno de Optuna sobre el
train-pool, que no es exactamente el mismo protocolo que el GroupKFold sobre todo el dataset.

| Modelo | AUC GroupKFold antes → ahora | AUC OOT antes → ahora | Recall@0.5 antes → ahora | Lift decil antes → ahora | Desv. AUC/mes antes → ahora |
|---|---:|---:|---:|---:|---:|
| XGBoost (final) | 0.7422 → 0.7476 | 0.7648 → 0.7637 | 0.776 → 0.736 | 2.44 → 2.29 | 0.018 → 0.022 |
| Random Forest | 0.7399 → 0.7451 | 0.7590 → 0.7650 | 0.729 → 0.736 | 2.34 → 2.41 | 0.021 → 0.026 |
| Regresión Logística (referencia) | — | — → 0.7647 | — | — | — |

Contra la referencia de la ablación (XGBoost con las 7 de campañas: GKF ≈ 0.748, OOT ≈ 0.765)
los números coinciden: 0.7476 / 0.7637.

Con el dataset vigente XGBoost y Random Forest tuneados quedan empatados (Δ OOT −0.001,
std mensual 0.02–0.03); la afirmación del capítulo de que "XGBoost superó a Random Forest en
todas las métricas relevantes" ya no se sostiene en OOT, aunque XGBoost sigue arriba en
GroupKFold, que es el protocolo principal. El modelo final sigue siendo XGBoost.

## 4. Evaluación del XGBoost final (`06_evaluation/`)

Test OOT antes: 803 filas, prevalencia 0.262; ahora: 885 filas, prevalencia 0.278.

| Cifra | Antes (capítulo) | Ahora |
|---|---:|---:|
| AUC-ROC OOT | 0.7648 | 0.7637 |
| PR-AUC OOT | 0.503 | 0.535 |
| Lift global (PR-AUC / prevalencia) | 1.92× | 1.93× |
| Brier score | 0.2012 | 0.2054 |
| Matriz de confusión a t=0.5 — TP (capturados) | 163 de 210 | 181 de 246 |
| — FN (perdidos) | 47 | 65 |
| — FP (falsas alarmas) | 217 | 230 |
| — TN | 376 | 409 |
| Recall / precision a t=0.5 | 0.78 / 0.43 | 0.74 / 0.44 |
| Contactadas a t=0.5 | 380 | 411 |
| Top 5 %: contactadas / churners / precision / recall / lift | 41 / 23 / 0.56 / 0.11 / 2.15× | 45 / 32 / 0.71 / 0.13 / 2.56× |
| Top 10 %: contactadas / churners / precision / recall / lift | 81 / 51 / 0.63 / 0.24 / 2.41× | 89 / 56 / 0.63 / 0.23 / 2.26× |
| Top 20 %: contactadas / churners / precision / recall / lift | 161 / 85 / 0.53 / 0.41 / 2.02× | 177 / 102 / 0.58 / 0.42 / 2.07× |
| Top 30 %: contactadas / churners / precision / recall / lift | 241 / 115 / 0.48 / 0.55 / 1.83× | 266 / 138 / 0.52 / 0.56 / 1.87× |
| Top 50 %: contactadas / churners / precision / recall / lift | 402 / 165 / 0.41 / 0.79 / 1.57× | 443 / 192 / 0.43 / 0.78 / 1.56× |
| Top 3 por \|SHAP\| | `compras_hist`, `n_ped_u12`, `n_prod_u12` | `n_ped_u12`, `monto_u3`, `compras_hist` |
| Top 3 por permutación (caída de AUC) | — | `n_ped_u12` (0.018), `compras_hist` (0.009), `camp_part_u12` (0.007) |
| Caso de mayor / menor riesgo (SHAP local) | p=0.87 → churn / p=0.01 → no churn | p=0.89 → churn / p=0.02 → no churn |

Las features de campañas entran en el top de todas las lentes: `camp_part_u12` es la
4.ª por gain y 3.ª por permutación; `camp_saltadas` 7.ª por gain y 5.ª por permutación;
`tasa_camp_u3/u6/u12` entre las 15 primeras por gain.

## 5. Qué sigue igual

- Target, población, granularidad mensual, split OOT (v=6, 4 meses de test) y los dos
  protocolos de validación.
- Hiperparámetros tuneados (`05_modelling/*_best_params.json`).
- Conclusión de fondo: techo de señal ~0.76–0.77 en OOT, los tres algoritmos empatan
  dentro del ruido mensual, y las probabilidades siguen mal calibradas (Brier peor que la
  prevalencia constante) por el `scale_pos_weight`.
