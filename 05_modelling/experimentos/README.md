# Experimentos (2026-09) — no forman parte del pipeline principal

Scripts de un solo pase que reutilizan `pipeline.py` (réplica de los notebooks 02→05 y de
los hiperparámetros de `05_modelling/*_best_params.json`, sin Optuna). Cada uno responde
una pregunta y deja su reporte en `reports/`. Correr desde la raíz del repo:
`uv run python 05_modelling/experimentos/<script>.py`.

| Script | Pregunta | Reporte | Número clave |
|---|---|---|---|
| `05_horizonte_churn.py` | ¿Cambia el AUC si el horizonte de churn k no es 6 meses? | `reports/horizonte_churn_modelos.md` | AUC GKF de XGBoost entre 0.736 (k=3) y 0.744 (k=6) para k ∈ [3, 12]: k=6 es el máximo y el horizonte no mueve el AUC. |
| `06_features_nuevas.py` | ¿Cuál de las 4 familias nuevas del SQL (pago, campañas, red, mix) aporta? | `reports/features_nuevas_ablation.md` | XGBoost base 0.7444 GKF / 0.7625 OOT → +campañas 0.7476 / 0.7650; pago y mix ≤ +0.0005; red sube GKF (+0.0055) pero no OOT (−0.0000). Se conservó solo campañas. *Corrió sobre la revisión del SQL con las 4 familias; hoy solo existen las columnas de campañas y el script filtra `GRUPOS` a las presentes.* |
| `07_ablacion_slices.py` | ¿Sirve el multi-slicing (entrenar con muchos meses de observación)? | `reports/ablacion_slices.md` | AUC OOT 0.7379 con K=1 mes de train → 0.7638 con todo el histórico (+0.026); a igual n, la variante *downsized* también sube (0.7186 → 0.7626): la ganancia es diversidad temporal, no solo volumen. |
| `08_poblacion.py` | ¿Mejora el modelo si se exige más historia (`compras_hist ≥ k`)? | `reports/poblacion_filtros.md` | `compras_hist ≥ 3` sube el AUC GKF (0.7523 → 0.7638) pero no el OOT (0.7638 → 0.7629) y recorta la población al 66 % de las filas. Se mantiene `compras_hist ≥ 1`. |
| `09_supervivencia.py` | ¿Gana algo formular el problema como tiempo hasta la próxima compra? | `reports/supervivencia.md` | XGBoost Cox AUC@6 OOT 0.7616 vs clasificador 0.7638 (Δ −0.002, dentro de la std por mes 0.020): empate. Ventaja solo operativa (un modelo para cualquier horizonte). |
| `10_tabfm.py` | ¿Un modelo fundacional tabular (BigQuery `AI.PREDICT`, TabFM) supera a XGBoost? | `reports/tabfm.md` | Sobre las 20 features top (tope de `AI.PREDICT`): TabFM 0.7627 AUC OOT vs XGBoost 0.7684 con las mismas 20 (0.7638 con las 110). No supera. |
| `11_ensemble.py` | ¿Aportan CatBoost, LightGBM o TabFM, o un ensemble de los 3 mejores? | `reports/ensemble.md` | No: los 6 modelos caben en 0.005 de AUC GKF (CatBoost 0.748, XGB 0.748, LightGBM 0.746, RF 0.744, LogReg 0.743, TabFM 0.743); el rank-average del top 3 da 0.748 GKF / 0.763 OOT, igual que XGBoost solo. Modelo final sin cambio. |

`pipeline.py`: funciones compartidas (split OOT, preprocessing fit-en-train, features derivadas,
selección por permutación, `make_model` tuneado, métricas GroupKFold + OOT). `BASE` es la raíz del repo.
