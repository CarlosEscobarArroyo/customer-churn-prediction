# Comparación ampliada de modelos y ensemble (LogReg, RF, XGB, LightGBM, CatBoost, TabFM)

> Generado por `05_modelling/experimentos/11_ensemble.py` el 2026-09-10 22:14.
> 91 features, hiperparámetros del repo para LogReg/RF/XGB; LightGBM y CatBoost con defaults regularizados (sin Optuna); TabFM zero-shot con 20 features (tope de `AI.PREDICT`). GroupKFold(5) por vendedora + OOT (885 filas, últimos 4 meses, gap 6). ★ = top 3 por `gkf_AUC`.

## Resultados

|                          |   gkf_AUC |   gkf_liftPR |   gkf_lift10 |   oot_AUC |   oot_AUCstd |   oot_PRAUC |   oot_lift10 |   segundos |
|:-------------------------|----------:|-------------:|-------------:|----------:|-------------:|------------:|-------------:|-----------:|
| LogReg                   |    0.7429 |       1.7125 |       1.9776 |    0.7619 |       0.0092 |      0.5277 |       2.2485 |    61.5793 |
| RandomForest             |    0.7442 |       1.7272 |       2.0227 |    0.7659 |       0.0213 |      0.5357 |       2.3711 |    10.9213 |
| XGBoost ★                |    0.7477 |       1.7409 |       2.0091 |    0.7643 |       0.0221 |      0.5391 |       2.3302 |     7.9281 |
| LightGBM ★               |    0.7464 |       1.7404 |       2.0185 |    0.7588 |       0.0183 |      0.5287 |       2.3711 |    12.7947 |
| CatBoost ★               |    0.7479 |       1.7448 |       2.0269 |    0.7608 |       0.0176 |      0.5293 |       2.2076 |    17.8819 |
| TabFM                    |    0.7428 |       1.713  |       1.986  |    0.7614 |       0.0176 |      0.5264 |       2.3711 |   478.673  |
| Ensemble top3 rank-avg   |    0.7484 |       1.7495 |       2.0091 |    0.7626 |       0.0191 |      0.5339 |       2.2894 |   nan      |
| Ensemble top3 prob-avg   |    0.7483 |       1.7471 |       2.0091 |    0.7626 |       0.0192 |      0.5324 |       2.2894 |   nan      |
| Ensemble 6 rank-avg      |    0.7483 |       1.746  |       2.0185 |    0.765  |       0.0174 |      0.5336 |       2.2894 |   nan      |
| Stacking top3 (solo OOT) |  nan      |     nan      |     nan      |    0.7633 |       0.0194 |      0.5358 |       2.2894 |   nan      |

`segundos` = 5 folds + OOT (fit+predict); para TabFM, latencia acumulada de las 6 consultas `AI.PREDICT` (0 s si vino de caché). `gkf_liftPR` = PR-AUC / prevalencia; `lift10` = lift del decil top; `oot_AUCstd` = std del AUC por mes del bloque OOT.

Orden por `gkf_AUC`: CatBoost > XGBoost > LightGBM > RandomForest > LogReg > TabFM. Ensembles sin ajuste (rank-avg = rango normalizado por fold/bloque; prob-avg = promedio de probabilidades). El stacking (regresión logística sobre las OOF del top 3, ajustada solo en las filas de train del OOT) se reporta **solo en OOT**: su métrica GKF no sería honesta porque el meta-modelo vería las OOF de todos los folds. Coeficientes: {'CatBoost': np.float64(2.169), 'XGBoost': np.float64(1.962), 'LightGBM': np.float64(0.573)}.

## TabFM

6 llamadas a `AI.PREDICT` (5 folds + OOT) con las 20 features top por ganancia del XGBoost ajustado en train OOT: `monto_u12, n_ped_u6, n_prod_u12, monto_u6, n_ped_u12, n_ped_u3, monto_u3, monto_acum, n_ped_acum, camp_saltadas, camp_part_u12, intensidad_u3, compras_hist, n_prod_acum, tasa_camp_u3, es_nueva_u12, tasa_camp_u6, meses_activos_u12, tasa_camp_u12, meses_activos_u6`. Esta corrida: 4 llamadas nuevas, 83,886,080 bytes facturados (~USD 0.0005); el resto salió de `reports/tabfm_predicciones.csv`. **No es bit-reproducible** entre corridas (sin semilla ni control de `n_ensembles`): por eso se cachean las predicciones. Las 20 features se eligen con el train del OOT, que solapa con los folds GKF (sesgo mínimo, solo selección de columnas). Tablas temporales borradas en cada llamada; costo recurrente cero.

## Lectura

- **Modelos nuevos vs XGBoost tuneado.** El mejor nuevo por GKF es **CatBoost** (0.7479 GKF / 0.7608 OOT) vs XGBoost 0.7477 / 0.7643: Δ +0.0002 GKF, -0.0035 OOT → **empata** (dentro del ruido: < 0.005 en GKF, < std por mes en OOT). Los 6 modelos individuales caben en un rango de 0.0051 de AUC GKF y 0.0071 de AUC OOT (std por mes del OOT ≈ 0.022).
- **Ensemble.** El rank-average de los 3 mejores (CatBoost, XGBoost, LightGBM) da 0.7484 GKF / 0.7626 OOT: Δ +0.0005 / -0.0033 contra el mejor individual (CatBoost) → **empata** (dentro del ruido: < 0.005 en GKF, < std por mes en OOT). Prob-avg: 0.7483 / 0.7626. Rank-avg de los 6: 0.7483 / 0.7650. Stacking (solo OOT): 0.7633.
- **Recomendación: no cambia.** XGBoost tuneado sigue como modelo final. Que seis familias distintas (lineal, bagging, tres boostings y un modelo fundacional zero-shot) y sus combinaciones queden en la misma franja es la firma de un **techo de señal**: las features capturan lo que hay en el historial de compra y el resto del churn es idiosincrático (no está en los datos). Un ensemble solo suma cuando los errores de los miembros están descorrelacionados; acá todos aprenden la misma función de la recencia/frecuencia y sus errores coinciden. Agregar complejidad de modelo no mueve la aguja; lo que la movería es información nueva (contacto de la coordinadora, motivo de baja, uplift medido de la acción de retención).
