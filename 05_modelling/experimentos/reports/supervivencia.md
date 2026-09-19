# Supervivencia (tiempo hasta la próxima compra) vs. clasificador de churn

> Generado por `05_modelling/09_supervivencia.py` el 2026-09-06 19:05. Dataset: variante en memoria de `qry_churn.sql` (gap_prox, meses_futuro). 110 features tras preprocessing; hiperparámetros de XGBoost de `xgboost_best_params.json` (sin Optuna).

## Diseño

- **Evento** = volver a comprar; **tiempo** = meses hasta la próxima compra (`gap_prox`), censurado a la derecha en `meses_futuro` si no se observa ninguna. Tasa de eventos: 0.811.
- **Split** idéntico al modelo de churn: `oot_split` sobre las filas con `meses_futuro >= 6` (train ≤ rank 101, gap de 6 meses, test = ranks 108-111 = 885 filas). GroupKFold(5) por vendedora sobre las mismas filas.
- **Filas recientes censuradas** (`meses_futuro < 6`): 1,026. Quedan después del test en el tiempo, así que se dejan fuera de train y test (leakage temporal). En producción sí serían filas de train censuradas: es la ventaja práctica del enfoque (el clasificador las descarta).
- **AUC@h** = AUC de la etiqueta observada "no compra en h meses" sobre filas con `meses_futuro >= h`, sin IPCW. AUC@6 es el comparable con el modelo de churn. El test OOT tiene `meses_futuro` entre 6 y 9, por eso **AUC@12 solo existe en GroupKFold (OOF)**, no en OOT.
- **C-index** de Harrell (`concordance_index_censored`) respecto del evento "volver".

## Sanity check del target

Sobre las 30,821 filas con `meses_futuro >= 6`: `churn6 = (gap_prox IS NULL OR gap_prox > 6)` coincide al 100% con `churn` del SQL y con `data/processed/churn_dataset.csv` (mismas filas por `(id_vendedor, mes_rank)`, mismos valores en las 70 columnas). Verificado con asserts en el script.

## ADVERTENCIA de signo

El evento es **volver a comprar**: hazard alto = vuelve pronto = riesgo de churn **bajo**. En el script, el score de riesgo de churn (AUC@h, lift, GroupKFold) es `-(log hazard ratio)` (Cox: `-predict`; XGBoost Cox: `-log(predict)`; RSF: `-predict`), y el C-index se calcula con el signo opuesto (`+hazard`). Para el clasificador: riesgo = `p(churn)`, C-index con `-p`. No invertir ninguno de los dos sin invertir el otro.

## Resultados

Referencia = `XGB clasificador (ref)`: XGBoost clasificador vigente (churn a 6 meses) evaluado en la misma corrida. Sus AUC@3 y AUC@12 usan `p(churn a 6 meses)` como score (modelo de un horizonte aplicado a otro).

| modelo                 |   C_gkf |   C_oot |   AUC3_oot |   AUC6_gkf |   AUC6_oot |   oot_AUCstd |   AUC12_gkf |   gkf_liftPR |   oot_lift10 |
|:-----------------------|--------:|--------:|-----------:|-----------:|-----------:|-------------:|------------:|-------------:|-------------:|
| Cox PH (lineal)        |  0.6794 |  0.6914 |     0.7299 |     0.7431 |     0.7439 |       0.0162 |      0.7492 |       1.7202 |       2.2076 |
| XGBoost Cox            |  0.6867 |  0.7059 |     0.7448 |     0.7511 |     0.7616 |       0.0266 |      0.7558 |       1.7726 |       2.2076 |
| RandomSurvivalForest   |  0.6663 |  0.6906 |     0.7291 |     0.7342 |     0.7562 |       0.0314 |      0.744  |       1.6993 |       2.2894 |
| XGB clasificador (ref) |  0.687  |  0.7032 |     0.7432 |     0.7523 |     0.7638 |       0.0202 |      0.7537 |       1.7742 |       2.1667 |

### n y prevalencia por horizonte (iguales para todos los modelos)

|   h |   n_gkf |   prev_gkf |   n_oot |   prev_oot |
|----:|--------:|-----------:|--------:|-----------:|
|   3 |   30821 |      0.399 |     885 |      0.385 |
|   6 |   30821 |      0.309 |     885 |      0.278 |
|  12 |   29553 |      0.246 |       0 |    nan     |

### Tabla completa

| modelo                 |   C_gkf |   C_oot |   gkf_AUC |   gkf_liftPR |   oot_AUC |   oot_AUCstd |   oot_lift10 |   AUC3_gkf |   n3_gkf |   prev3_gkf |   AUC3_oot |   n3_oot |   prev3_oot |   AUC6_gkf |   n6_gkf |   prev6_gkf |   AUC6_oot |   n6_oot |   prev6_oot |   AUC12_gkf |   n12_gkf |   prev12_gkf |   AUC12_oot |   n12_oot |   prev12_oot |
|:-----------------------|--------:|--------:|----------:|-------------:|----------:|-------------:|-------------:|-----------:|---------:|------------:|-----------:|---------:|------------:|-----------:|---------:|------------:|-----------:|---------:|------------:|------------:|----------:|-------------:|------------:|----------:|-------------:|
| Cox PH (lineal)        |  0.6794 |  0.6914 |    0.7431 |       1.7202 |    0.7439 |       0.0162 |       2.2076 |     0.7293 |    30821 |      0.3991 |     0.7299 |      885 |      0.3853 |     0.7431 |    30821 |      0.3091 |     0.7439 |      885 |       0.278 |      0.7492 |     29553 |       0.2457 |         nan |         0 |          nan |
| XGBoost Cox            |  0.6867 |  0.7059 |    0.7511 |       1.7726 |    0.7616 |       0.0266 |       2.2076 |     0.7373 |    30821 |      0.3991 |     0.7448 |      885 |      0.3853 |     0.7511 |    30821 |      0.3091 |     0.7616 |      885 |       0.278 |      0.7558 |     29553 |       0.2457 |         nan |         0 |          nan |
| RandomSurvivalForest   |  0.6663 |  0.6906 |    0.7342 |       1.6993 |    0.7562 |       0.0314 |       2.2894 |     0.7148 |    30821 |      0.3991 |     0.7291 |      885 |      0.3853 |     0.7342 |    30821 |      0.3091 |     0.7562 |      885 |       0.278 |      0.744  |     29553 |       0.2457 |         nan |         0 |          nan |
| XGB clasificador (ref) |  0.687  |  0.7032 |    0.7523 |       1.7742 |    0.7638 |       0.0202 |       2.1667 |     0.7404 |    30821 |      0.3991 |     0.7432 |      885 |      0.3853 |     0.7523 |    30821 |      0.3091 |     0.7638 |      885 |       0.278 |      0.7537 |     29553 |       0.2457 |         nan |         0 |          nan |

## Lectura

- **h = 6 (comparable)**: el mejor modelo de supervivencia (XGBoost Cox) empata al clasificador en OOT (0.7616 vs 0.7638, Δ = -0.0022) y empata en GroupKFold (0.7511 vs 0.7523, Δ = -0.0012). La std por mes del OOT es 0.020: diferencias de ±0.01 están dentro del ruido.
- **h = 3**: mejor supervivencia 0.7448 vs clasificador 0.7432 (Δ = +0.0016) → empata.
- **h = 12 (GKF)**: mejor supervivencia 0.7558 vs clasificador 0.7537 (Δ = +0.0021) → empata.
- **C-index**: XGBoost Cox 0.7059 vs clasificador 0.7032 en OOT. El clasificador, entrenado solo con la etiqueta binaria a 6 meses, ordena los tiempos de retorno casi tan bien como los modelos que los ven explícitamente.
- **Implicación para la tesis**: el cambio de formulación (binaria → tiempo hasta el evento) no mueve la capacidad discriminativa de forma material; refuerza la conclusión previa de que el cuello de botella es la información disponible en las features, no el algoritmo ni la forma del target. Lo que sí aporta supervivencia es operativo: un solo modelo sirve para cualquier horizonte, y en producción incorpora al train las 1,026 filas recientes censuradas que el clasificador descarta.
- Limitaciones: el RSF entrena con un submuestreo de 12,000 filas por tiempo de cómputo; Cox es lineal y sin interacciones; no hay tuning específico para supervivencia (se reutilizan los hiperparámetros del clasificador).
