# Ensemble de 42 variables en el test OOT del capítulo

Documento del 26 de septiembre de 2026. Script: [`scripts/ensemble_oot.py`](../scripts/ensemble_oot.py).

## Pregunta

El ensemble `mean_logreg_xgboost` ([variables_mejor_ensemble.md](variables_mejor_ensemble.md))
reportó AUC **0,790** como media de cuatro bloques de validación (oct-2023 a ene-2025). Los modelos
del capítulo reportan ~0,765 en el test *out-of-period*. ¿La diferencia viene del modelo o de la evaluación?

## Protocolo

- Test OOT = el mismo de `pipeline.oot_split`: últimos 4 meses del CSV vigente (oct-2025 a ene-2026),
  **885 filas, prevalencia 27,8 %**; entrenamiento hasta mar-2025 (gap de 6 meses).
- Ensemble: 42 variables transaccionales, logística con los últimos 36 meses del train, XGBoost con
  todo el train, promedio 50/50. Hiperparámetros fijos de `reports/master_features_v1/reconstructed_tuning`,
  buscados con observaciones hasta ene-2025: el OOT no participó en la búsqueda.
- Referencia: modelos tuneados del capítulo (`models/*_tuned.joblib`, 71 variables con datos maestros
  y campañas), reentrenados por `retrain_tuned.py` sobre el mismo train. Sus AUC reproducen
  [cambios_tras_features_campanas.md](cambios_tras_features_campanas.md).

## Resultados

| Modelo | Variables | AUC OOT | Desv. AUC/mes | PR-AUC | Recall@0.5 | Precisión@0.5 | Lift decil |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Ensemble 42 (LogReg 36m + XGB)** | 42 | **0,7636** | 0,014 | 0,533 | 0,711 | 0,458 | 2,37 |
| · componente LogReg 36m | 42 | 0,7600 | 0,010 | 0,530 | 0,720 | 0,453 | 2,25 |
| · componente XGBoost | 42 | 0,7643 | 0,020 | 0,544 | 0,720 | 0,455 | 2,37 |
| XGBoost tuneado (capítulo) | 71 | 0,7637 | 0,022 | 0,535 | 0,736 | 0,440 | 2,29 |
| Random Forest tuneado (capítulo) | 71 | 0,7650 | 0,026 | 0,536 | 0,736 | 0,455 | 2,41 |
| LogReg tuneada (capítulo) | 71 | 0,7647 | 0,007 | 0,533 | 0,724 | 0,461 | 2,29 |

## Lectura

- En el mismo bloque OOT, **el ensemble rinde igual que los modelos del capítulo** (0,7636 frente a
  0,764–0,765). Las diferencias son de milésimas, muy por debajo de la desviación mensual (0,01–0,03).
- El 0,790 **no refleja un mejor modelo**, sino que los bloques oct-2023 a ene-2025 son más fáciles
  de predecir, y además se reutilizaron para elegir hiperparámetros, ventanas y ensemble (sesgo de selección).
- Las 42 variables transaccionales, sin datos maestros ni campañas, alcanzan el mismo techo que las 71.
  Es un modelo más simple con el mismo desempeño.
- Recall@0.5 y precisión@0.5 dependen del umbral y de scores no calibrados, así que no se deben usar
  para comparar modelos. Para ordenar, la métrica comparable es el lift de decil.
