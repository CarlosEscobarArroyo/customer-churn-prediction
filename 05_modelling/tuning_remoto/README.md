# Tuning remoto con Optuna (LogReg, RF, XGBoost, LightGBM, CatBoost)

Carpeta autocontenida para correr el tuning en otra máquina. No necesita el resto del repo,
ni `gcloud`, ni BigQuery. TabFM queda fuera. Qué hace y con qué protocolo: ver la primera celda de
`tuning_optuna.ipynb`.

```
tuning_remoto/
  tuning_optuna.ipynb             # el notebook
  data/churn_dataset_features.csv # dataset vigente (salida de 04, 71 features) — NO se versiona
  pyproject.toml, uv.lock         # entorno (versiones = las del repo, para que los .joblib carguen de vuelta)
  resultados/                     # se crea al correr
```

## En la máquina prestada

```bash
# 1. Copiar la carpeta completa (incluye data/). Instalar uv si no está:
curl -LsSf https://astral.sh/uv/install.sh | sh
# macOS: además `brew install libomp` (lo necesitan xgboost y lightgbm)

cd tuning_remoto
uv sync

# 2. Prueba de instalación (~5 min, 3 trials por modelo, sale en resultados_smoke/)
SMOKE=1 uv run jupyter nbconvert --to notebook --execute tuning_optuna.ipynb \
  --output smoke.ipynb --ExecutePreprocessor.timeout=-1

# 3. Corrida real, desacoplada de la terminal (sobrevive a que se cierre la sesión SSH)
nohup uv run jupyter nbconvert --to notebook --execute tuning_optuna.ipynb \
  --output tuning_optuna_ejecutado.ipynb --ExecutePreprocessor.timeout=-1 > run.log 2>&1 &

# 4. Seguir el progreso
tail -f resultados/progreso.log
```

En Windows: `uv run jupyter lab` y ejecutar todas las celdas, o el comando 3 sin `nohup` ni `&`.

**Si se corta** (apagón, te piden la máquina): relanzar el mismo comando 3. Cada estudio se
guarda en `resultados/optuna_<modelo>.journal` y solo corren los trials que faltan.

## Qué ajustar (celda de configuración)

- `N_TRIALS`: por defecto 100 / 300 / 500 / 500 / 400 (logreg / rf / xgboost / lightgbm / catboost).
- `THREADS_POR_TRIAL` (4) y `N_PARALELO` (= núcleos / 4): cuántos trials corren a la vez.
  Con 28 mil filas conviene más paralelizar trials que darle muchos hilos a cada modelo.
- `MODELOS`: para correr solo algunos.

## Qué traer de vuelta

Toda la carpeta `resultados/`:

| Archivo | Qué es |
|---|---|
| `resultados.md` / `.csv` | Tabla comparativa: tuneado vs config previa de cada modelo, GroupKFold + OOT |
| `<modelo>_best_params.json` | Mejores hiperparámetros, mismo formato que `05_modelling/*_best_params.json` |
| `modelos/<modelo>_tuned.joblib` | Modelo entrenado con los mejores hiperparámetros sobre el train del OOT (como `models/*_tuned.joblib`) |
| `trials_<modelo>.csv` | Todos los trials (para analizar la importancia de los hiperparámetros) |
| `predicciones.csv` | OOF GroupKFold + predicción OOT por modelo (para ensembles sin reentrenar) |

Al terminar, **borrar `data/` de la máquina prestada**: el CSV trae `id_vendedor` reales.
