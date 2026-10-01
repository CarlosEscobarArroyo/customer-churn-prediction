# sem2 — versión limpia del modelo de churn

Reconstrucción de lo que mejor funcionó en `sem1`, siguiendo el flujo metodológico del documento
(§4.1 → §4.4). Cada fase es una carpeta numerada con un notebook y, cuando aplica, un script.
Las funciones compartidas viven en `src/`.

| Fase | Sección | Qué hace |
|---|---|---|
| `00_datos/` | 4.1 | `qry_churn.sql` + `build_dataset.py`: re-extrae del DW y preprocesa (imputación a cero, 6 derivadas). |
| `01_eda/` | — | `eda.ipynb`: balance, estabilidad temporal, panel, nulos, señal univariada, redundancia → `reports/eda.md`. |
| `02_particion/` | 4.3 | `particion.ipynb`: OOT (últimos 4 meses) + brecha 6 m + 4 bloques de validación expansiva → `particion.json`. |
| `03_variables/` | 4.2 | `variables.ipynb`: permutación dentro del entrenamiento + ablación forward sobre los 4 bloques → `variables_seleccionadas.json`, `reports/variables.md`. |
| `04_modelado/` | 4.1 + 4.3 | `modelado.ipynb` + `tuning.py`: desbalance (4 estrategias) → Optuna 5 algoritmos × {6, 42} variables (100 trials, mismo presupuesto) → ventanas 24/36/48/todo → ensembles por promedio → `modelo_final.json`, `reports/modelado.md`. Estudios en `optuna.db` (no versionado, reanudable). |
| `05_evaluacion/` | 4.3 | `evaluacion.ipynb`: verificación GroupKFold por vendedora (vs StratifiedKFold) → entrena el ensemble final con todo el pool → OOT una sola vez (AUC, PR-AUC, ROC, matriz, precisión/recall/lift por % contactado) → `oot_metricas.json`, `models/ensemble_final.joblib`, `reports/evaluacion.md`. |
| `06_interpretacion/` | 4.4 | pendiente |

```bash
uv run python sem2_draft/00_datos/build_dataset.py        # requiere acceso a glamour-peru-dw (cuenta gmail)
uv run jupyter nbconvert --to notebook --execute --inplace sem2_draft/01_eda/eda.ipynb sem2_draft/02_particion/particion.ipynb
```

`data/` no se versiona. Los notebooks asumen que el kernel corre desde su propia carpeta.
