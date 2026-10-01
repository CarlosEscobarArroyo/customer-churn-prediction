# CLAUDE.md

Predicción de *silent churn* de vendedoras de Glamour Perú (venta directa). Tesis de Carlos Escobar.
Todo en español.

## Dos carpetas

- `sem1/` — primera fase (seminario 1). Experimentación histórica; **no se modifica**. Su `CLAUDE.md`
  y `README.md` están desactualizados respecto del modelo ganador; lo que vale está en
  `sem1/07_results/ensemble_oot.md` y `variables_mejor_ensemble.md`.
- `sem2_draft/` — versión limpia, en construcción. **Trabajar aquí.** Sigue `sem2_draft/metodologia.md`
  (§4.1–4.4) fase por fase. El estado actual y los números vigentes están en **`sem2_draft/AVANCE.md`**:
  leerlo primero.

## Setup en una máquina nueva

```bash
uv sync                                   # entorno (raíz del repo)
brew install libomp                       # macOS: xgboost/lightgbm lo necesitan
gcloud auth login carlos.escobar.arroyo@gmail.com   # la cuenta con acceso a glamour-peru-dw
uv run python sem2_draft/00_datos/build_dataset.py  # re-extrae data/ (no versionada)
```

`src/datos.bq_client()` toma el token con `gcloud auth print-access-token --account=<gmail>`;
la cuenta se puede cambiar con `BQ_ACCOUNT`. Si la ADC activa es otra cuenta, no hace falta cambiarla.

Luego ejecutar los notebooks en orden (`01_eda`, `02_particion`, `03_variables`, …):

```bash
uv run jupyter nbconvert --to notebook --execute --inplace sem2_draft/<fase>/<nb>.ipynb
```

Los notebooks asumen que el kernel corre desde su propia carpeta (`sys.path.insert(0, Path.cwd().parent)`).

## Convenciones de sem2

- Una carpeta numerada por fase, con un notebook y, si hace falta, un script. Funciones compartidas en
  `sem2_draft/src/`. Notebooks escritos directo (sin builders).
- Cada notebook termina escribiendo un `reports/<fase>.md` con los números reales; nunca escribir
  números a mano en la documentación.
- Partición temporal única en `src/particion.py` (OOT = últimos 4 meses, brecha 6 m, 4 bloques
  expansivos). **El OOT se evalúa una sola vez con el modelo final**; ninguna fase intermedia lo toca.
- Métrica de selección: AUC medio de los 4 bloques; desempate por Top Decile Lift mensual; a igualdad,
  el modelo con menos variables.
- Al cerrar cada fase: actualizar `AVANCE.md` y la tabla de `sem2_draft/README.md`.
- No commitear `data/`, `models/`, CSV/parquet/joblib (ya en `.gitignore`). Revisar outputs de
  notebooks antes de commitear: no deben mostrar `id_vendedor` ni filas crudas.
- Git (add/commit/push) lo ejecuta Carlos; Claude deja el bloque listo.
