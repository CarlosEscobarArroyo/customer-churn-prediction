# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Proyecto

Modelo de predicción de **silent churn** para vendedoras de Glamour Peru (venta directa, cosmética).

## Stack

- Python 3.12, deps gestionadas con `uv` (`uv sync` para instalar; `pyproject.toml` + `uv.lock`).
- BigQuery: `glamour-peru-dw.glamour_dw` (8 tablas crudas).
- Modelado: scikit-learn, XGBoost, LightGBM, SHAP, Optuna.

## Comandos

```bash
# Setup
uv sync                                  # instalar deps (incluye dev group)
uv run pre-commit install                # opcional: hooks pre-commit
uvx nbstripout --install                 # opcional: limpiar outputs de notebooks

# Notebook / scripts
uv run jupyter lab                       # levantar JupyterLab
uv run python -m src.<modulo>            # ejecutar módulo de src/

# Calidad
uv run ruff check .                      # lint
uv run ruff format .                     # format
uv run mypy src/                         # type check
uv run pytest                            # suite completa
uv run pytest tests/test_x.py::test_y    # un solo test
uv run pytest --cov=src                  # con cobertura
```

## Estructura

```
data/
  qry_churn_v2.sql              # query autoritativa que construye el dataset
  diccionario_churn_data.md     # diccionario de variables (referencia)
  glamour_metadata.md           # schema BigQuery
notebooks/
  01_eda.ipynb                  # EDA inicial (validación + elección de k + viabilidad)
src/                            # módulos Python (vacío por ahora)
models/                         # binarios entrenados (no versionados)
reports/                        # outputs auto-generados (no versionados)
```

## Definición de churn (v2)

**Granularidad**: una fila por `(id_vendedor, id_campana_obs)`. Features = RFM + tendencia + producto + contexto, con ventanas 3/6/12 campañas.

`churn = 1` si la vendedora no compra en las **6 campañas consecutivas** posteriores a la campaña observada. Filtro de población: `compras_historicas >= 4`.

La elección de la ventana `k=6` está abierta a revisión — la EDA en `notebooks/01_eda.ipynb` (Bloque 2) la cuestiona con análisis de gaps + sensibilidad. Si el codo aparece en otro k, regenerar `qry_churn_v3.sql`.

Cambios v1 → v2 documentados al inicio de `data/qry_churn_v2.sql`.

## Cómo extraer el dataset

```bash
# Re-extracción (los CSV no se versionan):
gcloud auth application-default login        # una vez por máquina
# Ejecutar qry_churn_v2.sql desde el notebook o:
bq query --use_legacy_sql=false < data/qry_churn_v2.sql
bq extract glamour-peru-dw.glamour_dw.training_churn_v2 \
  gs://<bucket>/churn_v2.csv
```

O bien correr la primera celda del notebook que carga las tablas crudas via `google-cloud-bigquery`.

## Reglas del repo

- **No commitear data**: `.gitignore` excluye `data/*` excepto `*.md` y `*.sql`. También excluye `*.csv/parquet/feather` en cualquier subdirectorio.
- **No commitear credenciales**: `.env`, `service-account*.json`, `*.key`, etc. están bloqueados.
- **No commitear `reports/`**: son outputs auto-generados; pueden contener estadísticos del dataset. Si querés versionar uno puntualmente: `git add -f reports/<archivo>`.
- **Notebooks con outputs**: las celdas pueden mostrar datos (`df.head()`, conteos) → revisá los outputs antes de commitear, o usá `nbstripout` (`uvx nbstripout --install` en el repo).

## Convenciones

- Comentar el SQL cuando se cambia el target o el filtro de población; documentar el motivo en el header.
- En notebooks de EDA: dejar la última celda generando un reporte markdown en `reports/` con los números reales (no escribir números a mano).
- Para evitar leakage: split temporal por `campana_rank_obs`, nunca aleatorio. Variables `compro_t1..t6` y `monto_t1..t6` son target — nunca features.
