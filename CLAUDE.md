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
uv run python scripts/build_nb08.py      # regenerar un notebook desde su script
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
  qry_churn_v5.sql              # query autoritativa (vigente, granularidad mensual)
  qry_churn_v{2,3,4,6,6_all}.sql  # versiones históricas / experimentos
  diccionario_churn_data.md     # diccionario de variables (referencia v2)
  glamour_metadata.md           # schema BigQuery
notebooks/
  clean/                              # entregables finales (con marco teórico)
    modelo_final_v5.ipynb             # modelo vigente (HGB tuneado, sec 9 viabilidad)
    ablation_temporal_v5.ipynb        # ablation que sacó mes_num/anio_mes_num
    tuning_optuna_v5.ipynb            # tuning de hiperparámetros con Optuna
  drafts/                             # iteración exploratoria (ver VERSIONES.md)
    01_eda.ipynb                      # EDA inicial sobre v2
    02_validacion_v3.ipynb            # validación schema/leakage v3
    03_baselines.ipynb                # baselines v3 (campaña, k=6)
    04_split_temporal.ipynb           # detección del drift → v4 (mensual)
    05_baselines_v4.ipynb             # baselines v4 (mes, k=4)
    06_horizonte_v4.ipynb             # codo de horizonte → v5 (k=6 mensual)
    07_baselines_v5.ipynb             # baselines v5
    08_baselines_v6.ipynb             # comparación v5 vs v6 vs v6_all
scripts/                              # builders idempotentes (build_nb*.py)
src/                            # módulos Python reutilizables (vacío por ahora)
models/                         # binarios entrenados (no versionados)
reports/                        # outputs auto-generados (no versionados)
VERSIONES.md                    # bitácora de versiones del dataset (canónico)
STATUS.md                       # contexto general + checklist de leakage
V4_MENSUAL.md                   # racional de la migración a granularidad mensual
LEAKEAGE.md, CORRELACION.md     # discusiones sobre leakage / correlación de filas
```

## Definición de churn (v5 vigente)

**Granularidad**: una fila por `(id_vendedor, mes_obs)` (mes calendario, no campaña). `mes_rank_obs` es el ordenador temporal.

`churn = 1` si la vendedora **no compra en los próximos 6 meses consecutivos** después del mes observado. NULL si no hay 6 meses de futuro disponibles.

**Filtro de población**: `compras_historicas >= 3` (en mensual). Excluye vendedoras de historia muy corta para evitar la regla trivial "1 compra ⇒ churn".

**Blacklist de campañas atípicas**: 20102 (COVID), 20201 (fechas invertidas), 23105 (curso no-retail).

**Features**: RFM (recencia, frecuencia, monetario) en ventanas u3/u6/u12 **meses**, tendencias normalizadas en [-1, 1], diversidad de producto, contexto (coordinadora, ubicación). Ver header de `data/qry_churn_v5.sql` y `data/diccionario_churn_data.md` (ojo: el diccionario es de v2 — el delta vs v5 está en `V4_MENSUAL.md` y los headers de SQL).

**Features excluidas del modelo final** (presentes en el SQL pero no se entrenan): `id_vendedor`, `mes_obs`, `mes_rank_obs`, `fecha_ingreso`, `id_coordinadora`, `ccodrelacion`, `ccodubigeo`, `distrito`, `mes_num`, `anio_mes_num`. Las dos últimas se sacaron tras `notebooks/clean/ablation_temporal_v5.ipynb`: en split forward el modelo es igual o mejor sin ellas y se evita extrapolación de `anio_mes_num` futuros.

**Modelo final vigente**: `HistGradientBoostingClassifier` con `class_weight='balanced'` y hiperparámetros tuneados con Optuna (`learning_rate=0.0175, max_iter=750, max_depth=4, max_leaf_nodes=22, min_samples_leaf=100`). Ver `notebooks/clean/tuning_optuna_v5.ipynb`.

**Métricas vigentes (modelo final tuneado)**: AUC GroupKFold **0.7485 ± 0.008**, AUC split forward **0.7537** (std por mes 0.030), PR-AUC OOF 0.502 (lift 1.83×). Punto operativo recomendado: t=0.50 → recall 0.73, precision 0.43 (lift 1.56× sobre prevalencia). Ver `notebooks/clean/modelo_final_v5.ipynb` §9 para discusión de viabilidad de negocio. Ver `VERSIONES.md` para el cuadro comparativo entre versiones.

## Versionado del dataset

Cada cambio sustantivo del SQL (target, granularidad, filtro de población, schema de features) bumpea la versión y se documenta en `VERSIONES.md`. La versión vigente está marcada con ✅. Las versiones históricas se mantienen en BigQuery para auditoría.

Para crear una versión nueva: ver la sección "Cómo agregar una nueva versión" al final de `VERSIONES.md`.

## Cómo extraer el dataset

```bash
gcloud auth application-default login        # una vez por máquina

# Re-extracción desde el SQL (los CSV no se versionan):
bq query --use_legacy_sql=false < data/qry_churn_v5.sql

# Sanity check rápido:
bq query --use_legacy_sql=false 'SELECT COUNT(*) AS n_rows,
  COUNT(DISTINCT id_vendedor) AS n_vendedoras,
  AVG(churn) AS churn_rate
  FROM `glamour-peru-dw.glamour_dw.training_churn_v5`'
```

O bien correr la primera celda de un notebook que carga la tabla via `google-cloud-bigquery`.

## Reglas del repo

- **No commitear data**: `.gitignore` excluye `data/*` excepto `*.md` y `*.sql`. También excluye `*.csv/parquet/feather` en cualquier subdirectorio.
- **No commitear credenciales**: `.env`, `service-account*.json`, `*.key`, etc. están bloqueados.
- **No commitear `reports/`**: son outputs auto-generados; pueden contener estadísticos del dataset. Si querés versionar uno puntualmente: `git add -f reports/<archivo>`.
- **Notebooks con outputs**: las celdas pueden mostrar datos (`df.head()`, conteos) → revisá los outputs antes de commitear, o usá `nbstripout` (`uvx nbstripout --install` en el repo).

## Convenciones

- **Notebooks generados desde scripts**: a partir del NB 06, cada notebook tiene un builder idempotente en `scripts/build_nb*.py`. Editar el script y regenerar el `.ipynb`; no editar el JSON del notebook a mano. Los entregables limpios (`notebooks/clean/*.ipynb`) llevan marco teórico ML pedagógico; los de iteración (`notebooks/drafts/`) no.
- **EDA → reporte**: en notebooks de EDA, dejar la última celda generando un reporte markdown en `reports/` con los números reales (no escribir números a mano).
- **Validación**: dos protocolos siempre que se reporte una métrica:
  1. **GroupKFold por vendedora** (5 folds) — métrica honesta, evita leakage de identidad. Es la principal a reportar.
  2. **Split temporal forward** (último bloque de 6 unidades como test, GAP = horizonte+1) — escenario de producción. Reportar también std por unidad temporal del bloque test.
- **Leakage**: ver `LEAKEAGE.md`. Variables de target intermedio (`compro_t1..t6`, `monto_t1..t6` en v2/v3) están excluidas desde v3. SCD-1 sospechosos (`estado_coordinadora`) también excluidos.
- **SQL**: comentar cuando se cambia target o filtro de población; documentar el motivo en el header del archivo. La motivación profunda y los números crudos viven en el header del SQL, no en `VERSIONES.md`.
