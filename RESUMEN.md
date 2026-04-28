# Resumen del proyecto — handoff

> Snapshot al 2026-04-26. Solo lo vigente y relevante para arrancar otra sesión.
> Para historia completa: `VERSIONES.md`, `STATUS.md`, headers de SQL.

## 1. Objetivo

Predicción de **silent churn** para vendedoras de **Glamour Perú** (venta directa por catálogo, modelo tipo Avon/Natura, cosmética).

Una vendedora "churnea" cuando deja de comprar sin avisar. El equipo de retención necesita una lista priorizada mensual de quiénes están en riesgo, para contactarlas antes de que se vayan.

## 2. Definición de churn (v5 vigente)

> Una fila representa el estado de una vendedora en un mes calendario. El target mira los 6 meses siguientes.

`churn = 1` si la vendedora **no compra en ninguno de los 6 meses calendario siguientes** al mes observado (M+1 … M+6). `churn = NULL` si no hay 6 meses de futuro disponibles → esas filas se filtran.

- **Horizonte k=6**: ¿por qué 6 meses y no 4 u 8? Para cada k posible se calcula la probabilidad de que una vendedora *vuelva* a comprar después de estar exactamente k meses sin comprar (eso es el "hazard de retorno"). Si esperaste 1 mes, todavía es muy probable que vuelva; si esperaste 6, ya casi no vuelve nadie. Graficando esa probabilidad contra k aparece un "codo": el punto donde la curva se aplana y esperar más meses ya no aporta certeza adicional. En este dataset el codo está en k=6 (hazard 7.3%) → k=7 (5.9%): después de 6 meses sin compra, la probabilidad de que vuelva es marginal, así que llamarla "churn" es seguro. Validado en NB 06.
- **Filtro de población**: `compras_historicas >= 3`. Excluye vendedoras de historia muy corta para evitar la regla trivial "1 compra ⇒ churn".
- **Granularidad mensual** (no por campaña). Razón: desde 2025-06 las "campañas" pasaron de 1/mes a 2-3/mes simultáneas, y dejaron de ser unidad temporal consistente. Detalle en `V4_MENSUAL.md`.

## 3. Dataset vigente

- **Tabla**: `glamour-peru-dw.glamour_dw.training_churn_v5` (BigQuery).
- **SQL**: `data/qry_churn_v5.sql` (autoritativo).
- **Forma**: panel longitudinal — una fila por `(id_vendedor, mes_obs)`.
- **Tamaño**: ~23 700 filas / ~4 200 vendedoras únicas.
- **Cobertura temporal**: 2017-01 → 2025-10 (106 meses).
- **Tasa de churn**: 27.5%.
- **Blacklist de campañas** excluidas del cálculo de pedidos: 20102 (COVID), 20201 (fechas invertidas), 23105 (curso no-retail).

### Features (39 al modelo, tras pulido)

Todas miran solo hacia atrás (`ROWS BETWEEN N PRECEDING AND CURRENT ROW`):

- **RFM en ventanas de 3, 6 y 12 meses**: `num_compras_u{3,6,12}`, `monto_total_u{3,6,12}`, `monto_pagado_u{3,6,12}`, `tasa_compra_u{3,6,12}`, `ticket_promedio_u{3,6,12}`, `ratio_pago_u{3,6,12}`.
- **Recencia**: `meses_desde_compra_previa` (gap a la compra anterior; muy predictiva).
- **Histórico acumulado**: `compras_historicas`, `monto_historico`, `antiguedad_meses`.
- **Diversidad de producto**: `avg_categorias_u6`, `avg_productos_u6`, `unidades_u6`, `num_categorias_obs`, `num_productos_obs`.
- **Tendencias normalizadas en [-1, 1]**: `delta_monto_u3_vs_prev3`, `delta_compras_u3_vs_prev3`, `ticket_u3_vs_u12`. Fórmula: `(2·u3 − u6) / u6`.
- **Mes actual**: `num_pedidos_obs`, `monto_total_obs`, `monto_pagado_obs`.
- **Contexto vendedora** (SCD-1, snapshot actual): `edad_vendedor`, `sexo_vendedor`, `tipo_vendedor`, `edad_coordinadora`, `provincia`, `departamento`.

### Columnas presentes en el SQL pero EXCLUIDAS del modelo

`id_vendedor`, `mes_obs`, `mes_rank_obs`, `fecha_ingreso`, `id_coordinadora`, `ccodrelacion`, `ccodubigeo`, `distrito`, `mes_num`, `anio_mes_num`. Las dos últimas se sacaron tras el ablation (`ablation_temporal_v5.ipynb`): el modelo es igual o mejor sin ellas en split forward y se evita extrapolación de `anio_mes_num` futuros.

## 4. Cómo se construye el dataset (pipeline del SQL)

El SQL `qry_churn_v5.sql` arma el panel en 11 CTEs encadenadas. Resumido:

1. **`rango_fechas` + `meses_ordenados`**: serie continua de meses calendario desde el primer pedido hasta el último, cada mes con un `mes_rank` global. Se usa `GENERATE_DATE_ARRAY` para no depender de `dim_campana` (cuyas fechas son inconsistentes).
2. **`pedidos_limpios`**: pedidos con su mes calendario, excluyendo la blacklist de campañas atípicas.
3. **`pedidos_agg`**: pedidos agregados a nivel `(id_vendedor, mes)` — `num_pedidos`, `monto_total`, `monto_pagado`.
4. **`detalle_limpio` + `producto_agg`**: features de producto (`num_productos_distintos`, `num_categorias_distintas`, etc.) por `(id_vendedor, mes)`.
5. **`vida_vendedor`**: para cada vendedora, su `primer_compra_rank` y `ultima_compra_rank`. Define la ventana de vida activa.
6. **`panel`**: producto cartesiano `(vendedoras × meses)` restringido a la ventana de vida activa de cada una, **extendida hasta `ultima_compra_rank + 6`** para que el último mes observado tenga 6 meses de futuro (y por tanto target calculable). LEFT JOIN con `pedidos_agg` y `producto_agg` rellena con ceros los meses sin compra → así surgen las filas con `compro = 0` que son la base del churn.
7. **`features`**: window functions sobre `panel` para construir RFM u3/u6/u12 y métricas históricas.
8. **`features_derivadas`**: tendencias normalizadas en [-1, 1].
9. **`target`**: por cada `(id_vendedor, mes_rank)` calcula `LEAD(compro, k)` para k=1..6. Si alguno es NULL → no hay 6 meses de futuro → target = NULL. Si todos son 0 → `churn = 1`. Si alguno es 1 → `churn = 0`.
10. **`ensamble`**: junta features + target + atributos SCD de vendedora/coordinadora/ubicación.
11. **SELECT final**: filtra `churn IS NOT NULL` (target calculable), `_compro_en_obs = 1` (consistente con producción: solo scoreamos vendedoras que compraron en el mes), `compras_historicas >= 3`.

### Cómo se obtienen las ventanas de 6 meses (la pieza clave)

```sql
LEAD(compro, k) OVER (PARTITION BY id_vendedor ORDER BY mes_rank)
```

para `k = 1..6`. Eso da, para cada `(vendedora, mes_obs)`, si compró en cada uno de los 6 meses siguientes. La regla es:

```sql
CASE
  WHEN cualquier compro_t1..t6 IS NULL THEN NULL  -- no hay futuro completo
  WHEN SUM(compro_t1..t6) = 0 THEN 1              -- silent churn
  ELSE 0
END
```

El "encaje" del horizonte con el panel se asegura en el CTE `panel`:
`WHERE m.mes_rank <= vv.ultima_compra_rank + 6`. Sin esa cota superior, los últimos 6 meses de cada vendedora tendrían target NULL y se perderían filas observables.

Las ventanas RFM (`u3`, `u6`, `u12`) son **diferentes**: miran *hacia atrás* desde el mes observado (`ROWS BETWEEN N PRECEDING AND CURRENT ROW`). El "6" del target y el "u6" de las features son conceptualmente independientes.

## 5. Cómo extraer el dataset

```bash
gcloud auth application-default login          # una vez por máquina

# Re-extracción desde el SQL (los CSV no se versionan):
bq query --use_legacy_sql=false < data/qry_churn_v5.sql

# Sanity check rápido:
bq query --use_legacy_sql=false 'SELECT COUNT(*) AS n_rows,
  COUNT(DISTINCT id_vendedor) AS n_vendedoras,
  AVG(churn) AS churn_rate
  FROM `glamour-peru-dw.glamour_dw.training_churn_v5`'
```

O cargarlo desde un notebook con `google-cloud-bigquery`.

## 6. Modelo vigente

- **Algoritmo**: `HistGradientBoostingClassifier` (sklearn) con `class_weight='balanced'`.
- **Hiperparámetros** (Optuna, 50 trials):
  ```
  learning_rate=0.0175
  max_iter=750
  max_depth=4
  max_leaf_nodes=22
  min_samples_leaf=100
  ```
  `learning_rate` explica 79% de la varianza durante la búsqueda.
- **Features**: 39 (ver §3).
- **Entregable**: `notebooks/clean/modelo_final_v5.ipynb`.

### Métricas

| Métrica | Valor | Lectura |
|---|---:|---|
| AUC GroupKFold (mean ± std, 5 folds) | **0.7485 ± 0.008** | Métrica honesta — vendedora nunca en train+test simultáneamente |
| AUC split forward | **0.7537** | Escenario de producción (entrenar con histórico, predecir futuro) |
| PR-AUC GroupKFold OOF | 0.502 | Lift 1.83× sobre prevalencia (0.275) |
| Std AUC por mes (test forward) | 0.030 | Estable temporalmente |

### Punto operativo recomendado: t = 0.50

Recall **0.73** / precision **0.43** / lift 1.56× sobre prevalencia / 47% de la base contactada.

Argumento de viabilidad: contactando al mismo 47% al azar atraparíamos al 47% de los churners; con el modelo, atrapamos al 73%. Esos +26 pp de recall son el valor incremental.

## 7. Protocolos de validación (siempre los dos)

Como el dataset es un panel longitudinal (la misma vendedora aparece en muchos meses), un k-fold normal mezclaría filas de María en train y test al mismo tiempo. El modelo aprendería a reconocer a María, no a predecir churn — y la métrica saldría inflada. Por eso usamos dos protocolos diseñados para que esto no pase:

1. **GroupKFold por `id_vendedor`** (5 folds) — **métrica principal**.
   - Cómo funciona: divide a las vendedoras (no las filas) en 5 grupos. En cada fold, todas las filas de una vendedora caen *o* en train *o* en test, nunca en ambos.
   - Por qué es honesto: el modelo se evalúa siempre sobre vendedoras que *nunca vio* durante el entrenamiento. La AUC que sale (0.7485) es la que esperarías si te llega una vendedora nueva mañana.
   - Qué NO controla: el tiempo. El fold de test puede contener meses que en el calendario son anteriores a meses del fold de train. Es válido para medir generalización a personas nuevas, pero no replica el escenario real ("entrenar con el pasado, predecir el futuro").

2. **Split temporal forward** — **escenario de producción**.
   - Cómo funciona: train = todos los meses hasta cierta fecha de corte; test = los **últimos 6 meses** del dataset. Entre ambos se deja un **GAP de 7 meses** (horizonte + 1) sin usar, para evitar que una fila de train con `mes_obs` cercano al corte tenga su ventana de target solapando con meses de test (eso sería un leakage temporal sutil).
   - Por qué importa: replica exactamente cómo se va a usar el modelo en producción — entrenado con todo lo que pasó, scoreando el próximo mes.
   - Métrica a reportar: AUC del bloque test + std por mes (para ver si rinde estable o se cae en algún mes específico). En el modelo vigente: AUC 0.7537, std por mes 0.030.

Ambas métricas tienen que ser similares para confiar en el modelo. Si GroupKFold sale alto pero forward sale bajo → el modelo aprendió patrones que no se sostienen en el tiempo (drift). Si forward sale alto pero GroupKFold no → puede estar memorizando vendedoras. En v5 ambas dan ~0.75, lo cual es buena señal.

## 8. Estado de leakage (resumen)

Resueltos: identidad (GroupKFold), targets intermedios (sacados desde v3), `estado_coordinadora` SCD-1 (sacado desde v3), categóricas de alta cardinalidad (`distrito`, `ccodubigeo`), `ccodrelacion` (ID disfrazado, sacado en esta iteración), `mes_num` y `anio_mes_num` (ablation forward, sacados en esta iteración).

Abiertos menores: correlación entre filas de la misma vendedora dentro de train (no es leakage técnico, ver `CORRELACION.md`); patrón de NaN de `tendencia_*` en inactividad (HGB lo aprovecha — verificar consistencia en producción).

Detalle completo: `STATUS.md` (checklist) y `LEAKEAGE.md` (auditoría histórica).

## 9. Notebooks relevantes

```
notebooks/clean/                              # entregables (con marco teórico ML)
  modelo_final_v5.ipynb                       # modelo vigente — leer §9 para viabilidad de negocio
  ablation_temporal_v5.ipynb                  # justifica sacar mes_num/anio_mes_num
  tuning_optuna_v5.ipynb                      # tuning de hiperparámetros

notebooks/drafts/                             # iteración exploratoria
  06_horizonte_v4.ipynb                       # justifica k=6
  07_baselines_v5.ipynb                       # baselines v5 (pre-tuning)
  08_baselines_v6.ipynb                       # comparación v5 vs v6 (descartado)
```

## 10. Stack y comandos

- Python 3.12, deps con `uv`. `uv sync` para instalar.
- BigQuery: `glamour-peru-dw.glamour_dw`.
- Modelado: scikit-learn, XGBoost, LightGBM, SHAP, Optuna.
- `uv run jupyter lab` / `uv run pytest` / `uv run ruff check .` / `uv run mypy src/`.
- Notebooks generados desde `scripts/build_nb*.py` (idempotentes). **Editar el script, no el JSON.**
