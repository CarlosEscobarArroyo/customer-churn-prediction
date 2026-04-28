# Versiones del dataset de churn

Bitácora de cada versión del dataset de entrenamiento. Cada cambio sustantivo
del SQL (granularidad, target, filtro de población, esquema de features) bumpea
la versión y crea una entrada nueva al principio de este archivo.

**Convención**: la versión vigente se marca con ✅. Las versiones históricas se
mantienen en BigQuery para auditoría hasta que se confirme que no se necesitan.

Para detalles que no caben acá:
- Motivación profunda y números crudos: header del SQL correspondiente
  (`data/qry_churn_v<N>.sql`).
- Validaciones y métricas reproducibles: notebooks asociados (`notebooks/`).
- Discusiones puntuales: docs raíz (`STATUS.md`, `LEAKEAGE.md`,
  `CORRELACION.md`, `V4_MENSUAL.md`).

---

## Resumen comparativo

| Versión | Estado | Granularidad | Horizonte | Población | Churn rate | n filas | AUC GKF (HGB) | AUC fwd (HGB) | Std AUC fwd |
|---------|--------|--------------|-----------|-----------|-----------:|--------:|--------------:|--------------:|------------:|
| **v5 (modelo final tuneado)** | ✅ vigente | mes | 6 meses | full data + hist ≥ 3 | 27.5% | 23 684 | **0.7485** | **0.7537** | 0.030 |
| v5 (baseline pre-tuning) | reemplazado por tuneado | mes | 6 meses | full data + hist ≥ 3 | 27.5% | 23 684 | 0.7465 | 0.7509 | 0.031 |
| v6_all  | descartada (target trampa) | mes | 6 meses | post-2022, sin filtro hist | 36.6% | 14 362 | 0.7910 | 0.7614 | 0.029 |
| v6      | candidata (no domina v5) | mes | 6 meses | post-2022 + hist ≥ 3 | 25.9% | 9 841 | 0.7639 | 0.7424 | 0.037 |
| v4      | histórica | mes | 4 meses | full data + hist ≥ 3 | 32.6% | 24 159 | 0.7413 | 0.7286 | 0.030 |
| v3      | histórica | campaña | 6 campañas | full data + hist ≥ 4 | 27.9% | 21 911 | 0.7541 | 0.7294 | 0.082 |
| v2      | retirada (leakage) | campaña | 6 campañas | full data + hist ≥ 4 | 27.9% | — | — | — | — |
| v1      | retirada | campaña | 3 campañas | sin filtro | — | — | — | — | — |

`AUC GKF` = AUC bajo GroupKFold por vendedora (5 folds). `AUC fwd` = AUC del
split temporal forward (último bloque de 6 unidades como test, GAP =
horizonte+1). `Std AUC fwd` = desviación del AUC entre las unidades del bloque
test (mide estabilidad temporal).

---

## v6 / v6_all — Filtro post-pandemia (experimento, no migración)

- **Estado**: experimento concluido el 2026-04-26. **No se migra**: v5 sigue vigente.
- **SQL**: `data/qry_churn_v6.sql`, `data/qry_churn_v6_all.sql`
- **Tablas BQ**: `training_churn_v6`, `training_churn_v6_all`
- **Notebook**: `notebooks/drafts/08_baselines_v6.ipynb`

### Hipótesis probadas

1. **Filtrar pre-pandemia mejora la generalización** (v6 vs v5).
2. **El filtro `compras_historicas >= 3` sigue siendo necesario en régimen
   post-pandemia** (v6 vs v6_all).
3. **El volumen de v5 compensa el ruido pre-pandemia** (v5 vs v6 en
   GroupKFold).

### Configuraciones

- **v6**: cutoff `mes_obs >= '2022-01-01'` + filtro habitual `hist ≥ 3`.
- **v6_all**: mismo cutoff, **sin** `hist ≥ 3` (incluye vendedoras con 1-2
  meses de historia).

### Resultados

| Métrica | v5 | v6 | v6_all |
|---|---:|---:|---:|
| AUC GroupKFold (HGB) | 0.7465 | 0.7639 | **0.7910** |
| AUC split forward (HGB) | **0.7509** | 0.7424 | 0.7614 |
| PR-AUC / prevalencia (GroupKFold) | 1.83× | **1.94×** | 1.80× |
| PR-AUC / prevalencia (forward) | **1.89×** | 1.80× | 1.81× |
| AUC heurística (gap previo) | 0.6568 | **0.6939** | 0.4778 |
| Drift train↔test | 1.9pp | **0.36pp** | 4.8pp |
| n filas | 23 684 | 9 841 | 14 362 |

### Conclusiones

**v6_all (sin filtro de población) — descartada por target trampa**:
- AUC alto (0.7910) es engañoso: viene de juntar dos poblaciones
  distintas (regulares con historia vs nuevas). El modelo aprende a
  separarlas, no a predecir churn dentro de cada grupo.
- Heurística "gap previo" cae a 0.4778 (peor que aleatorio) → la regla
  de recencia se rompe al mezclar.
- PR-AUC sube en absoluto (0.66) pero el lift sobre prevalencia BAJA
  (1.80× vs 1.94× en v6).
- **Confirma que `compras_historicas >= 3` sigue siendo necesario.**

**v6 (post-pandemia + filtro habitual) — viable pero no domina v5**:
- ✓ Mejora AUC GroupKFold (+1.7pp) y reduce drift drásticamente
  (1.9pp → 0.36pp). Régimen más limpio.
- ✗ Empeora AUC split forward (−0.85pp) por menor volumen de
  entrenamiento (2.4× menos muestra).
- Empate técnico en lift forward (1.89× vs 1.80×, <1pp).
- **No justifica migración**: el costo en volumen no se compensa con
  ganancia operacional clara.

**Decisión: mantener v5 como vigente.**

### Hallazgo accionable

La heurística "gap previo" funciona mejor en v6 (0.6939 vs 0.6568 en v5)
y se rompe en v6_all (0.4778). Esto sugiere:
- El régimen post-pandemia es más predecible por recencia simple.
- Vendedoras pre-pandemia agregan ruido al patrón de recencia (probable
  efecto cuarentena/reactivación con gaps largos pero retorno).
- Mezclar nuevas + establecidas rompe la fenomenología.

→ **Próximo experimento sugerido**: feature de interacción
`compras_historicas × meses_desde_compra_previa`, o modelo de dos etapas
(segmentar primero "nueva vs establecida", aplicar modelos distintos).
Esto puede aportar más que seguir iterando sobre filtros temporales.

---

## v5 — HORIZON_CHURN = 6 meses ✅

- **Vigente desde**: 2026-04-26
- **SQL**: `data/qry_churn_v5.sql`
- **Tabla BQ**: `glamour-peru-dw.glamour_dw.training_churn_v5`
- **Notebooks de iteración**: `notebooks/drafts/06_horizonte_v4.ipynb` (validación del horizonte),
  `notebooks/drafts/07_baselines_v5.ipynb` (baselines + comparación contra v4)
- **Notebooks entregables**: `notebooks/clean/modelo_final_v5.ipynb` (modelo vigente),
  `notebooks/clean/ablation_temporal_v5.ipynb` (eliminación de features temporales),
  `notebooks/clean/tuning_optuna_v5.ipynb` (tuning de hiperparámetros)

### Pulido del modelo (2026-04-26, posterior al baseline v5)

Tres iteraciones que mejoraron el modelo sin cambiar el dataset:

1. **Eliminación de `ccodrelacion`**: aparecía top-4 en permutación pero
   era un ID disfrazado. Costo: −0.005 AUC (ruido).
2. **Ablation de `mes_num` y `anio_mes_num`** (`ablation_temporal_v5.ipynb`):
   en split forward el modelo es igual o mejor sin ellas (PR-AUC fwd
   +4.6%). Reduce el riesgo de extrapolación de `anio_mes_num` futuros.
3. **Tuning con Optuna** (`tuning_optuna_v5.ipynb`, 50 trials):
   `learning_rate=0.0175, max_iter=750, max_depth=4, max_leaf_nodes=22, min_samples_leaf=100`.
   `learning_rate` explica 79% de la varianza durante la búsqueda.

| Métrica | v5 baseline | v5 modelo final | Δ |
|---|---:|---:|---:|
| Features (post-OHE excluidas) | 42 | **39** | −3 |
| AUC GroupKFold (mean ± std) | 0.7465 ± 0.007 | **0.7485 ± 0.008** | +0.0020 |
| AUC split forward | 0.7509 | **0.7537** | +0.0028 |
| PR-AUC OOF | 0.5029 | **0.5020** | ≈ |
| Std AUC por mes (test) | 0.034 | 0.030 | mejor estabilidad |

### Punto operativo recomendado: t = 0.50

Recall 0.73, precision 0.43 (lift 1.56× sobre prevalencia), 47% de la
base contactada. Discusión completa de viabilidad de negocio en la
Parte 9 de `notebooks/clean/modelo_final_v5.ipynb`.

### Motivación

El horizonte k=4 de v4 era una elección heurística marcada como "abierta a
revisión". El NB 06 replica el Bloque 2 del NB 01 sobre la granularidad
mensual y muestra:

- **Codo automático (Δtasa < 2pp) cae en k=6**, igual que en v3 — el cambio de
  granularidad campaña → mes no movió el codo del horizonte.
- **Hazard de retorno se aplana entre k=6 (7.3%) y k=7 (5.9%)**: después de
  6 meses sin compra, la probabilidad de retorno es marginal.
- **Cohortes pre-2025 vs post-2025 convergen exactamente en k=6** (27.5% vs
  27.5%). El régimen post-2025 NO requería un horizonte más corto;
  k=4 estaba sobre-ajustado al supuesto de cambio de régimen.

### Cambios vs v4

1. **Target**: `HORIZON_CHURN` pasa de 4 a 6 meses. CTE `target` extiende a
   LEAD 1..6 y la suma de control en `ensamble` se hace sobre t1..t6.
2. **CTE `panel`**: cota superior de `mes_rank` pasa de
   `ultima_compra_rank + 4` a `ultima_compra_rank + 6`.
3. **Tabla de salida**: `training_churn_v5` (v4 se mantiene intacta).

Sin cambios: granularidad mensual, blacklist de campañas, filtro `hist ≥ 3`,
schema de features.

### Métricas

| Métrica | v4 | v5 | Δ |
|---|---:|---:|---:|
| AUC GroupKFold (HGB) | 0.7413 | 0.7465 | +0.0052 |
| AUC split forward (HGB) | 0.7286 | **0.7509** | **+0.0223** |
| PR-AUC GroupKFold (HGB) | 0.5531 | 0.5029 | −0.0502 |
| Lift PR-AUC / prevalencia | 1.69× | **1.84×** | +0.15× |
| Std AUC por mes (test) | 0.030 | 0.034 | +0.004 |
| \|Δ churn rate train↔test\| | 0.36pp | 1.9pp | +1.5pp |
| Falsos churn (gaps que vuelven) | 24.5% | 16.1% | −8.4pp |

### Decisión

**v5 reemplaza a v4** como dataset productivo. AUC sube en ambos protocolos
(notablemente +2.2pp en el split forward), el target tiene menos ruido, y el
modelo generaliza mejor temporalmente (AUC forward casi iguala al GroupKFold,
mientras que en v4 quedaba 1.3pp por debajo). PR-AUC absoluta baja por la
menor prevalencia, pero el lift sobre prevalencia mejora.

---

## v4 — Granularidad mensual

- **Estado**: histórica (reemplazada por v5 el 2026-04-26)
- **SQL**: `data/qry_churn_v4.sql`
- **Tabla BQ**: `glamour-peru-dw.glamour_dw.training_churn_v4`
- **Notebooks**: `notebooks/drafts/05_baselines_v4.ipynb`
- **Doc dedicado**: `V4_MENSUAL.md`

### Motivación

Las "campañas" de v3 dejaron de ser una unidad temporal consistente. Desde
2025-06 el negocio pasó de 1 campaña/mes a 2-3 campañas/mes simultáneas:
"6 campañas sin compra" equivale a 6 meses pre-2025 pero a 2-3 meses en
2025+. Eso explica el drift y la inestabilidad por campaña detectados en
NB 04 (v3): AUC test 0.729 vs GroupKFold 0.754, churn rate test 35.6% vs
train 27.4%, std AUC por campaña 0.082.

### Cambios vs v3

1. **Granularidad**: una fila por (id_vendedor, mes_obs) en vez de
   (id_vendedor, id_campana_obs). `mes_rank_obs` reemplaza a
   `campana_rank_obs`.
2. **Agregación**: por mes calendario usando `dim_fecha`, no por id_campana.
3. **Blacklist de campañas atípicas**: 20102 (COVID), 20201 (fechas
   invertidas), 23105 (curso no-retail).
4. **Horizonte**: 6 campañas → 4 meses (heurístico, "señal robusta de
   pérdida sin esperar 6 meses").
5. **Filtro de población**: `compras_historicas >= 3` (en v3 era ≥ 4 en
   campañas).
6. **Renombres**: `meses_desde_compra_previa`, `mes_rank_obs`, etc.

### Métricas

| Métrica | v3 | v4 | Δ |
|---|---:|---:|---:|
| AUC GroupKFold (HGB) | 0.7541 | 0.7413 | −0.0128 |
| AUC split forward (HGB) | 0.7294 | 0.7286 | −0.0008 |
| Std AUC por unidad temporal | 0.082 (campaña) | **0.030** (mes) | **−0.052** |
| Δ churn rate train↔test | +8.2pp | +0.36pp | mucho mejor |

### Decisión

v4 reemplazó a v3 por consistencia operativa: la granularidad mensual
estabiliza el target temporalmente (std 2.7× más baja, drift casi
eliminado) sin degradar AUC en términos prácticos. Quedó pendiente
validar el horizonte k=4 — esa validación motivó v5.

---

## v3 — Limpieza de leakage + tendencias normalizadas

- **Estado**: histórica
- **SQL**: `data/qry_churn_v3.sql`
- **Tabla BQ**: `training_churn_v3` (puede o no estar en BQ; ver `bq ls`)
- **Notebooks**: `notebooks/drafts/02_validacion_v3.ipynb`,
  `notebooks/drafts/03_baselines.ipynb`, `notebooks/drafts/04_split_temporal.ipynb`

### Cambios vs v2

1. **Eliminadas 6 columnas de leakage**: `compro_t1..t6`, `monto_t1..t6`
   (información del horizonte de target filtrada como features).
2. **Eliminadas 3 pseudo-features constantes**: `compro_en_obs` (=1 siempre
   tras filtrar), `es_nueva_vendedora` (=0), `campanas_desde_ultima_compra`
   (=0).
3. **Eliminada `estado_coordinadora`**: leakage SCD-1 (snapshot del DW
   actual, no histórico).
4. **Tendencias reformuladas como deltas normalizadas en [-1, 1]**:
   `(u3 − prev3) / (u3 + prev3) = (2·u3 − u6) / u6`. Resuelve el problema
   de NaN masivos en v2.
5. **Feature nueva**: `campanas_desde_compra_previa` — gap entre observación
   y compra anterior. Muy predictiva (~36pp de spread en churn rate entre
   gap=1 y gap≥7).

Sin cambios: granularidad campaña, k=6, filtro `hist ≥ 4`, target
`compras_historicas >= 4`.

### Métricas

| Modelo | AUC GroupKFold | PR-AUC |
|---|---:|---:|
| Dummy | 0.5014 | 0.2795 |
| Heurística (gap previo) | 0.6771 | 0.4404 |
| LogReg | 0.7456 | 0.5023 |
| **HGB (ganador)** | **0.7541** | **0.5229** |
| LightGBM | 0.7446 | 0.5054 |
| XGBoost | 0.7505 | 0.5176 |

Split forward (HGB): AUC 0.7294, std AUC por campaña **0.082** (alto),
drift churn rate +8.2pp train→test.

### Hallazgos asociados

- `LEAKEAGE.md`: auditoría completa de leakage v2 → v3.
- `CORRELACION.md`: análisis de redundancia entre features RFM.

---

## v2 — Primera versión funcional

- **Estado**: retirada por leakage múltiple
- **SQL**: `data/qry_churn_v2.sql`
- **Notebooks**: `notebooks/drafts/01_eda.ipynb`

### Forma

- Granularidad: una fila por (id_vendedor, id_campana_obs).
- Target: churn = 1 si no compra en las próximas 6 campañas (k=6 elegido
  vía codo en NB 01 Bloque 2: 30.7% de gaps ≥ 4 campañas, hazard se
  aplana en k=6-7).
- Filtro: `compras_historicas >= 4` (en v1 no había filtro y el 45% de
  vendedoras con 1 sola compra dominaba trivialmente el target).
- Features: RFM con ventanas u3/u6/u12 sobre campañas, recencia,
  histórico, producto.

### Por qué se retiró

- 6 columnas de target intermedio filtradas como features (`compro_t1..t6`,
  `monto_t1..t6`).
- 3 pseudo-features constantes (irrelevantes pero ocupaban schema).
- `estado_coordinadora` con leakage SCD-1.
- Tendencias `delta_*_u3_vs_prev3` con NaN correlacionados con el target
  (los NaN aparecían justo cuando había churn).

Todo eso motivó v3 sin cambiar la definición del problema.

---

## v1 — Iteración descartada

- **Estado**: retirada (no quedó SQL versionado)
- **Por qué**: target k=3 campañas + sin filtro de población. El 45% de
  las vendedoras tenía 1 sola compra y el target era trivialmente 1 para
  ellas. La regla "1 compra → churn" dominaba el aprendizaje. Análisis en
  NB 01 (Bloque 3) lo documenta.

---

## Cómo agregar una nueva versión

1. Crear `data/qry_churn_v<N>.sql` con header documentando el cambio
   sustantivo.
2. Correr la query en BigQuery (`bq query --use_legacy_sql=false`).
3. Crear notebook de validación (`notebooks/drafts/0X_<descripcion>.ipynb`) que
   reproduzca los baselines clave (GroupKFold + split forward) y compare
   contra la versión anterior.
4. Agregar entrada nueva al principio de este archivo (después del
   resumen comparativo): motivación, cambios, métricas, decisión.
5. Actualizar la tabla resumen y la marca ✅.
6. Si se confirma que la versión anterior queda obsoleta, dejarla en BQ
   con etiqueta `histórica` por al menos una iteración más antes de
   considerar drop.
