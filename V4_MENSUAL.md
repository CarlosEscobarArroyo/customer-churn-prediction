# Migración a granularidad mensual — v4

Este documento explica por qué `qry_churn_v4.sql` reemplaza la granularidad de campañas por **mes calendario**, qué decisiones se tomaron y qué falta validar.

---

## Síntesis ejecutiva

La unidad temporal de v3 — la **campaña** — no es consistente:

- Duración: mediana 28 días, **std 81 días**, rango [-15, 1006].
- Mezcla campañas regulares (~30 días, 200-400 pedidos) con liquidaciones (4-15 días, 30-250 pedidos) y outliers de carga.
- Desde **2025-06** el negocio operativo cambió: pasó de 1 campaña/mes a **2-3 campañas/mes simultáneas**.

Esto rompe la definición de churn: "no compra en 6 campañas consecutivas" significa **6 meses calendario** en el régimen 2017-2024, pero **2-3 meses** en el régimen 2025+. El modelo entrenado mayormente con datos pre-2025 aprende un horizonte y se le pide predecir bajo otro. Ese es el origen del drift detectado en `notebooks/04_split_temporal.ipynb` (AUC test 0.729 vs GroupKFold 0.754, churn rate test 35.6% vs train 27.4%).

La solución: **agregar a mes calendario**, donde cada periodo mide ~30 días sin importar cuántas campañas estén activas.

---

## Evidencia

### Distribución de duración de campañas

Sobre las 156 campañas en `dim_campana`:

| Estadística | Días |
|---|---|
| min | -15 |
| p25 | 21 |
| **mediana** | **28** |
| p75 | 33 |
| max | 1006 |
| **std** | **81** |

22 campañas duran ≤ 13 días (liquidaciones cortas). 3 campañas son outliers extremos (-15, 262, 1006 días).

### Cambio de régimen 2025

Campañas por mes calendario:

| Periodo | Campañas/mes típicas |
|---|---|
| 2017 – 2024 | 1 |
| 2025-06 → 2026-04 | 2 a 3 |

Detalle últimos 12 meses:

```
2025-05  1 campaña       2025-11  3 campañas
2025-06  2 campañas      2025-12  3 campañas
2025-07  1 campaña       2026-01  3 campañas
2025-08  1 campaña       2026-02  2 campañas
2025-09  1 campaña       2026-03  3 campañas
2025-10  2 campañas      2026-04  2 campañas
```

### Volumen mensual vs por campaña

Vendedoras únicas que compran (distintas, no suma):

| Granularidad | mediana | std | min |
|---|---|---|---|
| Por campaña (training_churn_v3) | ~150 | ~80 | 19 |
| **Por mes calendario** | **332** | **241** | **30** |

Por mes hay ~2× más muestras y la varianza relativa es menor.

### Datos sucios identificados

| id_campana | Nombre | Problema | Acción en v4 |
|---|---|---|---|
| 20102 | "SANITY BIOSEGURIDAD GLAMOUR" | dur 262 días, 39 pedidos, contexto COVID | excluida |
| 20201 | "PRIVILEGIOS GLAMOUR SAN VALENT" | dur **-15 días** (fecha_fin < fecha_inicio) | excluida |
| 23105 | "CURSOS PRALEMY" | dur 1006 días, 4 pedidos, no es retail | excluida |
| 19202, 21112 | varios | 0 pedidos | quedan fuera por INNER JOIN |

---

## Decisiones de diseño v4

### 1. Granularidad

Una fila por `(id_vendedor, mes_obs)` donde `mes_obs` es un `DATE` con el primer día del mes. Reemplaza `(id_vendedor, id_campana_obs)`.

`mes_rank_obs` reemplaza a `campana_rank_obs` (entero monótono ordenando los meses).

### 2. Agregación por fecha de pedido, no por campaña

El SQL extrae el mes calendario desde `dim_fecha.date` joineado con `fact_pedidos.id_fecha`. Una vendedora que compra en una "regular" + "liquidación" del mismo mes cuenta como **un solo punto** en el panel (1 mes con compra). Esto resuelve la doble-señal contradictoria que generaba v3.

### 3. Limpieza de campañas atípicas

Blacklist al inicio del CTE de pedidos: `WHERE id_campana NOT IN (20102, 20201, 23105)`. Las liquidaciones cortas legítimas se mantienen — sus pedidos suceden en algún mes calendario y se agregan ahí.

### 4. Definición de churn

`churn = 1` si la vendedora **no compra en los próximos 4 meses consecutivos**. NULL si no hay 4 meses de futuro disponibles.

Justificación de `HORIZON_CHURN = 4`:

- En venta directa, 4 meses sin compra es señal robusta de pérdida real.
- 6 meses (equivalente al horizonte viejo en régimen 1 camp/mes) es ya tarde para acción de retención.
- Mantiene la idea conceptual de v3 (~6 campañas en régimen viejo ≈ 6 meses) pero acortada a un horizonte operativo más útil.

**Esta elección está abierta a revisión**. Antes de tomarla como definitiva, replicar en mensual el análisis de gaps de `notebooks/01_eda.ipynb` (Bloque 2) para v3 y elegir el codo de la curva de sensibilidad.

### 5. Ventanas RFM (mismas N, distinta unidad)

Se mantienen los nombres `u3 / u6 / u12`, ahora **en meses**. Hay un trade-off: `u12 meses` es una ventana más larga que `u12 campañas` en régimen 2025+ (12 meses vs 4-6 meses). Eso le da al modelo más memoria histórica, lo cual es deseable.

### 6. Filtro de población

`compras_historicas >= 3` (en mensual; v3 usaba `>= 4` en campañas).

Razonamiento: 3 meses con compra cubre el equivalente mínimo de actividad. Bajar de 4 a 3 mantiene tamaño de muestra cuando se reduce de "campañas" a "meses". Validar en el NB de validación que la "regla trivial" (1 mes con compra ⇒ 99%+ churn) ya no domine.

### 7. Atributos SCD-1

Sin cambios respecto a v3. Mismas notas en el header del SQL: `edad_vendedor`, `tipo_vendedor`, `ccodubigeo`, `id_coordinadora`, `edad_coordinadora` son snapshots. Si el DW expone versiones historizadas en el futuro, regenerar.

---

## Diccionario delta vs v3

### Renombradas

| v3 | v4 | Tipo |
|---|---|---|
| `id_campana_obs` | `mes_obs` | DATE (primer día del mes) |
| `campana_rank_obs` | `mes_rank_obs` | INT |
| `anio_campana` | `anio_mes_num` | INT (año del mes_obs) |
| `numero_campana` | `mes_num` | INT (1-12) |
| `antiguedad_campanas` | `antiguedad_meses` | INT |
| `campanas_desde_compra_previa` | `meses_desde_compra_previa` | INT |

### Sin cambios de nombre (cambia la unidad)

`num_compras_u3/u6/u12`, `monto_total_u3/u6/u12`, `monto_pagado_u3/u6/u12`, `tasa_compra_u3/u6/u12`, `ticket_promedio_u3/u6/u12`, `ratio_pago_u3/u6/u12`, `avg_categorias_u6`, `avg_productos_u6`, `unidades_u6`, `delta_monto_u3_vs_prev3`, `delta_compras_u3_vs_prev3`, `ticket_u3_vs_u12`. Ahora `u3 = últimos 3 meses`, etc.

### Eliminadas

- `fecha_inicio_campana`: el mes ya está representado por `mes_obs` (DATE).

### Sin cambio

`fecha_ingreso`, `edad_vendedor`, `sexo_vendedor`, `tipo_vendedor`, `ccodrelacion`, `id_coordinadora`, `edad_coordinadora`, `ccodubigeo`, `distrito`, `provincia`, `departamento`, `num_pedidos_obs`, `monto_total_obs`, `monto_pagado_obs`, `num_categorias_obs`, `num_productos_obs`, `compras_historicas`, `monto_historico`, `churn`.

### Eliminadas igual que v3 (no se proyectan)

`estado_coordinadora` (leakage SCD-1), `compro_en_obs` (constante post-filtro), `es_nueva_vendedora` (constante post-filtro), `compro_t1..t6` y `monto_t1..t6` (target intermedio).

---

## Cómo regenerar

```bash
# Una vez por máquina:
gcloud auth application-default login

# Ejecutar el SQL (crea/reemplaza training_churn_v4):
bq query --use_legacy_sql=false < data/qry_churn_v4.sql

# Validar conteos rápidos:
bq query --use_legacy_sql=false 'SELECT COUNT(*) AS n_rows,
  COUNT(DISTINCT id_vendedor) AS n_vendedoras,
  COUNT(DISTINCT mes_obs) AS n_meses,
  AVG(churn) AS churn_rate
  FROM `glamour-peru-dw.glamour_dw.training_churn_v4`'
```

O bien correr la primera celda del notebook de validación que se cree (siguiente paso).

---

## Próximos pasos

1. **Notebook de validación `05_validacion_v4.ipynb`** análogo al `02_validacion_v3.ipynb`:
   - Schema esperado vs encontrado (delta vs v3 según este doc).
   - Conteos básicos (n_filas, n_vendedoras, n_meses, churn rate).
   - Granularidad: `(id_vendedor, mes_obs)` debe ser PK.
   - No-leakage: invariantes monotónicos sobre acumulados y ventanas u3 ≤ u6 ≤ u12.
   - SCD-1: confirmar que atributos de vendedor son constantes intra-vendedor.
   - Distribución del churn rate por subgrupos (`compras_historicas` bucketed, `tipo_vendedor`).
   - Comparación v3 vs v4 sobre vendedoras presentes en ambos: ¿cómo cambia el % de churn?
   - Validar que las campañas blacklist no aparecen en ningún pedido del dataset.

2. **Análisis de gaps en mensual** (Bloque 2 del NB 01 replicado): estudiar la distribución del gap entre meses con compra para confirmar/ajustar `HORIZON_CHURN = 4`.

3. **Re-correr baselines `03_baselines.ipynb`** apuntando a `training_churn_v4`. Hipótesis a verificar:
   - AUC GroupKFold ≥ 0.75 (al menos igual que v3).
   - **AUC del split temporal forward (NB 04 reentrenado) sube** respecto a v3 (esperado: el drift se reduce al usar unidad consistente).
   - Std del AUC por mes en el bloque test es **menor** que el std por campaña (esperado: meses tienen muestras más uniformes).

4. **Decidir si v4 reemplaza a v3** como dataset productivo. Criterios:
   - Si el AUC forward sube y la inestabilidad temporal cae → migrar.
   - Si solo cambia ligeramente, mantener v3 y documentar v4 como experimento.

---

## Riesgos abiertos

| Riesgo | Severidad | Mitigación |
|---|---|---|
| Pérdida de resolución intra-mes | Media | Aceptable: el negocio actúa con cadencia mensual; capturar churn quincenal no es crítico. |
| `HORIZON_CHURN = 4` sin justificación cuantitativa todavía | Media | Replicar análisis de gaps del NB 01 antes de cerrar. |
| Meses sin actividad de la vendedora (huecos largos) | Baja | El `WHERE _compro_en_obs = 1` filtra correctamente: solo vemos meses donde la vendedora estuvo activa. |
| Que el modelo migrado pierda performance vs v3 en GroupKFold | Media | Si pasa, indicar que la "señal" de v3 venía en parte del régimen mixto y migrar igual por consistencia operativa. |

---

## Referencias cruzadas

- `STATUS.md` — contexto general del proyecto y baselines anteriores.
- `LEAKEAGE.md` y `CORRELACION.md` — discusión de leakage que sigue aplicando en v4 (GroupKFold por vendedora sigue siendo el método de validación correcto).
- `data/diccionario_churn_data.md` — diccionario v2 (referencia histórica; el de v3 está implícito en el SQL).
- `data/qry_churn_v3.sql` — versión anterior, mantener para comparaciones A/B.
- `notebooks/04_split_temporal.ipynb` — análisis que motivó esta migración.
