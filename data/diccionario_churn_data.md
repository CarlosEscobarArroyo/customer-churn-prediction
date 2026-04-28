# Diccionario de Datos — Dataset de Silent Churn (v2)

> ⚠️ **Este diccionario describe el schema del dataset v2** (granularidad campaña, hist ≥ 4). El dataset vigente es **v5** (granularidad mensual, hist ≥ 3). Ver `V4_MENSUAL.md` y `VERSIONES.md` para los renombres y diferencias.
>
> **Features excluidas del modelo final v5** (presentes en el SQL pero no se entrenan): `id_vendedor`, `mes_obs`, `mes_rank_obs`, `fecha_ingreso`, `id_coordinadora`, `ccodrelacion` (ID disfrazado, ver §6 de `STATUS.md`), `ccodubigeo`, `distrito`, `mes_num`, `anio_mes_num` (eliminadas por `notebooks/clean/ablation_temporal_v5.ipynb`). Ver `notebooks/clean/modelo_final_v5.ipynb` §2.2 para el detalle.

## Resumen del Proyecto

| Campo | Detalle |
|---|---|
| **Proyecto** | Modelo de predicción de silent churn para venta directa |
| **Versión** | v2 |
| **Granularidad** | Una fila por `(id_vendedor, id_campana_obs)` |
| **Definición de churn** | `churn = 1` si la vendedora no compra en las **6 campañas consecutivas** posteriores a la campaña observada |
| **Población final** | Filas con target válido, donde la vendedora compró en la campaña observada **y con `compras_historicas >= 4`** |
| **Advertencia principal** | ⚠️ No entrenar con variables futuras: `compro_t1`…`compro_t6`, `monto_t1`…`monto_t6` |
| **Recomendación** | Excluir también identificadores directos y variables constantes como `compro_en_obs` |

### Changelog

| Versión | Cambio | Motivo |
|---|---|---|
| v2 | Ventana de churn ampliada de **3 a 6 campañas** | El 30.7% de los gaps entre compras consecutivas son ≥ 4 campañas. Con ventana de 3 se marcaba como churn a vendedoras legítimamente esporádicas. |
| v2 | Filtro de historia mínima: **`compras_historicas >= 4`** | El 45% de vendedoras tenía una sola compra → churn trivial del 99.94%. El filtro elimina ese ruido para que el modelo aprenda patrones reales. |

---

## Diccionario de Variables

### 🔑 Claves e Identificadores

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Irrelevante | `id_vendedor` | Identificador | Identificador único de la vendedora. | Viene de `dim_vendedor` y se cruza con el panel vendedor-campaña. | `dim_vendedor / panel` | No usar como feature directa; puede generar sobreajuste. |
| Irrelevante | `id_campana_obs` | Identificador temporal | Campaña de observación desde la cual se predice el churn. | Es `fd.id_campana` renombrado en el ensamble final. | `features_derivadas / ensamble` | Puede mantenerse para trazabilidad; cuidado si se usa como feature directa. |
| Débilmente Relevante | `campana_rank_obs` | Orden temporal | Orden consecutivo global de campañas. | `ROW_NUMBER() OVER (ORDER BY anio, numero_campana)` en `dim_campana`. | `campanas_ordenadas` | Recomendado ordenar también por `id_campana` para robustez. |

---

### 🕐 Tiempo

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Débilmente Relevante | `anio_campana` | Feature temporal | Año de la campaña observada. | Tomado de `dim_campana.anio`. | `campanas_ordenadas / panel` | Útil para capturar estacionalidad o cambios por año. |
| Débilmente Relevante | `numero_campana` | Feature temporal | Número de campaña dentro del año. | Tomado de `dim_campana.numero_campana`. | `campanas_ordenadas / panel` | Útil para estacionalidad por campaña. |
| Débilmente Relevante | `fecha_inicio_campana` | Trazabilidad temporal | Fecha de inicio de la campaña observada. | Tomado de `dim_campana.fecha_inicio`. | `campanas_ordenadas / panel` | Puede usarse para partición/análisis; evitar usar fecha cruda sin transformación. |

---

### 👤 Vendedora

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Fuertemente Relevante | `fecha_ingreso` | Feature contextual | Fecha de ingreso de la vendedora. | Tomado de `dim_vendedor.fecha_ingreso`. | `dim_vendedor / panel` | Validar si representa la fecha real histórica de ingreso. |
| Fuertemente Relevante | `edad_vendedor` | Feature contextual | Edad de la vendedora. | Tomado de `dim_vendedor.edad`. | `dim_vendedor / panel` | Riesgo: si es edad actual, no representa edad histórica en campañas pasadas. |
| Fuertemente Relevante | `sexo_vendedor` | Feature categórica | Sexo registrado de la vendedora. | Tomado de `dim_vendedor.csexpersona`. | `dim_vendedor / panel` | Validar calidad y valores nulos. |
| Fuertemente Relevante | `tipo_vendedor` | Feature categórica | Tipo o categoría de vendedora. | Tomado de `dim_vendedor.tipo_vendedor`. | `dim_vendedor / panel` | Riesgo si refleja estado actual y no histórico. |
| Irrelevante | `ccodrelacion` | Feature categórica | Código de relación asociado a la vendedora. | Tomado de `dim_vendedor.ccodrelacion`. | `dim_vendedor / panel` | Validar significado de negocio antes de modelar. |
| Fuertemente Relevante | `antiguedad_campanas` | Feature numérica | Cantidad de campañas desde la primera compra observada hasta la campaña actual. | `ROW_NUMBER()` por vendedor ordenado por `campana_rank`. | `features` | Mide antigüedad dentro del panel, no necesariamente desde `fecha_ingreso`. |
| Fuertemente Relevante | `es_nueva_vendedora` | Feature binaria | Indica si la vendedora tiene menos de 3 campañas de historia en el panel. | `IF(ROW_NUMBER() < 3, 1, 0)`. | `features` | Definición analítica; puede diferir del flag de negocio en `fact_pedidos`. |

---

### 🧑‍💼 Coordinadora

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Irrelevante | `id_coordinadora` | Feature categórica / identificador | Coordinadora asignada a la vendedora. | Tomado de `dim_vendedor.id_coordinadora`. | `dim_vendedor / panel` | Riesgo si es asignación actual y no histórica. |
| Irrelevante | `edad_coordinadora` | Feature contextual | Edad de la coordinadora. | Join con `dim_coordinadora` usando `id_coordinadora`. | `ensamble` | Validar si edad es actual o histórica. |
| Irrelevante | `estado_coordinadora` | Feature categórica | Estado de la coordinadora. | Join con `dim_coordinadora` usando `id_coordinadora`. | `ensamble` | Riesgo si estado es actual y no histórico. |

---

### 📍 Ubicación

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Irrelevante | `ccodubigeo` | Feature categórica / clave | Código de ubicación de la vendedora. | Tomado de `dim_vendedor.ccodubigeo`. | `dim_vendedor / panel` | Riesgo si ubicación es actual y no histórica. |
| Fuertemente Relevante | `distrito` | Feature categórica | Distrito asociado al ubigeo. | Join con `dim_ubicacion` usando `ccodubigeo`. | `ensamble` | Puede tener alta cardinalidad. |
| Fuertemente Relevante | `provincia` | Feature categórica | Provincia asociada al ubigeo. | Join con `dim_ubicacion` usando `ccodubigeo`. | `ensamble` | Útil para segmentación geográfica. |
| Fuertemente Relevante | `departamento` | Feature categórica | Departamento asociado al ubigeo. | Join con `dim_ubicacion` usando `ccodubigeo`. | `ensamble` | Útil para segmentación geográfica. |

---

### 🛒 Campaña Observada

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Irrelevante | `compro_en_obs` | Variable constante / trazabilidad | Indica si compró en la campaña observada. | `IF(pa.num_pedidos > 0, 1, 0)`. | `panel` | En el dataset final siempre vale 1 por el filtro `compro_en_obs = 1`; no aporta al modelo. |
| Fuertemente Relevante | `num_pedidos_obs` | Feature numérica | Número de pedidos realizados en la campaña observada. | `COUNT(DISTINCT id_pedido)` agregado por vendedor-campaña. | `pedidos_agg / panel` | Feature válida si el modelo se ejecuta después del cierre de campaña. |
| — | `monto_total_obs` | Feature numérica | Monto total pedido en la campaña observada. | `SUM(monto_total_pedido)` por vendedor-campaña. | `pedidos_agg / panel` | Feature válida post-campaña. |
| Fuertemente Relevante | `monto_pagado_obs` | Feature numérica | Monto pagado en la campaña observada. | `SUM(monto_pagado)` por vendedor-campaña. | `pedidos_agg / panel` | Feature válida post-campaña. |
| Fuertemente Relevante | `num_categorias_obs` | Feature numérica | Número de categorías distintas compradas en la campaña observada. | `COUNT(DISTINCT p.categoria)` desde detalle + `dim_producto`. | `producto_agg / panel` | Mide diversidad de compra en la campaña actual. |
| Fuertemente Relevante | `num_productos_obs` | Feature numérica | Número de productos distintos comprados en la campaña observada. | `COUNT(DISTINCT d.id_producto)` desde `fact_pedidos_detalle`. | `producto_agg / panel` | Mide diversidad de productos comprados. |

---

### 📊 RFM — Últimas 3 Campañas

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Fuertemente Relevante | `num_compras_u3` | Feature numérica | Número de campañas con compra en las últimas 3 campañas, incluyendo la actual. | `SUM(compro) OVER ventana w3`. | `features` | Ventana actual + 2 campañas previas. |
| — | `monto_total_u3` | Feature numérica | Monto total pedido en las últimas 3 campañas. | `SUM(monto_total) OVER w3`. | `features` | Incluye campaña observada. |
| Fuertemente Relevante | `monto_pagado_u3` | Feature numérica | Monto pagado en las últimas 3 campañas. | `SUM(monto_pagado) OVER w3`. | `features` | Incluye campaña observada. |
| Débilmente Relevante | `tasa_compra_u3` | Feature numérica | Proporción de campañas con compra en las últimas 3 campañas. | `AVG(compro) OVER w3`. | `features` | Rango aproximado 0 a 1. |
| Débilmente Relevante | `ticket_promedio_u3` | Feature numérica | Monto promedio por campaña con compra en las últimas 3 campañas. | `SAFE_DIVIDE(SUM(monto_total) OVER w3, NULLIF(SUM(compro) OVER w3, 0))`. | `features` | Puede ser NULL si no hubo compras en ventana, aunque con `compro_en_obs=1` debería existir. |
| Débilmente Relevante | `ratio_pago_u3` | Feature numérica | Proporción del monto pedido que fue pagado en las últimas 3 campañas. | `SAFE_DIVIDE(SUM(monto_pagado) OVER w3, NULLIF(SUM(monto_total) OVER w3, 0))`. | `features` | Revisar valores extremos o mayores a 1. |

---

### 📊 RFM — Últimas 6 Campañas

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Débilmente Relevante | `num_compras_u6` | Feature numérica | Número de campañas con compra en las últimas 6 campañas, incluyendo la actual. | `SUM(compro) OVER ventana w6`. | `features` | Ventana actual + 5 campañas previas. |
| Débilmente Relevante | `monto_total_u6` | Feature numérica | Monto total pedido en las últimas 6 campañas. | `SUM(monto_total) OVER w6`. | `features` | Incluye campaña observada. |
| Débilmente Relevante | `monto_pagado_u6` | Feature numérica | Monto pagado en las últimas 6 campañas. | `SUM(monto_pagado) OVER w6`. | `features` | Incluye campaña observada. |
| Débilmente Relevante | `tasa_compra_u6` | Feature numérica | Proporción de campañas con compra en las últimas 6 campañas. | `AVG(compro) OVER w6`. | `features` | Rango aproximado 0 a 1. |
| Débilmente Relevante | `ticket_promedio_u6` | Feature numérica | Monto promedio por campaña con compra en las últimas 6 campañas. | `SAFE_DIVIDE(SUM(monto_total) OVER w6, NULLIF(SUM(compro) OVER w6, 0))`. | `features` | Mide ticket promedio reciente. |
| Débilmente Relevante | `ratio_pago_u6` | Feature numérica | Proporción del monto pedido que fue pagado en las últimas 6 campañas. | `SAFE_DIVIDE(SUM(monto_pagado) OVER w6, NULLIF(SUM(monto_total) OVER w6, 0))`. | `features` | Revisar calidad de `monto_pagado`. |

---

### 📊 RFM — Últimas 12 Campañas

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Débilmente Relevante | `num_compras_u12` | Feature numérica | Número de campañas con compra en las últimas 12 campañas, incluyendo la actual. | `SUM(compro) OVER ventana w12`. | `features` | Ventana actual + 11 campañas previas. |
| Débilmente Relevante | `monto_total_u12` | Feature numérica | Monto total pedido en las últimas 12 campañas. | `SUM(monto_total) OVER w12`. | `features` | Incluye campaña observada. |
| Débilmente Relevante | `monto_pagado_u12` | Feature numérica | Monto pagado en las últimas 12 campañas. | `SUM(monto_pagado) OVER w12`. | `features` | Incluye campaña observada. |
| Débilmente Relevante | `tasa_compra_u12` | Feature numérica | Proporción de campañas con compra en las últimas 12 campañas. | `AVG(compro) OVER w12`. | `features` | Rango aproximado 0 a 1. |
| Débilmente Relevante | `ticket_promedio_u12` | Feature numérica | Monto promedio por campaña con compra en las últimas 12 campañas. | `SAFE_DIVIDE(SUM(monto_total) OVER w12, NULLIF(SUM(compro) OVER w12, 0))`. | `features` | Referencia de ticket de largo plazo. |
| Débilmente Relevante | `ratio_pago_u12` | Feature numérica | Proporción del monto pedido que fue pagado en las últimas 12 campañas. | `SAFE_DIVIDE(SUM(monto_pagado) OVER w12, NULLIF(SUM(monto_total) OVER w12, 0))`. | `features` | Referencia de comportamiento de pago de largo plazo. |

---

### 📦 Producto

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Débilmente Relevante | `avg_categorias_u6` | Feature numérica | Promedio de categorías distintas compradas en las últimas 6 campañas. | `AVG(num_categorias) OVER w6`. | `features` | Promedia ceros en campañas sin compra. |
| Débilmente Relevante | `avg_productos_u6` | Feature numérica | Promedio de productos distintos comprados en las últimas 6 campañas. | `AVG(num_productos) OVER w6`. | `features` | Promedia ceros en campañas sin compra. |
| Débilmente Relevante | `unidades_u6` | Feature numérica | Total de unidades compradas en las últimas 6 campañas. | `SUM(unidades) OVER w6`. | `features` | Feature de volumen reciente. |

---

### 📈 Recencia e Historial

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Fuertemente Relevante | `campanas_desde_ultima_compra` | Feature numérica | Número de campañas transcurridas desde la última compra hasta la campaña observada. | `campana_rank - MAX(IF(compro=1, campana_rank, NULL))` acumulado por vendedor. | `features` | Como se filtra `compro_en_obs=1`, normalmente será 0 en la fila final. |
| Fuertemente Relevante | `compras_historicas` | Feature numérica | Total acumulado de campañas con compra hasta la campaña observada. | `SUM(compro)` acumulado por vendedor hasta la campaña actual. | `features` | Mide historial de actividad. |
| Fuertemente Relevante | `monto_historico` | Feature numérica | Monto total acumulado hasta la campaña observada. | `SUM(monto_total)` acumulado por vendedor hasta la campaña actual. | `features` | Puede correlacionarse con antigüedad. |

---

### 📉 Tendencias

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Débilmente Relevante | `tendencia_monto_u3_vs_prev3` | Feature numérica | Compara monto de últimas 3 campañas contra las 3 campañas previas. | `monto_total_u3 / (monto_total_u6 - monto_total_u3)`. | `features_derivadas` | Puede ser NULL si el periodo previo tuvo monto 0. |
| Débilmente Relevante | `tendencia_compras_u3_vs_prev3` | Feature numérica | Compara frecuencia de compra reciente contra frecuencia de las 3 campañas previas. | `num_compras_u3 / (num_compras_u6 - num_compras_u3)`. | `features_derivadas` | Puede ser NULL si no hubo compras en las 3 previas. |
| Débilmente Relevante | `ticket_u3_vs_u12` | Feature numérica | Compara ticket promedio reciente contra ticket promedio de largo plazo. | `ticket_promedio_u3 / ticket_promedio_u12`. | `features_derivadas` | Mide aceleración o caída de gasto. |

---

### 🚫 Futuro / Leakage (NO usar como features)

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Irrelevante | `compro_t1` | No usar como feature | Indica si la vendedora compró en la campaña t+1. | `LEAD(compro, 1)` por vendedor ordenado por `campana_rank`. | `target` | ⚠️ Leakage: información futura. |
| Irrelevante | `compro_t2` | No usar como feature | Indica si compró en la campaña t+2. | `LEAD(compro, 2)` por vendedor ordenado por `campana_rank`. | `target` | ⚠️ Leakage: información futura. |
| Irrelevante | `compro_t3` | No usar como feature | Indica si compró en la campaña t+3. | `LEAD(compro, 3)` por vendedor ordenado por `campana_rank`. | `target` | ⚠️ Leakage: información futura. |
| Irrelevante | `compro_t4` | No usar como feature | Indica si compró en la campaña t+4. | `LEAD(compro, 4)` por vendedor ordenado por `campana_rank`. | `target` | ⚠️ Leakage: información futura. Nuevo en v2. |
| Irrelevante | `compro_t5` | No usar como feature | Indica si compró en la campaña t+5. | `LEAD(compro, 5)` por vendedor ordenado por `campana_rank`. | `target` | ⚠️ Leakage: información futura. Nuevo en v2. |
| Irrelevante | `compro_t6` | No usar como feature | Indica si compró en la campaña t+6. | `LEAD(compro, 6)` por vendedor ordenado por `campana_rank`. | `target` | ⚠️ Leakage: información futura. Nuevo en v2. |
| Irrelevante | `monto_t1` | No usar como feature | Monto comprado en la campaña t+1. | `LEAD(monto_total, 1)` por vendedor ordenado por `campana_rank`. | `target` | ⚠️ Leakage: información futura. |
| Irrelevante | `monto_t2` | No usar como feature | Monto comprado en la campaña t+2. | `LEAD(monto_total, 2)` por vendedor ordenado por `campana_rank`. | `target` | ⚠️ Leakage: información futura. |
| Irrelevante | `monto_t3` | No usar como feature | Monto comprado en la campaña t+3. | `LEAD(monto_total, 3)` por vendedor ordenado por `campana_rank`. | `target` | ⚠️ Leakage: información futura. |
| Irrelevante | `monto_t4` | No usar como feature | Monto comprado en la campaña t+4. | `LEAD(monto_total, 4)` por vendedor ordenado por `campana_rank`. | `target` | ⚠️ Leakage: información futura. Nuevo en v2. |
| Irrelevante | `monto_t5` | No usar como feature | Monto comprado en la campaña t+5. | `LEAD(monto_total, 5)` por vendedor ordenado por `campana_rank`. | `target` | ⚠️ Leakage: información futura. Nuevo en v2. |
| Irrelevante | `monto_t6` | No usar como feature | Monto comprado en la campaña t+6. | `LEAD(monto_total, 6)` por vendedor ordenado por `campana_rank`. | `target` | ⚠️ Leakage: información futura. Nuevo en v2. |

---

### 🎯 Target

| Relevancia | Columna | Tipo de Rol | Descripción | Cómo se obtuvo | Fuente / CTE | Observaciones |
|---|---|---|---|---|---|---|
| Fuertemente Relevante | `churn` | Variable objetivo | Vale 1 si no compró en ninguna de las **6 campañas** siguientes; vale 0 si compró al menos una vez. | `CASE WHEN compro_t1+…+compro_t6 = 0 THEN 1 ELSE 0 END`; NULL si falta futuro completo. | `ensamble` | Es el target. No usar como feature. |

---

## Variables a Excluir del Modelo

| Columna | Motivo | Acción recomendada |
|---|---|---|
| `compro_t1` … `compro_t6` | Información futura usada para construir el target. | Eliminar antes de entrenar. |
| `monto_t1` … `monto_t6` | Información futura usada para construir el target. | Eliminar antes de entrenar. |
| `churn` | Es la variable objetivo. | Separarla como `y`/target, no incluirla en `X`/features. |
| `compro_en_obs` | Constante por el filtro final `compro_en_obs = 1`. | Eliminar como feature; mantener solo si se desea trazabilidad. |
| `id_vendedor` | Identificador directo de la persona/vendedora. | No usar como feature directa; puede causar sobreajuste. |
| `id_campana_obs` | Identificador temporal crudo. | Usar con cuidado; preferible transformar a año, número de campaña o estacionalidad. |
| `campana_rank_obs` | Índice temporal crudo. | Usar con cuidado; puede capturar tendencia temporal sin generalizar. |

> **Filtro de población (v2):** aplicar `WHERE compras_historicas >= 4` antes de entrenar para excluir vendedoras con historia insuficiente.

---

## Metadata de Fuentes (Schema BigQuery)

### `dim_campana`
| Columna | Posición | Tipo | Nullable |
|---|---|---|---|
| `id_campana` | 1 | INT64 | YES |
| `nombre_campana` | 2 | STRING | YES |
| `estado` | 3 | STRING | YES |
| `fecha_inicio` | 4 | DATE | YES |
| `fecha_fin` | 5 | DATE | YES |
| `anio` | 6 | INT64 | YES |
| `numero_campana` | 7 | INT64 | YES |

### `dim_coordinadora`
| Columna | Posición | Tipo | Nullable |
|---|---|---|---|
| `id_coordinadora` | 1 | STRING | YES |
| `nombre_coordinadora` | 2 | STRING | YES |
| `estado` | 3 | STRING | YES |
| `edad` | 4 | FLOAT64 | YES |

### `dim_fecha`
| Columna | Posición | Tipo | Nullable |
|---|---|---|---|
| `date` | 1 | DATE | YES |
| `id_fecha` | 2 | INT64 | YES |
| `calendaryear` | 3 | INT64 | YES |
| `calendarmonth` | 4 | INT64 | YES |
| `calendardayinmonth` | 5 | INT64 | YES |
| `calendarquarter` | 6 | INT64 | YES |

### `dim_producto`
| Columna | Posición | Tipo | Nullable |
|---|---|---|---|
| `id_producto` | 1 | STRING | YES |
| `nombre_producto` | 2 | STRING | YES |
| `categoria` | 3 | STRING | YES |
| `subcategoria` | 4 | STRING | YES |

### `dim_ubicacion`
| Columna | Posición | Tipo | Nullable |
|---|---|---|---|
| `ccodubigeo` | 1 | STRING | YES |
| `id_ubicacion` | 2 | INT64 | YES |
| `distrito` | 3 | STRING | YES |
| `provincia` | 4 | STRING | YES |
| `departamento` | 5 | STRING | YES |

### `dim_vendedor`
| Columna | Posición | Tipo | Nullable |
|---|---|---|---|
| `id_vendedor` | 1 | INT64 | YES |
| `nombre_vendedor` | 2 | STRING | YES |
| `csexpersona` | 3 | STRING | YES |
| `ccodubigeo` | 4 | STRING | YES |
| `nmovil` | 5 | STRING | YES |
| `ccodrelacion` | 6 | INT64 | YES |
| `id_coordinadora` | 7 | STRING | YES |
| `fecha_ingreso` | 8 | DATE | YES |
| `edad` | 9 | FLOAT64 | YES |
| `tipo_vendedor` | 10 | STRING | YES |

### `fact_pedidos`
| Columna | Posición | Tipo | Nullable |
|---|---|---|---|
| `id_pedido` | 1 | INT64 | YES |
| `id_campana` | 2 | INT64 | YES |
| `id_vendedor` | 3 | INT64 | YES |
| `id_fecha` | 4 | INT64 | YES |
| `id_coordinadora` | 5 | STRING | YES |
| `id_ubicacion` | 6 | FLOAT64 | YES |
| `monto_total_pedido` | 7 | FLOAT64 | YES |
| `monto_pagado` | 8 | FLOAT64 | YES |
| `es_nueva_vendedora` | 9 | INT64 | YES |
| `tipo` | 10 | STRING | YES |

### `fact_pedidos_detalle`
| Columna | Posición | Tipo | Nullable |
|---|---|---|---|
| `id_pedido` | 1 | INT64 | YES |
| `id_campana` | 2 | INT64 | YES |
| `id_vendedor` | 3 | INT64 | YES |
| `id_fecha` | 4 | INT64 | YES |
| `id_ubicacion` | 5 | FLOAT64 | YES |
| `id_pedido_detalle` | 6 | INT64 | YES |
| `id_producto` | 7 | STRING | YES |
| `cantidad` | 8 | FLOAT64 | YES |
| `importe_producto` | 9 | FLOAT64 | YES |

### `training_churn` (60 columnas)
| Columna | Posición | Tipo | Nullable |
|---|---|---|---|
| `id_vendedor` | 1 | INT64 | YES |
| `id_campana_obs` | 2 | INT64 | YES |
| `campana_rank_obs` | 3 | INT64 | YES |
| `anio_campana` | 4 | INT64 | YES |
| `numero_campana` | 5 | INT64 | YES |
| `fecha_inicio_campana` | 6 | DATE | YES |
| `fecha_ingreso` | 7 | DATE | YES |
| `edad_vendedor` | 8 | FLOAT64 | YES |
| `sexo_vendedor` | 9 | STRING | YES |
| `tipo_vendedor` | 10 | STRING | YES |
| `ccodrelacion` | 11 | INT64 | YES |
| `antiguedad_campanas` | 12 | INT64 | YES |
| `es_nueva_vendedora` | 13 | INT64 | YES |
| `id_coordinadora` | 14 | STRING | YES |
| `edad_coordinadora` | 15 | FLOAT64 | YES |
| `estado_coordinadora` | 16 | STRING | YES |
| `ccodubigeo` | 17 | STRING | YES |
| `distrito` | 18 | STRING | YES |
| `provincia` | 19 | STRING | YES |
| `departamento` | 20 | STRING | YES |
| `compro_en_obs` | 21 | INT64 | YES |
| `num_pedidos_obs` | 22 | INT64 | YES |
| `monto_total_obs` | 23 | FLOAT64 | YES |
| `monto_pagado_obs` | 24 | FLOAT64 | YES |
| `num_categorias_obs` | 25 | INT64 | YES |
| `num_productos_obs` | 26 | INT64 | YES |
| `num_compras_u3` | 27 | INT64 | YES |
| `monto_total_u3` | 28 | FLOAT64 | YES |
| `monto_pagado_u3` | 29 | FLOAT64 | YES |
| `tasa_compra_u3` | 30 | FLOAT64 | YES |
| `ticket_promedio_u3` | 31 | FLOAT64 | YES |
| `ratio_pago_u3` | 32 | FLOAT64 | YES |
| `num_compras_u6` | 33 | INT64 | YES |
| `monto_total_u6` | 34 | FLOAT64 | YES |
| `monto_pagado_u6` | 35 | FLOAT64 | YES |
| `tasa_compra_u6` | 36 | FLOAT64 | YES |
| `ticket_promedio_u6` | 37 | FLOAT64 | YES |
| `ratio_pago_u6` | 38 | FLOAT64 | YES |
| `num_compras_u12` | 39 | INT64 | YES |
| `monto_total_u12` | 40 | FLOAT64 | YES |
| `monto_pagado_u12` | 41 | FLOAT64 | YES |
| `tasa_compra_u12` | 42 | FLOAT64 | YES |
| `ticket_promedio_u12` | 43 | FLOAT64 | YES |
| `ratio_pago_u12` | 44 | FLOAT64 | YES |
| `avg_categorias_u6` | 45 | FLOAT64 | YES |
| `avg_productos_u6` | 46 | FLOAT64 | YES |
| `unidades_u6` | 47 | FLOAT64 | YES |
| `campanas_desde_ultima_compra` | 48 | INT64 | YES |
| `compras_historicas` | 49 | INT64 | YES |
| `monto_historico` | 50 | FLOAT64 | YES |
| `tendencia_monto_u3_vs_prev3` | 51 | FLOAT64 | YES |
| `tendencia_compras_u3_vs_prev3` | 52 | FLOAT64 | YES |
| `ticket_u3_vs_u12` | 53 | FLOAT64 | YES |
| `compro_t1` | 54 | INT64 | YES |
| `compro_t2` | 55 | INT64 | YES |
| `compro_t3` | 56 | INT64 | YES |
| `monto_t1` | 57 | FLOAT64 | YES |
| `monto_t2` | 58 | FLOAT64 | YES |
| `monto_t3` | 59 | FLOAT64 | YES |
| `churn` | 60 | INT64 | YES |

### `vw_pedidos_analytics`
| Columna | Posición | Tipo | Nullable |
|---|---|---|---|
| `monto_total_pedido` | 1 | FLOAT64 | YES |
| `date` | 2 | DATE | YES |
| `calendaryear` | 3 | INT64 | YES |
| `calendarmonth` | 4 | INT64 | YES |
| `calendarquarter` | 5 | INT64 | YES |