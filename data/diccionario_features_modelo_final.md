# Diccionario de features — modelo final v5

Variables que **efectivamente entran al modelo** `HistGradientBoostingClassifier`
(modelo vigente, ver `notebooks/clean/modelo_final_v5.ipynb`).

- **Dataset fuente**: `glamour-peru-dw.glamour_dw.training_churn_v5` (ver `data/qry_churn_v5.sql`).
- **Granularidad**: una fila por `(id_vendedor, mes_obs)` — mes calendario, no campaña.
- **Total**: **39 features** = 35 numéricas + 4 categóricas.
- **Ventanas RFM**: `u3` / `u6` / `u12` = últimos 3 / 6 / 12 **meses** (`ROWS BETWEEN N PRECEDING AND CURRENT ROW`, incluye el mes actual).
- **Target**: `churn` (no es feature). `=1` si la vendedora no compra en los próximos 6 meses consecutivos.

> **Nota de scope**: el SQL produce más columnas, pero el notebook las descarta en el
> set `EXCLUDE` (claves, IDs, calendario). El detalle de qué se excluye y por qué está
> al final de este documento.

---

## 1. Atributos de la vendedora

| Feature | Tipo | Definición | Origen SQL |
|---|---|---|---|
| `edad_vendedor` | numérica | Edad de la vendedora (años). | `dim_vendedor.edad` |
| `sexo_vendedor` | **categórica** | Sexo de la vendedora. | `dim_vendedor.csexpersona` |
| `tipo_vendedor` | **categórica** | Tipo/segmento de vendedora. | `dim_vendedor.tipo_vendedor` |
| `antiguedad_meses` | numérica | Nº de meses transcurridos desde el primer mes en el panel hasta el mes observado (rank dentro de la vida activa). | `ROW_NUMBER() OVER (PARTITION BY id_vendedor ORDER BY mes_rank)` |

## 2. Contexto (coordinadora y ubicación)

| Feature | Tipo | Definición | Origen SQL |
|---|---|---|---|
| `edad_coordinadora` | numérica | Edad de la coordinadora a cargo. | `dim_coordinadora.edad` |
| `provincia` | **categórica** | Provincia de la vendedora. Se usa en vez de `distrito`/`ccodubigeo` por menor cardinalidad. | `dim_ubicacion.provincia` |
| `departamento` | **categórica** | Departamento de la vendedora. | `dim_ubicacion.departamento` |

## 3. Actividad del mes observado (estado puntual)

| Feature | Tipo | Definición | Origen SQL |
|---|---|---|---|
| `num_pedidos_obs` | numérica | Nº de pedidos distintos en el mes observado. | `COUNT(DISTINCT id_pedido)` |
| `monto_total_obs` | numérica | Monto total facturado en el mes observado. | `SUM(monto_total_pedido)` |
| `monto_pagado_obs` | numérica | Monto efectivamente pagado en el mes observado. | `SUM(monto_pagado)` |
| `num_categorias_obs` | numérica | Nº de categorías distintas compradas en el mes. | `COUNT(DISTINCT categoria)` |
| `num_productos_obs` | numérica | Nº de productos distintos comprados en el mes. | `COUNT(DISTINCT id_producto)` |

## 4. RFM — ventana 3 meses (`u3`)

| Feature | Tipo | Definición |
|---|---|---|
| `num_compras_u3` | numérica | Nº de meses con compra en los últimos 3 meses (frecuencia). |
| `monto_total_u3` | numérica | Monto total facturado en los últimos 3 meses (monetario). |
| `monto_pagado_u3` | numérica | Monto pagado en los últimos 3 meses. |
| `tasa_compra_u3` | numérica | Fracción de meses con compra en la ventana = `AVG(compro)`. |
| `ticket_promedio_u3` | numérica | Monto total / nº de meses con compra en la ventana. |
| `ratio_pago_u3` | numérica | `monto_pagado_u3 / monto_total_u3` (salud de cobranza). |

## 5. RFM — ventana 6 meses (`u6`)

| Feature | Tipo | Definición |
|---|---|---|
| `num_compras_u6` | numérica | Nº de meses con compra en los últimos 6 meses. |
| `monto_total_u6` | numérica | Monto total facturado en los últimos 6 meses. |
| `monto_pagado_u6` | numérica | Monto pagado en los últimos 6 meses. |
| `tasa_compra_u6` | numérica | Fracción de meses con compra en la ventana de 6m. |
| `ticket_promedio_u6` | numérica | Monto total / nº de meses con compra (6m). |
| `ratio_pago_u6` | numérica | `monto_pagado_u6 / monto_total_u6`. |

## 6. RFM — ventana 12 meses (`u12`)

| Feature | Tipo | Definición |
|---|---|---|
| `num_compras_u12` | numérica | Nº de meses con compra en los últimos 12 meses. |
| `monto_total_u12` | numérica | Monto total facturado en los últimos 12 meses. |
| `monto_pagado_u12` | numérica | Monto pagado en los últimos 12 meses. |
| `tasa_compra_u12` | numérica | Fracción de meses con compra en la ventana de 12m. |
| `ticket_promedio_u12` | numérica | Monto total / nº de meses con compra (12m). |
| `ratio_pago_u12` | numérica | `monto_pagado_u12 / monto_total_u12`. |

## 7. Diversidad de producto (ventana 6 meses)

| Feature | Tipo | Definición |
|---|---|---|
| `avg_categorias_u6` | numérica | Promedio de categorías distintas por mes en los últimos 6 meses. |
| `avg_productos_u6` | numérica | Promedio de productos distintos por mes en los últimos 6 meses. |
| `unidades_u6` | numérica | Total de unidades compradas en los últimos 6 meses. |

## 8. Recencia e histórico acumulado

| Feature | Tipo | Definición |
|---|---|---|
| `meses_desde_compra_previa` | numérica | Gap (en meses) hasta la compra **anterior**. `NULL` en la primera observación con compra → imputado con la mediana. Mira solo hacia atrás (recencia). |
| `compras_historicas` | numérica | Nº acumulado de meses con compra desde el inicio de la vida activa (solo mira atrás). También es el filtro de población: `>= 3`. |
| `monto_historico` | numérica | Monto total acumulado histórico (solo mira atrás). |

## 9. Tendencias normalizadas (en [-1, 1])

| Feature | Tipo | Definición / fórmula |
|---|---|---|
| `delta_monto_u3_vs_prev3` | numérica | Tendencia de monto: `(2·monto_u3 − monto_u6) / monto_u6`. >0 acelera, <0 desacelera. |
| `delta_compras_u3_vs_prev3` | numérica | Tendencia de frecuencia: `(2·num_compras_u3 − num_compras_u6) / num_compras_u6`. |
| `ticket_u3_vs_u12` | numérica | Ratio del ticket reciente vs el de largo plazo: `ticket_promedio_u3 / ticket_promedio_u12`. |

---

## Preprocesamiento (pipeline del modelo final)

- **Numéricas (35)**: imputación por **mediana** (`SimpleImputer`).
- **Categóricas (4)**: NaN → `'NA'`, luego **One-Hot** (`handle_unknown='ignore'`).
- Modelo: `HistGradientBoostingClassifier(class_weight='balanced')` con hiperparámetros
  tuneados con Optuna (`learning_rate=0.0175, max_iter=750, max_depth=4,
  max_leaf_nodes=22, min_samples_leaf=100`).

## Columnas del SQL excluidas del modelo (`EXCLUDE`)

No entran al entrenamiento aunque estén en `training_churn_v5`:

| Columna | Motivo de exclusión |
|---|---|
| `id_vendedor` | Identificador. Se usa como `groups` en GroupKFold, no como feature. |
| `mes_obs`, `mes_rank_obs` | Claves temporales (ordenador), no comportamiento. |
| `fecha_ingreso` | Fecha cruda; el comportamiento se captura vía `antiguedad_meses`. |
| `id_coordinadora` | ID de alta cardinalidad → riesgo de ID disfrazado. |
| `ccodrelacion` | Identificador de relación; top en permutación pero es un ID, no señal de conducta. |
| `ccodubigeo`, `distrito` | Cardinalidad ~cientos → riesgo de ID disfrazado. Se usan `provincia`/`departamento`. |
| `mes_num`, `anio_mes_num` | Calendario. El `ablation_temporal_v5.ipynb` mostró que en split forward el modelo es igual o mejor sin ellas (PR-AUC +4.6%); `anio_mes_num` además extrapola fuera de rango en producción. |
| `churn` | Es el target. |

> Variables de target intermedio (`compro_t1..t6`) y SCD-1 sospechosos
> (`estado_coordinadora`) ya están excluidas desde el propio SQL — ver `LEAKEAGE.md`.
