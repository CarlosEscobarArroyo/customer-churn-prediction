# Glamour Metadata

**Total de tablas:** 8  
**Total de columnas:** 55

## Cómo hacer queries

Para consultar cualquier tabla en BigQuery, usar la siguiente ruta:

```googlesql
`glamour-peru-dw.glamour_dw.{tabla}`
```

**Ejemplo:**
```googlesql
SELECT * FROM `glamour-peru-dw.glamour_dw.fact_pedidos` LIMIT 10;
```

---

## `dim_campana`

| # | Columna | Tipo de Dato | Nullable |
|---|---------|--------------|----------|
| 1 | `id_campana` | `INT64` | ✅ |
| 2 | `nombre_campana` | `STRING` | ✅ |
| 3 | `estado` | `STRING` | ✅ |
| 4 | `fecha_inicio` | `DATE` | ✅ |
| 5 | `fecha_fin` | `DATE` | ✅ |
| 6 | `anio` | `INT64` | ✅ |
| 7 | `numero_campana` | `INT64` | ✅ |

## `dim_coordinadora`

| # | Columna | Tipo de Dato | Nullable |
|---|---------|--------------|----------|
| 1 | `id_coordinadora` | `STRING` | ✅ |
| 2 | `nombre_coordinadora` | `STRING` | ✅ |
| 3 | `estado` | `STRING` | ✅ |
| 4 | `edad` | `FLOAT64` | ✅ |

## `dim_fecha`

| # | Columna | Tipo de Dato | Nullable |
|---|---------|--------------|----------|
| 1 | `date` | `DATE` | ✅ |
| 2 | `id_fecha` | `INT64` | ✅ |
| 3 | `calendaryear` | `INT64` | ✅ |
| 4 | `calendarmonth` | `INT64` | ✅ |
| 5 | `calendardayinmonth` | `INT64` | ✅ |
| 6 | `calendarquarter` | `INT64` | ✅ |

## `dim_producto`

| # | Columna | Tipo de Dato | Nullable |
|---|---------|--------------|----------|
| 1 | `id_producto` | `STRING` | ✅ |
| 2 | `nombre_producto` | `STRING` | ✅ |
| 3 | `categoria` | `STRING` | ✅ |
| 4 | `subcategoria` | `STRING` | ✅ |

## `dim_ubicacion`

| # | Columna | Tipo de Dato | Nullable |
|---|---------|--------------|----------|
| 1 | `ccodubigeo` | `STRING` | ✅ |
| 2 | `id_ubicacion` | `INT64` | ✅ |
| 3 | `distrito` | `STRING` | ✅ |
| 4 | `provincia` | `STRING` | ✅ |
| 5 | `departamento` | `STRING` | ✅ |

## `dim_vendedor`

| # | Columna | Tipo de Dato | Nullable |
|---|---------|--------------|----------|
| 1 | `id_vendedor` | `INT64` | ✅ |
| 2 | `nombre_vendedor` | `STRING` | ✅ |
| 3 | `csexpersona` | `STRING` | ✅ |
| 4 | `ccodubigeo` | `STRING` | ✅ |
| 5 | `nmovil` | `STRING` | ✅ |
| 6 | `ccodrelacion` | `INT64` | ✅ |
| 7 | `id_coordinadora` | `STRING` | ✅ |
| 8 | `fecha_ingreso` | `DATE` | ✅ |
| 9 | `edad` | `FLOAT64` | ✅ |
| 10 | `tipo_vendedor` | `STRING` | ✅ |

## `fact_pedidos`

| # | Columna | Tipo de Dato | Nullable |
|---|---------|--------------|----------|
| 1 | `id_pedido` | `INT64` | ✅ |
| 2 | `id_campana` | `INT64` | ✅ |
| 3 | `id_vendedor` | `INT64` | ✅ |
| 4 | `id_fecha` | `INT64` | ✅ |
| 5 | `id_coordinadora` | `STRING` | ✅ |
| 6 | `id_ubicacion` | `FLOAT64` | ✅ |
| 7 | `monto_total_pedido` | `FLOAT64` | ✅ |
| 8 | `monto_pagado` | `FLOAT64` | ✅ |
| 9 | `es_nueva_vendedora` | `INT64` | ✅ |
| 10 | `tipo` | `STRING` | ✅ |

## `fact_pedidos_detalle`

| # | Columna | Tipo de Dato | Nullable |
|---|---------|--------------|----------|
| 1 | `id_pedido` | `INT64` | ✅ |
| 2 | `id_campana` | `INT64` | ✅ |
| 3 | `id_vendedor` | `INT64` | ✅ |
| 4 | `id_fecha` | `INT64` | ✅ |
| 5 | `id_ubicacion` | `FLOAT64` | ✅ |
| 6 | `id_pedido_detalle` | `INT64` | ✅ |
| 7 | `id_producto` | `STRING` | ✅ |
| 8 | `cantidad` | `FLOAT64` | ✅ |
| 9 | `importe_producto` | `FLOAT64` | ✅ |