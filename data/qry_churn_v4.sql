-- ============================================================================
-- Dataset de entrenamiento para modelo de churn de vendedoras  (v4)
--
-- CAMBIO ESTRUCTURAL vs. v3:
--   La unidad temporal pasa de **campaña** a **mes calendario**.
--
-- MOTIVACIÓN (ver V4_MENSUAL.md para el análisis completo):
--   Las "campañas" no son una unidad temporal consistente:
--     - Duración mediana 28 días, std 81 días, rango [-15, 1006].
--     - Mezclan campañas regulares (~30 días) con liquidaciones (4-15 días) y
--       outliers de carga.
--     - Desde 2025-06 el negocio pasó de 1 campaña/mes a 2-3 campañas/mes
--       simultáneas. La definición de churn de v3 ("no compra en 6 campañas")
--       deja de ser un horizonte consistente: equivale a 6 meses en el régimen
--       viejo y a 2-3 meses en el nuevo. Eso explica el drift temporal y la
--       inestabilidad por campaña detectados en el NB 04.
--
-- CAMBIOS PUNTUALES vs. v3:
--
--   1. Granularidad: una fila por (id_vendedor, mes_obs) donde mes_obs es el
--      primer día del mes calendario (DATE).
--      `mes_rank_obs` reemplaza a `campana_rank_obs`.
--
--   2. Agregación de pedidos: se hace por mes calendario usando
--      `dim_fecha.calendaryear/calendarmonth`, NO por id_campana. Una
--      vendedora con compras en una "regular" + "liquidación" del mismo mes
--      cuenta como 1 mes con compra (no como 2 puntos en el panel).
--
--   3. Limpieza de campañas atípicas (blacklist al inicio del CTE de pedidos):
--      - 20102 "SANITY BIOSEGURIDAD" (262 días, contexto COVID, no retail).
--      - 20201 "PRIVILEGIOS GLAMOUR SAN VALENT" (fecha_fin < fecha_inicio).
--      - 23105 "CURSOS PRALEMY" (1006 días, 4 pedidos, no retail).
--      - id_campana con 0 pedidos (ya quedan fuera al hacer INNER JOIN).
--
--   4. Definición de churn: vendedora churnea si NO compra en los próximos
--      `HORIZON_CHURN = 4` meses consecutivos. Justificación: en venta directa
--      4 meses sin compra es señal robusta de pérdida; 6 meses sería tarde
--      para retención. La elección está abierta a revisión — ver TODO en
--      V4_MENSUAL.md sección "Próximos pasos".
--
--   5. Ventanas RFM cambian de "últimas N campañas" a "últimos N meses".
--      Se mantienen los nombres u3/u6/u12 (significan meses ahora).
--
--   6. Recencia: `meses_desde_compra_previa` reemplaza a
--      `campanas_desde_compra_previa`.
--
--   7. Filtro de población: `compras_historicas >= 3` (en mensual; antes 4 en
--      campañas). Razonamiento: 3 meses con compra cubre el equivalente
--      mínimo de actividad y mantiene tamaño de muestra.
--
-- SIN CAMBIOS vs. v3:
--   - Mismas exclusiones en el SELECT final (estado_coordinadora,
--     pseudo-features constantes, columnas de target intermedio).
--   - Mismas tendencias normalizadas en [-1, 1].
--   - Mismas notas SCD-1 (atributos de vendedor/coordinadora son snapshot).
-- ============================================================================

CREATE OR REPLACE TABLE `glamour-peru-dw.glamour_dw.training_churn_v4`
PARTITION BY mes_obs
CLUSTER BY id_vendedor
AS

WITH

-- ----------------------------------------------------------------------------
-- 1. Serie continua de meses con su rank global
--    Cubre desde el primer mes con datos hasta el último mes con datos.
--    Usar GENERATE_DATE_ARRAY para no depender de dim_campana (cuyas fechas
--    son inconsistentes).
-- ----------------------------------------------------------------------------
rango_fechas AS (
  SELECT
    DATE_TRUNC(MIN(d.date), MONTH) AS primer_mes,
    DATE_TRUNC(MAX(d.date), MONTH) AS ultimo_mes
  FROM `glamour-peru-dw.glamour_dw.fact_pedidos` p
  JOIN `glamour-peru-dw.glamour_dw.dim_fecha` d ON p.id_fecha = d.id_fecha
),
meses_ordenados AS (
  SELECT
    mes,
    ROW_NUMBER() OVER (ORDER BY mes) AS mes_rank
  FROM rango_fechas, UNNEST(GENERATE_DATE_ARRAY(primer_mes, ultimo_mes, INTERVAL 1 MONTH)) AS mes
),

-- ----------------------------------------------------------------------------
-- 2. Pedidos limpios (excluye campañas blacklist) con su mes calendario
-- ----------------------------------------------------------------------------
pedidos_limpios AS (
  SELECT
    p.id_pedido,
    p.id_vendedor,
    p.id_campana,
    p.monto_total_pedido,
    p.monto_pagado,
    DATE_TRUNC(d.date, MONTH) AS mes
  FROM `glamour-peru-dw.glamour_dw.fact_pedidos` p
  JOIN `glamour-peru-dw.glamour_dw.dim_fecha` d ON p.id_fecha = d.id_fecha
  WHERE p.id_campana NOT IN (20102, 20201, 23105)  -- ver header
),

-- ----------------------------------------------------------------------------
-- 3. Pedidos agregados a nivel (vendedor, mes)
-- ----------------------------------------------------------------------------
pedidos_agg AS (
  SELECT
    id_vendedor,
    mes,
    COUNT(DISTINCT id_pedido) AS num_pedidos,
    SUM(monto_total_pedido)   AS monto_total,
    SUM(monto_pagado)         AS monto_pagado
  FROM pedidos_limpios
  GROUP BY id_vendedor, mes
),

-- ----------------------------------------------------------------------------
-- 4. Features de producto por (vendedor, mes)
-- ----------------------------------------------------------------------------
detalle_limpio AS (
  SELECT
    d.id_vendedor,
    DATE_TRUNC(f.date, MONTH) AS mes,
    d.id_producto,
    p.categoria,
    p.subcategoria,
    d.cantidad
  FROM `glamour-peru-dw.glamour_dw.fact_pedidos_detalle` d
  JOIN `glamour-peru-dw.glamour_dw.dim_fecha` f ON d.id_fecha = f.id_fecha
  LEFT JOIN `glamour-peru-dw.glamour_dw.dim_producto` p ON d.id_producto = p.id_producto
  WHERE d.id_campana NOT IN (20102, 20201, 23105)
),
producto_agg AS (
  SELECT
    id_vendedor,
    mes,
    COUNT(DISTINCT id_producto)   AS num_productos_distintos,
    COUNT(DISTINCT categoria)     AS num_categorias_distintas,
    COUNT(DISTINCT subcategoria)  AS num_subcategorias_distintas,
    SUM(cantidad)                 AS unidades_totales
  FROM detalle_limpio
  GROUP BY id_vendedor, mes
),

-- ----------------------------------------------------------------------------
-- 5. Ventana de vida activa por vendedor.
--    El panel cubre desde su primer mes con compra hasta ultimo_mes_compra + 4
--    para que el último mes observado tenga horizonte de target calculable.
-- ----------------------------------------------------------------------------
vida_vendedor AS (
  SELECT
    pa.id_vendedor,
    MIN(m.mes_rank) AS primer_compra_rank,
    MAX(m.mes_rank) AS ultima_compra_rank
  FROM pedidos_agg pa
  JOIN meses_ordenados m ON pa.mes = m.mes
  GROUP BY pa.id_vendedor
),

-- ----------------------------------------------------------------------------
-- 6. Panel: (vendedor × meses) restringido a la ventana de vida activa
-- ----------------------------------------------------------------------------
panel AS (
  SELECT
    v.id_vendedor,
    v.fecha_ingreso,
    v.edad              AS edad_vendedor,
    v.csexpersona       AS sexo_vendedor,
    v.tipo_vendedor,
    v.ccodrelacion,
    v.id_coordinadora,
    v.ccodubigeo,

    m.mes,
    m.mes_rank,
    EXTRACT(YEAR  FROM m.mes) AS anio_mes_num,
    EXTRACT(MONTH FROM m.mes) AS mes_num,

    COALESCE(pa.num_pedidos, 0)        AS num_pedidos,
    COALESCE(pa.monto_total, 0)        AS monto_total,
    COALESCE(pa.monto_pagado, 0)       AS monto_pagado,
    IF(pa.num_pedidos > 0, 1, 0)       AS compro,

    COALESCE(pr.num_productos_distintos, 0)     AS num_productos,
    COALESCE(pr.num_categorias_distintas, 0)    AS num_categorias,
    COALESCE(pr.num_subcategorias_distintas, 0) AS num_subcategorias,
    COALESCE(pr.unidades_totales, 0)            AS unidades

  FROM `glamour-peru-dw.glamour_dw.dim_vendedor` v
  INNER JOIN vida_vendedor vv ON v.id_vendedor = vv.id_vendedor
  CROSS JOIN meses_ordenados m
  LEFT JOIN pedidos_agg pa
    ON v.id_vendedor = pa.id_vendedor AND m.mes = pa.mes
  LEFT JOIN producto_agg pr
    ON v.id_vendedor = pr.id_vendedor AND m.mes = pr.mes
  WHERE m.mes_rank >= vv.primer_compra_rank
    AND m.mes_rank <= vv.ultima_compra_rank + 4   -- HORIZON_CHURN
),

-- ----------------------------------------------------------------------------
-- 7. Features con window functions: RFM + recencia + histórico
--    Todas las ventanas son `ROWS BETWEEN N PRECEDING AND CURRENT ROW`
--    sobre meses (no campañas).
-- ----------------------------------------------------------------------------
features AS (
  SELECT
    p.*,

    ROW_NUMBER() OVER (PARTITION BY id_vendedor ORDER BY mes_rank)
      AS antiguedad_meses,

    -- Ventana de 3 meses
    SUM(compro)        OVER w3  AS num_compras_u3,
    SUM(monto_total)   OVER w3  AS monto_total_u3,
    SUM(monto_pagado)  OVER w3  AS monto_pagado_u3,
    AVG(compro)        OVER w3  AS tasa_compra_u3,
    SAFE_DIVIDE(SUM(monto_total) OVER w3, NULLIF(SUM(compro) OVER w3, 0))
      AS ticket_promedio_u3,
    SAFE_DIVIDE(SUM(monto_pagado) OVER w3, NULLIF(SUM(monto_total) OVER w3, 0))
      AS ratio_pago_u3,

    -- Ventana de 6 meses
    SUM(compro)        OVER w6  AS num_compras_u6,
    SUM(monto_total)   OVER w6  AS monto_total_u6,
    SUM(monto_pagado)  OVER w6  AS monto_pagado_u6,
    AVG(compro)        OVER w6  AS tasa_compra_u6,
    SAFE_DIVIDE(SUM(monto_total) OVER w6, NULLIF(SUM(compro) OVER w6, 0))
      AS ticket_promedio_u6,
    SAFE_DIVIDE(SUM(monto_pagado) OVER w6, NULLIF(SUM(monto_total) OVER w6, 0))
      AS ratio_pago_u6,

    -- Ventana de 12 meses
    SUM(compro)        OVER w12 AS num_compras_u12,
    SUM(monto_total)   OVER w12 AS monto_total_u12,
    SUM(monto_pagado)  OVER w12 AS monto_pagado_u12,
    AVG(compro)        OVER w12 AS tasa_compra_u12,
    SAFE_DIVIDE(SUM(monto_total) OVER w12, NULLIF(SUM(compro) OVER w12, 0))
      AS ticket_promedio_u12,
    SAFE_DIVIDE(SUM(monto_pagado) OVER w12, NULLIF(SUM(monto_total) OVER w12, 0))
      AS ratio_pago_u12,

    -- Diversidad de producto
    AVG(num_categorias) OVER w6 AS avg_categorias_u6,
    AVG(num_productos)  OVER w6 AS avg_productos_u6,
    SUM(unidades)       OVER w6 AS unidades_u6,

    -- Recencia: gap a la compra ANTERIOR (NULL en la primera observación
    -- con compra de la vendedora). Ver nota equivalente en v3.
    mes_rank - LAST_VALUE(
      IF(compro = 1, mes_rank, NULL) IGNORE NULLS
    ) OVER (PARTITION BY id_vendedor ORDER BY mes_rank
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
      AS meses_desde_compra_previa,

    -- Histórico acumulado (solo mira atrás)
    SUM(compro) OVER (PARTITION BY id_vendedor
                      ORDER BY mes_rank
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
      AS compras_historicas,

    SUM(monto_total) OVER (PARTITION BY id_vendedor
                           ORDER BY mes_rank
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
      AS monto_historico

  FROM panel p
  WINDOW
    w3  AS (PARTITION BY id_vendedor ORDER BY mes_rank
            ROWS BETWEEN 2  PRECEDING AND CURRENT ROW),
    w6  AS (PARTITION BY id_vendedor ORDER BY mes_rank
            ROWS BETWEEN 5  PRECEDING AND CURRENT ROW),
    w12 AS (PARTITION BY id_vendedor ORDER BY mes_rank
            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)
),

-- ----------------------------------------------------------------------------
-- 8. Features derivadas: tendencias normalizadas en [-1, 1]
--    Misma fórmula que v3: (u3 - prev3) / (u3 + prev3) = (2·u3 - u6) / u6
-- ----------------------------------------------------------------------------
features_derivadas AS (
  SELECT
    f.*,

    COALESCE(
      SAFE_DIVIDE(2 * monto_total_u3 - monto_total_u6, NULLIF(monto_total_u6, 0)),
      0
    ) AS delta_monto_u3_vs_prev3,

    COALESCE(
      SAFE_DIVIDE(2 * num_compras_u3 - num_compras_u6, NULLIF(num_compras_u6, 0)),
      0
    ) AS delta_compras_u3_vs_prev3,

    SAFE_DIVIDE(ticket_promedio_u3, NULLIF(ticket_promedio_u12, 0))
      AS ticket_u3_vs_u12

  FROM features f
),

-- ----------------------------------------------------------------------------
-- 9. Target: churn en los próximos 4 meses (HORIZON_CHURN)
-- ----------------------------------------------------------------------------
target AS (
  SELECT
    id_vendedor,
    mes_rank,
    LEAD(compro, 1) OVER w_fwd AS compro_t1,
    LEAD(compro, 2) OVER w_fwd AS compro_t2,
    LEAD(compro, 3) OVER w_fwd AS compro_t3,
    LEAD(compro, 4) OVER w_fwd AS compro_t4
  FROM panel
  WINDOW w_fwd AS (PARTITION BY id_vendedor ORDER BY mes_rank)
),

-- ----------------------------------------------------------------------------
-- 10. Ensamble final
-- ----------------------------------------------------------------------------
ensamble AS (
  SELECT
    -- Claves
    fd.id_vendedor,
    fd.mes                                      AS mes_obs,
    fd.mes_rank                                 AS mes_rank_obs,
    fd.anio_mes_num,
    fd.mes_num,

    -- Vendedora (atributos SCD-1)
    fd.fecha_ingreso,
    fd.edad_vendedor,
    fd.sexo_vendedor,
    fd.tipo_vendedor,
    fd.ccodrelacion,
    fd.antiguedad_meses,

    -- Coordinadora (sin estado_coordinadora)
    fd.id_coordinadora,
    co.edad                                     AS edad_coordinadora,

    -- Ubicación
    fd.ccodubigeo,
    ub.distrito,
    ub.provincia,
    ub.departamento,

    -- Métricas del mes actual
    fd.num_pedidos                              AS num_pedidos_obs,
    fd.monto_total                              AS monto_total_obs,
    fd.monto_pagado                             AS monto_pagado_obs,
    fd.num_categorias                           AS num_categorias_obs,
    fd.num_productos                            AS num_productos_obs,

    -- RFM ventana 3 meses
    fd.num_compras_u3,
    fd.monto_total_u3,
    fd.monto_pagado_u3,
    fd.tasa_compra_u3,
    fd.ticket_promedio_u3,
    fd.ratio_pago_u3,

    -- RFM ventana 6 meses
    fd.num_compras_u6,
    fd.monto_total_u6,
    fd.monto_pagado_u6,
    fd.tasa_compra_u6,
    fd.ticket_promedio_u6,
    fd.ratio_pago_u6,

    -- RFM ventana 12 meses
    fd.num_compras_u12,
    fd.monto_total_u12,
    fd.monto_pagado_u12,
    fd.tasa_compra_u12,
    fd.ticket_promedio_u12,
    fd.ratio_pago_u12,

    -- Producto
    fd.avg_categorias_u6,
    fd.avg_productos_u6,
    fd.unidades_u6,

    -- Recencia e histórico
    fd.meses_desde_compra_previa,
    fd.compras_historicas,
    fd.monto_historico,

    -- Tendencias normalizadas
    fd.delta_monto_u3_vs_prev3,
    fd.delta_compras_u3_vs_prev3,
    fd.ticket_u3_vs_u12,

    -- Auxiliar para filtro/target
    fd.compro                                   AS _compro_en_obs,

    -- Target
    CASE
      WHEN t.compro_t1 IS NULL
        OR t.compro_t2 IS NULL
        OR t.compro_t3 IS NULL
        OR t.compro_t4 IS NULL
        THEN NULL
      WHEN (t.compro_t1 + t.compro_t2 + t.compro_t3 + t.compro_t4) = 0 THEN 1
      ELSE 0
    END AS churn

  FROM features_derivadas fd
  LEFT JOIN target t
    ON fd.id_vendedor = t.id_vendedor
   AND fd.mes_rank = t.mes_rank
  LEFT JOIN `glamour-peru-dw.glamour_dw.dim_coordinadora` co
    ON fd.id_coordinadora = co.id_coordinadora
  LEFT JOIN `glamour-peru-dw.glamour_dw.dim_ubicacion` ub
    ON fd.ccodubigeo = ub.ccodubigeo
)

-- ----------------------------------------------------------------------------
-- 11. Resultado final
--     Filtros (equivalentes a v3 pero con escala mensual):
--       - churn IS NOT NULL: target calculable (4 meses siguientes presentes).
--       - _compro_en_obs = 1: solo observaciones donde la vendedora compró
--         en el mes (consistente con producción).
--       - compras_historicas >= 3: excluye historia muy corta.
-- ----------------------------------------------------------------------------
SELECT
  id_vendedor, mes_obs, mes_rank_obs, anio_mes_num, mes_num,
  fecha_ingreso, edad_vendedor, sexo_vendedor,
  tipo_vendedor, ccodrelacion, antiguedad_meses, id_coordinadora,
  edad_coordinadora, ccodubigeo, distrito, provincia, departamento,
  num_pedidos_obs, monto_total_obs, monto_pagado_obs, num_categorias_obs,
  num_productos_obs,
  num_compras_u3, monto_total_u3, monto_pagado_u3, tasa_compra_u3,
  ticket_promedio_u3, ratio_pago_u3,
  num_compras_u6, monto_total_u6, monto_pagado_u6, tasa_compra_u6,
  ticket_promedio_u6, ratio_pago_u6,
  num_compras_u12, monto_total_u12, monto_pagado_u12, tasa_compra_u12,
  ticket_promedio_u12, ratio_pago_u12,
  avg_categorias_u6, avg_productos_u6, unidades_u6,
  meses_desde_compra_previa, compras_historicas, monto_historico,
  delta_monto_u3_vs_prev3, delta_compras_u3_vs_prev3, ticket_u3_vs_u12,
  churn
FROM ensamble
WHERE churn IS NOT NULL
  AND _compro_en_obs = 1
  AND compras_historicas >= 3
;
