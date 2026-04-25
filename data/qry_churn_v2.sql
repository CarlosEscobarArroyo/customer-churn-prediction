-- ============================================================================
-- Dataset de entrenamiento para modelo de churn de vendedores  (v2)
--
-- CAMBIOS vs. versión original:
--   1. Ventana de churn ampliada de 3 a 6 campañas.
--        - Motivo: 30.7% de los gaps entre compras consecutivas son >= 4
--          campañas. Una ventana de 3 marca como "churn" a vendedoras
--          legítimamente esporádicas. Con 6 campañas reducimos ese falso
--          churn sin dejar de capturar a las realmente inactivas.
--   2. Filtro de historia mínima: compras_historicas >= 4.
--        - Motivo: 45% de vendedoras tenía una sola compra en el dataset
--          original → churn = 99.94% trivial. Excluyendo compradoras con
--          muy poca historia, el modelo aprende patrones reales de
--          comportamiento en vez de memorizar la regla "1 compra = churn".
--
-- Definición actualizada: churn = vendedor no compra en las 6 campañas
--                         CONSECUTIVAS siguientes a la campaña de observación.
--
-- Granularidad: una fila por (id_vendedor, id_campana_obs). Sin cambios.
--
-- Features:    RFM + tendencia + producto + contexto, con ventanas 3, 6 y 12.
--              Sin cambios en features, solo en el target y el filtro final.
-- ============================================================================

CREATE OR REPLACE TABLE `glamour-peru-dw.glamour_dw.training_churn_v2`
PARTITION BY RANGE_BUCKET(id_campana_obs, GENERATE_ARRAY(202001, 202700, 1))
CLUSTER BY id_vendedor
AS

WITH

-- ----------------------------------------------------------------------------
-- 1. Ranking global de campañas
--    Índice ordinal continuo para que "N campañas consecutivas" sea
--    rank+1, rank+2, ..., rank+N sin importar saltos de año.
-- ----------------------------------------------------------------------------
campanas_ordenadas AS (
  SELECT
    id_campana,
    nombre_campana,
    anio,
    numero_campana,
    fecha_inicio,
    fecha_fin,
    ROW_NUMBER() OVER (ORDER BY anio, numero_campana) AS campana_rank
  FROM `glamour-peru-dw.glamour_dw.dim_campana`
  WHERE estado IS NULL OR estado != 'CANCELADA'
),

-- ----------------------------------------------------------------------------
-- 2. Pedidos agregados a nivel (vendedor, campaña)
-- ----------------------------------------------------------------------------
pedidos_agg AS (
  SELECT
    id_vendedor,
    id_campana,
    COUNT(DISTINCT id_pedido)          AS num_pedidos,
    SUM(monto_total_pedido)            AS monto_total,
    SUM(monto_pagado)                  AS monto_pagado,
    MAX(es_nueva_vendedora)            AS fue_nueva_en_campana
  FROM `glamour-peru-dw.glamour_dw.fact_pedidos`
  GROUP BY id_vendedor, id_campana
),

-- ----------------------------------------------------------------------------
-- 3. Features de producto por (vendedor, campaña)
-- ----------------------------------------------------------------------------
producto_agg AS (
  SELECT
    d.id_vendedor,
    d.id_campana,
    COUNT(DISTINCT d.id_producto)      AS num_productos_distintos,
    COUNT(DISTINCT p.categoria)        AS num_categorias_distintas,
    COUNT(DISTINCT p.subcategoria)     AS num_subcategorias_distintas,
    SUM(d.cantidad)                    AS unidades_totales
  FROM `glamour-peru-dw.glamour_dw.fact_pedidos_detalle` d
  LEFT JOIN `glamour-peru-dw.glamour_dw.dim_producto` p
    ON d.id_producto = p.id_producto
  GROUP BY d.id_vendedor, d.id_campana
),

-- ----------------------------------------------------------------------------
-- 4. Ventana de vida activa por vendedor (Filosofía B)
--    CAMBIO: ahora el panel se extiende hasta ultima_compra_rank + 6
--    en vez de + 3, para que la última compra tenga 6 campañas siguientes
--    observables y el nuevo target sea calculable en todas las filas.
-- ----------------------------------------------------------------------------
vida_vendedor AS (
  SELECT
    pa.id_vendedor,
    MIN(c.campana_rank) AS primera_compra_rank,
    MAX(c.campana_rank) AS ultima_compra_rank,
    COUNT(DISTINCT c.campana_rank) AS campanas_con_compra_totales
  FROM pedidos_agg pa
  JOIN campanas_ordenadas c USING (id_campana)
  GROUP BY pa.id_vendedor
),

-- ----------------------------------------------------------------------------
-- 5. Panel: (vendedor × campaña) restringido a la ventana de vida activa
--    CAMBIO: el límite superior pasa de +3 a +6.
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

    c.id_campana,
    c.campana_rank,
    c.anio              AS anio_campana,
    c.numero_campana,
    c.fecha_inicio      AS fecha_inicio_campana,

    -- Métricas de la campaña (0 si no compró)
    COALESCE(pa.num_pedidos, 0)        AS num_pedidos,
    COALESCE(pa.monto_total, 0)        AS monto_total,
    COALESCE(pa.monto_pagado, 0)       AS monto_pagado,
    IF(pa.num_pedidos > 0, 1, 0)       AS compro,

    -- Producto (0 si no compró)
    COALESCE(pr.num_productos_distintos, 0)     AS num_productos,
    COALESCE(pr.num_categorias_distintas, 0)    AS num_categorias,
    COALESCE(pr.num_subcategorias_distintas, 0) AS num_subcategorias,
    COALESCE(pr.unidades_totales, 0)            AS unidades

  FROM `glamour-peru-dw.glamour_dw.dim_vendedor` v
  INNER JOIN vida_vendedor vv
    ON v.id_vendedor = vv.id_vendedor
  CROSS JOIN campanas_ordenadas c
  LEFT JOIN pedidos_agg pa
    ON v.id_vendedor = pa.id_vendedor
   AND c.id_campana  = pa.id_campana
  LEFT JOIN producto_agg pr
    ON v.id_vendedor = pr.id_vendedor
   AND c.id_campana  = pr.id_campana
  -- Filosofía B ampliada: primera compra hasta última compra + 6
  WHERE c.campana_rank >= vv.primera_compra_rank
    AND c.campana_rank <= vv.ultima_compra_rank + 6
),

-- ----------------------------------------------------------------------------
-- 6. Features con window functions: RFM + tendencia
--    Sin cambios: todas las ventanas siguen siendo 3/6/12 y solo usan
--    información hasta la campaña actual (cero leakage).
-- ----------------------------------------------------------------------------
features AS (
  SELECT
    p.*,

    -- Antigüedad en campañas
    ROW_NUMBER() OVER (PARTITION BY id_vendedor ORDER BY campana_rank)
      AS antiguedad_campanas,

    -- Flag de nueva vendedora: <3 campañas de historia en la observación
    IF(
      ROW_NUMBER() OVER (PARTITION BY id_vendedor ORDER BY campana_rank) < 3,
      1, 0
    ) AS es_nueva_vendedora,

    -- Ventana de 3 campañas
    SUM(compro) OVER w3        AS num_compras_u3,
    SUM(monto_total) OVER w3   AS monto_total_u3,
    SUM(monto_pagado) OVER w3  AS monto_pagado_u3,
    AVG(compro) OVER w3        AS tasa_compra_u3,
    SAFE_DIVIDE(SUM(monto_total) OVER w3, NULLIF(SUM(compro) OVER w3, 0))
      AS ticket_promedio_u3,
    SAFE_DIVIDE(SUM(monto_pagado) OVER w3, NULLIF(SUM(monto_total) OVER w3, 0))
      AS ratio_pago_u3,

    -- Ventana de 6 campañas
    SUM(compro) OVER w6        AS num_compras_u6,
    SUM(monto_total) OVER w6   AS monto_total_u6,
    SUM(monto_pagado) OVER w6  AS monto_pagado_u6,
    AVG(compro) OVER w6        AS tasa_compra_u6,
    SAFE_DIVIDE(SUM(monto_total) OVER w6, NULLIF(SUM(compro) OVER w6, 0))
      AS ticket_promedio_u6,
    SAFE_DIVIDE(SUM(monto_pagado) OVER w6, NULLIF(SUM(monto_total) OVER w6, 0))
      AS ratio_pago_u6,

    -- Ventana de 12 campañas
    SUM(compro) OVER w12       AS num_compras_u12,
    SUM(monto_total) OVER w12  AS monto_total_u12,
    SUM(monto_pagado) OVER w12 AS monto_pagado_u12,
    AVG(compro) OVER w12       AS tasa_compra_u12,
    SAFE_DIVIDE(SUM(monto_total) OVER w12, NULLIF(SUM(compro) OVER w12, 0))
      AS ticket_promedio_u12,
    SAFE_DIVIDE(SUM(monto_pagado) OVER w12, NULLIF(SUM(monto_total) OVER w12, 0))
      AS ratio_pago_u12,

    -- Diversidad de producto (ventanas)
    AVG(num_categorias) OVER w6   AS avg_categorias_u6,
    AVG(num_productos)  OVER w6   AS avg_productos_u6,
    SUM(unidades)       OVER w6   AS unidades_u6,

    -- Recencia: campañas desde la última compra
    campana_rank - MAX(IF(compro = 1, campana_rank, NULL))
      OVER (PARTITION BY id_vendedor
            ORDER BY campana_rank
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
      AS campanas_desde_ultima_compra,

    -- Total histórico de compras (usado también para el filtro de historia)
    SUM(compro) OVER (PARTITION BY id_vendedor
                      ORDER BY campana_rank
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
      AS compras_historicas,

    SUM(monto_total) OVER (PARTITION BY id_vendedor
                           ORDER BY campana_rank
                           ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
      AS monto_historico

  FROM panel p
  WINDOW
    w3  AS (PARTITION BY id_vendedor ORDER BY campana_rank
            ROWS BETWEEN 2  PRECEDING AND CURRENT ROW),
    w6  AS (PARTITION BY id_vendedor ORDER BY campana_rank
            ROWS BETWEEN 5  PRECEDING AND CURRENT ROW),
    w12 AS (PARTITION BY id_vendedor ORDER BY campana_rank
            ROWS BETWEEN 11 PRECEDING AND CURRENT ROW)
),

-- ----------------------------------------------------------------------------
-- 7. Features derivadas: tendencias (ratios entre ventanas). Sin cambios.
-- ----------------------------------------------------------------------------
features_derivadas AS (
  SELECT
    f.*,

    SAFE_DIVIDE(monto_total_u3, NULLIF(monto_total_u6 - monto_total_u3, 0))
      AS tendencia_monto_u3_vs_prev3,

    SAFE_DIVIDE(num_compras_u3, NULLIF(num_compras_u6 - num_compras_u3, 0))
      AS tendencia_compras_u3_vs_prev3,

    SAFE_DIVIDE(ticket_promedio_u3, NULLIF(ticket_promedio_u12, 0))
      AS ticket_u3_vs_u12

  FROM features f
),

-- ----------------------------------------------------------------------------
-- 8. Target: churn en las 6 campañas siguientes
--    CAMBIO: antes eran 3 LEADs (t+1, t+2, t+3). Ahora son 6 (t+1..t+6).
--    Si alguna de las 6 siguientes no existe en el DW → churn = NULL.
--    Si la suma de compro_t1..t6 es 0 → churn = 1.
-- ----------------------------------------------------------------------------
target AS (
  SELECT
    id_vendedor,
    campana_rank,

    -- Compras en las 6 siguientes
    LEAD(compro, 1) OVER w_fwd AS compro_t1,
    LEAD(compro, 2) OVER w_fwd AS compro_t2,
    LEAD(compro, 3) OVER w_fwd AS compro_t3,
    LEAD(compro, 4) OVER w_fwd AS compro_t4,
    LEAD(compro, 5) OVER w_fwd AS compro_t5,
    LEAD(compro, 6) OVER w_fwd AS compro_t6,

    -- Montos de las 6 siguientes
    LEAD(monto_total, 1) OVER w_fwd AS monto_t1,
    LEAD(monto_total, 2) OVER w_fwd AS monto_t2,
    LEAD(monto_total, 3) OVER w_fwd AS monto_t3,
    LEAD(monto_total, 4) OVER w_fwd AS monto_t4,
    LEAD(monto_total, 5) OVER w_fwd AS monto_t5,
    LEAD(monto_total, 6) OVER w_fwd AS monto_t6

  FROM panel
  WINDOW w_fwd AS (PARTITION BY id_vendedor ORDER BY campana_rank)
),

-- ----------------------------------------------------------------------------
-- 9. Ensamble final: features + target + info de coordinadora y ubicación
-- ----------------------------------------------------------------------------
ensamble AS (
  SELECT
    -- Claves
    fd.id_vendedor,
    fd.id_campana                               AS id_campana_obs,
    fd.campana_rank                             AS campana_rank_obs,
    fd.anio_campana,
    fd.numero_campana,
    fd.fecha_inicio_campana,

    -- Contexto del vendedor
    fd.fecha_ingreso,
    fd.edad_vendedor,
    fd.sexo_vendedor,
    fd.tipo_vendedor,
    fd.ccodrelacion,
    fd.antiguedad_campanas,
    fd.es_nueva_vendedora,

    -- Coordinadora
    fd.id_coordinadora,
    co.edad                                     AS edad_coordinadora,
    co.estado                                   AS estado_coordinadora,

    -- Ubicación
    fd.ccodubigeo,
    ub.distrito,
    ub.provincia,
    ub.departamento,

    -- Métricas de la campaña actual
    fd.compro                                   AS compro_en_obs,
    fd.num_pedidos                              AS num_pedidos_obs,
    fd.monto_total                              AS monto_total_obs,
    fd.monto_pagado                             AS monto_pagado_obs,
    fd.num_categorias                           AS num_categorias_obs,
    fd.num_productos                            AS num_productos_obs,

    -- Features RFM - ventana 3
    fd.num_compras_u3,
    fd.monto_total_u3,
    fd.monto_pagado_u3,
    fd.tasa_compra_u3,
    fd.ticket_promedio_u3,
    fd.ratio_pago_u3,

    -- Features RFM - ventana 6
    fd.num_compras_u6,
    fd.monto_total_u6,
    fd.monto_pagado_u6,
    fd.tasa_compra_u6,
    fd.ticket_promedio_u6,
    fd.ratio_pago_u6,

    -- Features RFM - ventana 12
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
    fd.campanas_desde_ultima_compra,
    fd.compras_historicas,
    fd.monto_historico,

    -- Tendencias
    fd.tendencia_monto_u3_vs_prev3,
    fd.tendencia_compras_u3_vs_prev3,
    fd.ticket_u3_vs_u12,

    -- Target (ahora con 6 leads)
    t.compro_t1,
    t.compro_t2,
    t.compro_t3,
    t.compro_t4,
    t.compro_t5,
    t.compro_t6,
    t.monto_t1,
    t.monto_t2,
    t.monto_t3,
    t.monto_t4,
    t.monto_t5,
    t.monto_t6,

    -- churn = 1 si NO compró en ninguna de las 6 siguientes
    -- NULL si falta alguna campaña posterior (bordes del DW)
    CASE
      WHEN t.compro_t1 IS NULL
        OR t.compro_t2 IS NULL
        OR t.compro_t3 IS NULL
        OR t.compro_t4 IS NULL
        OR t.compro_t5 IS NULL
        OR t.compro_t6 IS NULL
        THEN NULL
      WHEN (t.compro_t1 + t.compro_t2 + t.compro_t3
          + t.compro_t4 + t.compro_t5 + t.compro_t6) = 0 THEN 1
      ELSE 0
    END AS churn

  FROM features_derivadas fd
  LEFT JOIN target t
    ON fd.id_vendedor = t.id_vendedor
   AND fd.campana_rank = t.campana_rank
  LEFT JOIN `glamour-peru-dw.glamour_dw.dim_coordinadora` co
    ON fd.id_coordinadora = co.id_coordinadora
  LEFT JOIN `glamour-peru-dw.glamour_dw.dim_ubicacion` ub
    ON fd.ccodubigeo = ub.ccodubigeo
)

-- ----------------------------------------------------------------------------
-- 10. Resultado final
--     Filtros:
--       - churn IS NOT NULL: target calculable (hay 6 campañas siguientes)
--       - compro_en_obs = 1: solo observaciones post-compra (consistente con
--         producción; igual que el dataset original)
--       - compras_historicas >= 4: NUEVO. Exige al menos 4 compras
--         acumuladas hasta la campaña de observación (inclusive). Esto
--         excluye a las vendedoras de historia corta que generaban la regla
--         trivial "1 compra = churn" del dataset original.
-- ----------------------------------------------------------------------------
SELECT *
FROM ensamble
WHERE churn IS NOT NULL
  AND compro_en_obs = 1
  AND compras_historicas >= 4
;