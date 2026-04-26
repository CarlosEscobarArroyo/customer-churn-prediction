-- ============================================================================
-- Dataset de entrenamiento para modelo de churn de vendedores  (v3)
--
-- CAMBIOS vs. v2 (auditoría de leakage + limpieza):
--
--   1. Eliminada `estado_coordinadora`.
--        - Motivo: `dim_coordinadora.estado` es snapshot actual (SCD-1, sin
--          historial). Para una fila de campaña C50 estaríamos usando el
--          estado de la coordinadora a la fecha de extracción del DW, no el
--          que tenía en C50. Es leakage temporal real.
--
--   2. Eliminadas pseudo-features constantes (consecuencia directa de los
--      filtros finales):
--        - `compro_en_obs`              → siempre 1 con `compro_en_obs = 1`.
--        - `es_nueva_vendedora`         → siempre 0 con `compras_historicas >= 4`
--                                         (antiguedad < 3 ⇒ compras < 4).
--        - `campanas_desde_ultima_compra` → siempre 0 con `compro_en_obs = 1`
--                                           (la "última compra" es la actual).
--
--   3. Nueva feature de recencia: `campanas_desde_compra_previa`.
--        - Gap entre la campaña de observación y la compra inmediatamente
--          ANTERIOR. Esta sí varía y mide regularidad de compra.
--        - NULL en la primera observación de cada vendedora (no hay previa).
--
--   4. Eliminadas las columnas de target intermedio del SELECT final
--      (`compro_t1..t6`, `monto_t1..t6`).
--        - Motivo: información post-observación. Aunque el código de modelado
--          las excluía manualmente, mantenerlas en el dataset aumentaba el
--          riesgo de uso accidental como feature. Solo queda `churn`.
--
--   5. Tendencias re-expresadas como diferencias normalizadas en [-1, 1]:
--        - `delta_monto_u3_vs_prev3`    (reemplaza `tendencia_monto_u3_vs_prev3`)
--        - `delta_compras_u3_vs_prev3`  (reemplaza `tendencia_compras_u3_vs_prev3`)
--        Fórmula: (u3 - prev3) / (u3 + prev3); 0 cuando ambas ventanas son 0.
--        Evita el patrón de NaN correlacionado con el target que aparecía en v2.
--        Se mantiene `ticket_u3_vs_u12` (denominador no se anula post-filtro).
--
-- SIN CAMBIOS vs. v2:
--   - Definición de churn: 6 campañas consecutivas siguientes sin compra.
--   - Filtros de población: `churn IS NOT NULL`, `compro_en_obs = 1`,
--     `compras_historicas >= 4`.
--   - Granularidad: una fila por (id_vendedor, id_campana_obs).
--   - Ventanas RFM 3/6/12 (todas miran solo hacia atrás).
--
-- NOTAS DE SCD-1 (atributos snapshot actual del DW, no históricos):
--   `edad_vendedor`, `edad_coordinadora`, `tipo_vendedor`, `id_coordinadora`,
--   `ccodubigeo` y derivados (`distrito`, `provincia`, `departamento`).
--   Mantenidos como features porque son razonablemente estables en el
--   horizonte temporal del dataset, pero si el DW expone versiones
--   historizadas en el futuro, conviene regenerar.
-- ============================================================================

CREATE OR REPLACE TABLE `glamour-peru-dw.glamour_dw.training_churn_v3`
PARTITION BY RANGE_BUCKET(id_campana_obs, GENERATE_ARRAY(202001, 202700, 1))
CLUSTER BY id_vendedor
AS

WITH

-- ----------------------------------------------------------------------------
-- 1. Ranking global de campañas
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
    SUM(monto_pagado)                  AS monto_pagado
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
-- 4. Ventana de vida activa por vendedor
--    Panel hasta ultima_compra_rank + 6 para que la última compra observada
--    tenga 6 campañas siguientes calculables (target).
-- ----------------------------------------------------------------------------
vida_vendedor AS (
  SELECT
    pa.id_vendedor,
    MIN(c.campana_rank) AS primera_compra_rank,
    MAX(c.campana_rank) AS ultima_compra_rank
  FROM pedidos_agg pa
  JOIN campanas_ordenadas c USING (id_campana)
  GROUP BY pa.id_vendedor
),

-- ----------------------------------------------------------------------------
-- 5. Panel: (vendedor × campaña) restringido a la ventana de vida activa
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

    COALESCE(pa.num_pedidos, 0)        AS num_pedidos,
    COALESCE(pa.monto_total, 0)        AS monto_total,
    COALESCE(pa.monto_pagado, 0)       AS monto_pagado,
    IF(pa.num_pedidos > 0, 1, 0)       AS compro,

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
  WHERE c.campana_rank >= vv.primera_compra_rank
    AND c.campana_rank <= vv.ultima_compra_rank + 6
),

-- ----------------------------------------------------------------------------
-- 6. Features con window functions: RFM + recencia + histórico
--    Todas las ventanas son `ROWS BETWEEN N PRECEDING AND CURRENT ROW`
--    (cero leakage temporal por construcción).
-- ----------------------------------------------------------------------------
features AS (
  SELECT
    p.*,

    ROW_NUMBER() OVER (PARTITION BY id_vendedor ORDER BY campana_rank)
      AS antiguedad_campanas,

    -- Ventana de 3 campañas
    SUM(compro)        OVER w3  AS num_compras_u3,
    SUM(monto_total)   OVER w3  AS monto_total_u3,
    SUM(monto_pagado)  OVER w3  AS monto_pagado_u3,
    AVG(compro)        OVER w3  AS tasa_compra_u3,
    SAFE_DIVIDE(SUM(monto_total) OVER w3, NULLIF(SUM(compro) OVER w3, 0))
      AS ticket_promedio_u3,
    SAFE_DIVIDE(SUM(monto_pagado) OVER w3, NULLIF(SUM(monto_total) OVER w3, 0))
      AS ratio_pago_u3,

    -- Ventana de 6 campañas
    SUM(compro)        OVER w6  AS num_compras_u6,
    SUM(monto_total)   OVER w6  AS monto_total_u6,
    SUM(monto_pagado)  OVER w6  AS monto_pagado_u6,
    AVG(compro)        OVER w6  AS tasa_compra_u6,
    SAFE_DIVIDE(SUM(monto_total) OVER w6, NULLIF(SUM(compro) OVER w6, 0))
      AS ticket_promedio_u6,
    SAFE_DIVIDE(SUM(monto_pagado) OVER w6, NULLIF(SUM(monto_total) OVER w6, 0))
      AS ratio_pago_u6,

    -- Ventana de 12 campañas
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

    -- Recencia v3: gap a la compra ANTERIOR (no la actual).
    -- BigQuery no soporta IGNORE NULLS en LAG/LEAD, así que usamos
    -- LAST_VALUE con frame estrictamente anterior a la fila actual
    -- (UNBOUNDED PRECEDING ... 1 PRECEDING) para tomar el rank de la
    -- compra más reciente ANTES de la fila. En la primera observación
    -- con compra de la vendedora vale NULL (no hay compra previa).
    campana_rank - LAST_VALUE(
      IF(compro = 1, campana_rank, NULL) IGNORE NULLS
    ) OVER (PARTITION BY id_vendedor ORDER BY campana_rank
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)
      AS campanas_desde_compra_previa,

    -- Histórico acumulado (solo mira atrás: UNBOUNDED PRECEDING → CURRENT ROW)
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
-- 7. Features derivadas: tendencias normalizadas en [-1, 1]
--    Fórmula: (u3 - prev3) / (u3 + prev3) = (2·u3 - u6) / u6
--    - +1: toda la actividad de u6 está concentrada en u3 (aceleración).
--    - -1: toda la actividad de u6 está en las 3 campañas previas (caída).
--    -  0: distribución uniforme o ambas ventanas son 0 (COALESCE).
--    Reemplaza las tendencias-ratio de v2 que producían NaN cuando
--    `monto_u6 - monto_u3 = 0` (justo el caso correlacionado con churn).
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

    -- Ratio ticket reciente vs largo plazo. Post-filtro `compro_en_obs = 1`
    -- el denominador (ticket_promedio_u12) no se anula porque la fila actual
    -- tiene compra ⇒ num_compras_u12 ≥ 1.
    SAFE_DIVIDE(ticket_promedio_u3, NULLIF(ticket_promedio_u12, 0))
      AS ticket_u3_vs_u12

  FROM features f
),

-- ----------------------------------------------------------------------------
-- 8. Target: churn en las 6 campañas siguientes
--    Solo se usa para construir `churn`. Las columnas LEAD individuales
--    NO se exponen en el SELECT final (cambio v3).
-- ----------------------------------------------------------------------------
target AS (
  SELECT
    id_vendedor,
    campana_rank,
    LEAD(compro, 1) OVER w_fwd AS compro_t1,
    LEAD(compro, 2) OVER w_fwd AS compro_t2,
    LEAD(compro, 3) OVER w_fwd AS compro_t3,
    LEAD(compro, 4) OVER w_fwd AS compro_t4,
    LEAD(compro, 5) OVER w_fwd AS compro_t5,
    LEAD(compro, 6) OVER w_fwd AS compro_t6
  FROM panel
  WINDOW w_fwd AS (PARTITION BY id_vendedor ORDER BY campana_rank)
),

-- ----------------------------------------------------------------------------
-- 9. Ensamble final
--    Eliminadas vs v2:
--      - estado_coordinadora        (leakage SCD-1)
--      - compro_en_obs              (constante con filtro)
--      - es_nueva_vendedora         (constante con filtro)
--      - campanas_desde_ultima_compra (constante con filtro)
--      - compro_t1..t6, monto_t1..t6 (target intermedio, riesgo de uso
--                                     accidental como feature)
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

    -- Vendedora (atributos SCD-1: ver header)
    fd.fecha_ingreso,
    fd.edad_vendedor,
    fd.sexo_vendedor,
    fd.tipo_vendedor,
    fd.ccodrelacion,
    fd.antiguedad_campanas,

    -- Coordinadora (sin estado_coordinadora)
    fd.id_coordinadora,
    co.edad                                     AS edad_coordinadora,

    -- Ubicación
    fd.ccodubigeo,
    ub.distrito,
    ub.provincia,
    ub.departamento,

    -- Métricas de la campaña actual
    fd.num_pedidos                              AS num_pedidos_obs,
    fd.monto_total                              AS monto_total_obs,
    fd.monto_pagado                             AS monto_pagado_obs,
    fd.num_categorias                           AS num_categorias_obs,
    fd.num_productos                            AS num_productos_obs,

    -- RFM ventana 3
    fd.num_compras_u3,
    fd.monto_total_u3,
    fd.monto_pagado_u3,
    fd.tasa_compra_u3,
    fd.ticket_promedio_u3,
    fd.ratio_pago_u3,

    -- RFM ventana 6
    fd.num_compras_u6,
    fd.monto_total_u6,
    fd.monto_pagado_u6,
    fd.tasa_compra_u6,
    fd.ticket_promedio_u6,
    fd.ratio_pago_u6,

    -- RFM ventana 12
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
    fd.campanas_desde_compra_previa,
    fd.compras_historicas,
    fd.monto_historico,

    -- Tendencias normalizadas
    fd.delta_monto_u3_vs_prev3,
    fd.delta_compras_u3_vs_prev3,
    fd.ticket_u3_vs_u12,

    -- Variables auxiliares solo para construir el filtro/target. NO usar
    -- como features (se filtran en el SELECT final, no aparecen en la tabla).
    fd.compro                                   AS _compro_en_obs,

    -- Target
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
--     Filtros (sin cambios vs v2):
--       - churn IS NOT NULL: target calculable (hay 6 campañas siguientes).
--       - _compro_en_obs = 1: solo observaciones donde la vendedora compró
--         en la campaña de observación (consistente con producción).
--       - compras_historicas >= 4: excluye historia corta que generaba la
--         regla trivial "1 compra = churn".
--     `_compro_en_obs` se usa solo como filtro y NO se proyecta.
-- ----------------------------------------------------------------------------
SELECT
  id_vendedor, id_campana_obs, campana_rank_obs, anio_campana, numero_campana,
  fecha_inicio_campana, fecha_ingreso, edad_vendedor, sexo_vendedor,
  tipo_vendedor, ccodrelacion, antiguedad_campanas, id_coordinadora,
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
  campanas_desde_compra_previa, compras_historicas, monto_historico,
  delta_monto_u3_vs_prev3, delta_compras_u3_vs_prev3, ticket_u3_vs_u12,
  churn
FROM ensamble
WHERE churn IS NOT NULL
  AND _compro_en_obs = 1
  AND compras_historicas >= 4
;
