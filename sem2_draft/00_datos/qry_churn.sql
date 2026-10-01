-- ============================================================================
-- qry_churn.sql — Dataset de churn de vendedoras de Glamour Perú (sem2)
-- ============================================================================
-- Hereda de sem1/00_dataset_construction/qry_churn.sql. Cambios en sem2:
--   • Se eliminan los datos maestros (sexo, edad, antigüedad, tipo, ubicación):
--     solo existe su valor vigente, no su historial (metodología §4.2).
--   • Se eliminan las variables de campañas: no mejoraron el AUC en sem1
--     (ensemble de 42 variables == modelos de 71 en el test OOT).
--   • Corte al último mes COMPLETO de pedidos: el mes en curso del warehouse
--     está incompleto y contaminaría la ventana de etiqueta de los últimos
--     meses observados.
--
-- Definiciones (sin cambio respecto de sem1):
--   • Una fila por (id_vendedor, mes_obs); esquema multi-slice mensual.
--   • Ventana de features w = 12 meses hacia atrás, incluye t.
--   • Etiqueta: churn = 1 si no compra en ninguno de los meses t+1..t+6.
--     Filas sin 6 meses de futuro completos se excluyen.
--   • Población en t: compró en el mes t Y tiene >= 1 compra previa.
--   • Variables: RFM por ventana u3/u6/u12, tendencias, acumuladas (CF) y
--     deltas (DF) según Mirkovic et al. (2022). 36 variables.
--   • Productos huérfanos (sin ficha en dim_producto) se imputan como 'Ropa'.
--
-- Salida: id_vendedor, mes_obs, mes_rank, churn + 36 variables.
-- ============================================================================

WITH
-- Primer mes que NO entra: el del último pedido, salvo que ese pedido caiga en
-- el último día del mes (entonces el mes se considera completo) ---------------
corte AS (
  SELECT IF(MAX(df.date) = LAST_DAY(MAX(df.date)),
            DATE_ADD(DATE_TRUNC(MAX(df.date), MONTH), INTERVAL 1 MONTH),
            DATE_TRUNC(MAX(df.date), MONTH)) AS mes_incompleto
  FROM `glamour-peru-dw.glamour_dw.fact_pedidos` fp
  JOIN `glamour-peru-dw.glamour_dw.dim_fecha`    df USING (id_fecha)
),

-- Pedidos agregados a (vendedora, mes calendario) -----------------------------
ped AS (
  SELECT
    fp.id_vendedor,
    DATE_TRUNC(df.date, MONTH)        AS mes,
    COUNT(*)                          AS n_ped,
    SUM(fp.monto_total_pedido)        AS monto
  FROM `glamour-peru-dw.glamour_dw.fact_pedidos` fp
  JOIN `glamour-peru-dw.glamour_dw.dim_fecha`    df USING (id_fecha)
  CROSS JOIN corte
  WHERE DATE_TRUNC(df.date, MONTH) < corte.mes_incompleto
  GROUP BY 1, 2
),

-- Diversidad de producto por (vendedora, mes); huérfano -> 'Ropa' -------------
prod AS (
  SELECT
    fd.id_vendedor,
    DATE_TRUNC(df.date, MONTH)                              AS mes,
    COUNT(DISTINCT fd.id_producto)                          AS n_prod,
    COUNT(DISTINCT COALESCE(dp.categoria, 'Ropa'))          AS n_cat
  FROM `glamour-peru-dw.glamour_dw.fact_pedidos_detalle` fd
  JOIN `glamour-peru-dw.glamour_dw.dim_fecha`            df USING (id_fecha)
  LEFT JOIN `glamour-peru-dw.glamour_dw.dim_producto`    dp USING (id_producto)
  CROSS JOIN corte
  WHERE DATE_TRUNC(df.date, MONTH) < corte.mes_incompleto
  GROUP BY 1, 2
),

-- Espina de meses contigua entre el primer y último mes con datos -------------
bounds AS (
  SELECT DATE_TRUNC(MIN(mes), MONTH) AS gmin, DATE_TRUNC(MAX(mes), MONTH) AS gmax
  FROM ped
),
cal AS (
  SELECT mes, ROW_NUMBER() OVER (ORDER BY mes) AS mes_rank
  FROM bounds, UNNEST(GENERATE_DATE_ARRAY(gmin, gmax, INTERVAL 1 MONTH)) AS mes
),
rango_global AS ( SELECT MAX(mes_rank) AS max_rank FROM cal ),

-- Panel DENSO: vendedora x todos los meses desde su 1er mes hasta gmax ---------
rng AS (
  SELECT id_vendedor, MIN(mes) AS primer_mes FROM ped GROUP BY 1
),
dense AS (
  SELECT r.id_vendedor, c.mes, c.mes_rank
  FROM rng r
  JOIN cal c ON c.mes >= r.primer_mes
),
panel AS (
  SELECT
    d.id_vendedor, d.mes AS mes_obs, d.mes_rank,
    IFNULL(p.n_ped, 0)                    AS n_ped,
    IFNULL(p.monto, 0)                    AS monto,
    IFNULL(pr.n_prod, 0)                  AS n_prod,
    IFNULL(pr.n_cat, 0)                   AS n_cat,
    CASE WHEN p.n_ped > 0 THEN 1 ELSE 0 END AS activo
  FROM dense d
  LEFT JOIN ped  p  USING (id_vendedor, mes)
  LEFT JOIN prod pr ON pr.id_vendedor = d.id_vendedor AND pr.mes = d.mes
),

-- Features de ventana + etiqueta ---------------------------------------------
feat AS (
  SELECT
    p.*,
    -- ETIQUETA: actividad en t+1..t+6 -----------------------------------------
    SUM(activo) OVER fwd6                                          AS compras_fwd6,
    -- Historia previa (meses activos antes de t) ------------------------------
    SUM(activo) OVER hist                                          AS compras_hist,
    -- FRECUENCIA --------------------------------------------------------------
    SUM(activo) OVER w3   AS meses_activos_u3,
    SUM(activo) OVER w6   AS meses_activos_u6,
    SUM(activo) OVER w12  AS meses_activos_u12,
    SUM(n_ped)  OVER w3   AS n_ped_u3,
    SUM(n_ped)  OVER w6   AS n_ped_u6,
    SUM(n_ped)  OVER w12  AS n_ped_u12,
    -- MONETARIO ---------------------------------------------------------------
    SUM(monto)    OVER w3   AS monto_u3,
    SUM(monto)    OVER w6   AS monto_u6,
    SUM(monto)    OVER w12  AS monto_u12,
    AVG(monto)    OVER w12  AS monto_mean_u12,
    STDDEV(monto) OVER w12  AS monto_std_u12,
    -- DIVERSIDAD DE PRODUCTO --------------------------------------------------
    SUM(n_prod) OVER w12  AS n_prod_u12,
    MAX(n_cat)  OVER w12  AS n_cat_max_u12,
    -- RECENCIA: meses desde la compra previa ----------------------------------
    p.mes_rank - MAX(CASE WHEN activo = 1 THEN p.mes_rank END) OVER ant
                                                                  AS meses_desde_compra_previa,
    -- componentes para tendencias normalizadas -------------------------------
    SUM(monto) OVER prev3 AS monto_prev3,
    SUM(n_ped) OVER prev3 AS nped_prev3,
    -- CUMULATIVE FEATURES (CF): toda la historia [primer registro .. t] -------
    SUM(monto)  OVER acum AS monto_acum,
    SUM(n_ped)  OVER acum AS n_ped_acum,
    SUM(n_prod) OVER acum AS n_prod_acum,
    SUM(activo) OVER acum AS meses_activos_acum,
    MAX(n_cat)  OVER acum AS n_cat_max_acum,
    -- DELTA FEATURES (DF): mt vs mt-n, n ∈ {1,3,6,9,12} ------------------------
    monto - LAG(monto, 1)  OVER seq AS d_monto_m1,
    monto - LAG(monto, 3)  OVER seq AS d_monto_m3,
    monto - LAG(monto, 6)  OVER seq AS d_monto_m6,
    monto - LAG(monto, 9)  OVER seq AS d_monto_m9,
    monto - LAG(monto, 12) OVER seq AS d_monto_m12,
    n_ped - LAG(n_ped, 1)  OVER seq AS d_nped_m1,
    n_ped - LAG(n_ped, 3)  OVER seq AS d_nped_m3,
    n_ped - LAG(n_ped, 6)  OVER seq AS d_nped_m6,
    n_ped - LAG(n_ped, 9)  OVER seq AS d_nped_m9,
    n_ped - LAG(n_ped, 12) OVER seq AS d_nped_m12
  FROM panel p
  WINDOW
    fwd6  AS (PARTITION BY id_vendedor ORDER BY mes_rank ROWS BETWEEN 1 FOLLOWING AND 6 FOLLOWING),
    hist  AS (PARTITION BY id_vendedor ORDER BY mes_rank ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    ant   AS (PARTITION BY id_vendedor ORDER BY mes_rank ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
    acum  AS (PARTITION BY id_vendedor ORDER BY mes_rank ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
    seq   AS (PARTITION BY id_vendedor ORDER BY mes_rank),
    w3    AS (PARTITION BY id_vendedor ORDER BY mes_rank ROWS BETWEEN 2  PRECEDING AND CURRENT ROW),
    w6    AS (PARTITION BY id_vendedor ORDER BY mes_rank ROWS BETWEEN 5  PRECEDING AND CURRENT ROW),
    w12   AS (PARTITION BY id_vendedor ORDER BY mes_rank ROWS BETWEEN 11 PRECEDING AND CURRENT ROW),
    prev3 AS (PARTITION BY id_vendedor ORDER BY mes_rank ROWS BETWEEN 5  PRECEDING AND 3 PRECEDING)
)

-- Salida final: una fila por (vendedora, mes_obs) ----------------------------
SELECT
  -- Identificadores / temporales (NO features) --------------------------------
  f.id_vendedor,
  f.mes_obs,
  f.mes_rank,
  -- ETIQUETA ------------------------------------------------------------------
  CASE WHEN f.compras_fwd6 = 0 THEN 1 ELSE 0 END              AS churn,
  -- FRECUENCIA ----------------------------------------------------------------
  f.meses_activos_u3, f.meses_activos_u6, f.meses_activos_u12,
  f.n_ped_u3, f.n_ped_u6, f.n_ped_u12,
  -- MONETARIO -----------------------------------------------------------------
  f.monto_u3, f.monto_u6, f.monto_u12,
  f.monto_mean_u12, f.monto_std_u12,
  SAFE_DIVIDE(f.monto_std_u12, f.monto_mean_u12)              AS monto_cv_u12,
  SAFE_DIVIDE(f.monto, f.monto_mean_u12)                      AS monto_ult_vs_media,
  -- RECENCIA ------------------------------------------------------------------
  f.meses_desde_compra_previa,
  f.compras_hist,
  -- DIVERSIDAD ----------------------------------------------------------------
  f.n_prod_u12, f.n_cat_max_u12,
  -- TENDENCIAS normalizadas en [-1, 1]  (u3 vs prev3) -------------------------
  SAFE_DIVIDE(f.monto_u3 - f.monto_prev3, f.monto_u3 + f.monto_prev3) AS tend_monto_u3_vs_prev3,
  SAFE_DIVIDE(f.n_ped_u3 - f.nped_prev3, f.n_ped_u3 + f.nped_prev3)   AS tend_nped_u3_vs_prev3,
  -- CUMULATIVE FEATURES (toda la historia hasta t) ----------------------------
  f.monto_acum, f.n_ped_acum, f.n_prod_acum, f.n_cat_max_acum,
  SAFE_DIVIDE(f.monto_acum, f.n_ped_acum)            AS ticket_acum,
  SAFE_DIVIDE(f.monto_acum, f.n_prod_acum)           AS monto_por_prod_acum,
  SAFE_DIVIDE(f.monto_acum, f.meses_activos_acum)    AS monto_mensual_acum,
  -- DELTA FEATURES (mt vs mt-n) -----------------------------------------------
  f.d_monto_m1, f.d_monto_m3, f.d_monto_m6, f.d_monto_m9, f.d_monto_m12,
  f.d_nped_m1, f.d_nped_m3, f.d_nped_m6, f.d_nped_m9, f.d_nped_m12
FROM feat f
CROSS JOIN rango_global g
WHERE f.activo = 1                          -- (población) compró en el mes t
  AND f.compras_hist >= 1                   -- (población) tiene historia previa
  AND f.mes_rank <= g.max_rank - 6          -- (etiqueta) hay 6 meses de futuro observables
ORDER BY f.id_vendedor, f.mes_rank
