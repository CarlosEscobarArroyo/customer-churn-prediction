# Opción C — Google Trends

> Generado por `08_experimentation/03_opcion_C_google_trends.ipynb` el 2026-09-15 11:31.
> 20 términos (geo=PE, mensual, 2016-01 → 2026-02), 60 features. OOT = 885 filas.

## Calidad de las series

|                     | familia      |   n_meses |   pct_ceros |   media |   max |   std | primer_mes_>0   |   salto_2022_ratio |
|:--------------------|:-------------|----------:|------------:|--------:|------:|------:|:----------------|-------------------:|
| venta por catalogo  | catalogo     |       122 |       0.008 |    35.5 |   100 |  24.2 | 2016-01         |               0.78 |
| catalogo            | catalogo     |       122 |       0     |    50.5 |   100 |  23.5 | 2016-01         |               0.81 |
| catalogo de ropa    | catalogo     |       122 |       0.008 |    33.1 |   100 |  25.6 | 2016-01         |               0.81 |
| vender por catalogo | catalogo     |       122 |       0.508 |    28.1 |   100 |  31.5 | 2016-01         |               0.59 |
| glamour             | catalogo     |       122 |       0     |    61.2 |   100 |  12.5 | 2016-01         |               1.13 |
| yanbal              | competencia  |       122 |       0     |    42.3 |   100 |  28.2 | 2016-01         |               1.57 |
| natura              | competencia  |       122 |       0     |    69.3 |   100 |   7.3 | 2016-01         |               1.11 |
| unique              | competencia  |       122 |       0     |    50.3 |   100 |  28.5 | 2016-01         |               0.85 |
| esika               | competencia  |       122 |       0     |    68.2 |   100 |  14.9 | 2016-01         |               1.12 |
| belcorp             | competencia  |       122 |       0     |    57.5 |   100 |  19.3 | 2016-01         |               1.23 |
| shein               | alternativas |       122 |       0.008 |    26.7 |   100 |  23.1 | 2016-02         |               1.34 |
| temu                | alternativas |       122 |       0.721 |    12.2 |   100 |  26.8 | 2023-05         |             nan    |
| gamarra             | alternativas |       122 |       0     |    49.3 |   100 |   9.1 | 2016-01         |               1.19 |
| ropa por mayor      | alternativas |       122 |       0     |    24.3 |   100 |  12.7 | 2016-01         |               0.69 |
| trabajo desde casa  | alternativas |       122 |       0     |    46.6 |   100 |  18.4 | 2016-01         |               1.07 |
| trabajo             | macro        |       122 |       0     |    71   |   100 |  10.3 | 2016-01         |               1.18 |
| prestamo            | macro        |       122 |       0     |    62.1 |   100 |  14.6 | 2016-01         |               1.32 |
| ofertas             | macro        |       122 |       0     |    48.2 |   100 |  16.9 | 2016-01         |               0.89 |
| ropa                | macro        |       122 |       0     |    76.8 |   100 |   8.1 | 2016-01         |               1.08 |
| cts                 | macro        |       122 |       0     |    11.5 |   100 |  13   | 2016-01         |               1.13 |

`salto_2022_ratio` = media 2022 / media 2021 (cambio de metodología de Google en 2022-01).

## Nivel mensual

Mejor lag por término (Spearman con la tasa mensual de churn, meses de train, desestacionalizada):

| termino             |   lag |   rho_bruto |   rho_desestac |   n |
|:--------------------|------:|------------:|---------------:|----:|
| trabajo             |     6 |       0.464 |          0.472 | 100 |
| trabajo desde casa  |     6 |      -0.405 |         -0.456 | 100 |
| glamour             |     5 |       0.375 |          0.437 | 100 |
| yanbal              |     4 |      -0.353 |         -0.394 | 100 |
| ropa                |     2 |      -0.263 |         -0.385 | 100 |
| esika               |     5 |       0.348 |          0.376 | 100 |
| unique              |     0 |       0.372 |          0.361 | 100 |
| ofertas             |     1 |       0.36  |          0.336 | 100 |
| vender por catalogo |     0 |       0.274 |          0.335 | 100 |
| catalogo            |     6 |       0.245 |          0.307 | 100 |
| natura              |     2 |      -0.303 |         -0.305 | 100 |
| shein               |     6 |      -0.268 |         -0.305 | 100 |
| belcorp             |     6 |       0.27  |          0.304 | 100 |
| venta por catalogo  |     2 |       0.231 |          0.279 | 100 |
| catalogo de ropa    |     6 |       0.229 |          0.277 | 100 |
| gamarra             |     0 |       0.308 |          0.256 | 100 |
| ropa por mayor      |     4 |      -0.214 |         -0.252 | 100 |
| prestamo            |     0 |      -0.148 |         -0.231 | 100 |
| cts                 |     5 |      -0.075 |          0.131 | 100 |
| temu                |     1 |      -0.072 |         -0.092 | 100 |

SE(rho) ≈ 0.102 con n = 100 meses.

Pronóstico expansivo de la tasa mensual (22 meses, train ≤ t−7), RMSE:
|                   |      0 |
|:------------------|-------:|
| media_train       | 0.0404 |
| estacional        | 0.04   |
| estacional+trends | 0.066  |
Desvío de la tasa real: 0.0384. Términos usados: ['trabajo', 'trabajo desde casa', 'glamour', 'yanbal', 'ropa'].

## Nivel de fila

| variante             |   n_feats |   oot_lift10 |   gkf_AUC |   gkf_AUC_std |   gkf_PRAUC |   gkf_lift10 |   oot_AUC |   oot_AUCstd |   oot_PRAUC |   oot_prec10 |   oot_rec10 |   oot_prec30 |   oot_rec30 |   oot_brier |   oot_ece |
|:---------------------|----------:|-------------:|----------:|--------------:|------------:|-------------:|----------:|-------------:|------------:|-------------:|------------:|-------------:|------------:|------------:|----------:|
| base                 |        91 |       2.3302 |    0.7477 |        0.0065 |      0.5381 |       2.0091 |    0.7643 |       0.0221 |      0.5391 |       0.6477 |      0.2317 |       0.5188 |      0.561  |      0.2052 |    0.1801 |
| base+trends_catalogo |       106 |       2.2894 |    0.7514 |        0.0074 |      0.5455 |       2.0322 |    0.7681 |       0.021  |      0.5388 |       0.6364 |      0.2276 |       0.5075 |      0.5488 |      0.1997 |    0.1671 |
| base+trends_todos    |       151 |       2.3711 |    0.7526 |        0.0077 |      0.5471 |       2.0395 |    0.7691 |       0.0219 |      0.5448 |       0.6591 |      0.2358 |       0.5376 |      0.5813 |      0.219  |    0.2122 |
| base+oraculo_mes     |        92 |       2.412  |    0.7537 |        0.0076 |      0.5489 |       2.0448 |    0.7671 |       0.0192 |      0.5442 |       0.6705 |      0.2398 |       0.5526 |      0.5976 |      0.1914 |    0.1413 |

ΔAUC OOT pareado vs base (IC 95 %):

| variante             |   oot_delta |   oot_ci_lo |   oot_ci_hi |
|:---------------------|------------:|------------:|------------:|
| base+trends_catalogo |      0.0038 |     -0.0002 |      0.008  |
| base+trends_todos    |      0.0047 |     -0.0009 |      0.0106 |
| base+oraculo_mes     |      0.0027 |     -0.0022 |      0.0081 |

Rolling-origin (12 meses, train ≤ t−7):

|                      |   AUC_medio_dentro_del_mes |   AUC_agrupado_12m |   meses_gana_a_base |   Δ_agrupado_vs_base |   Δ_dentro_mes_vs_base |
|:---------------------|---------------------------:|-------------------:|--------------------:|---------------------:|-----------------------:|
| base                 |                     0.7712 |             0.7688 |                   0 |               0      |                 0      |
| base+trends_catalogo |                     0.7738 |             0.7711 |                   9 |               0.0023 |                 0.0027 |
| base+trends_todos    |                     0.7734 |             0.769  |                  10 |               0.0002 |                 0.0023 |
| base+oraculo_mes     |                     0.7735 |             0.7755 |                   7 |               0.0067 |                 0.0024 |

Detalle por mes:

|   mes_rank |   n_test |   prev |   base |   base+trends_catalogo |   base+trends_todos |   base+oraculo_mes |
|-----------:|---------:|-------:|-------:|-----------------------:|--------------------:|-------------------:|
|        100 |      210 | 0.3238 | 0.7987 |                 0.799  |              0.7997 |             0.7957 |
|        101 |      214 | 0.2944 | 0.7965 |                 0.8006 |              0.798  |             0.8071 |
|        102 |      235 | 0.3234 | 0.7613 |                 0.7637 |              0.7629 |             0.7651 |
|        103 |      213 | 0.3286 | 0.7375 |                 0.7448 |              0.7445 |             0.7421 |
|        104 |      199 | 0.3116 | 0.7913 |                 0.7916 |              0.7935 |             0.7951 |
|        105 |      171 | 0.2865 | 0.812  |                 0.8024 |              0.8041 |             0.8081 |
|        106 |      186 | 0.2581 | 0.777  |                 0.7745 |              0.7772 |             0.7729 |
|        107 |      197 | 0.2792 | 0.756  |                 0.7538 |              0.749  |             0.7552 |
|        108 |      184 | 0.2174 | 0.7248 |                 0.729  |              0.729  |             0.7307 |
|        109 |      236 | 0.2924 | 0.7755 |                 0.7812 |              0.7829 |             0.779  |
|        110 |      273 | 0.3114 | 0.7778 |                 0.784  |              0.7838 |             0.7764 |
|        111 |      192 | 0.2708 | 0.7456 |                 0.7611 |              0.7565 |             0.7547 |

## Lectura

- El oráculo mensual (`base+oraculo_mes`) marca el techo de cualquier variable constante por mes:
  OOT 0.7671 vs base 0.7643;
  agrupado 12 m 0.7755 vs 0.7688
  y dentro del mes 0.7735 vs 0.7712.
- Trends: OOT 0.7691 (todos) / 0.7681 (catálogo);
  rolling agrupado 0.7690, dentro del mes 0.7734.
- Una variable constante por mes solo puede mover el AUC agrupado (nivel del mes); si el AUC dentro
  del mes no cambia, no mejora la selección individual de vendedoras.
- GroupKFold con features mensuales es optimista (mismos meses en train y validación); se reporta pero no decide.
