# Opción A — LSTM como entrada para XGBoost

> Generado por `08_experimentation/01_opcion_A_lstm_xgboost.ipynb` el 2026-09-15 11:12.
> 30,821 filas, 91 features tabulares, secuencias de 24 meses × 6 canales.
> OOT = 885 filas (ranks 108–111), prevalencia 0.278.

## Resultados

| variante   |   oot_lift10 |   seg |   gkf_AUC |   gkf_AUC_std |   gkf_PRAUC |   gkf_lift10 |   oot_AUC |   oot_AUCstd |   oot_PRAUC |   oot_prec10 |   oot_rec10 |   oot_prec30 |   oot_rec30 |   oot_brier |   oot_ece |
|:-----------|-------------:|------:|----------:|--------------:|------------:|-------------:|----------:|-------------:|------------:|-------------:|------------:|-------------:|------------:|------------:|----------:|
| base       |       2.3302 |     8 |    0.7477 |        0.0065 |      0.5381 |       2.0091 |    0.7643 |       0.0221 |      0.5391 |       0.6477 |      0.2317 |       0.5188 |      0.561  |      0.2052 |    0.1801 |
| flat24     |       2.3302 |    10 |    0.7478 |        0.0066 |      0.5382 |       2.0164 |    0.7673 |       0.0199 |      0.5374 |       0.6477 |      0.2317 |       0.5226 |      0.565  |      0.2063 |    0.1832 |
| lstm_solo  |       2.4529 |    30 |    0.7454 |        0.0073 |      0.533  |       2.0143 |    0.7665 |       0.0309 |      0.5368 |       0.6818 |      0.2439 |       0.515  |      0.5569 |      0.2108 |    0.1939 |
| stack_lstm |       2.3302 |    64 |    0.7493 |        0.0071 |      0.5409 |       2.0154 |    0.7662 |       0.0244 |      0.5362 |       0.6477 |      0.2317 |       0.5338 |      0.5772 |      0.2114 |    0.1949 |
| emb_aux    |       2.2485 |   137 |    0.7482 |        0.0072 |      0.5402 |       2.0553 |    0.7614 |       0.0197 |      0.5313 |       0.625  |      0.2236 |       0.5113 |      0.5528 |      0.2154 |    0.2025 |
| emb_churn  |       2.2485 |    39 |    0.7491 |        0.0069 |      0.5419 |       2.0227 |    0.7609 |       0.029  |      0.5285 |       0.625  |      0.2236 |       0.515  |      0.5569 |      0.2125 |    0.1938 |
| todo       |       2.2076 |   197 |    0.7484 |        0.0071 |      0.5397 |       2.0248 |    0.7632 |       0.0223 |      0.531  |       0.6136 |      0.2195 |       0.5038 |      0.5447 |      0.2147 |    0.1999 |
| rank_avg   |       2.2485 |     0 |    0.7489 |      nan      |      0.5394 |       2.029  |    0.7682 |       0.0257 |      0.5418 |       0.625  |      0.2236 |       0.5188 |      0.561  |      0.2253 |    0.2226 |

## ΔAUC pareado vs base (IC 95 % bootstrap, 1000 réplicas)

| variante   |   ΔAUC_gkf |   gkf_lo |   gkf_hi |   ΔAUC_oot |   oot_lo |   oot_hi |
|:-----------|-----------:|---------:|---------:|-----------:|---------:|---------:|
| flat24     |     0.0002 |  -0.0006 |   0.0009 |     0.0029 |  -0.0011 |   0.0071 |
| lstm_solo  |    -0.0022 |  -0.004  |  -0.0003 |     0.0022 |  -0.0092 |   0.0153 |
| stack_lstm |     0.0017 |   0.0008 |   0.0026 |     0.0018 |  -0.0041 |   0.0077 |
| emb_aux    |     0.0005 |  -0.0006 |   0.0016 |    -0.003  |  -0.0117 |   0.0063 |
| emb_churn  |     0.0014 |   0.0001 |   0.0026 |    -0.0035 |  -0.0122 |   0.0052 |
| todo       |     0.0007 |  -0.0005 |   0.0018 |    -0.0011 |  -0.0103 |   0.0087 |
| rank_avg   |     0.0012 |   0.0003 |   0.0022 |     0.0038 |  -0.0021 |   0.0107 |

## Complementariedad base ↔ LSTM

|     |   spearman_base_lstm |   overlap_top10 |
|:----|---------------------:|----------------:|
| gkf |                0.962 |           0.756 |
| oot |                0.959 |           0.705 |

In-sample (bloque train OOT): AUC train base 0.7641 vs con `emb_churn`
0.7663; el embedding concentra 69.7% del gain.

## Lectura

- Mejor GroupKFold: `stack_lstm` 0.7493 vs base 0.7477
  (OOT 0.7662 vs 0.7643).
- La LSTM sola alcanza 0.7454 GKF /
  0.7665 OOT con solo 6 canales mensuales:
  la secuencia contiene casi la misma información que las 91 features (lo mismo se ve en `flat24`).
- Criterio de decisión: una variante "aporta" si su ΔAUC GKF tiene IC que excluye 0 **y** el ΔAUC OOT
  tiene el mismo signo. Ver tabla de deltas.
