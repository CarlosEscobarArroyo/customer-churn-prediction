# Opción B — Ritmo individual, pausas, rachas y campañas

> Generado por `08_experimentation/02_opcion_B_ritmo_pausas_campanas.ipynb` el 2026-09-15 11:13.
> 29 features `r_*` calculadas sobre el panel denso con historia ≤ t. OOT = 885 filas.

## Resultados

| variante          |   n_feats |   oot_lift10 |   gkf_AUC |   gkf_AUC_std |   gkf_PRAUC |   gkf_lift10 |   oot_AUC |   oot_AUCstd |   oot_PRAUC |   oot_prec10 |   oot_rec10 |   oot_prec30 |   oot_rec30 |   oot_brier |   oot_ece |
|:------------------|----------:|-------------:|----------:|--------------:|------------:|-------------:|----------:|-------------:|------------:|-------------:|------------:|-------------:|------------:|------------:|----------:|
| base              |        91 |       2.3302 |    0.7477 |        0.0065 |      0.5381 |       2.0091 |    0.7643 |       0.0221 |      0.5391 |       0.6477 |      0.2317 |       0.5188 |      0.561  |      0.2052 |    0.1801 |
| ritmo_solo        |        29 |       2.412  |    0.7274 |        0.0086 |      0.5075 |       1.9178 |    0.7646 |       0.0359 |      0.5428 |       0.6705 |      0.2398 |       0.5226 |      0.565  |      0.2082 |    0.1909 |
| base+ritmo        |       120 |       2.2076 |    0.7481 |        0.0069 |      0.5388 |       2.0259 |    0.7661 |       0.02   |      0.5446 |       0.6136 |      0.2195 |       0.515  |      0.5569 |      0.2062 |    0.183  |
| base+flat24       |       139 |       2.3302 |    0.7478 |        0.0066 |      0.5382 |       2.0164 |    0.7673 |       0.0199 |      0.5374 |       0.6477 |      0.2317 |       0.5226 |      0.565  |      0.2063 |    0.1832 |
| base+ritmo+flat24 |       168 |       2.2894 |    0.748  |        0.0069 |      0.5389 |       2.0227 |    0.7672 |       0.0197 |      0.5454 |       0.6364 |      0.2276 |       0.5263 |      0.5691 |      0.207  |    0.185  |
| base+ritmo+lstm   |       153 |       2.2894 |  nan      |      nan      |    nan      |     nan      |    0.7628 |       0.0226 |      0.5313 |       0.6364 |      0.2276 |       0.5075 |      0.5488 |      0.218  |    0.2075 |

## ΔAUC pareado vs base (IC 95 % bootstrap)

| variante          |   ΔAUC_gkf |   gkf_lo |   gkf_hi |   ΔAUC_oot |   oot_lo |   oot_hi |
|:------------------|-----------:|---------:|---------:|-----------:|---------:|---------:|
| ritmo_solo        |    -0.0203 |  -0.0231 |  -0.0173 |     0.0002 |  -0.0176 |   0.0186 |
| base+ritmo        |     0.0004 |  -0.0001 |   0.001  |     0.0017 |  -0.0023 |   0.0054 |
| base+flat24       |     0.0002 |  -0.0006 |   0.0009 |     0.0029 |  -0.0011 |   0.0071 |
| base+ritmo+flat24 |     0.0003 |  -0.0005 |   0.0011 |     0.0029 |  -0.0024 |   0.0083 |
| base+ritmo+lstm   |   nan      | nan      | nan      |    -0.0015 |  -0.0101 |   0.008  |

## Redundancia con las features vigentes (|rho| de Spearman máxima) y AUC univariado

| feature                    |   max_|rho|_vigente | con                       |   AUC_univariado |
|:---------------------------|--------------------:|:--------------------------|-----------------:|
| r_densidad_u12_vs_vida     |               0.304 | meses_activos_u6          |            0.546 |
| r_camp_tasa_u6_vs_vida     |               0.307 | d_nped_m12                |            0.525 |
| r_gap_trend                |               0.522 | meses_activos_u6          |            0.635 |
| r_racha_actual_vs_max      |               0.633 | compras_hist              |            0.539 |
| r_gap_hab_mean             |               0.678 | monto_cv_u12              |            0.596 |
| r_gap_median               |               0.679 | monto_cv_u12              |            0.647 |
| r_gap_last_vs_hab          |               0.691 | meses_desde_compra_previa |            0.579 |
| r_camp_saltadas_delta      |               0.692 | camp_saltadas             |            0.625 |
| r_frac_pausas              |               0.717 | monto_cv_u12              |            0.632 |
| r_camp_saltadas_media      |               0.73  | antiguedad_meses          |            0.506 |
| r_esp_gap                  |               0.748 | tasa_camp_u12             |            0.664 |
| r_camp_racha               |               0.752 | meses_activos_u3          |            0.625 |
| r_gap_cv                   |               0.753 | antiguedad_meses          |            0.541 |
| r_gap_last_z               |               0.765 | recencia_norm             |            0.565 |
| r_p_gap_le3                |               0.769 | meses_activos_u12         |            0.691 |
| r_gap_std                  |               0.777 | monto_cv_u12              |            0.615 |
| r_n_pausas_largas          |               0.802 | antiguedad_meses          |            0.525 |
| r_camp_saltadas_max        |               0.805 | antiguedad_meses          |            0.535 |
| r_es_reactivacion          |               0.807 | meses_desde_compra_previa |            0.607 |
| r_p_gap_le6                |               0.815 | meses_activos_u12         |            0.684 |
| r_racha_max                |               0.815 | n_ped_acum                |            0.677 |
| r_n_rachas                 |               0.834 | antiguedad_meses          |            0.542 |
| r_n_pausas                 |               0.834 | antiguedad_meses          |            0.542 |
| r_gap_max                  |               0.835 | antiguedad_meses          |            0.559 |
| r_meses_desde_reactivacion |               0.855 | meses_activos_u6          |            0.688 |
| r_racha_actual             |               0.868 | meses_desde_compra_previa |            0.635 |
| r_gap_mean                 |               0.876 | monto_cv_u12              |            0.627 |
| r_densidad_vida            |               0.879 | monto_cv_u12              |            0.613 |
| r_camp_tasa_vida           |               0.919 | tasa_camp_u12             |            0.63  |

## Importancia por permutación (OOT, modelo base+ritmo): top 15

| feature          |   ΔAUC_perm |    std | familia       |
|:-----------------|------------:|-------:|:--------------|
| n_ped_u12        |      0.0041 | 0.0025 | vigente       |
| n_prod_u12       |      0.0036 | 0.0023 | vigente       |
| monto_u12        |      0.003  | 0.0035 | vigente       |
| tasa_camp_u3     |      0.0027 | 0.0016 | vigente       |
| compras_hist     |      0.0025 | 0.001  | vigente       |
| n_ped_u6         |      0.0022 | 0.0022 | vigente       |
| camp_saltadas    |      0.0021 | 0.0022 | vigente       |
| r_p_gap_le6      |      0.0019 | 0.0004 | ritmo (nueva) |
| intensidad_u3    |      0.0016 | 0.002  | vigente       |
| n_ped_u3         |      0.0015 | 0.001  | vigente       |
| r_camp_tasa_vida |      0.0014 | 0.001  | ritmo (nueva) |
| r_gap_mean       |      0.0013 | 0.0005 | ritmo (nueva) |
| ticket_prom_u3   |      0.0012 | 0.0003 | vigente       |
| n_ped_acum       |      0.0011 | 0.0007 | vigente       |
| monto_std_u12    |      0.001  | 0.0006 | vigente       |

Gain acumulado de `r_*`: 15.9%.

## Tiempo hasta la próxima compra con el ritmo individual

| score                                           |   AUC_gkf(all) |   oot_AUC |   oot_prec10 |   oot_rec30 |
|:------------------------------------------------|---------------:|----------:|-------------:|------------:|
| r_p_gap_le6 (share hist. de gaps ≤ 6, encogido) |         0.6835 |    0.7365 |       0.6364 |      0.5041 |
| r_esp_gap (gap medio encogido)                  |         0.6637 |    0.731  |       0.5909 |      0.5325 |
| meses_activos_u12 (vigente, referencia)         |         0.6966 |    0.7537 |       0.5795 |      0.5407 |
| XGBoost Cox (09_supervivencia, referencia)      |         0.7511 |    0.7616 |     nan      |    nan      |

## Lectura

- `base+ritmo`: GKF 0.7481 vs base 0.7477;
  OOT 0.7661 vs 0.7643.
- `ritmo_solo` (29 columnas) llega a 0.7274 GKF: el ritmo resume
  la mayor parte de la señal que ya tienen las 91 features (ver redundancia).
- Las variantes con salidas de la Opción A (`flat24`, `lstm`) muestran si el ritmo a mano agrega algo
  sobre el detalle mensual crudo o aprendido: ver deltas.
