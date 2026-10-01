# Features nuevas (PAGO / CAMPAÑAS / RED / MIX) — ablación por grupo

> Generado por `05_modelling/06_features_nuevas.py` el 2026-09-06 15:23.
> Base = 84 features del schema anterior (sin selección por permutación); modelos con hiperparámetros de `05_modelling/*_best_params.json`.

## AUC GroupKFold(5) por vendedora

| config        |   LogReg |   RandomForest |   XGBoost |
|:--------------|---------:|---------------:|----------:|
| base          |   0.7414 |         0.742  |    0.7444 |
| base+pago     |   0.7415 |         0.7419 |    0.7444 |
| base+campanas |   0.7429 |         0.7443 |    0.7476 |
| base+red      |   0.7458 |         0.7453 |    0.7498 |
| base+mix      |   0.7417 |         0.742  |    0.7449 |
| todas         |   0.7474 |         0.7468 |    0.7521 |

### Δ vs base

| config        |   LogReg |   RandomForest |   XGBoost |
|:--------------|---------:|---------------:|----------:|
| base          |   0      |         0      |    0      |
| base+pago     |   0.0001 |        -0.0001 |    0      |
| base+campanas |   0.0015 |         0.0023 |    0.0032 |
| base+red      |   0.0044 |         0.0033 |    0.0055 |
| base+mix      |   0.0003 |        -0      |    0.0005 |
| todas         |   0.006  |         0.0048 |    0.0077 |

## AUC out-of-period (± std por mes)

| config        | LogReg         | RandomForest   | XGBoost        |
|:--------------|:---------------|:---------------|:---------------|
| base          | 0.7612 ± 0.01  | 0.7659 ± 0.022 | 0.7625 ± 0.019 |
| base+pago     | 0.7597 ± 0.008 | 0.7655 ± 0.021 | 0.7641 ± 0.019 |
| base+campanas | 0.7619 ± 0.009 | 0.7663 ± 0.022 | 0.765 ± 0.023  |
| base+red      | 0.7603 ± 0.01  | 0.7659 ± 0.021 | 0.7624 ± 0.016 |
| base+mix      | 0.7603 ± 0.009 | 0.7658 ± 0.024 | 0.7633 ± 0.02  |
| todas         | 0.7601 ± 0.011 | 0.7668 ± 0.023 | 0.7637 ± 0.02  |

### Δ vs base

| config        |   LogReg |   RandomForest |   XGBoost |
|:--------------|---------:|---------------:|----------:|
| base          |   0      |         0      |    0      |
| base+pago     |  -0.0015 |        -0.0004 |    0.0016 |
| base+campanas |   0.0007 |         0.0004 |    0.0026 |
| base+red      |  -0.0009 |         0      |   -0      |
| base+mix      |  -0.0009 |        -0      |    0.0008 |
| todas         |  -0.0011 |         0.0009 |    0.0013 |

## Lift PR-AUC / prevalencia (GroupKFold)

| config        |   LogReg |   RandomForest |   XGBoost |
|:--------------|---------:|---------------:|----------:|
| base          |    1.711 |          1.718 |     1.724 |
| base+pago     |    1.711 |          1.716 |     1.724 |
| base+campanas |    1.712 |          1.729 |     1.741 |
| base+red      |    1.739 |          1.738 |     1.76  |
| base+mix      |    1.713 |          1.719 |     1.729 |
| todas         |    1.743 |          1.743 |     1.773 |

## Tabla completa

|                                   |   n_feat |   gkf_AUC |   gkf_PRAUC |   gkf_liftPR |   gkf_lift10 |   oot_AUC |   oot_AUCstd |   oot_PRAUC |   oot_liftPR |   oot_F1 |   oot_prec |   oot_rec |   oot_lift10 |
|:----------------------------------|---------:|----------:|------------:|-------------:|-------------:|----------:|-------------:|------------:|-------------:|---------:|-----------:|----------:|-------------:|
| ('base', 'LogReg')                |       84 |    0.7414 |      0.5288 |       1.7108 |       1.9766 |    0.7612 |       0.0098 |      0.53   |       1.9067 |   0.5615 |     0.4747 |    0.687  |       2.2485 |
| ('base', 'RandomForest')          |       84 |    0.742  |      0.531  |       1.7179 |       2.0017 |    0.7659 |       0.022  |      0.544  |       1.957  |   0.559  |     0.4638 |    0.7033 |       2.412  |
| ('base', 'XGBoost')               |       84 |    0.7444 |      0.5329 |       1.724  |       2.0007 |    0.7625 |       0.0187 |      0.5337 |       1.92   |   0.5559 |     0.4497 |    0.7276 |       2.412  |
| ('base+pago', 'LogReg')           |       88 |    0.7415 |      0.5289 |       1.7109 |       1.9734 |    0.7597 |       0.0082 |      0.5287 |       1.902  |   0.5611 |     0.4722 |    0.6911 |       2.2485 |
| ('base+pago', 'RandomForest')     |       88 |    0.7419 |      0.5305 |       1.7161 |       2.0038 |    0.7655 |       0.0206 |      0.5405 |       1.9446 |   0.5627 |     0.4654 |    0.7114 |       2.3302 |
| ('base+pago', 'XGBoost')          |       88 |    0.7444 |      0.5329 |       1.7239 |       1.9954 |    0.7641 |       0.019  |      0.5365 |       1.9301 |   0.5505 |     0.4458 |    0.7195 |       2.3711 |
| ('base+campanas', 'LogReg')       |       91 |    0.7429 |      0.5293 |       1.7125 |       1.9776 |    0.7619 |       0.0092 |      0.5277 |       1.8986 |   0.5628 |     0.4621 |    0.7195 |       2.2485 |
| ('base+campanas', 'RandomForest') |       91 |    0.7443 |      0.5345 |       1.7292 |       2.028  |    0.7663 |       0.022  |      0.537  |       1.9317 |   0.5545 |     0.4495 |    0.7236 |       2.3711 |
| ('base+campanas', 'XGBoost')      |       91 |    0.7476 |      0.5383 |       1.7413 |       2.0133 |    0.765  |       0.0226 |      0.5419 |       1.9495 |   0.5658 |     0.4506 |    0.7602 |       2.3302 |
| ('base+red', 'LogReg')            |       93 |    0.7458 |      0.5376 |       1.7392 |       1.9954 |    0.7603 |       0.0098 |      0.5292 |       1.904  |   0.5541 |     0.4643 |    0.687  |       2.3302 |
| ('base+red', 'RandomForest')      |       93 |    0.7453 |      0.5372 |       1.7378 |       2.0416 |    0.7659 |       0.0212 |      0.5406 |       1.9449 |   0.5572 |     0.4613 |    0.7033 |       2.3302 |
| ('base+red', 'XGBoost')           |       93 |    0.7498 |      0.544  |       1.7599 |       2.0374 |    0.7624 |       0.0157 |      0.5311 |       1.9106 |   0.5607 |     0.4545 |    0.7317 |       2.1667 |
| ('base+mix', 'LogReg')            |       90 |    0.7417 |      0.5295 |       1.713  |       1.9902 |    0.7603 |       0.0091 |      0.5338 |       1.9205 |   0.5559 |     0.4669 |    0.687  |       2.2076 |
| ('base+mix', 'RandomForest')      |       90 |    0.742  |      0.5314 |       1.7192 |       2.0059 |    0.7658 |       0.0235 |      0.5486 |       1.9735 |   0.5654 |     0.4727 |    0.7033 |       2.4529 |
| ('base+mix', 'XGBoost')           |       90 |    0.7449 |      0.5343 |       1.7286 |       1.9975 |    0.7633 |       0.0202 |      0.5456 |       1.963  |   0.5615 |     0.4588 |    0.7236 |       2.412  |
| ('todas', 'LogReg')               |      110 |    0.7474 |      0.5389 |       1.7434 |       1.9944 |    0.7601 |       0.0106 |      0.5316 |       1.9123 |   0.558  |     0.4541 |    0.7236 |       2.2485 |
| ('todas', 'RandomForest')         |      110 |    0.7468 |      0.5388 |       1.7431 |       2.0448 |    0.7668 |       0.0231 |      0.5519 |       1.9854 |   0.5554 |     0.4506 |    0.7236 |       2.5346 |
| ('todas', 'XGBoost')              |      110 |    0.7521 |      0.548  |       1.7728 |       2.0584 |    0.7637 |       0.0205 |      0.5334 |       1.919  |   0.5594 |     0.4439 |    0.7561 |       2.2076 |

## Importancia por permutación de las features nuevas (XGBoost todas, AUC en test OOT)

Rank sobre 110 features. dAUC = caída de AUC al permutar la variable (10 repeticiones); negativo o ~0 = no aporta fuera de muestra.

|   rank | feature            | grupo    |    dAUC |    std |
|-------:|:-------------------|:---------|--------:|-------:|
|      4 | camp_saltadas      | campanas |  0.0037 | 0.0029 |
|      9 | camp_part_u12      | campanas |  0.0022 | 0.0015 |
|     10 | tasa_camp_u12      | campanas |  0.0015 | 0.001  |
|     11 | unidades_u12       | mix      |  0.0015 | 0.0015 |
|     12 | tasa_camp_u3       | campanas |  0.0015 | 0.0011 |
|     16 | unidades_u3        | mix      |  0.0008 | 0.0008 |
|     17 | pct_directo_u12    | campanas |  0.0007 | 0.0002 |
|     21 | red_size           | red      |  0.0005 | 0.0015 |
|     23 | equipo_size        | red      |  0.0003 | 0.0003 |
|     24 | lider_recencia     | red      |  0.0003 | 0.0003 |
|     25 | lider_act_u12      | red      |  0.0003 | 0.0001 |
|     28 | pct_accesorios_u12 | mix      |  0.0002 | 0.0001 |
|     30 | equipo_tasa_act_u3 | red      |  0.0002 | 0.0004 |
|     32 | tiene_lider        | red      |  0.0001 | 0.0012 |
|     33 | pct_no_ropa_u12    | mix      |  0.0001 | 0.0001 |
|     44 | lider_act_u3       | red      |  0      | 0      |
|     48 | pct_belleza_u12    | mix      |  0      | 0.0001 |
|     82 | tasa_camp_u6       | campanas | -0      | 0.0008 |
|     84 | es_nueva_u12       | campanas | -0.0001 | 0      |
|     89 | ratio_pago_u6      | pago     | -0.0001 | 0.0001 |
|     90 | red_tasa_act_u12   | red      | -0.0001 | 0.0002 |
|     92 | ratio_pago_u12     | pago     | -0.0001 | 0.0002 |
|     99 | ratio_pago_acum    | pago     | -0.0002 | 0.0001 |
|    101 | precio_unit_u12    | mix      | -0.0004 | 0.0005 |
|    103 | ratio_pago_u3      | pago     | -0.0009 | 0.0011 |
|    105 | red_tasa_act_u3    | red      | -0.001  | 0.0003 |

### Top 15 global (para contexto)

|   rank | feature        | grupo    |   dAUC |    std |
|-------:|:---------------|:---------|-------:|-------:|
|      1 | compras_hist   | base     | 0.0072 | 0.0017 |
|      2 | n_ped_u12      | base     | 0.0058 | 0.0031 |
|      3 | n_ped_acum     | base     | 0.0039 | 0.002  |
|      4 | camp_saltadas  | campanas | 0.0037 | 0.0029 |
|      5 | n_ped_u6       | base     | 0.0029 | 0.0028 |
|      6 | monto_u12      | base     | 0.0027 | 0.0031 |
|      7 | n_ped_u3       | base     | 0.0026 | 0.0019 |
|      8 | n_prod_u12     | base     | 0.0023 | 0.0017 |
|      9 | camp_part_u12  | campanas | 0.0022 | 0.0015 |
|     10 | tasa_camp_u12  | campanas | 0.0015 | 0.001  |
|     11 | unidades_u12   | mix      | 0.0015 | 0.0015 |
|     12 | tasa_camp_u3   | campanas | 0.0015 | 0.0011 |
|     13 | ticket_prom_u3 | base     | 0.001  | 0.0002 |
|     14 | monto_std_u12  | base     | 0.0009 | 0.0007 |
|     15 | intensidad_u3  | base     | 0.0008 | 0.0014 |
