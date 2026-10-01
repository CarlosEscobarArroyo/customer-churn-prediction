# Opción D — Modelos por segmento de vendedoras

> Generado por `08_experimentation/04_opcion_D_segmentos.ipynb` el 2026-09-15 11:43.
> 9 segmentaciones × 4 enfoques. Global de referencia: GKF 0.7477 / OOT 0.7643.
> Segmento con < 800 filas de train usa el global en `especializado` y `fine-tune`.

## Tamaño y prevalencia por segmento

| segmentacion        | segmento          |   n_rows |   n_train |   n_oot |   prev |   n_vend |
|:--------------------|:------------------|---------:|----------:|--------:|-------:|---------:|
| tipo_vendedor       | Asesora           |    27452 |     25573 |     788 |  0.32  |     5949 |
| tipo_vendedor       | DESCONOCIDO       |       30 |        30 |       0 |  0.467 |       11 |
| tipo_vendedor       | Líder             |     3339 |      3132 |      97 |  0.218 |      396 |
| red                 | Asesora con líder |    13192 |     12600 |     239 |  0.363 |     3431 |
| red                 | Asesora sin líder |    14290 |     13003 |     549 |  0.28  |     2529 |
| red                 | Líder             |     3339 |      3132 |      97 |  0.218 |      396 |
| region              | Lima-Callao       |     6099 |      5536 |     244 |  0.276 |     1196 |
| region              | Provincias        |    24722 |     23199 |     641 |  0.317 |     5160 |
| antiguedad_relacion | 1-2 compras       |    10628 |     10306 |     110 |  0.422 |     6356 |
| antiguedad_relacion | 10+               |     8107 |      7076 |     461 |  0.159 |      788 |
| antiguedad_relacion | 3-9               |    12086 |     11353 |     314 |  0.311 |     3096 |
| frecuencia_u12      | 1-3 meses         |    15667 |     14729 |     424 |  0.432 |     6356 |
| frecuencia_u12      | 4-8               |    12069 |     11205 |     342 |  0.215 |     2456 |
| frecuencia_u12      | 9-12              |     3085 |      2801 |     119 |  0.051 |      374 |
| nivel_compra_u12    | alto              |    10340 |      9578 |     295 |  0.124 |     1390 |
| nivel_compra_u12    | bajo              |    10260 |      9578 |     310 |  0.497 |     5177 |
| nivel_compra_u12    | medio             |    10221 |      9579 |     280 |  0.308 |     3414 |
| reactivacion        | continua (gap 1)  |    15223 |     14391 |     382 |  0.216 |     4406 |
| reactivacion        | pausa corta (2-3) |     7772 |      7275 |     184 |  0.311 |     3417 |
| reactivacion        | reactivada (4+)   |     7826 |      7069 |     319 |  0.489 |     3793 |
| regularidad_racha   | racha 1           |    15598 |     14344 |     503 |  0.4   |     5100 |
| regularidad_racha   | racha 2-3         |    10777 |     10213 |     265 |  0.267 |     4406 |
| regularidad_racha   | racha 4+          |     4446 |      4178 |     117 |  0.092 |     1061 |
| kmeans4             | C0                |     4406 |      3974 |     198 |  0.564 |     2817 |
| kmeans4             | C1                |     9649 |      9459 |      45 |  0.342 |     5088 |
| kmeans4             | C2                |     8511 |      7753 |     361 |  0.346 |     2548 |
| kmeans4             | C3                |     8255 |      7549 |     281 |  0.096 |     1016 |

## Estabilidad

| segmentacion        |   n_segmentos |   permanece_≤3m |   cambio_mix_anual_medio |   min_share_oot |
|:--------------------|--------------:|----------------:|-------------------------:|----------------:|
| tipo_vendedor       |             3 |           1     |                    0.028 |           0.11  |
| red                 |             3 |           1     |                    0.124 |           0.11  |
| region              |             2 |           1     |                    0.039 |           0.276 |
| antiguedad_relacion |             3 |           0.849 |                    0.189 |           0.124 |
| frecuencia_u12      |             3 |           0.821 |                    0.179 |           0.134 |
| nivel_compra_u12    |             3 |           0.805 |                    0.152 |           0.316 |
| reactivacion        |             3 |           0.511 |                    0.244 |           0.208 |
| regularidad_racha   |             3 |           0.503 |                    0.254 |           0.132 |
| kmeans4             |             4 |           0.798 |                    0.279 |           0.051 |

`permanece_≤3m`: fracción de pares de observaciones de la misma vendedora (≤ 3 meses de distancia) que quedan en el mismo segmento.
`cambio_mix_anual_medio`: suma de |Δ share| entre años consecutivos (0 = mix constante).

## AUC agrupado por segmentación y enfoque

| segmentacion        | enfoque       |   gkf_AUC |   oot_AUC |   oot_AUCstd |   oot_PRAUC |   oot_prec10 |   oot_rec30 |   oot_brier |
|:--------------------|:--------------|----------:|----------:|-------------:|------------:|-------------:|------------:|------------:|
| tipo_vendedor       | global        |    0.7477 |    0.7643 |       0.0221 |      0.5391 |       0.6477 |      0.561  |      0.2052 |
| tipo_vendedor       | global+seg    |    0.7475 |    0.765  |       0.0219 |      0.541  |       0.6364 |      0.565  |      0.2055 |
| tipo_vendedor       | especializado |    0.7438 |    0.7635 |       0.0158 |      0.5224 |       0.6477 |      0.561  |      0.2049 |
| tipo_vendedor       | fine-tune     |    0.7461 |    0.7645 |       0.0177 |      0.5341 |       0.6477 |      0.561  |      0.2049 |
| red                 | global        |    0.7477 |    0.7643 |       0.0221 |      0.5391 |       0.6477 |      0.561  |      0.2052 |
| red                 | global+seg    |    0.75   |    0.7648 |       0.02   |      0.5439 |       0.625  |      0.5407 |      0.2036 |
| red                 | especializado |    0.7419 |    0.7621 |       0.0204 |      0.5386 |       0.6932 |      0.5407 |      0.2066 |
| red                 | fine-tune     |    0.7452 |    0.7636 |       0.0207 |      0.5401 |       0.6364 |      0.5447 |      0.207  |
| region              | global        |    0.7477 |    0.7643 |       0.0221 |      0.5391 |       0.6477 |      0.561  |      0.2052 |
| region              | global+seg    |    0.7478 |    0.764  |       0.0215 |      0.5375 |       0.6591 |      0.561  |      0.2058 |
| region              | especializado |    0.7455 |    0.7555 |       0.0186 |      0.5166 |       0.625  |      0.5488 |      0.2093 |
| region              | fine-tune     |    0.7471 |    0.7627 |       0.0238 |      0.5356 |       0.6136 |      0.5488 |      0.2063 |
| antiguedad_relacion | global        |    0.7477 |    0.7643 |       0.0221 |      0.5391 |       0.6477 |      0.561  |      0.2052 |
| antiguedad_relacion | global+seg    |    0.7475 |    0.7647 |       0.022  |      0.5406 |       0.6364 |      0.5569 |      0.2055 |
| antiguedad_relacion | especializado |    0.7171 |    0.7491 |       0.016  |      0.5246 |       0.625  |      0.5407 |      0.2313 |
| antiguedad_relacion | fine-tune     |    0.7302 |    0.7562 |       0.0188 |      0.5334 |       0.6364 |      0.5366 |      0.2255 |
| frecuencia_u12      | global        |    0.7477 |    0.7643 |       0.0221 |      0.5391 |       0.6477 |      0.561  |      0.2052 |
| frecuencia_u12      | global+seg    |    0.7475 |    0.7648 |       0.0217 |      0.54   |       0.6364 |      0.561  |      0.2054 |
| frecuencia_u12      | especializado |    0.6977 |    0.7185 |       0.0206 |      0.4718 |       0.625  |      0.5041 |      0.2154 |
| frecuencia_u12      | fine-tune     |    0.718  |    0.7296 |       0.0256 |      0.5055 |       0.6705 |      0.5081 |      0.2134 |
| nivel_compra_u12    | global        |    0.7477 |    0.7643 |       0.0221 |      0.5391 |       0.6477 |      0.561  |      0.2052 |
| nivel_compra_u12    | global+seg    |    0.7474 |    0.7649 |       0.0212 |      0.5404 |       0.6364 |      0.5569 |      0.2055 |
| nivel_compra_u12    | especializado |    0.6627 |    0.7177 |       0.022  |      0.4553 |       0.5568 |      0.4959 |      0.2133 |
| nivel_compra_u12    | fine-tune     |    0.7007 |    0.7313 |       0.026  |      0.4868 |       0.6023 |      0.5081 |      0.2092 |
| reactivacion        | global        |    0.7477 |    0.7643 |       0.0221 |      0.5391 |       0.6477 |      0.561  |      0.2052 |
| reactivacion        | global+seg    |    0.7474 |    0.7648 |       0.0221 |      0.5404 |       0.6477 |      0.5569 |      0.2054 |
| reactivacion        | especializado |    0.7041 |    0.7118 |       0.0241 |      0.4353 |       0.5227 |      0.4919 |      0.2119 |
| reactivacion        | fine-tune     |    0.7253 |    0.7385 |       0.0146 |      0.4694 |       0.5682 |      0.5447 |      0.2057 |
| regularidad_racha   | global        |    0.7477 |    0.7643 |       0.0221 |      0.5391 |       0.6477 |      0.561  |      0.2052 |
| regularidad_racha   | global+seg    |    0.7475 |    0.7648 |       0.0218 |      0.5392 |       0.6364 |      0.5569 |      0.2054 |
| regularidad_racha   | especializado |    0.7165 |    0.7357 |       0.0223 |      0.4965 |       0.6023 |      0.5203 |      0.2053 |
| regularidad_racha   | fine-tune     |    0.731  |    0.7493 |       0.022  |      0.5147 |       0.5795 |      0.5528 |      0.2055 |
| kmeans4             | global        |    0.7477 |    0.7643 |       0.0221 |      0.5391 |       0.6477 |      0.561  |      0.2052 |
| kmeans4             | global+seg    |    0.7474 |    0.7629 |       0.0224 |      0.5343 |       0.6364 |      0.5569 |      0.2064 |
| kmeans4             | especializado |    0.6664 |    0.6898 |       0.0137 |      0.4172 |       0.4659 |      0.4919 |      0.2179 |
| kmeans4             | fine-tune     |    0.7037 |    0.7098 |       0.0315 |      0.4624 |       0.5341 |      0.5163 |      0.2147 |

## ΔAUC pareado vs global (IC 95 % bootstrap)

| segmentacion        | enfoque       |   ΔAUC_gkf |   gkf_lo |   gkf_hi |   ΔAUC_oot |   oot_lo |   oot_hi |
|:--------------------|:--------------|-----------:|---------:|---------:|-----------:|---------:|---------:|
| tipo_vendedor       | global+seg    |    -0.0002 |  -0.0004 |   0      |     0.0006 |  -0.0007 |   0.002  |
| tipo_vendedor       | especializado |    -0.0039 |  -0.005  |  -0.0026 |    -0.0008 |  -0.0064 |   0.0044 |
| tipo_vendedor       | fine-tune     |    -0.0015 |  -0.0021 |  -0.0009 |     0.0002 |  -0.0032 |   0.0035 |
| red                 | global+seg    |     0.0024 |   0.0014 |   0.0033 |     0.0005 |  -0.0042 |   0.0051 |
| red                 | especializado |    -0.0058 |  -0.0074 |  -0.0041 |    -0.0023 |  -0.0102 |   0.0059 |
| red                 | fine-tune     |    -0.0025 |  -0.0033 |  -0.0016 |    -0.0007 |  -0.0039 |   0.0029 |
| region              | global+seg    |     0.0001 |  -0.0002 |   0.0003 |    -0.0004 |  -0.0022 |   0.0013 |
| region              | especializado |    -0.0022 |  -0.0032 |  -0.0011 |    -0.0088 |  -0.0163 |  -0.0009 |
| region              | fine-tune     |    -0.0005 |  -0.0009 |  -0.0001 |    -0.0017 |  -0.0042 |   0.0009 |
| antiguedad_relacion | global+seg    |    -0.0002 |  -0.0004 |   0      |     0.0004 |  -0.001  |   0.0018 |
| antiguedad_relacion | especializado |    -0.0306 |  -0.0342 |  -0.0272 |    -0.0152 |  -0.0317 |   0.0024 |
| antiguedad_relacion | fine-tune     |    -0.0175 |  -0.0202 |  -0.0151 |    -0.0082 |  -0.0205 |   0.0041 |
| frecuencia_u12      | global+seg    |    -0.0002 |  -0.0004 |   0      |     0.0005 |  -0.001  |   0.0019 |
| frecuencia_u12      | especializado |    -0.0499 |  -0.0544 |  -0.0457 |    -0.0459 |  -0.0641 |  -0.0288 |
| frecuencia_u12      | fine-tune     |    -0.0296 |  -0.0329 |  -0.0265 |    -0.0348 |  -0.0488 |  -0.0214 |
| nivel_compra_u12    | global+seg    |    -0.0002 |  -0.0004 |  -0      |     0.0006 |  -0.0008 |   0.0019 |
| nivel_compra_u12    | especializado |    -0.0849 |  -0.0908 |  -0.0794 |    -0.0466 |  -0.0697 |  -0.0266 |
| nivel_compra_u12    | fine-tune     |    -0.047  |  -0.0513 |  -0.0429 |    -0.033  |  -0.0502 |  -0.0167 |
| reactivacion        | global+seg    |    -0.0002 |  -0.0004 |   0      |     0.0005 |  -0.0009 |   0.0019 |
| reactivacion        | especializado |    -0.0435 |  -0.0479 |  -0.0394 |    -0.0526 |  -0.0731 |  -0.0302 |
| reactivacion        | fine-tune     |    -0.0224 |  -0.0255 |  -0.0195 |    -0.0259 |  -0.0397 |  -0.0118 |
| regularidad_racha   | global+seg    |    -0.0002 |  -0.0004 |   0      |     0.0005 |  -0.0008 |   0.0018 |
| regularidad_racha   | especializado |    -0.0311 |  -0.0343 |  -0.0278 |    -0.0286 |  -0.0407 |  -0.0151 |
| regularidad_racha   | fine-tune     |    -0.0167 |  -0.019  |  -0.0144 |    -0.0151 |  -0.0243 |  -0.0057 |
| kmeans4             | global+seg    |    -0.0003 |  -0.0006 |   0      |    -0.0015 |  -0.0034 |   0.0004 |
| kmeans4             | especializado |    -0.0813 |  -0.0865 |  -0.076  |    -0.0746 |  -0.0998 |  -0.0481 |
| kmeans4             | fine-tune     |    -0.0439 |  -0.0477 |  -0.04   |    -0.0545 |  -0.0738 |  -0.0345 |

Enfoques con IC GKF > 0 **y** ΔOOT > 0: 1 de 27: red/global+seg.

## AUC por segmento (global vs enfoques)

|                                        |   AUC_gkf_especializado |   AUC_gkf_fine-tune |   AUC_gkf_global |   AUC_gkf_global+seg |   AUC_oot_especializado |   AUC_oot_fine-tune |   AUC_oot_global |   AUC_oot_global+seg |   n_rows |   n_oot |   prev | mejor_gkf   |   Δ_mejor_vs_global_gkf |
|:---------------------------------------|------------------------:|--------------------:|-----------------:|---------------------:|------------------------:|--------------------:|-----------------:|---------------------:|---------:|--------:|-------:|:------------|------------------------:|
| ('antiguedad_relacion', '1-2 compras') |                   0.671 |               0.676 |            0.676 |                0.676 |                   0.716 |               0.703 |            0.69  |                0.691 |    10628 |     110 |  0.422 | global      |                   0     |
| ('antiguedad_relacion', '10+')         |                   0.778 |               0.782 |            0.784 |                0.784 |                   0.737 |               0.746 |            0.747 |                0.746 |     8107 |     461 |  0.159 | global      |                   0     |
| ('antiguedad_relacion', '3-9')         |                   0.715 |               0.716 |            0.717 |                0.716 |                   0.731 |               0.728 |            0.727 |                0.73  |    12086 |     314 |  0.311 | global      |                   0     |
| ('frecuencia_u12', '1-3 meses')        |                   0.664 |               0.667 |            0.667 |                0.667 |                   0.667 |               0.664 |            0.665 |                0.665 |    15667 |     424 |  0.432 | fine-tune   |                   0     |
| ('frecuencia_u12', '4-8')              |                   0.689 |               0.694 |            0.694 |                0.694 |                   0.628 |               0.637 |            0.643 |                0.644 |    12069 |     342 |  0.215 | global      |                   0     |
| ('frecuencia_u12', '9-12')             |                   0.683 |               0.715 |            0.741 |                0.739 |                   0.411 |               0.596 |            0.721 |                0.747 |     3085 |     119 |  0.051 | global      |                   0     |
| ('kmeans4', 'C0')                      |                   0.597 |               0.608 |            0.613 |                0.613 |                   0.624 |               0.632 |            0.63  |                0.626 |     4406 |     198 |  0.564 | global      |                   0     |
| ('kmeans4', 'C1')                      |                   0.664 |               0.671 |            0.671 |                0.67  |                   0.634 |               0.675 |            0.657 |                0.652 |     9649 |      45 |  0.342 | fine-tune   |                   0     |
| ('kmeans4', 'C2')                      |                   0.63  |               0.64  |            0.641 |                0.64  |                   0.615 |               0.61  |            0.611 |                0.609 |     8511 |     361 |  0.346 | global      |                   0     |
| ('kmeans4', 'C3')                      |                   0.708 |               0.719 |            0.72  |                0.719 |                   0.645 |               0.642 |            0.661 |                0.654 |     8255 |     281 |  0.096 | global      |                   0     |
| ('nivel_compra_u12', 'alto')           |                   0.728 |               0.732 |            0.733 |                0.732 |                   0.681 |               0.695 |            0.697 |                0.704 |    10340 |     295 |  0.124 | global      |                   0     |
| ('nivel_compra_u12', 'bajo')           |                   0.616 |               0.622 |            0.622 |                0.622 |                   0.664 |               0.66  |            0.654 |                0.656 |    10260 |     310 |  0.497 | global      |                   0     |
| ('nivel_compra_u12', 'medio')          |                   0.64  |               0.643 |            0.643 |                0.643 |                   0.594 |               0.591 |            0.596 |                0.595 |    10221 |     280 |  0.308 | fine-tune   |                   0     |
| ('reactivacion', 'continua (gap 1)')   |                   0.751 |               0.753 |            0.753 |                0.753 |                   0.746 |               0.762 |            0.762 |                0.767 |    15223 |     382 |  0.216 | global      |                   0     |
| ('reactivacion', 'pausa corta (2-3)')  |                   0.683 |               0.691 |            0.692 |                0.691 |                   0.628 |               0.651 |            0.655 |                0.654 |     7772 |     184 |  0.311 | global      |                   0     |
| ('reactivacion', 'reactivada (4+)')    |                   0.657 |               0.661 |            0.659 |                0.659 |                   0.676 |               0.678 |            0.679 |                0.68  |     7826 |     319 |  0.489 | fine-tune   |                   0.001 |
| ('red', 'Asesora con líder')           |                   0.725 |               0.727 |            0.726 |                0.726 |                   0.723 |               0.729 |            0.729 |                0.729 |    13192 |     239 |  0.363 | fine-tune   |                   0     |
| ('red', 'Asesora sin líder')           |                   0.753 |               0.754 |            0.754 |                0.754 |                   0.787 |               0.786 |            0.787 |                0.787 |    14290 |     549 |  0.28  | fine-tune   |                   0     |
| ('red', 'Líder')                       |                   0.758 |               0.768 |            0.77  |                0.77  |                   0.701 |               0.714 |            0.71  |                0.715 |     3339 |      97 |  0.218 | global      |                   0     |
| ('region', 'Lima-Callao')              |                   0.763 |               0.769 |            0.769 |                0.769 |                   0.725 |               0.75  |            0.753 |                0.753 |     6099 |     244 |  0.276 | global+seg  |                   0     |
| ('region', 'Provincias')               |                   0.741 |               0.742 |            0.742 |                0.742 |                   0.765 |               0.766 |            0.766 |                0.765 |    24722 |     641 |  0.317 | fine-tune   |                   0     |
| ('regularidad_racha', 'racha 1')       |                   0.701 |               0.701 |            0.701 |                0.7   |                   0.692 |               0.695 |            0.699 |                0.698 |    15598 |     503 |  0.4   | fine-tune   |                   0     |
| ('regularidad_racha', 'racha 2-3')     |                   0.702 |               0.706 |            0.707 |                0.707 |                   0.72  |               0.743 |            0.742 |                0.747 |    10777 |     265 |  0.267 | global      |                   0     |
| ('regularidad_racha', 'racha 4+')      |                   0.747 |               0.767 |            0.772 |                0.771 |                   0.68  |               0.758 |            0.812 |                0.821 |     4446 |     117 |  0.092 | global      |                   0     |
| ('tipo_vendedor', 'Asesora')           |                   0.742 |               0.742 |            0.742 |                0.742 |                   0.769 |               0.769 |            0.769 |                0.77  |    27452 |     788 |  0.32  | global      |                   0     |
| ('tipo_vendedor', 'DESCONOCIDO')       |                   0.522 |               0.522 |            0.522 |                0.509 |                 nan     |             nan     |          nan     |              nan     |       30 |       0 |  0.467 | global      |                   0     |
| ('tipo_vendedor', 'Líder')             |                   0.758 |               0.768 |            0.77  |                0.77  |                   0.701 |               0.719 |            0.71  |                0.709 |     3339 |      97 |  0.218 | global      |                   0     |

## Lectura

- Top 5 por AUC GKF: red/global+seg 0.7500, region/global+seg 0.7478, tipo_vendedor/global 0.7477, region/global 0.7477, kmeans4/global 0.7477.
- Los segmentos con menor AUC del global señalan dónde falta información, no necesariamente dónde
  un modelo aparte ayuda: comparar `Δ_mejor_vs_global_gkf` con el IC de la tabla de deltas.
- El segmento debe ser conocido en t: todas las segmentaciones usan solo columnas ≤ t y los cortes /
  KMeans se fijan con train.
