# Selección de variables (sem2)

Generado por `03_variables/variables.ipynb`. Pool de desarrollo: 29,183 filas; 4 bloques de validación
temporal (`02_particion/particion.json`). Modelo de referencia: XGBoost con parámetros fijos.

## Importancia por permutación (dentro del entrenamiento, media de 4 bloques)
37 de 42 variables con importancia media > 0. Descartadas: `tend_monto_u3_vs_prev3`, `tend_nped_u3_vs_prev3`, `d_monto_m12`, `ticket_prom_u12`, `recencia_norm`.

Top 15:
|                     |   media |   n_positivos |
|:--------------------|--------:|--------------:|
| compras_hist        |  0.0089 |             4 |
| n_prod_u12          |  0.0071 |             4 |
| n_ped_acum          |  0.006  |             4 |
| monto_u6            |  0.0058 |             4 |
| monto_cv_u12        |  0.0044 |             4 |
| n_ped_u12           |  0.004  |             4 |
| n_ped_u6            |  0.0036 |             4 |
| monto_u12           |  0.0035 |             4 |
| d_monto_m1          |  0.0026 |             4 |
| meses_activos_u12   |  0.0026 |             4 |
| monto_u3            |  0.0025 |             4 |
| d_monto_m6          |  0.0023 |             3 |
| monto_por_prod_acum |  0.0013 |             3 |
| ticket_prom_u3      |  0.0013 |             3 |
| n_ped_u3            |  0.0012 |             4 |

## Ablación forward
Regla: conservar si el AUC medio sube más de 0.001.
**6 variables seleccionadas**: `compras_hist`, `n_prod_u12`, `monto_cv_u12`, `n_ped_u12`, `monto_u12`, `ticket_prom_u3`.

|   paso | variable                  | accion   |    auc |   bloques_mejoran |   n_vars |
|-------:|:--------------------------|:---------|-------:|------------------:|---------:|
|      1 | compras_hist              | inicio   | 0.6939 |               nan |        1 |
|      2 | n_prod_u12                | conserva | 0.776  |                 4 |        2 |
|      3 | n_ped_acum                | descarta | 0.7729 |                 1 |        2 |
|      4 | monto_u6                  | descarta | 0.7731 |                 1 |        2 |
|      5 | monto_cv_u12              | conserva | 0.7804 |                 4 |        3 |
|      6 | n_ped_u12                 | conserva | 0.7825 |                 4 |        4 |
|      7 | n_ped_u6                  | descarta | 0.7808 |                 1 |        4 |
|      8 | monto_u12                 | conserva | 0.7844 |                 3 |        5 |
|      9 | d_monto_m1                | descarta | 0.7852 |                 3 |        5 |
|     10 | meses_activos_u12         | descarta | 0.7844 |                 2 |        5 |
|     11 | monto_u3                  | descarta | 0.7851 |                 3 |        5 |
|     12 | d_monto_m6                | descarta | 0.7842 |                 1 |        5 |
|     13 | monto_por_prod_acum       | descarta | 0.7843 |                 2 |        5 |
|     14 | ticket_prom_u3            | conserva | 0.7879 |                 4 |        6 |
|     15 | n_ped_u3                  | descarta | 0.7859 |                 1 |        6 |
|     16 | intensidad_u3             | descarta | 0.7858 |                 1 |        6 |
|     17 | monto_std_u12             | descarta | 0.7867 |                 1 |        6 |
|     18 | monto_acum                | descarta | 0.7877 |                 2 |        6 |
|     19 | monto_mean_u12            | descarta | 0.7878 |                 2 |        6 |
|     20 | meses_desde_compra_previa | descarta | 0.7852 |                 0 |        6 |
|     21 | meses_activos_u6          | descarta | 0.7866 |                 1 |        6 |
|     22 | d_nped_m6                 | descarta | 0.7865 |                 1 |        6 |
|     23 | monto_ult_vs_media        | descarta | 0.7854 |                 0 |        6 |
|     24 | d_monto_m3                | descarta | 0.7883 |                 3 |        6 |
|     25 | d_nped_m12                | descarta | 0.7866 |                 2 |        6 |
|     26 | d_nped_m1                 | descarta | 0.7865 |                 1 |        6 |
|     27 | tasa_act_reciente_vs_hist | descarta | 0.7855 |                 1 |        6 |
|     28 | ticket_acum               | descarta | 0.7865 |                 1 |        6 |
|     29 | n_cat_max_u12             | descarta | 0.7877 |                 2 |        6 |
|     30 | monto_mensual_acum        | descarta | 0.7846 |                 0 |        6 |
|     31 | n_cat_max_acum            | descarta | 0.7873 |                 1 |        6 |
|     32 | d_nped_m9                 | descarta | 0.7867 |                 2 |        6 |
|     33 | d_monto_m9                | descarta | 0.787  |                 1 |        6 |
|     34 | basket_size_u12           | descarta | 0.7885 |                 2 |        6 |
|     35 | n_prod_acum               | descarta | 0.7875 |                 1 |        6 |
|     36 | d_nped_m3                 | descarta | 0.7874 |                 2 |        6 |
|     37 | meses_activos_u3          | descarta | 0.7873 |                 2 |        6 |

## Completo vs seleccionado
|                |   42 variables |   6 seleccionadas |
|:---------------|---------------:|------------------:|
| auc_b1         |         0.7876 |            0.7964 |
| auc_b2         |         0.7814 |            0.7824 |
| auc_b3         |         0.7894 |            0.7964 |
| auc_b4         |         0.773  |            0.7765 |
| auc_medio      |         0.7828 |            0.7879 |
| std            |         0.0064 |            0.0087 |
| lift10_mensual |         2.239  |            2.358  |

Lectura: la métrica es de selección (los mismos bloques se usan para elegir); no es una estimación final.
