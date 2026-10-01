# EDA — dataset de churn (sem2)

Generado por `01_eda/eda.ipynb`. Fuente: `data/churn_dataset_processed.csv`.

## Dataset
- 31,235 filas (vendedora-mes), 6,386 vendedoras, 42 variables.
- Meses observados: 2016-12 → 2026-03.
- Prevalencia de churn: **30.84%** (9,633 positivos).
- Filas por vendedora: mediana 2, máximo 88.

## Estabilidad temporal (prevalencia por año)
|   mes_obs |    n |   churn |
|----------:|-----:|--------:|
|      2016 |  514 |  0.2335 |
|      2017 | 5719 |  0.3029 |
|      2018 | 4195 |  0.349  |
|      2019 | 3700 |  0.3489 |
|      2020 | 1576 |  0.3693 |
|      2021 | 2997 |  0.2663 |
|      2022 | 3053 |  0.2784 |
|      2023 | 3149 |  0.2972 |
|      2024 | 3153 |  0.2953 |
|      2025 | 2573 |  0.2997 |
|      2026 |  606 |  0.2607 |

Últimos 4 meses (OOT): prevalencia 27.65% con 879 filas.

## Completitud
12 columnas con nulos en el SQL (máximo 48.2%); 0 nulos tras `preprocess`.

## Señal univariada (top 10 por AUC)
|                   |    auc | direccion   |
|:------------------|-------:|:------------|
| n_prod_u12        | 0.7216 | - churn     |
| monto_u12         | 0.7215 | - churn     |
| n_ped_u12         | 0.7178 | - churn     |
| monto_u6          | 0.7149 | - churn     |
| n_ped_u6          | 0.7077 | - churn     |
| meses_activos_u12 | 0.6977 | - churn     |
| monto_u3          | 0.6926 | - churn     |
| meses_activos_u6  | 0.69   | - churn     |
| monto_mean_u12    | 0.6878 | - churn     |
| n_prod_acum       | 0.6812 | - churn     |

## Redundancia
11 pares con |rho de Spearman| >= 0.9.
