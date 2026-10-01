# Modelado (sem2)

Generado por `04_modelado/modelado.ipynb`. Pool de desarrollo: 29,183 filas; 4 bloques de validación temporal
(`02_particion/particion.json`). Criterio: AUC medio 4 bloques; empate (<= 0.001) por lift top-10 % mensual; empate (<= 0.02) el más simple.

## 1 · Desbalance (§4.1) — XGBoost de referencia, 6 variables
| config         |   auc_b1 |   auc_b2 |   auc_b3 |   auc_b4 |    auc |    std |   lift10 |
|:---------------|---------:|---------:|---------:|---------:|-------:|-------:|---------:|
| pesos de clase |   0.7966 |   0.7822 |   0.7969 |   0.7779 | 0.7884 | 0.0085 |    2.354 |
| sin ponderar   |   0.7959 |   0.7815 |   0.7966 |   0.7775 | 0.7879 | 0.0085 |    2.274 |
| submuestreo    |   0.7933 |   0.7875 |   0.7908 |   0.7769 | 0.7871 | 0.0062 |    2.201 |
| SMOTE          |   0.7861 |   0.7748 |   0.7866 |   0.7689 | 0.7791 | 0.0075 |    2.234 |

**Elegido: pesos de clase.**

## 2 · Optuna (§4.3) — 100 trials por estudio, objetivo AUC medio, peso de clase en el espacio de búsqueda
| config      | algo     |   n_vars |   auc_b1 |   auc_b2 |   auc_b3 |   auc_b4 |    auc |    std |   lift10 |   best_trial |
|:------------|:---------|---------:|---------:|---------:|---------:|---------:|-------:|-------:|---------:|-------------:|
| logreg_6    | logreg   |        6 |   0.7787 |   0.774  |   0.7991 |   0.7728 | 0.7811 | 0.0106 |    2.192 |           95 |
| logreg_42   | logreg   |       42 |   0.7912 |   0.7785 |   0.7908 |   0.7775 | 0.7845 | 0.0065 |    2.267 |           91 |
| rf_6        | rf       |        6 |   0.7973 |   0.7836 |   0.799  |   0.779  | 0.7898 | 0.0086 |    2.287 |           20 |
| rf_42       | rf       |       42 |   0.7958 |   0.7791 |   0.7965 |   0.7763 | 0.7869 | 0.0093 |    2.249 |           99 |
| xgboost_6   | xgboost  |        6 |   0.8011 |   0.7845 |   0.7994 |   0.7792 | 0.791  | 0.0094 |    2.281 |           90 |
| xgboost_42  | xgboost  |       42 |   0.7936 |   0.7811 |   0.7933 |   0.775  | 0.7858 | 0.008  |    2.272 |           77 |
| lightgbm_6  | lightgbm |        6 |   0.7994 |   0.7847 |   0.7993 |   0.7808 | 0.7911 | 0.0084 |    2.312 |           98 |
| lightgbm_42 | lightgbm |       42 |   0.7935 |   0.7813 |   0.7944 |   0.7754 | 0.7862 | 0.0081 |    2.293 |           82 |
| catboost_6  | catboost |        6 |   0.7984 |   0.7836 |   0.8001 |   0.7793 | 0.7904 | 0.009  |    2.281 |           52 |
| catboost_42 | catboost |       42 |   0.7936 |   0.7795 |   0.7947 |   0.7757 | 0.7859 | 0.0084 |    2.296 |           53 |

## 3 · Conjunto de variables por algoritmo
| algoritmo   |   auc_6 |   auc_42 |   lift_6 |   lift_42 |   elegido |
|:------------|--------:|---------:|---------:|----------:|----------:|
| logreg      |  0.7811 |   0.7845 |    2.192 |     2.267 |        42 |
| rf          |  0.7898 |   0.7869 |    2.287 |     2.249 |         6 |
| xgboost     |  0.791  |   0.7858 |    2.281 |     2.272 |         6 |
| lightgbm    |  0.7911 |   0.7862 |    2.312 |     2.293 |         6 |
| catboost    |  0.7904 |   0.7859 |    2.281 |     2.296 |         6 |

## 4 · Ventanas de entrenamiento (hiperparámetros fijos)
AUC medio:
| algo     |    24m |    36m |    48m |   todo |
|:---------|-------:|-------:|-------:|-------:|
| catboost | 0.7871 | 0.789  | 0.7902 | 0.7904 |
| lightgbm | 0.7884 | 0.7902 | 0.7904 | 0.7911 |
| logreg   | 0.7866 | 0.7871 | 0.7859 | 0.7845 |
| rf       | 0.7883 | 0.7892 | 0.7901 | 0.7898 |
| xgboost  | 0.7885 | 0.7902 | 0.7898 | 0.791  |

Lift top-10 % mensual:
| algo     |   24m |   36m |   48m |   todo |
|:---------|------:|------:|------:|-------:|
| catboost | 2.2   | 2.232 | 2.278 |  2.281 |
| lightgbm | 2.253 | 2.332 | 2.294 |  2.312 |
| logreg   | 2.172 | 2.242 | 2.212 |  2.267 |
| rf       | 2.298 | 2.26  | 2.239 |  2.287 |
| xgboost  | 2.206 | 2.272 | 2.305 |  2.281 |

Ventana elegida: logreg → 36m, rf → todo, xgboost → todo, lightgbm → todo, catboost → todo.

## 5 · Candidatos individuales y ensembles por promedio
| config                              |   n_miembros |   n_vars | ventana     |   auc_b1 |   auc_b2 |   auc_b3 |   auc_b4 |    auc |    std |   lift10 |
|:------------------------------------|-------------:|---------:|:------------|---------:|---------:|---------:|---------:|-------:|-------:|---------:|
| lightgbm+xgboost+catboost+rf+logreg |            5 |       42 | por miembro |   0.7999 |   0.7846 |   0.8007 |   0.7828 | 0.792  | 0.0083 |    2.311 |
| lightgbm+xgboost                    |            2 |        6 | por miembro |   0.8004 |   0.7848 |   0.7995 |   0.7803 | 0.7913 | 0.0088 |    2.3   |
| lightgbm+xgboost+catboost           |            3 |        6 | por miembro |   0.8001 |   0.7846 |   0.7999 |   0.7803 | 0.7912 | 0.0089 |    2.344 |
| lightgbm+xgboost+catboost+rf        |            4 |        6 | por miembro |   0.7995 |   0.7845 |   0.7999 |   0.7803 | 0.7911 | 0.0088 |    2.315 |
| lightgbm                            |            1 |        6 | todo        |   0.7994 |   0.7847 |   0.7993 |   0.7808 | 0.7911 | 0.0084 |    2.312 |
| xgboost                             |            1 |        6 | todo        |   0.8011 |   0.7845 |   0.7994 |   0.7792 | 0.791  | 0.0094 |    2.281 |
| catboost                            |            1 |        6 | todo        |   0.7984 |   0.7836 |   0.8001 |   0.7793 | 0.7904 | 0.009  |    2.281 |
| rf                                  |            1 |        6 | todo        |   0.7973 |   0.7836 |   0.799  |   0.779  | 0.7898 | 0.0086 |    2.287 |
| logreg                              |            1 |       42 | 36m         |   0.7928 |   0.7776 |   0.796  |   0.7819 | 0.7871 | 0.0076 |    2.242 |

## Modelo final
**lightgbm+xgboost+catboost** (ensemble, 6 variables): AUC medio 0.7912,
lift top-10 % mensual 2.34.

Hiperparámetros de los miembros:
- `lightgbm` (6 vars, ventana todo): `{'w': 1.0809366442018964, 'n_estimators': 450, 'learning_rate': 0.01194484865470532, 'num_leaves': 31, 'min_child_samples': 6, 'subsample': 0.6490876456502913, 'colsample_bytree': 0.5675823318446789, 'reg_lambda': 3.6583631415514786, 'reg_alpha': 7.917674381996484}`
- `xgboost` (6 vars, ventana todo): `{'w': 1.2144133732796452, 'n_estimators': 300, 'learning_rate': 0.014300848118031843, 'max_depth': 6, 'min_child_weight': 16.564848321502605, 'subsample': 0.9996157411929472, 'colsample_bytree': 0.5347541959768086, 'gamma': 2.589573610083936, 'reg_lambda': 0.07911200387245458, 'reg_alpha': 5.919074950856576}`
- `catboost` (6 vars, ventana todo): `{'w': 1.0460297856090754, 'iterations': 200, 'learning_rate': 0.011828970101096912, 'depth': 6, 'l2_leaf_reg': 23.866719567196707, 'random_strength': 0.030589318465974465, 'bagging_temperature': 0.9967629508977666}`

Lectura: todas las métricas son de selección (los mismos bloques se usan para decidir); la estimación final es la
del OOT en `05_evaluacion`.
