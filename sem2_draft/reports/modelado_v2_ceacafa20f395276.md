# Modelado v2

Experimento: ceacafa20f395276. Fecha UTC: 2026-11-01T22:21:04.066953+00:00.

Presupuesto temporal: 120 minutos. Trials comparados por estudio: 382; 10 estudios. Objetivo: AUC promedio de 4 bloques.

## Tratamiento del desbalance

| estrategia | auc_promedio | desviacion_auc | lift_promedio |
| --- | --- | --- | --- |
| sin_tratamiento | 0.7873 | 0.0072 | 2.3230 |
| pesos_clase | 0.7874 | 0.0075 | 2.3258 |
| submuestreo | 0.7854 | 0.0066 | 2.2231 |
| smote | 0.7874 | 0.0082 | 2.2819 |

Mejor estrategia de referencia: sin_tratamiento; las cuatro estrategias participaron en la búsqueda conjunta.

## Optuna

| estudio | algoritmo | conjunto | numero_variables | estrategia | ventana | trials_completos | trials_comparados | mejor_trial | auc_promedio | desviacion_auc | lift_promedio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logreg__seleccionadas | logreg | seleccionadas | 6 | smote | 24_meses | 383 | 382 | 91 | 0.7857 | 0.0097 | 2.1488 |
| logreg__completas | logreg | completas | 42 | pesos_clase | 36_meses | 383 | 382 | 230 | 0.7876 | 0.0077 | 2.2577 |
| random_forest__seleccionadas | random_forest | seleccionadas | 6 | sin_tratamiento | 48_meses | 383 | 382 | 47 | 0.7912 | 0.0092 | 2.2882 |
| random_forest__completas | random_forest | completas | 42 | sin_tratamiento | toda_historia | 383 | 382 | 91 | 0.7869 | 0.0092 | 2.2903 |
| xgboost__seleccionadas | xgboost | seleccionadas | 6 | sin_tratamiento | 48_meses | 383 | 382 | 363 | 0.7930 | 0.0098 | 2.3025 |
| xgboost__completas | xgboost | completas | 42 | pesos_clase | 48_meses | 383 | 382 | 375 | 0.7872 | 0.0094 | 2.2621 |
| lightgbm__seleccionadas | lightgbm | seleccionadas | 6 | pesos_clase | 36_meses | 383 | 382 | 325 | 0.7916 | 0.0076 | 2.2784 |
| lightgbm__completas | lightgbm | completas | 42 | pesos_clase | 36_meses | 383 | 382 | 263 | 0.7879 | 0.0081 | 2.2857 |
| catboost__seleccionadas | catboost | seleccionadas | 6 | smote | 48_meses | 383 | 382 | 347 | 0.7912 | 0.0094 | 2.2636 |
| catboost__completas | catboost | completas | 42 | sin_tratamiento | 36_meses | 382 | 382 | 378 | 0.7880 | 0.0087 | 2.2774 |

## Búsqueda conjunta de estrategias y ventanas

| estudio | estrategia | ventana | trials | mejor_auc |
| --- | --- | --- | --- | --- |
| catboost__completas | pesos_clase | 24_meses | 2 | 0.7805 |
| catboost__completas | pesos_clase | 36_meses | 13 | 0.7869 |
| catboost__completas | pesos_clase | 48_meses | 2 | 0.7856 |
| catboost__completas | pesos_clase | toda_historia | 3 | 0.7840 |
| catboost__completas | sin_tratamiento | 24_meses | 12 | 0.7865 |
| catboost__completas | sin_tratamiento | 36_meses | 282 | 0.7880 |
| catboost__completas | sin_tratamiento | 48_meses | 14 | 0.7872 |
| catboost__completas | sin_tratamiento | toda_historia | 14 | 0.7859 |
| catboost__completas | smote | 24_meses | 4 | 0.7848 |
| catboost__completas | smote | 36_meses | 13 | 0.7856 |
| catboost__completas | smote | 48_meses | 4 | 0.7858 |
| catboost__completas | smote | toda_historia | 1 | 0.7590 |
| catboost__completas | submuestreo | 24_meses | 4 | 0.7860 |
| catboost__completas | submuestreo | 36_meses | 12 | 0.7857 |
| catboost__completas | submuestreo | 48_meses | 1 | 0.7802 |
| catboost__completas | submuestreo | toda_historia | 1 | 0.7726 |
| catboost__seleccionadas | pesos_clase | 24_meses | 9 | 0.7900 |
| catboost__seleccionadas | pesos_clase | 36_meses | 1 | 0.7530 |
| catboost__seleccionadas | pesos_clase | 48_meses | 7 | 0.7901 |
| catboost__seleccionadas | pesos_clase | toda_historia | 3 | 0.7884 |
| catboost__seleccionadas | sin_tratamiento | 24_meses | 9 | 0.7900 |
| catboost__seleccionadas | sin_tratamiento | 36_meses | 1 | 0.7887 |
| catboost__seleccionadas | sin_tratamiento | 48_meses | 9 | 0.7900 |
| catboost__seleccionadas | sin_tratamiento | toda_historia | 2 | 0.7886 |
| catboost__seleccionadas | smote | 24_meses | 173 | 0.7910 |
| catboost__seleccionadas | smote | 36_meses | 19 | 0.7903 |
| catboost__seleccionadas | smote | 48_meses | 122 | 0.7912 |
| catboost__seleccionadas | smote | toda_historia | 9 | 0.7890 |
| catboost__seleccionadas | submuestreo | 24_meses | 8 | 0.7899 |
| catboost__seleccionadas | submuestreo | 36_meses | 4 | 0.7898 |
| catboost__seleccionadas | submuestreo | 48_meses | 5 | 0.7900 |
| catboost__seleccionadas | submuestreo | toda_historia | 1 | 0.7803 |
| lightgbm__completas | pesos_clase | 24_meses | 9 | 0.7864 |
| lightgbm__completas | pesos_clase | 36_meses | 218 | 0.7879 |
| lightgbm__completas | pesos_clase | 48_meses | 7 | 0.7867 |
| lightgbm__completas | pesos_clase | toda_historia | 9 | 0.7855 |
| lightgbm__completas | sin_tratamiento | 24_meses | 6 | 0.7854 |
| lightgbm__completas | sin_tratamiento | 36_meses | 79 | 0.7868 |
| lightgbm__completas | sin_tratamiento | 48_meses | 9 | 0.7863 |
| lightgbm__completas | sin_tratamiento | toda_historia | 7 | 0.7860 |
| lightgbm__completas | smote | 24_meses | 3 | 0.7843 |
| lightgbm__completas | smote | 36_meses | 6 | 0.7849 |
| lightgbm__completas | smote | 48_meses | 9 | 0.7859 |
| lightgbm__completas | smote | toda_historia | 2 | 0.7839 |
| lightgbm__completas | submuestreo | 24_meses | 3 | 0.7652 |
| lightgbm__completas | submuestreo | 36_meses | 10 | 0.7848 |
| lightgbm__completas | submuestreo | 48_meses | 2 | 0.7730 |
| lightgbm__completas | submuestreo | toda_historia | 3 | 0.7840 |
| lightgbm__seleccionadas | pesos_clase | 24_meses | 10 | 0.7888 |
| lightgbm__seleccionadas | pesos_clase | 36_meses | 148 | 0.7916 |
| lightgbm__seleccionadas | pesos_clase | 48_meses | 8 | 0.7904 |
| lightgbm__seleccionadas | pesos_clase | toda_historia | 8 | 0.7892 |
| lightgbm__seleccionadas | sin_tratamiento | 24_meses | 5 | 0.7886 |
| lightgbm__seleccionadas | sin_tratamiento | 36_meses | 156 | 0.7916 |
| lightgbm__seleccionadas | sin_tratamiento | 48_meses | 8 | 0.7902 |
| lightgbm__seleccionadas | sin_tratamiento | toda_historia | 6 | 0.7897 |
| lightgbm__seleccionadas | smote | 24_meses | 1 | 0.7767 |
| lightgbm__seleccionadas | smote | 36_meses | 8 | 0.7893 |
| lightgbm__seleccionadas | smote | 48_meses | 4 | 0.7902 |
| lightgbm__seleccionadas | smote | toda_historia | 3 | 0.7884 |
| lightgbm__seleccionadas | submuestreo | 24_meses | 4 | 0.7866 |
| lightgbm__seleccionadas | submuestreo | 36_meses | 10 | 0.7897 |
| lightgbm__seleccionadas | submuestreo | 48_meses | 1 | 0.7126 |
| lightgbm__seleccionadas | submuestreo | toda_historia | 2 | 0.7875 |
| logreg__completas | pesos_clase | 24_meses | 13 | 0.7874 |
| logreg__completas | pesos_clase | 36_meses | 282 | 0.7876 |
| logreg__completas | pesos_clase | 48_meses | 13 | 0.7864 |
| logreg__completas | pesos_clase | toda_historia | 11 | 0.7841 |
| logreg__completas | sin_tratamiento | 24_meses | 2 | 0.7865 |
| logreg__completas | sin_tratamiento | 36_meses | 13 | 0.7875 |
| logreg__completas | sin_tratamiento | 48_meses | 5 | 0.7864 |
| logreg__completas | sin_tratamiento | toda_historia | 1 | 0.7844 |
| logreg__completas | smote | 24_meses | 4 | 0.7870 |
| logreg__completas | smote | 36_meses | 15 | 0.7874 |
| logreg__completas | smote | 48_meses | 2 | 0.7846 |
| logreg__completas | smote | toda_historia | 1 | 0.7825 |
| logreg__completas | submuestreo | 24_meses | 2 | 0.7850 |
| logreg__completas | submuestreo | 36_meses | 12 | 0.7855 |
| logreg__completas | submuestreo | 48_meses | 1 | 0.7832 |
| logreg__completas | submuestreo | toda_historia | 5 | 0.7840 |
| logreg__seleccionadas | pesos_clase | 24_meses | 13 | 0.7852 |
| logreg__seleccionadas | pesos_clase | 36_meses | 1 | 0.7851 |
| logreg__seleccionadas | pesos_clase | 48_meses | 4 | 0.7834 |
| logreg__seleccionadas | pesos_clase | toda_historia | 1 | 0.7808 |
| logreg__seleccionadas | sin_tratamiento | 24_meses | 13 | 0.7852 |
| logreg__seleccionadas | sin_tratamiento | 36_meses | 2 | 0.7856 |
| logreg__seleccionadas | sin_tratamiento | 48_meses | 5 | 0.7836 |
| logreg__seleccionadas | sin_tratamiento | toda_historia | 1 | 0.7810 |
| logreg__seleccionadas | smote | 24_meses | 289 | 0.7857 |
| logreg__seleccionadas | smote | 36_meses | 13 | 0.7854 |
| logreg__seleccionadas | smote | 48_meses | 11 | 0.7833 |
| logreg__seleccionadas | smote | toda_historia | 9 | 0.7801 |
| logreg__seleccionadas | submuestreo | 24_meses | 8 | 0.7845 |
| logreg__seleccionadas | submuestreo | 36_meses | 4 | 0.7848 |
| logreg__seleccionadas | submuestreo | 48_meses | 1 | 0.7817 |
| logreg__seleccionadas | submuestreo | toda_historia | 7 | 0.7802 |
| random_forest__completas | pesos_clase | 24_meses | 3 | 0.7863 |
| random_forest__completas | pesos_clase | 36_meses | 2 | 0.7854 |
| random_forest__completas | pesos_clase | 48_meses | 3 | 0.7859 |
| random_forest__completas | pesos_clase | toda_historia | 12 | 0.7863 |
| random_forest__completas | sin_tratamiento | 24_meses | 11 | 0.7867 |
| random_forest__completas | sin_tratamiento | 36_meses | 12 | 0.7865 |
| random_forest__completas | sin_tratamiento | 48_meses | 14 | 0.7868 |
| random_forest__completas | sin_tratamiento | toda_historia | 283 | 0.7869 |
| random_forest__completas | smote | 24_meses | 3 | 0.7852 |
| random_forest__completas | smote | 36_meses | 4 | 0.7812 |
| random_forest__completas | smote | 48_meses | 4 | 0.7860 |
| random_forest__completas | smote | toda_historia | 12 | 0.7860 |
| random_forest__completas | submuestreo | 24_meses | 3 | 0.7836 |
| random_forest__completas | submuestreo | 36_meses | 2 | 0.7821 |
| random_forest__completas | submuestreo | 48_meses | 2 | 0.7851 |
| random_forest__completas | submuestreo | toda_historia | 12 | 0.7852 |
| random_forest__seleccionadas | pesos_clase | 24_meses | 2 | 0.7869 |
| random_forest__seleccionadas | pesos_clase | 36_meses | 2 | 0.7876 |
| random_forest__seleccionadas | pesos_clase | 48_meses | 13 | 0.7905 |
| random_forest__seleccionadas | pesos_clase | toda_historia | 3 | 0.7893 |
| random_forest__seleccionadas | sin_tratamiento | 24_meses | 13 | 0.7882 |
| random_forest__seleccionadas | sin_tratamiento | 36_meses | 13 | 0.7905 |
| random_forest__seleccionadas | sin_tratamiento | 48_meses | 279 | 0.7912 |
| random_forest__seleccionadas | sin_tratamiento | toda_historia | 14 | 0.7897 |
| random_forest__seleccionadas | smote | 24_meses | 2 | 0.7877 |
| random_forest__seleccionadas | smote | 36_meses | 4 | 0.7836 |
| random_forest__seleccionadas | smote | 48_meses | 14 | 0.7899 |
| random_forest__seleccionadas | smote | toda_historia | 3 | 0.7881 |
| random_forest__seleccionadas | submuestreo | 24_meses | 3 | 0.7871 |
| random_forest__seleccionadas | submuestreo | 36_meses | 1 | 0.7877 |
| random_forest__seleccionadas | submuestreo | 48_meses | 15 | 0.7904 |
| random_forest__seleccionadas | submuestreo | toda_historia | 1 | 0.7874 |
| xgboost__completas | pesos_clase | 24_meses | 14 | 0.7861 |
| xgboost__completas | pesos_clase | 36_meses | 12 | 0.7860 |
| xgboost__completas | pesos_clase | 48_meses | 278 | 0.7872 |
| xgboost__completas | pesos_clase | toda_historia | 10 | 0.7854 |
| xgboost__completas | sin_tratamiento | 24_meses | 1 | 0.7829 |
| xgboost__completas | sin_tratamiento | 36_meses | 1 | 0.7521 |
| xgboost__completas | sin_tratamiento | 48_meses | 11 | 0.7857 |
| xgboost__completas | sin_tratamiento | toda_historia | 14 | 0.7852 |
| xgboost__completas | smote | 24_meses | 1 | 0.7407 |
| xgboost__completas | smote | 36_meses | 5 | 0.7827 |
| xgboost__completas | smote | 48_meses | 12 | 0.7856 |
| xgboost__completas | smote | toda_historia | 3 | 0.7834 |
| xgboost__completas | submuestreo | 24_meses | 3 | 0.7851 |
| xgboost__completas | submuestreo | 36_meses | 3 | 0.7828 |
| xgboost__completas | submuestreo | 48_meses | 13 | 0.7852 |
| xgboost__completas | submuestreo | toda_historia | 1 | 0.7416 |
| xgboost__seleccionadas | pesos_clase | 24_meses | 5 | 0.7881 |
| xgboost__seleccionadas | pesos_clase | 36_meses | 1 | 0.7410 |
| xgboost__seleccionadas | pesos_clase | 48_meses | 43 | 0.7911 |
| xgboost__seleccionadas | pesos_clase | toda_historia | 1 | 0.7681 |
| xgboost__seleccionadas | sin_tratamiento | 24_meses | 7 | 0.7881 |
| xgboost__seleccionadas | sin_tratamiento | 36_meses | 11 | 0.7899 |
| xgboost__seleccionadas | sin_tratamiento | 48_meses | 260 | 0.7930 |
| xgboost__seleccionadas | sin_tratamiento | toda_historia | 14 | 0.7896 |
| xgboost__seleccionadas | smote | 24_meses | 5 | 0.7853 |
| xgboost__seleccionadas | smote | 36_meses | 5 | 0.7885 |
| xgboost__seleccionadas | smote | 48_meses | 6 | 0.7897 |
| xgboost__seleccionadas | smote | toda_historia | 4 | 0.7883 |
| xgboost__seleccionadas | submuestreo | 24_meses | 2 | 0.7873 |
| xgboost__seleccionadas | submuestreo | 36_meses | 4 | 0.7893 |
| xgboost__seleccionadas | submuestreo | 48_meses | 13 | 0.7897 |
| xgboost__seleccionadas | submuestreo | toda_historia | 1 | 0.7529 |

## Candidatos individuales

| estudio | algoritmo | conjunto | numero_variables | estrategia | ventana | trials_completos | trials_comparados | mejor_trial | auc_promedio | desviacion_auc | lift_promedio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logreg__seleccionadas | logreg | seleccionadas | 6 | smote | 24_meses | 383 | 382 | 91 | 0.7857 | 0.0097 | 2.1488 |
| logreg__completas | logreg | completas | 42 | pesos_clase | 36_meses | 383 | 382 | 230 | 0.7876 | 0.0077 | 2.2577 |
| random_forest__seleccionadas | random_forest | seleccionadas | 6 | sin_tratamiento | 48_meses | 383 | 382 | 47 | 0.7912 | 0.0092 | 2.2882 |
| random_forest__completas | random_forest | completas | 42 | sin_tratamiento | toda_historia | 383 | 382 | 91 | 0.7869 | 0.0092 | 2.2903 |
| xgboost__seleccionadas | xgboost | seleccionadas | 6 | sin_tratamiento | 48_meses | 383 | 382 | 363 | 0.7930 | 0.0098 | 2.3025 |
| xgboost__completas | xgboost | completas | 42 | pesos_clase | 48_meses | 383 | 382 | 375 | 0.7872 | 0.0094 | 2.2621 |
| lightgbm__seleccionadas | lightgbm | seleccionadas | 6 | pesos_clase | 36_meses | 383 | 382 | 325 | 0.7916 | 0.0076 | 2.2784 |
| lightgbm__completas | lightgbm | completas | 42 | pesos_clase | 36_meses | 383 | 382 | 263 | 0.7879 | 0.0081 | 2.2857 |
| catboost__seleccionadas | catboost | seleccionadas | 6 | smote | 48_meses | 383 | 382 | 347 | 0.7912 | 0.0094 | 2.2636 |
| catboost__completas | catboost | completas | 42 | sin_tratamiento | 36_meses | 382 | 382 | 378 | 0.7880 | 0.0087 | 2.2774 |

## Selección entre modelos individuales

| estudio | algoritmo | conjunto | numero_variables | estrategia | ventana | trials_completos | trials_comparados | mejor_trial | auc_promedio | desviacion_auc | lift_promedio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost__seleccionadas | xgboost | seleccionadas | 6 | sin_tratamiento | 48_meses | 383 | 382 | 363 | 0.7930 | 0.0098 | 2.3025 |
| lightgbm__seleccionadas | lightgbm | seleccionadas | 6 | pesos_clase | 36_meses | 383 | 382 | 325 | 0.7916 | 0.0076 | 2.2784 |
| random_forest__seleccionadas | random_forest | seleccionadas | 6 | sin_tratamiento | 48_meses | 383 | 382 | 47 | 0.7912 | 0.0092 | 2.2882 |
| catboost__seleccionadas | catboost | seleccionadas | 6 | smote | 48_meses | 383 | 382 | 347 | 0.7912 | 0.0094 | 2.2636 |
| catboost__completas | catboost | completas | 42 | sin_tratamiento | 36_meses | 382 | 382 | 378 | 0.7880 | 0.0087 | 2.2774 |
| lightgbm__completas | lightgbm | completas | 42 | pesos_clase | 36_meses | 383 | 382 | 263 | 0.7879 | 0.0081 | 2.2857 |
| logreg__completas | logreg | completas | 42 | pesos_clase | 36_meses | 383 | 382 | 230 | 0.7876 | 0.0077 | 2.2577 |
| xgboost__completas | xgboost | completas | 42 | pesos_clase | 48_meses | 383 | 382 | 375 | 0.7872 | 0.0094 | 2.2621 |
| random_forest__completas | random_forest | completas | 42 | sin_tratamiento | toda_historia | 383 | 382 | 91 | 0.7869 | 0.0092 | 2.2903 |
| logreg__seleccionadas | logreg | seleccionadas | 6 | smote | 24_meses | 383 | 382 | 91 | 0.7857 | 0.0097 | 2.1488 |

## Configuración seleccionada

Modelo: xgboost__seleccionadas. AUC promedio: 0.792966. Desviación entre bloques: 0.009830. Lift mensual: 2.302536.

Regla fijada antes de entrenar: tolerancia AUC 0.001, tolerancia lift 0.04; después menos variables.

| mes | filas_validacion | seleccionadas | tasa_churn_mes | precision_seleccionadas | lift |
| --- | --- | --- | --- | --- | --- |
| 2024-02-01 00:00:00 | 239 | 23 | 0.2385 | 0.4348 | 1.8230 |
| 2024-03-01 00:00:00 | 242 | 24 | 0.2893 | 0.7083 | 2.4488 |
| 2024-04-01 00:00:00 | 258 | 25 | 0.3178 | 0.7600 | 2.3912 |
| 2024-05-01 00:00:00 | 267 | 26 | 0.2846 | 0.7308 | 2.5673 |
| 2024-06-01 00:00:00 | 250 | 25 | 0.2240 | 0.5600 | 2.5000 |
| 2024-07-01 00:00:00 | 283 | 28 | 0.2580 | 0.6786 | 2.6306 |
| 2024-08-01 00:00:00 | 296 | 29 | 0.2905 | 0.5862 | 2.0176 |
| 2024-09-01 00:00:00 | 281 | 28 | 0.3381 | 0.6786 | 2.0071 |
| 2024-10-01 00:00:00 | 247 | 24 | 0.2672 | 0.7083 | 2.6509 |
| 2024-11-01 00:00:00 | 265 | 26 | 0.3245 | 0.7308 | 2.2518 |
| 2024-12-01 00:00:00 | 272 | 27 | 0.3787 | 0.8519 | 2.2496 |
| 2025-01-01 00:00:00 | 255 | 25 | 0.3373 | 0.8400 | 2.4907 |
| 2025-02-01 00:00:00 | 210 | 21 | 0.3238 | 0.8571 | 2.6471 |
| 2025-03-01 00:00:00 | 214 | 21 | 0.2944 | 0.6667 | 2.2646 |
| 2025-04-01 00:00:00 | 235 | 23 | 0.3234 | 0.6522 | 2.0166 |
| 2025-05-01 00:00:00 | 213 | 21 | 0.3286 | 0.6190 | 1.8837 |

## Artefactos

Configuración: /home/carlos/projects/glamour/customer-churn-prediction/sem2_draft/04_modelado/resultados_v2/ceacafa20f395276/modelo_final.json.

Pipeline entrenado sobre desarrollo: /home/carlos/projects/glamour/customer-churn-prediction/sem2_draft/04_modelado/resultados_v2/ceacafa20f395276/modelo_final.joblib.

Estudios y trials: /home/carlos/projects/glamour/customer-churn-prediction/sem2_draft/04_modelado/resultados_v2/ceacafa20f395276/optuna.db y /home/carlos/projects/glamour/customer-churn-prediction/sem2_draft/04_modelado/resultados_v2/ceacafa20f395276/trials.

Estas métricas se usaron para seleccionar configuraciones; no son una estimación independiente del desempeño final. El OOT no se evaluó en esta corrida. Las evaluaciones históricas del OOT están documentadas en AVANCE.md. Queda pendiente evaluar el artefacto v2 y estimar incertidumbre en la fase 05.
