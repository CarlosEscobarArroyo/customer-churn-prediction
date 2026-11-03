# Modelado v2

Experimento: b2ad959cb4c424b4. Fecha UTC: 2026-11-03T05:59:09.986921+00:00.

Búsqueda sin límite de tiempo. Trials comparados por estudio: 1000; 10 estudios. Objetivo: AUC promedio de 4 bloques.

## Optuna

| estudio | algoritmo | conjunto | numero_variables | estrategia | ventana | trials_completos | trials_comparados | mejor_trial | auc_promedio | desviacion_auc | lift_promedio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logreg__seleccionadas | logreg | seleccionadas | 6 | smote | 24_meses | 1000 | 1000 | 729 | 0.7857 | 0.0097 | 2.1488 |
| logreg__completas | logreg | completas | 42 | pesos_clase | 36_meses | 1000 | 1000 | 525 | 0.7876 | 0.0077 | 2.2577 |
| random_forest__seleccionadas | random_forest | seleccionadas | 6 | sin_tratamiento | 48_meses | 1000 | 1000 | 587 | 0.7915 | 0.0092 | 2.3187 |
| random_forest__completas | random_forest | completas | 42 | sin_tratamiento | toda_historia | 1000 | 1000 | 91 | 0.7869 | 0.0092 | 2.2903 |
| xgboost__seleccionadas | xgboost | seleccionadas | 6 | sin_tratamiento | 48_meses | 1000 | 1000 | 363 | 0.7930 | 0.0098 | 2.3025 |
| xgboost__completas | xgboost | completas | 42 | sin_tratamiento | 48_meses | 1000 | 1000 | 638 | 0.7882 | 0.0079 | 2.2421 |
| lightgbm__seleccionadas | lightgbm | seleccionadas | 6 | sin_tratamiento | 36_meses | 1000 | 1000 | 822 | 0.7924 | 0.0085 | 2.3058 |
| lightgbm__completas | lightgbm | completas | 42 | pesos_clase | 36_meses | 1000 | 1000 | 972 | 0.7880 | 0.0083 | 2.2567 |
| catboost__seleccionadas | catboost | seleccionadas | 6 | smote | 48_meses | 1000 | 1000 | 886 | 0.7912 | 0.0095 | 2.2256 |
| catboost__completas | catboost | completas | 42 | sin_tratamiento | 36_meses | 1000 | 1000 | 378 | 0.7880 | 0.0087 | 2.2774 |

## Búsqueda conjunta de estrategias y ventanas

| estudio | estrategia | ventana | trials | mejor_auc |
| --- | --- | --- | --- | --- |
| catboost__completas | pesos_clase | 24_meses | 2 | 0.7805 |
| catboost__completas | pesos_clase | 36_meses | 38 | 0.7875 |
| catboost__completas | pesos_clase | 48_meses | 2 | 0.7856 |
| catboost__completas | pesos_clase | toda_historia | 3 | 0.7840 |
| catboost__completas | sin_tratamiento | 24_meses | 29 | 0.7865 |
| catboost__completas | sin_tratamiento | 36_meses | 758 | 0.7880 |
| catboost__completas | sin_tratamiento | 48_meses | 38 | 0.7872 |
| catboost__completas | sin_tratamiento | toda_historia | 39 | 0.7859 |
| catboost__completas | smote | 24_meses | 4 | 0.7848 |
| catboost__completas | smote | 36_meses | 37 | 0.7858 |
| catboost__completas | smote | 48_meses | 5 | 0.7858 |
| catboost__completas | smote | toda_historia | 1 | 0.7590 |
| catboost__completas | submuestreo | 24_meses | 12 | 0.7862 |
| catboost__completas | submuestreo | 36_meses | 30 | 0.7861 |
| catboost__completas | submuestreo | 48_meses | 1 | 0.7802 |
| catboost__completas | submuestreo | toda_historia | 1 | 0.7726 |
| catboost__seleccionadas | pesos_clase | 24_meses | 10 | 0.7900 |
| catboost__seleccionadas | pesos_clase | 36_meses | 2 | 0.7905 |
| catboost__seleccionadas | pesos_clase | 48_meses | 28 | 0.7904 |
| catboost__seleccionadas | pesos_clase | toda_historia | 4 | 0.7887 |
| catboost__seleccionadas | sin_tratamiento | 24_meses | 9 | 0.7900 |
| catboost__seleccionadas | sin_tratamiento | 36_meses | 2 | 0.7903 |
| catboost__seleccionadas | sin_tratamiento | 48_meses | 32 | 0.7903 |
| catboost__seleccionadas | sin_tratamiento | toda_historia | 3 | 0.7889 |
| catboost__seleccionadas | smote | 24_meses | 176 | 0.7910 |
| catboost__seleccionadas | smote | 36_meses | 43 | 0.7906 |
| catboost__seleccionadas | smote | 48_meses | 618 | 0.7912 |
| catboost__seleccionadas | smote | toda_historia | 30 | 0.7893 |
| catboost__seleccionadas | submuestreo | 24_meses | 8 | 0.7899 |
| catboost__seleccionadas | submuestreo | 36_meses | 5 | 0.7898 |
| catboost__seleccionadas | submuestreo | 48_meses | 26 | 0.7901 |
| catboost__seleccionadas | submuestreo | toda_historia | 4 | 0.7886 |
| lightgbm__completas | pesos_clase | 24_meses | 31 | 0.7864 |
| lightgbm__completas | pesos_clase | 36_meses | 683 | 0.7880 |
| lightgbm__completas | pesos_clase | 48_meses | 33 | 0.7872 |
| lightgbm__completas | pesos_clase | toda_historia | 29 | 0.7857 |
| lightgbm__completas | sin_tratamiento | 24_meses | 8 | 0.7860 |
| lightgbm__completas | sin_tratamiento | 36_meses | 108 | 0.7874 |
| lightgbm__completas | sin_tratamiento | 48_meses | 9 | 0.7863 |
| lightgbm__completas | sin_tratamiento | toda_historia | 9 | 0.7860 |
| lightgbm__completas | smote | 24_meses | 3 | 0.7843 |
| lightgbm__completas | smote | 36_meses | 28 | 0.7856 |
| lightgbm__completas | smote | 48_meses | 12 | 0.7859 |
| lightgbm__completas | smote | toda_historia | 3 | 0.7848 |
| lightgbm__completas | submuestreo | 24_meses | 4 | 0.7835 |
| lightgbm__completas | submuestreo | 36_meses | 32 | 0.7849 |
| lightgbm__completas | submuestreo | 48_meses | 3 | 0.7825 |
| lightgbm__completas | submuestreo | toda_historia | 5 | 0.7840 |
| lightgbm__seleccionadas | pesos_clase | 24_meses | 32 | 0.7888 |
| lightgbm__seleccionadas | pesos_clase | 36_meses | 598 | 0.7917 |
| lightgbm__seleccionadas | pesos_clase | 48_meses | 30 | 0.7904 |
| lightgbm__seleccionadas | pesos_clase | toda_historia | 30 | 0.7898 |
| lightgbm__seleccionadas | sin_tratamiento | 24_meses | 7 | 0.7886 |
| lightgbm__seleccionadas | sin_tratamiento | 36_meses | 201 | 0.7924 |
| lightgbm__seleccionadas | sin_tratamiento | 48_meses | 9 | 0.7906 |
| lightgbm__seleccionadas | sin_tratamiento | toda_historia | 8 | 0.7897 |
| lightgbm__seleccionadas | smote | 24_meses | 1 | 0.7767 |
| lightgbm__seleccionadas | smote | 36_meses | 33 | 0.7896 |
| lightgbm__seleccionadas | smote | 48_meses | 5 | 0.7902 |
| lightgbm__seleccionadas | smote | toda_historia | 3 | 0.7884 |
| lightgbm__seleccionadas | submuestreo | 24_meses | 5 | 0.7866 |
| lightgbm__seleccionadas | submuestreo | 36_meses | 33 | 0.7905 |
| lightgbm__seleccionadas | submuestreo | 48_meses | 2 | 0.7886 |
| lightgbm__seleccionadas | submuestreo | toda_historia | 3 | 0.7887 |
| logreg__completas | pesos_clase | 24_meses | 38 | 0.7874 |
| logreg__completas | pesos_clase | 36_meses | 756 | 0.7876 |
| logreg__completas | pesos_clase | 48_meses | 37 | 0.7864 |
| logreg__completas | pesos_clase | toda_historia | 31 | 0.7845 |
| logreg__completas | sin_tratamiento | 24_meses | 2 | 0.7865 |
| logreg__completas | sin_tratamiento | 36_meses | 37 | 0.7875 |
| logreg__completas | sin_tratamiento | 48_meses | 6 | 0.7864 |
| logreg__completas | sin_tratamiento | toda_historia | 1 | 0.7844 |
| logreg__completas | smote | 24_meses | 4 | 0.7870 |
| logreg__completas | smote | 36_meses | 40 | 0.7874 |
| logreg__completas | smote | 48_meses | 2 | 0.7846 |
| logreg__completas | smote | toda_historia | 1 | 0.7825 |
| logreg__completas | submuestreo | 24_meses | 2 | 0.7850 |
| logreg__completas | submuestreo | 36_meses | 32 | 0.7855 |
| logreg__completas | submuestreo | 48_meses | 1 | 0.7832 |
| logreg__completas | submuestreo | toda_historia | 10 | 0.7840 |
| logreg__seleccionadas | pesos_clase | 24_meses | 36 | 0.7852 |
| logreg__seleccionadas | pesos_clase | 36_meses | 1 | 0.7851 |
| logreg__seleccionadas | pesos_clase | 48_meses | 6 | 0.7834 |
| logreg__seleccionadas | pesos_clase | toda_historia | 1 | 0.7808 |
| logreg__seleccionadas | sin_tratamiento | 24_meses | 37 | 0.7852 |
| logreg__seleccionadas | sin_tratamiento | 36_meses | 2 | 0.7856 |
| logreg__seleccionadas | sin_tratamiento | 48_meses | 5 | 0.7836 |
| logreg__seleccionadas | sin_tratamiento | toda_historia | 1 | 0.7810 |
| logreg__seleccionadas | smote | 24_meses | 769 | 0.7857 |
| logreg__seleccionadas | smote | 36_meses | 38 | 0.7854 |
| logreg__seleccionadas | smote | 48_meses | 34 | 0.7833 |
| logreg__seleccionadas | smote | toda_historia | 25 | 0.7801 |
| logreg__seleccionadas | submuestreo | 24_meses | 24 | 0.7845 |
| logreg__seleccionadas | submuestreo | 36_meses | 4 | 0.7848 |
| logreg__seleccionadas | submuestreo | 48_meses | 1 | 0.7817 |
| logreg__seleccionadas | submuestreo | toda_historia | 16 | 0.7802 |
| random_forest__completas | pesos_clase | 24_meses | 3 | 0.7863 |
| random_forest__completas | pesos_clase | 36_meses | 2 | 0.7854 |
| random_forest__completas | pesos_clase | 48_meses | 4 | 0.7859 |
| random_forest__completas | pesos_clase | toda_historia | 36 | 0.7866 |
| random_forest__completas | sin_tratamiento | 24_meses | 37 | 0.7867 |
| random_forest__completas | sin_tratamiento | 36_meses | 38 | 0.7866 |
| random_forest__completas | sin_tratamiento | 48_meses | 37 | 0.7868 |
| random_forest__completas | sin_tratamiento | toda_historia | 750 | 0.7869 |
| random_forest__completas | smote | 24_meses | 3 | 0.7852 |
| random_forest__completas | smote | 36_meses | 4 | 0.7812 |
| random_forest__completas | smote | 48_meses | 5 | 0.7860 |
| random_forest__completas | smote | toda_historia | 36 | 0.7860 |
| random_forest__completas | submuestreo | 24_meses | 3 | 0.7836 |
| random_forest__completas | submuestreo | 36_meses | 2 | 0.7821 |
| random_forest__completas | submuestreo | 48_meses | 2 | 0.7851 |
| random_forest__completas | submuestreo | toda_historia | 38 | 0.7852 |
| random_forest__seleccionadas | pesos_clase | 24_meses | 2 | 0.7869 |
| random_forest__seleccionadas | pesos_clase | 36_meses | 5 | 0.7904 |
| random_forest__seleccionadas | pesos_clase | 48_meses | 38 | 0.7910 |
| random_forest__seleccionadas | pesos_clase | toda_historia | 3 | 0.7893 |
| random_forest__seleccionadas | sin_tratamiento | 24_meses | 38 | 0.7887 |
| random_forest__seleccionadas | sin_tratamiento | 36_meses | 36 | 0.7910 |
| random_forest__seleccionadas | sin_tratamiento | 48_meses | 748 | 0.7915 |
| random_forest__seleccionadas | sin_tratamiento | toda_historia | 39 | 0.7904 |
| random_forest__seleccionadas | smote | 24_meses | 2 | 0.7877 |
| random_forest__seleccionadas | smote | 36_meses | 5 | 0.7887 |
| random_forest__seleccionadas | smote | 48_meses | 37 | 0.7901 |
| random_forest__seleccionadas | smote | toda_historia | 3 | 0.7881 |
| random_forest__seleccionadas | submuestreo | 24_meses | 3 | 0.7871 |
| random_forest__seleccionadas | submuestreo | 36_meses | 1 | 0.7877 |
| random_forest__seleccionadas | submuestreo | 48_meses | 39 | 0.7904 |
| random_forest__seleccionadas | submuestreo | toda_historia | 1 | 0.7874 |
| xgboost__completas | pesos_clase | 24_meses | 17 | 0.7861 |
| xgboost__completas | pesos_clase | 36_meses | 15 | 0.7860 |
| xgboost__completas | pesos_clase | 48_meses | 342 | 0.7880 |
| xgboost__completas | pesos_clase | toda_historia | 13 | 0.7854 |
| xgboost__completas | sin_tratamiento | 24_meses | 21 | 0.7855 |
| xgboost__completas | sin_tratamiento | 36_meses | 23 | 0.7867 |
| xgboost__completas | sin_tratamiento | 48_meses | 451 | 0.7882 |
| xgboost__completas | sin_tratamiento | toda_historia | 35 | 0.7863 |
| xgboost__completas | smote | 24_meses | 2 | 0.7724 |
| xgboost__completas | smote | 36_meses | 5 | 0.7827 |
| xgboost__completas | smote | 48_meses | 32 | 0.7867 |
| xgboost__completas | smote | toda_historia | 3 | 0.7834 |
| xgboost__completas | submuestreo | 24_meses | 3 | 0.7851 |
| xgboost__completas | submuestreo | 36_meses | 3 | 0.7828 |
| xgboost__completas | submuestreo | 48_meses | 33 | 0.7852 |
| xgboost__completas | submuestreo | toda_historia | 2 | 0.7838 |
| xgboost__seleccionadas | pesos_clase | 24_meses | 6 | 0.7881 |
| xgboost__seleccionadas | pesos_clase | 36_meses | 8 | 0.7902 |
| xgboost__seleccionadas | pesos_clase | 48_meses | 60 | 0.7921 |
| xgboost__seleccionadas | pesos_clase | toda_historia | 1 | 0.7681 |
| xgboost__seleccionadas | sin_tratamiento | 24_meses | 28 | 0.7889 |
| xgboost__seleccionadas | sin_tratamiento | 36_meses | 29 | 0.7903 |
| xgboost__seleccionadas | sin_tratamiento | 48_meses | 739 | 0.7930 |
| xgboost__seleccionadas | sin_tratamiento | toda_historia | 39 | 0.7896 |
| xgboost__seleccionadas | smote | 24_meses | 8 | 0.7889 |
| xgboost__seleccionadas | smote | 36_meses | 5 | 0.7885 |
| xgboost__seleccionadas | smote | 48_meses | 28 | 0.7902 |
| xgboost__seleccionadas | smote | toda_historia | 4 | 0.7883 |
| xgboost__seleccionadas | submuestreo | 24_meses | 2 | 0.7873 |
| xgboost__seleccionadas | submuestreo | 36_meses | 4 | 0.7893 |
| xgboost__seleccionadas | submuestreo | 48_meses | 38 | 0.7898 |
| xgboost__seleccionadas | submuestreo | toda_historia | 1 | 0.7529 |

## Candidatos individuales

| estudio | algoritmo | conjunto | numero_variables | estrategia | ventana | trials_completos | trials_comparados | mejor_trial | auc_promedio | desviacion_auc | lift_promedio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logreg__seleccionadas | logreg | seleccionadas | 6 | smote | 24_meses | 1000 | 1000 | 729 | 0.7857 | 0.0097 | 2.1488 |
| logreg__completas | logreg | completas | 42 | pesos_clase | 36_meses | 1000 | 1000 | 525 | 0.7876 | 0.0077 | 2.2577 |
| random_forest__seleccionadas | random_forest | seleccionadas | 6 | sin_tratamiento | 48_meses | 1000 | 1000 | 587 | 0.7915 | 0.0092 | 2.3187 |
| random_forest__completas | random_forest | completas | 42 | sin_tratamiento | toda_historia | 1000 | 1000 | 91 | 0.7869 | 0.0092 | 2.2903 |
| xgboost__seleccionadas | xgboost | seleccionadas | 6 | sin_tratamiento | 48_meses | 1000 | 1000 | 363 | 0.7930 | 0.0098 | 2.3025 |
| xgboost__completas | xgboost | completas | 42 | sin_tratamiento | 48_meses | 1000 | 1000 | 638 | 0.7882 | 0.0079 | 2.2421 |
| lightgbm__seleccionadas | lightgbm | seleccionadas | 6 | sin_tratamiento | 36_meses | 1000 | 1000 | 822 | 0.7924 | 0.0085 | 2.3058 |
| lightgbm__completas | lightgbm | completas | 42 | pesos_clase | 36_meses | 1000 | 1000 | 972 | 0.7880 | 0.0083 | 2.2567 |
| catboost__seleccionadas | catboost | seleccionadas | 6 | smote | 48_meses | 1000 | 1000 | 886 | 0.7912 | 0.0095 | 2.2256 |
| catboost__completas | catboost | completas | 42 | sin_tratamiento | 36_meses | 1000 | 1000 | 378 | 0.7880 | 0.0087 | 2.2774 |

## Selección entre modelos individuales

| estudio | algoritmo | conjunto | numero_variables | estrategia | ventana | trials_completos | trials_comparados | mejor_trial | auc_promedio | desviacion_auc | lift_promedio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| xgboost__seleccionadas | xgboost | seleccionadas | 6 | sin_tratamiento | 48_meses | 1000 | 1000 | 363 | 0.7930 | 0.0098 | 2.3025 |
| lightgbm__seleccionadas | lightgbm | seleccionadas | 6 | sin_tratamiento | 36_meses | 1000 | 1000 | 822 | 0.7924 | 0.0085 | 2.3058 |
| random_forest__seleccionadas | random_forest | seleccionadas | 6 | sin_tratamiento | 48_meses | 1000 | 1000 | 587 | 0.7915 | 0.0092 | 2.3187 |
| catboost__seleccionadas | catboost | seleccionadas | 6 | smote | 48_meses | 1000 | 1000 | 886 | 0.7912 | 0.0095 | 2.2256 |
| xgboost__completas | xgboost | completas | 42 | sin_tratamiento | 48_meses | 1000 | 1000 | 638 | 0.7882 | 0.0079 | 2.2421 |
| lightgbm__completas | lightgbm | completas | 42 | pesos_clase | 36_meses | 1000 | 1000 | 972 | 0.7880 | 0.0083 | 2.2567 |
| catboost__completas | catboost | completas | 42 | sin_tratamiento | 36_meses | 1000 | 1000 | 378 | 0.7880 | 0.0087 | 2.2774 |
| logreg__completas | logreg | completas | 42 | pesos_clase | 36_meses | 1000 | 1000 | 525 | 0.7876 | 0.0077 | 2.2577 |
| random_forest__completas | random_forest | completas | 42 | sin_tratamiento | toda_historia | 1000 | 1000 | 91 | 0.7869 | 0.0092 | 2.2903 |
| logreg__seleccionadas | logreg | seleccionadas | 6 | smote | 24_meses | 1000 | 1000 | 729 | 0.7857 | 0.0097 | 2.1488 |

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

Configuración: /home/carlos/projects/glamour/customer-churn-prediction/sem2_draft/04_modelado/resultados_v2/b2ad959cb4c424b4/modelo_final.json.

Pipeline entrenado sobre desarrollo: /home/carlos/projects/glamour/customer-churn-prediction/sem2_draft/04_modelado/resultados_v2/b2ad959cb4c424b4/modelo_final.joblib.

Estudios y trials: /home/carlos/projects/glamour/customer-churn-prediction/sem2_draft/04_modelado/resultados_v2/b2ad959cb4c424b4/optuna.db y /home/carlos/projects/glamour/customer-churn-prediction/sem2_draft/04_modelado/resultados_v2/b2ad959cb4c424b4/trials.

Estas métricas se usaron para seleccionar configuraciones; no son una estimación independiente del desempeño final. El OOT no se evaluó en esta corrida. Las evaluaciones históricas del OOT están documentadas en AVANCE.md. Queda pendiente evaluar el artefacto v2 y estimar incertidumbre en la fase 05.
