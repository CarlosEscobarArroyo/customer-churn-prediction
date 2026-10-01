# Aporte de sexo, edad, antigüedad y ubicación al ensemble

La base con **42 variables transaccionales** obtuvo el mayor AUC temporal medio: **0,78988**. Ninguna de las ocho ampliaciones la superó con los parámetros fijos de esta comparación. Con las cinco variables nuevas juntas, el AUC fue **0,78572** (cambio: **−0,00416**).

## Comparación controlada

| variant                   |   auc_mean |   delta_auc |   ap_mean |   precision_top10 |   recall_top10 |   lift_top10 |
|:--------------------------|-----------:|------------:|----------:|------------------:|---------------:|-------------:|
| transaccional             |    0.78988 |     0.00000 |   0.59621 |           0.68445 |        0.21901 |      2.25675 |
| mas_sexo                  |    0.78983 |    -0.00005 |   0.59590 |           0.68677 |        0.21975 |      2.26490 |
| mas_antiguedad            |    0.78973 |    -0.00015 |   0.59523 |           0.68213 |        0.21826 |      2.25065 |
| mas_demografia_antiguedad |    0.78899 |    -0.00089 |   0.59371 |           0.67749 |        0.21678 |      2.23063 |
| mas_edad                  |    0.78893 |    -0.00095 |   0.59493 |           0.67749 |        0.21678 |      2.22598 |
| mas_departamento          |    0.78892 |    -0.00096 |   0.58771 |           0.67749 |        0.21678 |      2.22507 |
| mas_provincia             |    0.78658 |    -0.00330 |   0.58421 |           0.68910 |        0.22049 |      2.25560 |
| mas_ubicacion             |    0.78648 |    -0.00340 |   0.58302 |           0.67981 |        0.21752 |      2.21958 |
| mas_todas                 |    0.78572 |    -0.00416 |   0.58136 |           0.66821 |        0.21381 |      2.17542 |

Todas las variantes usan las mismas 42 variables transaccionales, logística con 36 meses,
XGBoost con toda la historia permitida y promedio 50/50. Los parámetros se fijan antes
de la ablación; no se retunean por variante. Las columnas de campañas no se incorporan.
Origen de parámetros: `supplied`; consultar `parameter_source.json`.
La búsqueda reconstruida terminó 100 trials de logística (77 completos y 23 podados) y 85 de XGBoost (45 completos y 40 podados). Se interrumpió intencionalmente el siguiente trial de XGBoost para priorizar la comparación; ese trial no participó en la selección. Se exportó el mejor trial completo con las 42 variables para cada algoritmo.
No es necesario ejecutar 100 trials para esta comparación: basta con parámetros fijos
y una base evaluada en los mismos períodos. La búsqueda previa se conserva como antecedente.

Cuatro bloques de validación de cuatro meses entre octubre de 2023 y enero de 2025,
gap de seis meses y 4379 observaciones vendedora-mes. El corte se fija por fecha;
no se desplaza con los meses adicionales del CSV actual. No se evalúa el OOT.

## AUC por bloque

| variant                   |       0 |       1 |       2 |       3 |
|:--------------------------|--------:|--------:|--------:|--------:|
| mas_antiguedad            | 0.78564 | 0.79449 | 0.78188 | 0.79692 |
| mas_demografia_antiguedad | 0.78521 | 0.79440 | 0.78141 | 0.79495 |
| mas_departamento          | 0.78568 | 0.79224 | 0.78256 | 0.79520 |
| mas_edad                  | 0.78454 | 0.79480 | 0.78124 | 0.79515 |
| mas_provincia             | 0.78050 | 0.78864 | 0.78138 | 0.79582 |
| mas_sexo                  | 0.78584 | 0.79483 | 0.78192 | 0.79674 |
| mas_todas                 | 0.78101 | 0.78846 | 0.78081 | 0.79261 |
| mas_ubicacion             | 0.78106 | 0.78930 | 0.78133 | 0.79423 |
| transaccional             | 0.78535 | 0.79493 | 0.78187 | 0.79737 |

## Resultado de cada componente

| variant                   |   logreg_36m |   xgboost_all |
|:--------------------------|-------------:|--------------:|
| transaccional             |      0.78763 |       0.78834 |
| mas_sexo                  |      0.78758 |       0.78815 |
| mas_edad                  |      0.78628 |       0.78790 |
| mas_antiguedad            |      0.78769 |       0.78808 |
| mas_departamento          |      0.78582 |       0.78815 |
| mas_provincia             |      0.77792 |       0.78829 |
| mas_ubicacion             |      0.77770 |       0.78833 |
| mas_demografia_antiguedad |      0.78630 |       0.78786 |
| mas_todas                 |      0.77666 |       0.78769 |

La caída al añadir ubicación se concentra en logística: de **0,78763** a **0,77770** al añadir departamento y provincia. XGBoost se mantiene prácticamente igual: **0,78834** frente a **0,78833**. Esta prueba mantiene la regularización y los otros parámetros fijos; no demuestra que esas variables sean inútiles bajo cualquier configuración.

Las métricas de contacto no cambian todas en el mismo sentido: sexo mejora ligeramente el lift top 10 % (2,26490× frente a 2,25675×), aunque no el AUC. La elección de esta comparación sigue el criterio original de AUC temporal medio; no se ha demostrado significancia estadística de esas diferencias.

## Tratamiento y límites

- Edad y antigüedad: mediana aprendida exclusivamente en cada entrenamiento e indicador
  explícito de dato ausente. Edad se refiere al mes observado, a partir de fecha de nacimiento;
  antigüedad es meses desde la fecha de ingreso registrada.
- Sexo, departamento y provincia: normalización de espacios, mayúsculas y tildes;
  one-hot aprendido en entrenamiento, categorías nuevas ignoradas. Faltantes tienen categoría propia.
- La imputación, vocabulario y escalado se ajustan por separado para cada modelo y fold.
- Ubicación es departamento/provincia del snapshot actual. Una mejora retrospectiva no
  acredita que esa ubicación estuviera disponible en la fecha histórica. Lo mismo exige
  verificar la calidad de sexo, nacimiento e ingreso antes de una decisión operativa.
- Son métricas de selección sobre validaciones reutilizadas. Las pequeñas diferencias no
  demuestran superioridad estadística ni equivalen a una evaluación final independiente.
- La referencia histórica de AUC 0,78961 procede de otra corrida. La comparación del aporte
  de añadir variables se realiza contra `transaccional` de esta misma ejecución.

## Calidad de los datos de desarrollo

| variable         |   missing |   faltantes_pct |
|:-----------------|----------:|----------------:|
| edad             |     16381 |           57.86 |
| antiguedad_meses |        88 |            0.31 |
| sexo             |         0 |            0    |
| departamento     |        30 |            0.11 |
| provincia        |        30 |            0.11 |

## Verificación

Pasaron 18 pruebas automatizadas. Se recalcularon los AUC desde las predicciones guardadas, se verificó la alineación de las 4.379 filas entre las nueve variantes y el promedio 50/50. El hash del candidato coincide con su manifiesto y su inferencia funciona al cargarlo en un proceso nuevo.

## Artefactos

En `reports/master_features_comparison_v1`: protocolo con hashes/versiones,
parámetros, perfil de faltantes, fechas de folds, predicciones por variante/fold,
`comparison.csv`, `block_metrics.csv`, `monthly_metrics.csv`, `candidate.json` y
`candidate.joblib`. El candidato se guarda por separado del ensemble histórico y su recarga
se comprueba. Las métricas de precisión y recall son agregadas; el lift es promedio mensual.

Ejecución y análisis: [notebook 08](../../05_modelling/08_ablacion_datos_maestros.ipynb).
